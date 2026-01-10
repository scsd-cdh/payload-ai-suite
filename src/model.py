"""Resnet50 base model, 2 classes. First layers removed

Features:
- train
- validate
- export to onnx
- mixed resolution operations for resolution-agnostic training

TODO:
- optimize
- more data!

"""

import os
import tensorflow as tf
import keras
import preprocess as preprocess
import tf2onnx
import onnx
import numpy as np
import onnxruntime as rt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import logging
from mlops import GCSHandler
from metrics import ModelEvaluator
from paths import resolve_path, get_model_path
from mixed_res_config import DEFAULT_MIXED_RES_CONFIG
import experiment_tracker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train(validate=True, epochs=50, use_nir=False, use_gcs=False,
          use_mixed_res=False, mixed_res_config=None, fusion_technique='enhanced_red',
          fusion_alpha=0.5, degrade_gsd=False):
    """
    Train CNN ResNet50 model on labeled data.

    Args:
        validate (bool): Whether to validate the model after training.
        epochs (int): Number of epochs to train the model.
        use_nir (bool): Whether to use NIR channel.
        use_gcs (bool): Whether to stream data from GCS.
        use_mixed_res (bool): Whether to apply mixed resolution operations during training.
        mixed_res_config (dict): Configuration for mixed resolution operations.
        fusion_technique (str): RGB-NIR fusion technique ('enhanced_red', 'hsv', 'none').
        fusion_alpha (float): Alpha parameter for enhanced_red fusion.
        degrade_gsd (bool): Whether to degrade imagery to CubeSat GSD (~85m from 10m).

    Returns:
        str: Experiment ID for this training run
    """
    # Generate experiment ID
    experiment_id = experiment_tracker.generate_experiment_id()
    logger.info(f"Starting training experiment: {experiment_id}")

    X = []
    y = []

    # Use default config if none provided
    if use_mixed_res and mixed_res_config is None:
        mixed_res_config = DEFAULT_MIXED_RES_CONFIG

    if use_gcs:
        # Stream images from GCS
        try:
            gcs = GCSHandler()

            # Pass GCS handler and mixed_res options to populate function
            X, y = preprocess.populate(X, y, "labeled/yes", use_nir=use_nir,
                          gcs_handler=gcs, use_mixed_res=use_mixed_res,
                          mixed_res_config=mixed_res_config,
                          fusion_technique=fusion_technique,
                          fusion_alpha=fusion_alpha,
                          degrade_gsd=degrade_gsd)
            X, y = preprocess.populate(X, y, "labeled/no", use_nir=use_nir,
                          end=True, gcs_handler=gcs,
                          use_mixed_res=use_mixed_res,
                          mixed_res_config=mixed_res_config,
                          fusion_technique=fusion_technique,
                          fusion_alpha=fusion_alpha,
                          degrade_gsd=degrade_gsd)

        except Exception as e:
            logger.error(f"Failed to load data from GCS: {str(e)}")
            raise
    else:
        # Use local files with mixed resolution options
        X, y = preprocess.populate(X, y, resolve_path("data/labeled/yes"), use_nir=use_nir,
                      use_mixed_res=use_mixed_res,
                      mixed_res_config=mixed_res_config,
                      fusion_technique=fusion_technique,
                      fusion_alpha=fusion_alpha,
                      degrade_gsd=degrade_gsd)
        X, y = preprocess.populate(X, y, resolve_path("data/labeled/no"), use_nir=use_nir,
                      end=True, use_mixed_res=use_mixed_res,
                      mixed_res_config=mixed_res_config,
                      fusion_technique=fusion_technique,
                      fusion_alpha=fusion_alpha,
                      degrade_gsd=degrade_gsd)

    # TODO: Use numpy instead here
    X = [X[i] for i in range(min(len(X), len(y)))]
    y = [y[i] for i in range(min(len(X), len(y)))]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    label_encoder = preprocessing.LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    # Should this be just fit instead of fit_transform?
    y_test = label_encoder.fit_transform(y_test)

    # Compute balanced class weights
    class_weights_array = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )

    y_train = np.array(y_train)
    X_train = np.array(X_train)
    y_test = np.array(y_test)
    X_test = np.array(X_test)

    y_train = tf.keras.utils.to_categorical(y_train, num_classes=2)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=2)

    class_weights = {i: weight for i, weight in enumerate(class_weights_array)}
    logger.info(f"Computed class weights: {class_weights}")

    input_channels = 4 if use_nir else 3
    input_shape = (224, 224, input_channels)

    weights = 'imagenet' if input_channels == 3 else None

    resnet50 = tf.keras.applications.resnet.ResNet50(
        weights=weights,
        include_top=False,
        input_shape=input_shape
    )

    # Here we freeze the last 4 layers
    # Layers are set to trainable as True by default
    for layer in resnet50.layers:
        layer.trainable = False

    for (i, layer) in enumerate(resnet50.layers):
        logger.info(f"{i} {layer.__class__.__name__} {layer.trainable}")

    def create_top(bottom_model, num_classes):
        top_model = bottom_model.output
        top_model = keras.layers.GlobalAveragePooling2D()(top_model)
        top_model = keras.layers.Dense(1024, activation='relu')(top_model)
        top_model = keras.layers.Dropout(0.2)(top_model)
        top_model = keras.layers.Dense(1024, activation='relu')(top_model)
        top_model = keras.layers.Dense(512, activation='relu')(top_model)
        output = keras.layers.Dense(num_classes, activation='softmax')(top_model)
        return output

    num_classes = 2
    head = create_top(resnet50, num_classes)
    model = keras.models.Model(inputs=resnet50.input, outputs=head)

    model.summary(print_fn=logger.info)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )

    checkpoint_path = "training_checkpoints/cp.weights.h5"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    latest = tf.train.latest_checkpoint(checkpoint_dir)
    if latest:
        model.load_weights(latest)
        logger.info("Loaded weights from checkpoint.")

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )

    history = model.fit(X_train, y_train,
                        epochs=50,
                        validation_data=(X_test, y_test),
                        verbose=1,
                        initial_epoch=0,
                        shuffle=True,
                        class_weight=class_weights,
                        callbacks=[cp_callback, early_stopping, reduce_lr])

    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    if validate:
        test_loss, test_accuracy = model.evaluate(X_test, y_test)
        logger.info(f'Test accuracy: {test_accuracy:.4f}')
        epochs_range = range(len(accuracy))
        plt.plot(epochs_range, accuracy, 'r', label='Training accuracy')
        plt.plot(epochs_range, val_accuracy, 'b', label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.legend(loc=0)

        exp_dir = experiment_tracker.create_experiment_directory(experiment_id)
        plot_filename = os.path.join(exp_dir, "training_plot.png")
        plt.savefig(plot_filename)
        # plt.close()

        model_filename = os.path.join(exp_dir, f"model_{experiment_id}.onnx")
        export_to_onnx(model, model_filename)

        evaluator = ModelEvaluator(experiment_id, exp_dir)


        evaluation_metrics = evaluator.evaluate_and_save_all(X_test, y_test, model)

        experiment_tracker.save_experiment_config(
            experiment_id,
            {
                "use_nir": use_nir,
                "dataset": {"total_samples": len(X), "train_samples": len(X_train), "test_samples": len(X_test), "source": "gcs" if use_gcs else "local"},
                "mixed_resolution": {"enabled": use_mixed_res, **(mixed_res_config if use_mixed_res else {})},
                "rgb_nir_fusion": {"technique": fusion_technique, "alpha": fusion_alpha},
                "gsd_degradation": degrade_gsd,
                "normalization": "dyn_zscore"
            },
            {
                "base_model": resnet50.__class__.__name__,
                "input_shape": list(model.input_shape[1:]),
                "pretrained_weights": weights
            },
            {
                "epochs": len(accuracy),
                "optimizer": model.optimizer.__class__.__name__.lower(),
                "loss": model.loss,
                "train_test_split": len(X_test) / len(X)
            }
        )

        experiment_tracker.save_experiment_results(
            experiment_id,
            {
                "final_train_accuracy": float(accuracy[-1]),
                "final_val_accuracy": float(val_accuracy[-1]),
                "final_train_loss": float(loss[-1]),
                "final_val_loss": float(val_loss[-1]),
                "best_val_accuracy": float(max(val_accuracy)),
                "best_val_accuracy_epoch": int(val_accuracy.index(max(val_accuracy))),
                "test_accuracy": float(test_accuracy),
                "test_loss": float(test_loss)
            },
            model_filename,
            plot_filename,
            evaluation_metrics
        )

        logger.info(f"Experiment {experiment_id} saved to {exp_dir}")


    return experiment_id

def export_to_onnx(model, filename=None):
    """
    Export the trained model to ONNX format.

    Args:
        model (tf.keras.Model): The trained Keras model to be exported.
        filename (str): Path to save the ONNX model. If None, uses default path.

    Returns:
        None
    """
    if filename is None:
        filename = get_model_path('zetane.onnx')

    input_signature = [
        tf.TensorSpec(
            shape=model.input_shape,
            dtype=tf.float32,
            name='input'
        )
    ]
    onnx_model, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=input_signature,
        opset=13
    )
    onnx.save(onnx_model, filename)

def run_inference(onnx_model=None, data_target=None):
    if onnx_model is None:
        onnx_model = get_model_path('zetane.onnx')
    if not data_target.any():
        logger.error("Please provide a test target")
        return
    session = rt.InferenceSession(onnx_model, providers=rt.get_available_providers)
    input_name = session.get_inputs()[0].name
    prediction_onnx = session.run(None, {input_name: data_target.astype(np.float32)})[0]
    logger.info(f"Prediction: {prediction_onnx}")