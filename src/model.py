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
import matplotlib.pyplot as plt
import logging
from mlops import GCSHandler
from paths import resolve_path, get_model_path
from mixed_res_config import DEFAULT_MIXED_RES_CONFIG
import experiment_tracker
import keras_tuner as kt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train(validate=True, epochs=12, use_nir=False, use_gcs=False,
          use_mixed_res=False, mixed_res_config=None, fusion_technique='enhanced_red',
          fusion_alpha=0.5, degrade_gsd=False, tune_first=True):
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
    y_test = label_encoder.fit_transform(y_test)

    y_train = tf.keras.utils.to_categorical(y_train, num_classes=2)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=2)

    y_train = np.array(y_train)
    X_train = np.array(X_train)
    y_test = np.array(y_test)
    X_test = np.array(X_test)

    logger.info(f"Shape of an image in X_train: {X_train.shape}")
    logger.info(f"Shape of an image in X_test: {X_test.shape}")
    logger.info(f"y_train Shape: {y_train.shape}")
    logger.info(f"y_test Shape: {y_test.shape}")

    if tune_first:
        # run tuning to find the best hyperparameters
        logger.info("--- STARTING HYPERPARAMETER TUNING ---")
        best_hps = tune_hyperparameters(
            X_train, y_train, X_test, y_test,
            use_nir=use_nir,
            epochs_per_trial=10,  # Quick search
            max_trials=15
        )
        
        # build the final model using the best hps
        logger.info("--- BUILDING FINAL MODEL WITH OPTIMAL HYPERPARAMETERS ---")
        model, weights, base_model_name = build_hypermodel(best_hps, use_nir=use_nir)
        
    else:
        # build model with the default hps
        logger.info("--- BUILDING MODEL WITH DEFAULT HYPERPARAMETERS (Tuning skipped) ---")
        default_hps = kt.HyperParameters() 
        model, weights, base_model_name = build_hypermodel(default_hps, use_nir=use_nir)

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

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

    history = model.fit(X_train, y_train,
                        epochs=epochs,
                        validation_data=(X_test, y_test),
                        verbose=1,
                        initial_epoch=0,
                        shuffle=True,
                        callbacks=[cp_callback, early_stopping],
                        )

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

        # Save experiment
        exp_dir = experiment_tracker.create_experiment_directory(experiment_id)
        plot_filename = os.path.join(exp_dir, "training_plot.png")
        plt.savefig(plot_filename)

        model_filename = os.path.join(exp_dir, f"model_{experiment_id}.onnx")
        export_to_onnx(model, model_filename)

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
                "base_model": base_model_name,
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
            plot_filename
        )

        logger.info(f"Experiment {experiment_id} saved to {exp_dir}")

        # plt.figure()
        # plt.show()

    return experiment_id

def build_hypermodel(hp, use_nir: bool):
    """
    Builds a hypermodel for Keras Tuner.
    
    This function defines the search space for hyperparameters.
    
    Args:
        hp (kerastuner.HyperParameters): The hyperparameter object.
        use_nir (bool): Whether to use the NIR channel (affects input shape).

    Returns:
        (tf.keras.Model, str, str): A compiled Keras model, the weights string, and the base model name.
    """
    input_channels = 4 if use_nir else 3
    input_shape = (224, 224, input_channels)
    
    # Use 'imagenet' weights only if we have 3 channels, otherwise start from scratch
    weights = 'imagenet' if input_channels == 3 else None
    base_model_name = 'ResNet50'

    resnet50 = tf.keras.applications.resnet.ResNet50(
        weights=weights,
        include_top=False,
        input_shape=input_shape
    )

    # Start with all layers frozen
    for layer in resnet50.layers:
        layer.trainable = False

    # Hyperparameter: Number of ResNet blocks to unfreeze (from the end)
    # 0 = all frozen, 1 = unfreeze 'conv5', 2 = unfreeze 'conv4' and 'conv5'
    hp_unfreeze_blocks = hp.Choice('unfreeze_blocks', values=[0, 1, 2])

    if hp_unfreeze_blocks > 0:
        for layer in resnet50.layers:
            if layer.name.startswith('conv5'):
                layer.trainable = True
    if hp_unfreeze_blocks > 1:
        for layer in resnet50.layers:
            if layer.name.startswith('conv4'):
                layer.trainable = True
                
    # Build the classification head 
    top_model = resnet50.output
    top_model = keras.layers.GlobalAveragePooling2D()(top_model)

    # Hyperparameter: Units in the first Dense layer
    hp_units_1 = hp.Choice('units_1', values=[512, 1024])
    top_model = keras.layers.Dense(hp_units_1, activation='relu')(top_model)
    
    # Hyperparameter: Dropout rate for the first Dense layer
    hp_dropout_1 = hp.Float('dropout_1', min_value=0.3, max_value=0.6, step=0.1)
    top_model = keras.layers.Dropout(hp_dropout_1)(top_model)

    # Hyperparameter: Units in the second Dense layer
    hp_units_2 = hp.Choice('units_2', values=[512, 1024])
    top_model = keras.layers.Dense(hp_units_2, activation='relu')(top_model)

    # Hyperparameter: Dropout rate for the second Dense layer
    hp_dropout_2 = hp.Float('dropout_2', min_value=0.3, max_value=0.6, step=0.1)
    top_model = keras.layers.Dropout(hp_dropout_2)(top_model)
    
    # Hyperparameter: Units in the third Dense layer
    hp_units_3 = hp.Choice('units_3', values=[256, 512])
    top_model = keras.layers.Dense(hp_units_3, activation='relu')(top_model)
    
    # Hyperparameter: Dropout rate for the third Dense layer
    hp_dropout_3 = hp.Float('dropout_3', min_value=0.2, max_value=0.4, step=0.1)
    top_model = keras.layers.Dropout(hp_dropout_3)(top_model)
    
    # Output layer
    num_classes = 2
    output = keras.layers.Dense(num_classes, activation='softmax')(top_model)
    
    model = keras.models.Model(inputs=resnet50.input, outputs=output)

    # Hyperparameter: Learning rate
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-3, 1e-4, 1e-5])
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.summary(print_fn=logger.info)

    return model, weights, base_model_name

def tune_hyperparameters(X_train, y_train, X_val, y_val, use_nir: bool, 
                         epochs_per_trial=10, max_trials=15):
    """
    Runs the Keras Tuner search to find the best hyperparameters.

    Args:
        X_train, y_train: Training data and labels.
        X_val, y_val: Validation data and labels.
        use_nir (bool): Whether to use the NIR channel.
        epochs_per_trial (int): Number of epochs to train each model during search.
        max_trials (int): Total number of hyperparameter combinations to test.

    Returns:
        kerastuner.HyperParameters: The set of best performing hyperparameters.
    """
    # We use a lambda to pass the `use_nir` argument to our model builder
    model_builder = lambda hp: build_hypermodel(hp, use_nir=use_nir)[0] 

    tuner = kt.RandomSearch(
        model_builder,
        objective='val_accuracy',
        max_trials=max_trials,
        executions_per_trial=1,  # Train each model once
        directory='hyperparam_tuning',
        project_name='resnet50_classification'
    )

    # Add an EarlyStopping callback to speed up the search
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

    logger.info(f"Starting hyperparameter tuning... (max_trials={max_trials}, epochs_per_trial={epochs_per_trial})")
    
    tuner.search(
        X_train, y_train,
        epochs=epochs_per_trial,
        validation_data=(X_val, y_val),
        callbacks=[stop_early],
    )

    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    logger.info(f"""
    Optimal hyperparameters found:
    - Learning Rate: {best_hps.get('learning_rate')}
    - Unfreeze Blocks: {best_hps.get('unfreeze_blocks')}
    - Units 1: {best_hps.get('units_1')}
    - Dropout 1: {best_hps.get('dropout_1'):.2f}
    - Units 2: {best_hps.get('units_2')}
    - Dropout 2: {best_hps.get('dropout_2'):.2f}
    - Units 3: {best_hps.get('units_3')}
    - Dropout 3: {best_hps.get('dropout_3'):.2f}
    """)

    return best_hps

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