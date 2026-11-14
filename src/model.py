"""Resnet50 base model, 3 classes (fire / smoke / neither). First layers removed

Features:
- train
- validate
- export to onnx
- predict with an exported onnx model
- mixed resolution operations for resolution-agnostic training

TODO:
- optimize
- more data!

"""

import os
import json
import cv2
import tensorflow as tf
import keras
import preprocess as preprocess
import tf2onnx
import onnx
import numpy as np
import onnxruntime as rt
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

# Model output order: softmax index i is the probability of CLASS_NAMES[i].
CLASS_NAMES = ["fire", "smoke", "neither"]

# Where each class comes from: (labeled subfolder, include glob, exclude globs, class).
# labeled/yes holds both fire images and smoke_* images; labeled/no holds
# no-fire images, including haze_* (hazy but no smoke), which are all 'neither'.
DATA_SOURCES = [
    ("labeled/yes", None, ["smoke_*", "haze_*"], "fire"),
    ("labeled/yes", "smoke_*", None, "smoke"),
    ("labeled/no", None, None, "neither"),
]

def train(validate=True, epochs=50, use_nir=False, use_gcs=False,
          use_mixed_res=False, mixed_res_config=None, fusion_technique='enhanced_red',
          fusion_alpha=0.5, degrade_gsd=False, use_class_weights=False, seed=42):
    """
    Train a CNN ResNet50 model to classify images as fire, smoke or neither.

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
        use_class_weights (bool): Whether to use balanced class weights during training.
        seed (int): Seed for training randomness (weight init, shuffling, augmentation),
            so the same settings reproduce the same model. The train/test split
            always uses a fixed seed so every run is tested on the same images.

    Returns:
        str: Experiment ID for this training run
    """
    # Generate experiment ID
    experiment_id = experiment_tracker.generate_experiment_id()
    logger.info(f"Starting training experiment: {experiment_id} (seed={seed})")

    # Seeds Python, NumPy and TensorFlow; op determinism makes TF results repeatable
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()

    X = []
    y = []
    sources = []  # file path of each image, used for the per-group report

    # Use default config if none provided
    if use_mixed_res and mixed_res_config is None:
        mixed_res_config = DEFAULT_MIXED_RES_CONFIG

    try:
        # Stream images from GCS, or use local files
        gcs = GCSHandler() if use_gcs else None

        for subdir, pattern, exclude, label in DATA_SOURCES:
            path = subdir if use_gcs else resolve_path(f"data/{subdir}")
            X, y = preprocess.populate(X, y, path, label, use_nir=use_nir,
                          gcs_handler=gcs,
                          use_mixed_res=use_mixed_res,
                          mixed_res_config=mixed_res_config,
                          fusion_technique=fusion_technique,
                          fusion_alpha=fusion_alpha,
                          degrade_gsd=degrade_gsd,
                          pattern=pattern, exclude=exclude,
                          sources=sources)

    except Exception as e:
        if use_gcs:
            logger.error(f"Failed to load data from GCS: {str(e)}")
        raise

    num_classes = len(CLASS_NAMES)
    class_counts = {name: y.count(name) for name in CLASS_NAMES}
    logger.info(f"Images per class: {class_counts}")
    missing = [name for name, count in class_counts.items() if count < 2]
    if missing:
        raise ValueError(f"Not enough training images for class(es) {missing} "
                         f"(need at least 2 each, got {class_counts})")

    # Explicit label -> index mapping so output order always matches CLASS_NAMES
    y = np.array([CLASS_NAMES.index(label) for label in y])
    X = np.array(X)

    X_train, X_test, y_train, y_test, _, sources_test = train_test_split(
        X, y, sources, test_size=0.33, random_state=42, stratify=y)

    # Compute balanced class weights
    class_weights_array = compute_class_weight(
        class_weight='balanced',
        classes=np.arange(num_classes),
        y=y_train
    )

    y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)

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

    head = create_top(resnet50, num_classes)
    model = keras.models.Model(inputs=resnet50.input, outputs=head)

    model.summary(print_fn=logger.info)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy'] + [
            metric(class_id=i, name=f"{metric_name}_{name}")
            for i, name in enumerate(CLASS_NAMES) if name != "neither"
            for metric, metric_name in ((tf.keras.metrics.Precision, "precision"),
                                        (tf.keras.metrics.Recall, "recall"))
        ]
    )

    checkpoint_path = "training_checkpoints/multiclass/cp.weights.h5"
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
                        epochs=epochs,
                        validation_data=(X_test, y_test),
                        verbose=1,
                        initial_epoch=0,
                        shuffle=True,
                        class_weight=class_weights if use_class_weights else None,
                        callbacks=[cp_callback, early_stopping, reduce_lr])

    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    if validate:
        evaluation_results = model.evaluate(X_test, y_test)
        test_loss = evaluation_results[0]
        test_accuracy = evaluation_results[1]
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


        evaluation_metrics = evaluator.evaluate_and_save_all(
            X_test, y_test, model, sources=sources_test, class_names=CLASS_NAMES)

        experiment_tracker.save_experiment_config(
            experiment_id,
            {
                "class_names": CLASS_NAMES,
                "use_nir": use_nir,
                "dataset": {"total_samples": len(X), "train_samples": len(X_train), "test_samples": len(X_test), "class_counts": class_counts, "source": "gcs" if use_gcs else "local"},
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
                "train_test_split": len(X_test) / len(X),
                "seed": seed
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
    session = rt.InferenceSession(onnx_model, providers=rt.get_available_providers())
    input_name = session.get_inputs()[0].name
    prediction_onnx = session.run(None, {input_name: data_target.astype(np.float32)})[0]
    logger.info(f"Prediction: {prediction_onnx}")

def _load_experiment_config(onnx_model):
    """Read the config.json saved next to an exported model, if there is one."""
    config_path = os.path.join(os.path.dirname(os.path.abspath(onnx_model)), "config.json")
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r") as f:
        return json.load(f)


def predict(image_path, onnx_model, fusion_technique='enhanced_red', fusion_alpha=0.5):
    """
    Classify one image, or every image in a directory, as fire, smoke or neither.

    Uses the same preprocessing as training (preprocess.prepare_for_model). The
    fusion settings and class names are read from the experiment's config.json
    next to the model when available; otherwise the arguments and CLASS_NAMES
    are used. NIR mode is inferred from the model's input channel count.

    Args:
        image_path (str): Image file or directory of images.
        onnx_model (str): Path to the exported ONNX model.
        fusion_technique (str): Fallback RGB-NIR fusion technique.
        fusion_alpha (float): Fallback alpha for enhanced_red fusion.

    Returns:
        list[dict]: One result per image with path, predicted class, its
        confidence and the probability of every class.
    """
    if os.path.isdir(image_path):
        paths = sorted(
            os.path.join(image_path, f) for f in os.listdir(image_path)
            if preprocess.is_image_file(f)
        )
    else:
        paths = [image_path]

    config = _load_experiment_config(onnx_model)
    preprocessing_config = config.get("preprocessing", {})
    fusion = preprocessing_config.get("rgb_nir_fusion", {})
    fusion_technique = fusion.get("technique", fusion_technique)
    fusion_alpha = fusion.get("alpha", fusion_alpha)
    class_names = preprocessing_config.get("class_names", CLASS_NAMES)

    session = rt.InferenceSession(onnx_model, providers=rt.get_available_providers())
    model_input = session.get_inputs()[0]
    use_nir = model_input.shape[-1] == 4

    results = []
    for path in paths:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            logger.warning(f"Could not read {path}, skipping...")
            continue
        img = preprocess.prepare_for_model(img, use_nir=use_nir,
                                           fusion_technique=fusion_technique,
                                           fusion_alpha=fusion_alpha,
                                           source=path)
        batch = np.expand_dims(img, axis=0).astype(np.float32)

        probabilities = session.run(None, {model_input.name: batch})[0][0]
        best = int(np.argmax(probabilities))
        result = {
            "path": path,
            "prediction": class_names[best],
            "confidence": float(probabilities[best]),
            "probabilities": {name: float(p) for name, p in zip(class_names, probabilities)},
        }
        logger.info(f"{path}: {result['prediction']} ({result['confidence']:.3f})")
        results.append(result)

    return results
