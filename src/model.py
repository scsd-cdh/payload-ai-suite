"""VGG base model, 2 classes. First layers removed

Features:
- train
- validate
- export to onnx
- mixed resolution operations for resolution-agnostic training
- dropout regularization to reduce overfitting
- class weight balancing for imbalanced datasets

"""

import os
import argparse
import tensorflow as tf
import keras
import numpy as np
import tf2onnx
import onnx
import onnxruntime as rt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import logging
import subprocess
from src.preprocess import populate, clean_zone_identifiers
from src.mlops import GCSHandler
from src.mixed_res_config import DEFAULT_MIXED_RES_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train(validate=False, epochs=12, use_nir=False, use_gcs=False, 
          use_mixed_res=False, mixed_res_config=None, unfreeze_top_layers=False,
          clean_data=False):
    """
    Train CNN VGG model on labeled data.

    Args:
        validate (bool): Whether to validate the model after training.
        epochs (int): Number of epochs to train the model.
        use_nir (bool): Whether to use NIR channel.
        use_gcs (bool): Whether to stream data from GCS.
        use_mixed_res (bool): Whether to apply mixed resolution operations during training.
        mixed_res_config (dict): Configuration for mixed resolution operations.
        unfreeze_top_layers (bool): Whether to unfreeze block5 layers for fine-tuning.
        clean_data (bool): Whether to clean Zone.Identifier files before training.

    Returns:
        None
    """
    # Clean Zone.Identifier files if requested
    if clean_data:
        logger.info("Cleaning Zone.Identifier files from data directory...")
        clean_zone_identifiers()
    
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
            X, y = populate(X, y, "labeled/yes", use_nir=use_nir, 
                          gcs_handler=gcs, use_mixed_res=use_mixed_res, 
                          mixed_res_config=mixed_res_config)
            X, y = populate(X, y, "labeled/no", use_nir=use_nir, 
                          end=True, gcs_handler=gcs, 
                          use_mixed_res=use_mixed_res,
                          mixed_res_config=mixed_res_config)

        except Exception as e:
            logger.error(f"Failed to load data from GCS: {str(e)}")
            raise
    else:
        # Use local files with mixed resolution options
        X, y = populate(X, y, "data/labeled/yes", use_nir=use_nir,
                      use_mixed_res=use_mixed_res, 
                      mixed_res_config=mixed_res_config)
        X, y = populate(X, y, "data/labeled/no", use_nir=use_nir, 
                      end=True, use_mixed_res=use_mixed_res,
                      mixed_res_config=mixed_res_config)

    # TODO: Use numpy instead here
    X = [X[i] for i in range(min(len(X), len(y)))]
    y = [y[i] for i in range(min(len(X), len(y)))]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    label_encoder = preprocessing.LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.fit_transform(y_test)

    # Calculate class weights to handle imbalance
    # Count each class
    class_counts = {}
    for label in y_train:
        if label not in class_counts:
            class_counts[label] = 0
        class_counts[label] += 1
    
    # Calculate class weights (inverse frequency)
    total_samples = len(y_train)
    n_classes = len(class_counts)
    class_weights = {}
    
    for cls, count in class_counts.items():
        class_weights[cls] = total_samples / (n_classes * count)
    
    logger.info(f"Class distribution: {class_counts}")
    logger.info(f"Class weights: {class_weights}")

    y_train = tf.keras.utils.to_categorical(y_train, num_classes=2)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=2)

    y_train = np.array(y_train)
    X_train = np.array(X_train)
    y_test = np.array(y_test)
    X_test = np.array(X_test)

    print("Shape of an image in X_train: ", X_train.shape)
    print("Shape of an image in X_test: ", X_test.shape)
    print("y_train Shape: ", y_train.shape)
    print("y_test Shape: ", y_test.shape)

    input_channels = 4 if use_nir else 3
    input_shape = (224, 224, input_channels)

    # Use ImageNet weights only for 3-channel inputs
    weights = 'imagenet' if input_channels == 3 else None

    vgg = tf.keras.applications.vgg16.VGG16(
        weights=weights,
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze layers by default
    for layer in vgg.layers:
        layer.trainable = False
        
    # Optionally unfreeze block5 layers for fine-tuning
    if unfreeze_top_layers:
        for layer in vgg.layers:
            if "block5" in layer.name:
                layer.trainable = True
                logger.info(f"Unfreezing layer: {layer.name}")

    for (i, layer) in enumerate(vgg.layers):
        print(str(i) + " " + layer.__class__.__name__, layer.trainable)

    def create_top(bottom_model, num_classes):
        """Create top layers with dropout for regularization"""
        top_model = bottom_model.output
        top_model = keras.layers.GlobalAveragePooling2D()(top_model)
        top_model = keras.layers.Dense(1024, activation='relu')(top_model)
        top_model = keras.layers.Dropout(0.3)(top_model)  # Add dropout for regularization
        top_model = keras.layers.Dense(1024, activation='relu')(top_model)
        top_model = keras.layers.Dropout(0.3)(top_model)  # Add dropout for regularization
        top_model = keras.layers.Dense(512, activation='relu')(top_model)
        top_model = keras.layers.Dropout(0.2)(top_model)  # Add dropout for regularization
        output = keras.layers.Dense(num_classes, activation='softmax')(top_model)
        return output

    num_classes = 2
    head = create_top(vgg, num_classes)
    model = keras.models.Model(inputs=vgg.input, outputs=head)

    print(model.summary())
    
    # Set learning rate based on whether we're fine-tuning or training from scratch
    learning_rate = 1e-5 if unfreeze_top_layers else 1e-3
    model.compile(optimizer=keras.optimizers.Adam(learning_rate), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

    checkpoint_path = "training_checkpoints/cp.weights.h5"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    
    # Create directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)

    latest = tf.train.latest_checkpoint(checkpoint_dir)
    if latest:
        model.load_weights(latest)
        print("Loaded weights from checkpoint.")

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)

    # Early stopping to prevent overfitting
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    # Use class_weights for handling imbalance
    history = model.fit(X_train, y_train,
                        epochs=epochs,
                        validation_data=(X_test, y_test),
                        verbose=1,
                        initial_epoch=0,
                        shuffle=True,
                        class_weight=class_weights,
                        callbacks=[cp_callback, early_stopping])

    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    # Export model
    export_to_onnx(model)

    if validate:
        test_loss, test_accuracy = model.evaluate(X_test, y_test)
        print(f'Test accuracy: {test_accuracy:.4f}')
        epochs_range = range(len(accuracy))
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, accuracy, 'r', label='Training accuracy')
        plt.plot(epochs_range, val_accuracy, 'b', label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.legend(loc=0)
        
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, 'r', label='Training loss')
        plt.plot(epochs_range, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend(loc=0)
        
        plt.tight_layout()
        plt.savefig('training_results.png')
        plt.show()

def export_to_onnx(model):
    """
    Export the trained model to ONNX format.

    Args:
        model (tf.keras.Model): The trained Keras model to be exported.

    Returns:
        None
    """
    input_signature = [
        tf.TensorSpec(
            shape=model.input_shape,
            dtype=tf.float32,
            name='input'
        )
    ]
    
    # Check if the model is large
    model_size_mb = model.count_params() * 4 / (1024 * 1024)  # Rough estimate in MB
    large_model = model_size_mb >= 64
    
    if large_model:
        logger.info(f"Model size is approximately {model_size_mb:.2f} MB, using large_model flag")
    
    onnx_model, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=input_signature,
        opset=13,  # Specify ONNX opset version
        large_model=large_model
    )
    onnx.save(onnx_model, 'zetane.onnx')

def run_inference(onnx_model="zetane.onnx", data_target=None):
    if data_target is None or not data_target.any():
        print("Please provide a test target")
        return
    session = rt.InferenceSession(onnx_model, providers=rt.get_available_providers)
    input_name = session.get_inputs()[0].name
    prediction_onnx = session.run(None, {input_name: data_target.astype(np.float32)})[0]
    print(prediction_onnx)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VGG model with optional mixed resolution ops")
    parser.add_argument("--validate", action="store_true", help="Validate after training")
    parser.add_argument("--epochs", type=int, default=12, help="Number of epochs")
    parser.add_argument("--use-nir", action="store_true", help="Use NIR channel")
    parser.add_argument("--use-gcs", action="store_true", help="Stream data from GCS")
    parser.add_argument("--use-mixed-res", action="store_true", help="Enable mixed resolution ops")
    parser.add_argument("--clean-data", action="store_true", help="Clean Zone.Identifier files before training")
    parser.add_argument("--unfreeze-top", action="store_true", help="Unfreeze block5 layers for fine-tuning")
    parser.add_argument("--min-scale", type=float, default=DEFAULT_MIXED_RES_CONFIG['min_scale'], 
                        help="Minimum scale for random resize")
    parser.add_argument("--max-scale", type=float, default=DEFAULT_MIXED_RES_CONFIG['max_scale'], 
                        help="Maximum scale for random resize")
    parser.add_argument("--mixup-alpha", type=float, default=DEFAULT_MIXED_RES_CONFIG['mixup_alpha'], 
                        help="Alpha parameter for resolution mixup")
    parser.add_argument("--resolution-scales", type=float, nargs='+', 
                        default=DEFAULT_MIXED_RES_CONFIG['resolution_scales'],
                        help="List of resolution scales for multi-resolution batch")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU acceleration if available")
    args = parser.parse_args()

    # Handle GPU visibility
    if not args.gpu:
        # Hide GPU from visible devices to avoid CUDA errors
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        logger.info("GPU disabled. Using CPU only.")
    else:
        logger.info("GPU enabled if available.")
        # Check if CUDA is available
        try:
            gpu_devices = tf.config.list_physical_devices('GPU')
            if gpu_devices:
                logger.info(f"Found {len(gpu_devices)} GPU(s): {gpu_devices}")
            else:
                logger.warning("No GPU found, falling back to CPU.")
        except Exception as e:
            logger.warning(f"Error checking GPU availability: {str(e)}")

    # Fix: Copy default config and only update if needed
    mixed_res_config = DEFAULT_MIXED_RES_CONFIG.copy()
    if args.use_mixed_res:
        mixed_res_config.update({
            'min_scale': args.min_scale,
            'max_scale': args.max_scale,
            'resolution_scales': args.resolution_scales,
            'mixup_alpha': args.mixup_alpha
        })

    train(
        validate=args.validate,
        epochs=args.epochs,
        use_nir=args.use_nir,
        use_gcs=args.use_gcs,
        use_mixed_res=args.use_mixed_res,
        mixed_res_config=mixed_res_config,
        unfreeze_top_layers=args.unfreeze_top,
        clean_data=args.clean_data
    )