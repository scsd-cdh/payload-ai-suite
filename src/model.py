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
from tensorflow import keras
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train(validate=True, epochs=12, use_nir=False, use_gcs=False, 
          use_mixed_res=False, mixed_res_config=None):
    """
    Train CNN ResNet50 model on labeled data.

    Args:
        validate (bool): Whether to validate the model after training.
        epochs (int): Number of epochs to train the model.
        use_nir (bool): Whether to use NIR channel.
        use_gcs (bool): Whether to stream data from GCS.
        use_mixed_res (bool): Whether to apply mixed resolution operations during training.
        mixed_res_config (dict): Configuration for mixed resolution operations.

    Returns:
        None
    """
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
                          mixed_res_config=mixed_res_config)
            X, y = preprocess.populate(X, y, "labeled/no", use_nir=use_nir, 
                          end=True, gcs_handler=gcs, 
                          use_mixed_res=use_mixed_res,
                          mixed_res_config=mixed_res_config)

        except Exception as e:
            logger.error(f"Failed to load data from GCS: {str(e)}")
            raise
    else:
        # Use local files with mixed resolution options
        X, y = preprocess.populate(X, y, resolve_path("data/labeled/yes"), use_nir=use_nir,
                      use_mixed_res=use_mixed_res, 
                      mixed_res_config=mixed_res_config)
        X, y = preprocess.populate(X, y, resolve_path("data/labeled/no"), use_nir=use_nir, 
                      end=True, use_mixed_res=use_mixed_res,
                      mixed_res_config=mixed_res_config)

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

    input_channels = 4 if use_nir else 3
    input_shape = (224, 224, input_channels)

    weights = 'imagenet' if input_channels == 3 else None

    resnet50 = tf.keras.applications.resnet.ResNet50(
        weights=weights,
        include_top=False,
        input_shape=input_shape
    )
    #since ResNet50 is pre-trained w/ 3-channel RGB images, this if-else ensure it runs on a 4-channel system

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
        top_model = keras.layers.Dense(1024, activation='relu')(top_model)
        top_model = keras.layers.Dense(512, activation='relu')(top_model)
        output = keras.layers.Dense(num_classes, activation='softmax')(top_model)
        return output

    num_classes = 2
    head = create_top(resnet50, num_classes)
    model = keras.models.Model(inputs=resnet50.input, outputs=head)

    model.summary(print_fn=logger.info)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

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

    history = model.fit(X_train, y_train,
                        epochs=epochs,
                        validation_data=(X_test, y_test),
                        verbose=1,
                        initial_epoch=0,
                        shuffle=True,
                        callbacks=[cp_callback])

    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    export_to_onnx(model)

    if validate:
        test_loss, test_accuracy = model.evaluate(X_test, y_test)
        logger.info(f'Test accuracy: {test_accuracy:.4f}')
        epochs = range(len(accuracy))
        plt.plot(epochs, accuracy, 'r', label='Training accuracy')
        plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.legend(loc=0)
        plt.figure()
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
    onnx_model, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=input_signature,
        opset=13  # Specify ONNX opset version
    )
    onnx.save(onnx_model, get_model_path('zetane.onnx'))

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