"""VGG base model, 2 classes. First layers removed

Features:
- train
- validate
- export to onnx

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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)

#  change to resent 50 from vgg check if keras has and figure out how to train it!!!!!!
def train(validate=True, epochs=12, use_nir=False, use_gcs=False):
    """
    Train CNN ResNet50 model on labeled data, 2 classes.

    Args:
        validate (bool): Whether to validate the model after training.
        epochs (int): Number of epochs to train the model.
        use_nir (bool): Whether to use NIR channel.
        use_gcs (bool): Whether to stream data from GCS.

    Returns:
        None
    """
    print(f" DEBUG: Starting train() with use_gcs={use_gcs}, use_nir={use_nir}")
    X = []
    y = []
    print(f" DEBUG: Initial X length: {len(X)}, y length: {len(y)}")

    if use_gcs:
        # Stream images from GCS
        try:
            print(" DEBUG: Creating GCS handler...")
            gcs = GCSHandler()
            print(" DEBUG: GCS handler created successfully")

            # Pass GCS handler to populate function to stream from bucket
            print(" DEBUG: About to call populate for 'labeled/yes'...")
            X, y = preprocess.populate(X, y, "labeled/yes", use_nir=use_nir, gcs_handler=gcs)
            print(f" DEBUG: After 'yes' populate - X length: {len(X)}, y length: {len(y)}")
            
            # Debug the contents
            if len(X) > 0:
                print(f" DEBUG: First X sample shape: {X[0].shape}")
                print(f" DEBUG: First X sample type: {type(X[0])}")
            if len(y) > 0:
                print(f" DEBUG: First few y labels: {y[:5]}")
                unique_labels = list(set(y))
                print(f" DEBUG: Unique labels so far: {unique_labels}")
            
            print(" DEBUG: About to call populate for 'labeled/no'...")
            X, y = preprocess.populate(X, y, "labeled/no", use_nir=use_nir, end=True, gcs_handler=gcs)
            print(f" DEBUG: After 'no' populate - X length: {len(X)}, y length: {len(y)}")
            
            # Debug final contents
            if len(y) > 0:
                unique_labels = list(set(y))
                print(f" DEBUG: Final unique labels: {unique_labels}")
                for label in unique_labels:
                    count = y.count(label)
                    print(f" DEBUG: Label '{label}': {count} samples")

        except Exception as e:
            print(f" DEBUG: Exception occurred: {e}")
            logger.error(f"Failed to load data from GCS: {str(e)}")
            raise
    else:
        # Use local files
        print(" DEBUG: Using local files...")
        print(" DEBUG: About to call populate for local 'yes' directory...")
        X, y = preprocess.populate(X, y, resolve_path("data/labeled/yes"), use_nir=use_nir)
        print(f" DEBUG: After local 'yes' populate - X length: {len(X)}, y length: {len(y)}")
        
        print(" DEBUG: About to call populate for local 'no' directory...")
        X, y = preprocess.populate(X, y, resolve_path("data/labeled/no"), use_nir=use_nir, end=True)
        print(f" DEBUG: After local 'no' populate - X length: {len(X)}, y length: {len(y)}")

    print(f" DEBUG: Final data summary:")
    print(f" DEBUG: Total X samples: {len(X)}")
    print(f" DEBUG: Total y labels: {len(y)}")
    
    # Check if we have any data at all
    if len(X) == 0:
        print("🚨 ERROR: No images were loaded!")
        return
    if len(y) == 0:
        print("🚨 ERROR: No labels were created!")
        return
        
    # TODO: Use numpy instead here
    print(" DEBUG: Before alignment - X length:", len(X), "y length:", len(y))
    X = [X[i] for i in range(min(len(X), len(y)))]
    y = [y[i] for i in range(min(len(X), len(y)))]
    print(" DEBUG: After alignment - X length:", len(X), "y length:", len(y))

    # Check label distribution before splitting
    if len(y) > 0:
        unique_labels = list(set(y))
        print(f" DEBUG: Labels before train_test_split: {unique_labels}")
        for label in unique_labels:
            count = y.count(label)
            print(f" DEBUG: Label '{label}': {count} samples")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    print(f" DEBUG: After train_test_split:")
    print(f" DEBUG: X_train: {len(X_train)}, X_test: {len(X_test)}")
    print(f" DEBUG: y_train: {len(y_train)}, y_test: {len(y_test)}")

    label_encoder = preprocessing.LabelEncoder()
    print(f" DEBUG: Before label encoding - unique y_train: {set(y_train)}")
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.fit_transform(y_test)
    print(f" DEBUG: After label encoding - y_train range: {y_train.min()}-{y_train.max()}")
    print(f" DEBUG: Label encoder classes: {label_encoder.classes_}")

    y_train = tf.keras.utils.to_categorical(y_train, num_classes=2)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=2)

    y_train = np.array(y_train)
    X_train = np.array(X_train)
    y_test = np.array(y_test)
    X_test = np.array(X_test)

    print(f" DEBUG: Final array shapes:")
    print(f" DEBUG: X_train.shape: {X_train.shape}")
    print(f" DEBUG: X_test.shape: {X_test.shape}")
    print(f" DEBUG: y_train.shape: {y_train.shape}")
    print(f" DEBUG: y_test.shape: {y_test.shape}")
    
    # Check class distribution in one-hot encoded labels
    print(f" DEBUG: y_train class distribution:")
    print(f" DEBUG: Class 0: {y_train[:, 0].sum()} samples")
    print(f" DEBUG: Class 1: {y_train[:, 1].sum()} samples")

    logger.info(f"Shape of an image in X_train: {X_train.shape}")
    logger.info(f"Shape of an image in X_test: {X_test.shape}")
    logger.info(f"y_train Shape: {y_train.shape}")
    logger.info(f"y_test Shape: {y_test.shape}")

    input_channels = 4 if use_nir else 3
    input_shape = (224, 224, input_channels)

    weights = None

    resnet50 = tf.keras.applications.resnet50.ResNet50(
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
        print(str(i) + " " + layer.__class__.__name__, layer.trainable)

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
