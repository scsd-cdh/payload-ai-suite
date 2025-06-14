"""VGG base model, 2 classes. First layers removed

Features:
- train
- validate
- export to onnx

TODO:
- optimize
- more data!

"""

import tensorflow as tf
import keras
import preprocess
import tf2onnx
import onnx
import numpy as np
import onnxruntime as rt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def train(validate=True, epochs=20, use_nir =False, destination_model_dir: Union[str, Path, None] = None):
    """
    Train CNN VGG model on labeled data.

    Args:
        validate (bool): Whether to validate the model after training.
        epochs (int): Number of epochs to train the model.

    Returns:
        None
    """
    X = []
    y = []

    X, y = preprocess.populate(X, y, "data/labeled/yes", use_nir=use_nir)
    X, y = preprocess.populate(X, y, "data/labeled/no", use_nir=use_nir , end=True)

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

    print("Shape of an image in X_train: ", X_train.shape)
    print("Shape of an image in X_test: ", X_test.shape)
    print("y_train Shape: ", y_train.shape)
    print("y_test Shape: ", y_test.shape)

    input_channels = 4 if use_nir else 3
    input_shape = (224, 224, input_channels)

    #weights = 'imagenet' if input_channels == 3 else None
    
    if destination_model_dir:
        weights = str(destination_model_dir)
    elif input_channels == 3:
        #Imagenet-pretrained VGG
        weights = 'imagenet'
    else:
        # 4-channel input
        weights = None

    vgg = tf.keras.applications.vgg16.VGG16(
        weights=weights,
        include_top=False,
        input_shape=input_shape
    )
    #since VGG16 is pre-trained w/ 3-channel RGB images, this if-else ensure it runs on a 4-channel system
    

    # Here we freeze the last 4 layers
    # Layers are set to trainable as True by default
    for layer in vgg.layers:
        layer.trainable = False

    for (i, layer) in enumerate(vgg.layers):
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
    head = create_top(vgg, num_classes)
    model = keras.models.Model(inputs=vgg.input, outputs=head)

    print(model.summary())
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train,
                        epochs=epochs,
                        validation_data=(X_test, y_test),
                        verbose=1,
                        initial_epoch=0)

    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    if validate:
        test_loss, test_accuracy = model.evaluate(X_test, y_test)
        print(f'Test accuracy: {test_accuracy:.4f}')
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
    onnx.save(onnx_model, 'zetane.onnx')

# not being run, here for testing with the onnx file to put on the hardware which has a program that calls it
# we need the file onnx first
def run_inference(onnx_model="zetane.onnx", data_target=None):
    if data_target is None or not data_target.any(): #error fixed 
        print("Please provide a test target")
        return
    session = rt.InferenceSession(onnx_model, providers=rt.get_available_providers)
    input_name = session.get_inputs()[0].name
    prediction_onnx = session.run(None, {input_name: data_target.astype(np.float32)})[0]
    print(prediction_onnx)

#research if we should make a separate file for the cloud model, so that we would 
# not break the tf model for the wildfire...
# test whether we can add a separate function where the user can try it out