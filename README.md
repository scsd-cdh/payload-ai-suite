# Payload AI Suite
Collection software tools for multispectral image analysis and model testing.

# Features
- Cross reference FIRMS fire events and query via Copernicus's process API
- Analyze downloaded multispectral data with ENVI
- Run VGG model on labeled (fire/no fire) data

# Mission Goals
The underlying goal of this project is to illustrate the use of an embedded AI classification model for onboard wildfire detection. The inference provided by the model enables us to discard erroneous images and selectively downlink only successful captures.

Our operational goal is to detect medium fires (10-1,000 acres). These events represent a critical transition phase where intervention is still effective, but urgency is high. This targeted monitoring fills the gap between in-situ ground methods and “big players” like MODIS and VIIRS. Given our quality control calculations, medium fire targets are well within our system's capabilities. By reducing false positives, we aim to increase stakeholder confidence in alerts.

# Preprocess Methodology
For effective wildfire detection, we are using a multispectral RGB-NIR camera from Spectral Devices. This choice is based on the fact that the visible light spectrum (i.e., RGB) shares the same limitations as the human visual system when directly detecting fires. Incidental smoke severely limits the visual contrast of active flames, and fire emits far more energy in the IR spectrum.

It has been shown that NIR wavelengths between 830 nm and 1000 nm, captured by COTS camera sensors, provide statistically significant advantages in fire detection. As commonly employed in the field of robotics, our thesis is that the accuracy of our model will increase with an RGB-NIR fusion image as an input to improve feature detection.

#### Resources
- [Deep Learning in OpenCV](https://github.com/opencv/opencv/wiki/Deep-Learning-in-OpenCV)
- [VGG Onnx How To](https://github.com/onnx/models/blob/main/validated/vision/classification/vgg/train_vgg.ipynb)
- [ImageNet Demo](https://navigu.net/#imagenet)
- [Satellite Deep Learning Techniques](https://github.com/satellite-image-deep-learning/techniques)
