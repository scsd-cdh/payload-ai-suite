# Payload AI Suite
Collection of software tools for multispectral image analysis and model testing.

# Features
- Cross-reference FIRMS fire event/EoNet and query via Copernicus's process API.
- Run VGG model on labeled (fire/no fire) data.
- Preprocess RGB-NIR algorithm with advanced fusion techniques
- Optional NIR channel support for RGB-NIR 4 channel tensor model.
- Cloud mask preprocessing to check cloud coverage threshold before wildfire inference.
- Image compression for satellite downlinking (AVIF and CCDS formats)
- NASA FIRMS integration for real-time fire data retrieval
- Sen2Fire dataset support for enhanced training data
- Multimodal quality control using Gemini 2.0 Flash
- Google Cloud Storage integration for scalable data management
- Automated cleanup utilities for dataset maintenance
- Progress tracking for batch processing operations

# Mission Goals
The underlying goal of this project is to illustrate the use of an embedded AI classification model for onboard wildfire detection. The inference provided by the model enables us to discard erroneous images and selectively downlink only successful captures.

Our operational goal is to detect medium fires (10-1,000 acres). These events represent a critical transition phase where intervention is still effective, but urgency is high. This targeted monitoring fills the gap between in-situ ground methods and “big players” like MODIS and VIIRS. Given our quality control calculations, medium fire targets are well within our system's capabilities. By reducing false positives, we aim to increase stakeholder confidence in alerts.

# Preprocess Methodology
For effective wildfire detection, we are using a multispectral RGB-NIR camera from Spectral Devices. This choice is based on the fact that the visible light spectrum (i.e., RGB) shares the same limitations as the human visual system when directly detecting fires. Incidental smoke severely limits the visual contrast of active flames, and fire emits far more energy in the IR spectrum.

It has been shown that NIR wavelengths between 830 nm and 1000 nm, captured by COTS camera sensors, provide statistically significant advantages in fire detection. As commonly employed in the field of robotics, our thesis is that the accuracy of our model will increase with an RGB-NIR fusion image as an input to improve feature detection.

## RGB-NIR Fusion Techniques
The preprocessing pipeline now includes two advanced fusion methods to combine RGB and NIR data:

1. **Enhanced Red Fusion**: Blends the NIR channel with the red channel using a weighted average (α=0.5), enhancing fire-related features while maintaining color balance.

2. **HSV Fusion**: Converts RGB to HSV color space and combines the NIR channel with the brightness (V) channel, preserving hue and saturation while enhancing intensity information.

These fusion techniques reduce the 4-channel input to 3 channels while retaining critical NIR information for improved fire detection.

If the `--use-nir` flag is used, preprocessing will maintain the additional NIR channel for R&D purposes. Currently, this NIR data can be found in the alpha channel of the test data. In production, the input would be the raw bayer output of the multispectral camera.

**Note**: The `--use-nir` flag is not currently supported when using `--use-gcs` due to channel consistency requirements when streaming from Google Cloud Storage.

# MLOps Quality Control
To ensure the reliability and accuracy of our models in production, the MLOps pipeline incorporates new multimodal quality control (QC) checks. These checks are designed to validate the integrity and consistency of incoming data and model outputs across different modalities, preventing issues such as data drift, sensor anomalies, and model performance degradation. Gemini 2.0 Flash is used under the hood for these checks.

# Cloud Mask Preprocessing
The cloud mask functionality serves as a critical preprocessing step to assess cloud coverage before wildfire detection inference. Using OmniCloudMask models, the system:
- Classifies each pixel as Clear, Thin Cloud, Thick Cloud, or Cloud Shadow
- Calculates total cloud coverage percentage to determine image suitability
- Filters out heavily clouded images that would compromise wildfire detection accuracy
- Ensures only clear or minimally clouded images proceed to the wildfire CNN

This preprocessing step significantly improves the reliability of wildfire detection by preventing false negatives caused by cloud occlusion.

# File Structure
The project is organized as follows:

```
payload-ai-suite/
├── src/                    # Source code directory
│   ├── main.py             # CLI entry point for running tools and workflows.
│   ├── model.py            # VGG-based wildfire classification model implementation.
│   ├── fetch.py            # Core utilities for data fetching (NASA FIRMS, Copernicus, EONET).
│   ├── preprocess.py       # Preprocessing utilities for input data.
│   ├── postprocess.py      # Image compression utilities (AVIF, CCDS formats).
│   ├── mlops.py            # MLOps utilities including GCS integration and multimodal QC.
│   ├── cloud.py            # Cloud mask detection preprocessing for cloud coverage assessment.
│   ├── clean_up_files.py   # Utilities for cleaning duplicate and empty files.
│   ├── paths.py            # Path management utilities for consistent file access.
│   └── training_checkpoints/  # Model checkpoint storage
├── events/                 # Directory for storing event-related data.
│   └── categories.json     # EONET wildfire events data.
├── data/                   # Directory for storing downloaded data (e.g., images, multispectral data).
│   ├── labeled/            # Training data directory
│   │   ├── yes/            # Positive fire samples
│   │   └── no/             # Negative (no-fire) samples
│   ├── eonet_fire_events/  # EONET fire event imagery
│   ├── nasa_firms/         # NASA FIRMS data
│   └── sen2fire/           # Sen2Fire dataset
├── models/                 # Pre-trained models directory
│   └── zetane.onnx         # ONNX model for wildfire detection
├── progress_counter/       # Progress tracking for data processing
├── requirements.txt        # Python dependencies for the project.
├── README.md               # Project documentation.
└── CLAUDE.md               # Instructions for Claude Code AI assistant.
```

# Environment Variables
The following environment variables are required for the project to function correctly:

- `NASA_KEY`: Your NASA FIRMS API key for accessing wildfire data. Request one at https://firms.modaps.eosdis.nasa.gov/api/map_key/
- `CLIENT_ID`: Client ID for Copernicus Data Space Ecosystem. Check out https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Overview/Authentication.html
- `CLIENT_SECRET`: Client secret for Copernicus Data Space Ecosystem.
- `GEMINI_API_KEY`: API key for Gemini 2.0 Flash, used for multimodal quality control checks.

## Google Cloud Storage (Optional)
The project supports Google Cloud Storage for training data and image storage. If you don't have access to GCS:

1. **For Local Development**: The project works fully with local file storage by default. Simply omit the `--use-gcs` flag.
2. **For Production Access**: Contact the code owner to request service account access for cloud storage.

If you have GCS access, set these additional environment variables:
- `GCS_BUCKET_NAME`: The storage bucket name (provided by code owner)
- `GCS_PROJECT_ID`: The GCP project ID (provided by code owner)
- `GOOGLE_APPLICATION_CREDENTIALS`: Path to your service account JSON key file

# How to Build and Run

## Prerequisites
1. **Python**: Ensure Python 3.8 or higher is installed.
2. **uv**: Install uv package manager:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
3. **Dependencies**: Install the required Python packages:
   ```bash
   uv sync
   ```

## Running the CLI
The project provides a CLI interface via `main.py`. Use the following commands to run specific tools:

### General Usage
```bash
# From the project root directory:
python3 src/main.py [OPTIONS]

# Or using uv:
uv run src/main.py [OPTIONS]
```

### Available Options
- `--run-model`: Run the wildfire classification model.
- `--nasa-firms`: Fetch data availability from NASA FIRMS API.
- `--setup-auth`: Set up OAuth2 authentication for Copernicus.
- `--batch-download`: Download images from Flickr using Selenium.
- `--eonet-crossref`: Fetch wildfire data from the EONET API and save it locally.
- `--copernicus-query`: Query Sentinel-2 and Sentinel-1 data from Copernicus.
- `--coordinates MIN_LON MIN_LAT MAX_LON MAX_LAT`: Specify a bounding box for the query.
- `--time-range FROM TO`: Specify a time range for the query (e.g., `2023-01-01T00:00:00Z 2023-01-03T23:59:59Z`).
- `--use-nir`: Enable the 4-channel (RGB+NIR) model (Note: Not supported when using `--use-gcs`)
- `--use-gcs`: Stream training data from Google Cloud Storage
- `--multimodal-qc`: Run multimodal quality control checks using Gemini 2.0
- `--qc-path PATH`: Path to folder for QC processing (e.g., 'sen2fire/to_qc' or 'eonet_fire_events/to_process')
- `--cloud-mask`: Export OmniCloudMask models to ONNX and test cloud coverage assessment on labeled data
- `--process-sen2fire`: Convert Sen2Fire dataset to a state to be processed by the pipeline
- `--upload-labeled`: Upload labeled data to GCS after running cleanup
- `--download-labeled`: Download labeled data from GCS to local filesystem

### Examples
1. **Run the wildfire classification model**:
   ```bash
   uv run main.py --run-model
   ```

2. **Fetch wildfire data from the EONET API**:
   ```bash
   uv run main.py --eonet-crossref
   ```

3. **Query Sentinel data with a bounding box and time range**:
   ```bash
   uv run main.py --copernicus-query --coordinates -59.75 -19.91 -58.72 -19.06 --time-range 2023-01-01T00:00:00Z 2023-01-03T23:59:59Z
   ```

4. **Download images using Selenium**:
   ```bash
   uv run main.py --batch-download
   ```
5. **Run the wildfire classification model with simulated NIR**:
   ```bash
   uv run main.py --run-model --use-nir
   ```

6. **Run the model using Google Cloud Storage**:
   ```bash
   uv run main.py --run-model --use-gcs
   ```

7. **Test cloud mask preprocessing on labeled data**:
   ```bash
   uv run main.py --cloud-mask
   ```
# Resources
- [Deep Learning in OpenCV](https://github.com/opencv/opencv/wiki/Deep-Learning-in-OpenCV)
- [VGG Onnx How To](https://github.com/onnx/models/blob/main/validated/vision/classification/vgg/train_vgg.ipynb)
- [ImageNet Demo](https://navigu.net/#imagenet)
- [Satellite Deep Learning Techniques](https://github.com/satellite-image-deep-learning/techniques)
