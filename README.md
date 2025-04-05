# Payload AI Suite
Collection of software tools for multispectral image analysis and model testing.

# Features
- Cross-reference FIRMS fire event/EoNet and query via Copernicus's process API.
- Run VGG model on labeled (fire/no fire) data.
- Preproces RGB-NIR algorithm

# Mission Goals
The underlying goal of this project is to illustrate the use of an embedded AI classification model for onboard wildfire detection. The inference provided by the model enables us to discard erroneous images and selectively downlink only successful captures.

Our operational goal is to detect medium fires (10-1,000 acres). These events represent a critical transition phase where intervention is still effective, but urgency is high. This targeted monitoring fills the gap between in-situ ground methods and “big players” like MODIS and VIIRS. Given our quality control calculations, medium fire targets are well within our system's capabilities. By reducing false positives, we aim to increase stakeholder confidence in alerts.

# Preprocess Methodology
For effective wildfire detection, we are using a multispectral RGB-NIR camera from Spectral Devices. This choice is based on the fact that the visible light spectrum (i.e., RGB) shares the same limitations as the human visual system when directly detecting fires. Incidental smoke severely limits the visual contrast of active flames, and fire emits far more energy in the IR spectrum.

It has been shown that NIR wavelengths between 830 nm and 1000 nm, captured by COTS camera sensors, provide statistically significant advantages in fire detection. As commonly employed in the field of robotics, our thesis is that the accuracy of our model will increase with an RGB-NIR fusion image as an input to improve feature detection.

# File Structure
The project is organized as follows:

```
payload-ai-suite/
├── fetch.py                # Core utilities for data fetching (e.g., NASA FIRMS, Copernicus API, EONET).
├── main.py                 # CLI entry point for running tools and workflows.
├── model.py                # VGG-based wildfire classification model implementation.
├── preprocess.py           # Preprocessing utilities for input data.
├── events/                 # Directory for storing event-related data.
│   └── categories.json     # EONET wildfire events data.
├── data/                   # Directory for storing downloaded data (e.g., images, multispectral data).
├── requirements.txt        # Python dependencies for the project.
├── README.md               # Project documentation.
└── .gitignore              # Git ignore file for excluding unnecessary files.
```

# Environment Variables
The following environment variables are required for the project to function correctly:

- `NASA_KEY`: Your NASA FIRMS API key for accessing wildfire data. Request one at https://firms.modaps.eosdis.nasa.gov/api/map_key/
- `CLIENT_ID`: Client ID for Copernicus Data Space Ecosystem. Check out https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Overview/Authentication.html
- `CLIENT_SECRET`: Client secret for Copernicus Data Space Ecosystem.

# How to Build and Run

## Prerequisites
1. **Python**: Ensure Python 3.8 or higher is installed.
2. **Dependencies**: Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Running the CLI
The project provides a CLI interface via `main.py`. Use the following commands to run specific tools:

### General Usage
```bash
python3 main.py [OPTIONS]
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

### Examples
1. **Run the wildfire classification model**:
   ```bash
   python3 main.py --run-model
   ```

2. **Fetch wildfire data from the EONET API**:
   ```bash
   python3 main.py --eonet-crossref
   ```

3. **Query Sentinel data with a bounding box and time range**:
   ```bash
   python3 main.py --copernicus-query --coordinates -59.75 -19.91 -58.72 -19.06 --time-range 2023-01-01T00:00:00Z 2023-01-03T23:59:59Z
   ```

4. **Download images using Selenium**:
   ```bash
   python3 main.py --batch-download
   ```

# Resources
- [Deep Learning in OpenCV](https://github.com/opencv/opencv/wiki/Deep-Learning-in-OpenCV)
- [VGG Onnx How To](https://github.com/onnx/models/blob/main/validated/vision/classification/vgg/train_vgg.ipynb)
- [ImageNet Demo](https://navigu.net/#imagenet)
- [Satellite Deep Learning Techniques](https://github.com/satellite-image-deep-learning/techniques)
