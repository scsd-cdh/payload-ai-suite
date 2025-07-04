"""Utilities for fetching data for the model
- Selenium based flickr webscraper
- EONET wildfire cross reference tool
"""

import requests
import json
import time
import os
from mlops import GCSHandler
from datetime import datetime


from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session
import numpy as np
import pandas as pd
import pygeohash as pgh
from selenium import webdriver
from selenium.webdriver.common.by import By
from pyproj import Proj, Transformer

from PIL import Image

import logging
logger = logging.getLogger(__name__)


class ProgressTracker:
    """Tracks progress of EONET data scraping at event and location level."""

    # Legacy workflow starts at event 125
    LEGACY_START_INDEX = 125

    def __init__(self, filepath='progress_counter/eonet.json'):
        self.filepath = filepath
        self.progress = self._load_progress()

    def _load_progress(self):
        """Load existing progress from file or create new."""
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load progress file: {e}. Starting fresh.")

        # Default structure - start at legacy index
        return {
            "current_event_index": self.LEGACY_START_INDEX,
            "events": {}
        }

    def save_progress(self):
        """Save current progress to file."""
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        with open(self.filepath, 'w') as f:
            json.dump(self.progress, f, indent=2)

    def get_event_progress(self, event_id):
        """Get progress for a specific event."""
        return self.progress["events"].get(event_id, {
            "locations_generated": 0,
            "locations_processed": 0,
            "location_status": {}
        })

    def update_event_progress(self, event_id, locations_generated=None, location_index=None, status=None):
        """Update progress for an event and/or specific location."""
        if event_id not in self.progress["events"]:
            self.progress["events"][event_id] = {
                "locations_generated": 0,
                "locations_processed": 0,
                "location_status": {}
            }

        event = self.progress["events"][event_id]

        if locations_generated is not None:
            event["locations_generated"] = locations_generated

        if location_index is not None and status is not None:
            event["location_status"][str(location_index)] = status
            # Update processed count
            event["locations_processed"] = sum(
                1 for s in event["location_status"].values()
                if s == "completed"
            )

        self.save_progress()

    def should_skip_location(self, event_id, location_index):
        """Check if a location has already been processed."""
        event = self.get_event_progress(event_id)
        return event["location_status"].get(str(location_index)) == "completed"

    def get_resume_point(self):
        """Get the event index to resume from."""
        return self.progress.get("current_event_index", 0)

    def update_current_event_index(self, index):
        """Update the current event index being processed."""
        self.progress["current_event_index"] = index
        self.save_progress()


def nasa_firms_api():
    """
    Fetches data availability from NASA FIRMS API.

    This function uses the NASA FIRMS API to retrieve data availability in CSV format.
    The API key is retrieved from the environment variable `NASA_KEY`.

    https://firms.modaps.eosdis.nasa.gov/api/ Fire Information for Resource Management System
    /api/area/csv/[NASA_KEY]/[SOURCE]/[AREA_COORDINATES]/[DAY_RANGE]/[DATE]

    Returns:
        pd.DataFrame: A DataFrame containing the data availability information for NASA FIRMS.
    """
    # TODO: Implement AREA_COORDINATES and DAY_RANGE etc. parameters for API call
    NASA_KEY = os.getenv("NASA_KEY")
    data_url = 'https://firms.modaps.eosdis.nasa.gov/api/data_availability/csv/' + NASA_KEY + '/all'
    data_frame = pd.read_csv(data_url)
    return data_frame


def setup_auth():
    """
    Sets up OAuth2 authentication for Copernicus Data Space Ecosystem.

    Retrieves an access token using client credentials and returns it for use in API requests.

    Returns:
        dict: A dictionary containing the access token and related information.
    """
    # Your client credentials
    # If not set, please check README for further information.
    client_id = os.getenv("CLIENT_ID")
    client_secret = os.getenv("CLIENT_SECRET")

    # Create a session
    client = BackendApplicationClient(client_id=client_id)
    oauth = OAuth2Session(client=client)

    # Get token for the session
    token = oauth.fetch_token(token_url='https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token',
                            client_secret=client_secret, include_client_id=True)

    # All requests using this session will have an access token automatically added
    return token['access_token']

def retrieve_eonet_cross_reference():
    """Fetches wildfire data from the EONET API and saves it to a JSON file.

    The data is retrieved from the EONET API's wildfire category and saved to
    `events/categories.json`.

    Returns:
        None
    """
    # Code used to gather cross referencing data
    wildfire_url = "https://eonet.gsfc.nasa.gov/api/v3/categories/wildfires"
    response = requests.get(url=wildfire_url)
    data = response.json()
    with open('events/categories.json', 'w', encoding='utf-8') as f:
         json.dump(data, f, ensure_ascii=False, indent=4)


def extract_eonet_coordinates(file_path='events/categories.json'):
    """Extracts coordinates from the EONET categories JSON file.

    Args:
        file_path (str): Path to the JSON file containing EONET wildfire events. Defaults to 'events/categories.json'.

    Returns:
        list: A list of coordinates in the format [[lon, lat], [lon, lat], ...].
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Extract coordinates from the events
        coordinates = []
        for event in data.get('events', []):
            if 'geometry' in event:
                for geo in event['geometry']:
                    if geo.get('type') == 'Point':
                        coordinates.append(geo.get('coordinates'))

        if not coordinates:
            raise ValueError("No valid coordinates found in the JSON file.")

        return [coordinates]

    except Exception as e:
        print(f"Error extracting coordinates: {e}")
        return None

def extract_time_ranges_from_eonet(file_path='events/categories.json'):
    """Extracts time ranges from the EONET categories JSON file and converts them to the required format.

    Args:
        file_path (str): Path to the JSON file containing EONET wildfire events. Defaults to 'events/categories.json'.

    Returns:
        list of dict: A list of time ranges in the format [{"from": "YYYY-MM-DDTHH:MM:SSZ", "to": "YYYY-MM-DDTHH:MM:SSZ"}].
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Extract time ranges from the events
        time_ranges = []
        for event in data.get('events', []):
            if 'geometry' in event:
                for geo in event['geometry']:
                    if 'date' in geo:
                        start_time = geo['date']
                        # Assuming a default duration of 2 days for the time range
                        end_time = (pd.to_datetime(start_time) + pd.Timedelta(days=2)).strftime('%Y-%m-%dT%H:%M:%SZ')
                        time_ranges.append({"from": start_time, "to": end_time})

        if not time_ranges:
            raise ValueError("No valid time ranges found in the JSON file.")

        return time_ranges

    except Exception as e:
        print(f"Error extracting time ranges: {e}")
        return []

def write_image(response, metadata, location=None, use_gcs=False):
    """Writes image data from an API response to a file or GCS.

    This function extracts image data (e.g., TIFF) from the response object and writes it to a file
    or uploads it to Google Cloud Storage.

    Args:
        response (requests.Response): The response object containing the image data.
        metadata (dict): A dictionary containing metadata about the request, such as output format,
                         dimensions, and other relevant information.
        location: Location object with geohash attribute
        use_gcs (bool): Whether to upload to GCS instead of saving locally

    Returns:
        None

    Raises:
        IOError: If there is an error writing the image to a file.
    """


    try:
        # Extract the output format from metadata (default to TIFF)
        output_format = metadata.get('output', {}).get('format', 'image/tiff').split('/')[-1]
        # TODO: write custom logic for filename to be populated by metadata satellite type and bands

        if use_gcs:
            # Upload to GCS

            try:
                gcs = GCSHandler()

                # Create GCS path
                date_str = datetime.now().strftime('%Y%m%d')
                gcs_path = f"raw_data/eonet/to_process/{date_str}/{location.geohash}.{output_format}"

                # Get content type
                content_type = metadata.get('output', {}).get('format', 'image/png')

                # Upload bytes directly
                success = gcs.upload_bytes(response.content, gcs_path, content_type=content_type)

                if success:
                    logger.info(f"Image successfully uploaded to gs://{gcs.bucket_name}/{gcs_path}")
                else:
                    logger.error(f"Failed to upload image to GCS")

            except Exception as e:
                logger.error(f"GCS upload failed: {str(e)}")
                # Don't fall back to local save - fail explicitly
                raise
        else:
            # Save locally
            filename = f"./data/eonet_fire_events/{location.geohash}.{output_format}"

            # Write the response content data to a image file
            with open(filename, 'wb') as f:
                f.write(response.content)
            print(f"Image successfully saved to {filename}")
    except Exception as e:
        print(f"Error writing image: {e}")

def convert_coords_to_bbox(longitude, latitude, buffer_distance=5000):
    wgs84 = Proj(proj="latlong", datum="WGS84")
    utm_zone = int((longitude + 180) / 6) + 1
    utm = Proj(proj="utm", zone=utm_zone, datum="WGS84")

    transformer_to_utm = Transformer.from_proj(wgs84, utm, always_xy=True)
    transformer_to_wgs84 = Transformer.from_proj(utm, wgs84, always_xy=True)

    x_center, y_center = transformer_to_utm.transform(longitude, latitude)

    if not all(map(lambda x: abs(x) < 1e8, [x_center, y_center])):
        raise ValueError("UTM transform failed. Check input coordinates.")

    x_min = x_center - buffer_distance
    x_max = x_center + buffer_distance
    y_min = y_center - buffer_distance
    y_max = y_center + buffer_distance

    sw_lon, sw_lat = transformer_to_wgs84.transform(x_min, y_min)
    ne_lon, ne_lat = transformer_to_wgs84.transform(x_max, y_max)

    return {
        "bbox": [sw_lon, sw_lat, ne_lon, ne_lat]
    }

class Location:
    def __init__(self, coordinates, time, geohash=None, bbox=None):
        self.coordinates = coordinates
        self.time = time
        print(self.time)
        self.geohash = self.create_geohash(coordinates)
        self.bbox = convert_coords_to_bbox(coordinates[0], coordinates[1])
        print(self.bbox)
    def create_geohash(self, coordinates):
        geohash = pgh.encode(latitude=coordinates[0], longitude=coordinates[1])
        return geohash

def create_locations(amount=135, progress_tracker=None):
    """Creates a list of Location objects based on EONET data.

    This function extracts time ranges and coordinates from the EONET wildfire data
    and uses them to create Location objects, starting from the last processed entry
    if a progress tracker is provided.

    Args:
        amount (int): The number of Location objects to process from the starting point. Defaults to 135.
        progress_tracker (ProgressTracker): Optional progress tracker instance.

    Returns:
        list: A list of Location objects.
    """
    # list of dict entries in the form {'from': '2023-08-05T17:59:00Z', 'to': '2023-08-07T17:59:00Z'}
    locations = []
    time_ranges = extract_time_ranges_from_eonet()
    coordinates = extract_eonet_coordinates()

    # Get resume point if using progress tracker
    start_entry = 0
    if progress_tracker:
        start_entry = progress_tracker.get_resume_point()
        print(f"Resuming from entry {start_entry}")
    
    print(f"Creating locations from {start_entry} to {start_entry + amount}")
    
    # Process 'amount' number of locations starting from start_entry
    for entry in range(start_entry, start_entry + amount):
        location = Location(coordinates[0][entry], time=time_ranges[entry])
        location.entry_index = entry

        # Skip if already processed (using geohash as unique ID)
        if progress_tracker and progress_tracker.should_skip_location(location.geohash, 0):
            print(f"Skipping already processed location: {location.geohash}")
            # Still update the current index even when skipping
            if progress_tracker:
                progress_tracker.update_current_event_index(entry + 1)
            continue

        locations.append(location)

        # Update progress tracker - set to next entry to process
        if progress_tracker:
            progress_tracker.update_current_event_index(entry + 1)

    return locations

def validate_query(target, auth):
    """Vaildiate data availibitly for the given location parameters.
    Args:
        target (Location): Location object to check database for (specifically time range and bbox)
        auth (dict): Dict of oauth HTTP header request info.
    """
    date = f"{target.time['from']}/.."
    data = {
        "bbox": target.bbox['bbox'],
        "datetime": date,
        "collections": ["sentinel-2-l2a"],
        "limit": 10,
        "fields": {"include": ["properties.gsd"]},
    }
    url = "https://sh.dataspace.copernicus.eu/api/v1/catalog/1.0.0/search"
    response = requests.post(url, json=data, headers=auth)
    print(response.content)

def copernicus_sentiel_query(use_gcs=False, amount=135):
    """Queries Sentinel-2 and Sentinel-1 data from the Copernicus Data Space Ecosystem.

    This function uses an inline evaluation script to process Sentinel-2 bands of interest
    and retrieves data for a specified bounding box and time range. If the option is used, the data is uploaded to GCS.

    Sentiel-2 bands of interest
    B02: Blue
    B03: Green
    B04: Red
    B08: Visible and Near Infared (VNIR)

    Sentiel-3 bands of interest

    Args:
        use_gcs (bool): Whether to upload images to GCS instead of saving locally
        amount (int): Total number of locations to process (default 135)

    Returns:
        None
    """
    # Need a valid eval script, specified bands, specified data range
    ACCESS_TOKEN = setup_auth()
    headers={f"Authorization" : f"Bearer {ACCESS_TOKEN}"}

    # Initialize progress tracker
    progress_tracker = ProgressTracker()

    locations = create_locations(amount=amount, progress_tracker=progress_tracker)

    print(f"Will process {len(locations)} unprocessed locations")
    
    # Example code how to query copernicus sentiel 2 data and do explcit image processing evals with inline script.
    # Currently reading from the eo_net wildfire json file.

    sensor = "sentinel-2-l2a"

    for location in locations:
        # Update progress for this specific location
        try:
            evalscript = """
            //VERSION=3
            function setup() {
            return {
                    input: [ "B08", "B04", "B03", "B02"],
                    output: {
                    bands: 4
                    },

                };
            }

            function evaluatePixel(sample) {
            return [2.5 * sample.B08, 2.5 * sample.B04, 2.5 * sample.B03, 2.5 * sample.B02];
            }
            """


            request = {
                "input": {
                    "bounds": {
                        # "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/32633"},
                        # "bbox": [13, 45, 14, 46],
                        "bbox": location.bbox['bbox']

                    },
                    "data": [
                        {
                            "type": sensor,
                            "dataFilter": {
                                "timeRange": {
                                    "from": location.time["from"],
                                    "to": location.time["to"],
                                }
                            },
                        }
                    ],
                },
                "output": {
                    "width": 512,
                    "height": 512,
                    "format": "image/tiff"
                },
                "evalscript": evalscript,
            }
            url = "https://sh.dataspace.copernicus.eu/api/v1/process"
            response = requests.post(url, json=request, headers=headers)

            if response.status_code == 200:
                write_image(response, metadata=request, location=location, use_gcs=use_gcs)
                # Mark location as completed
                progress_tracker.update_event_progress(location.geohash, location_index=0, status="completed")
                print(f"Completed location {location.geohash}")
            else:
                # Mark location as failed
                progress_tracker.update_event_progress(location.geohash, location_index=0, status="failed")
                print(f"{response.status_code}: error in request for {location.geohash}, outputting content for debugging {response.content}")
        except Exception as e:
            # Mark location as failed on any exception
            progress_tracker.update_event_progress(location.geohash, location_index=0, status="failed")
            print(f"Exception processing location {location.geohash}: {e}")

def batch_data_downloader_selenium(url=None, max_pages=9):
    """Downloads images from a Flickr album using Selenium.

    Args:
        url (str, optional): URL of the Flickr album. Defaults to a hardcoded URL.
        max_pages (int, optional): Maximum number of pages to scrape. Defaults to 9.

    Returns:
        int: The number of images downloaded.
    """
    # TODO: Hardcoded url for now, if needed expose this for customization
    url = "https://www.flickr.com/photos/esa_events/albums/72157716491073681/"
    destination = "./data/labeled/no"
    driver = webdriver.Chrome()  # Make sure you have chromedriver installed
    driver.get(url)
    downloaded = 0
    last_height = driver.execute_script("return document.body.scrollHeight")

    # TODO: figure out how to go past 100
    # either pagination or rate limit
    # might need to retrieve next page element.
    while True:
        # Scroll down
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)  # Wait for images to load
        # Get all image elements
        images = driver.find_elements(By.TAG_NAME, 'img')
        # Download new images
        for img in images[downloaded:]:
            src = img.get_attribute('src')
            if src and src.startswith('http'):
                try:
                    response = requests.get(src)
                    filepath = os.path.join(destination, f'image_{downloaded}.jpg')
                    with open(filepath, 'wb') as f:
                        f.write(response.content)
                    print(f"Downloaded: {filepath}")
                    downloaded += 1
                except Exception as e:
                    print(f"Error: {e}")
        # Check if we've reached the bottom
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height
    driver.quit()
    return downloaded


def convert_sen2fire_labeled(root_dir="data/sen2fire", output_dir="data/labeled",  use_nir=False):
    """
    Based on:
    Xu, Y., Berg, A., & Haglund, L. (2024). 
    Sen2Fire: A Challenging Benchmark Dataset for Wildfire Detection using Sentinel Data.
    arXiv preprint arXiv:2403.17884

    Converts Sen2Fire .npz files into RGB or RGB+NIR PNG images stored in labeled yes/no folders

    Args:
        root_dir (str): Path to Sen2Fire dataset 
        output_dir (str): Path to labeled data folder
    """
    #TODO: to access the npz files -> url:https://zenodo.org/records/10881058
    scenes = {
        # for training
        "scene1": "yes",
        "scene2": "yes",
        # for validation
        "scene3": "no",   
        "scene4": "no"
    }
    # dataset folders
    os.makedirs(os.path.join(output_dir, "yes"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "no"), exist_ok=True)

    for scene, label in scenes.items():
        scene_path = os.path.join(root_dir, scene)
        npz_files = [f for f in os.listdir(scene_path) if f.endswith(".npz")] # collect all the .npz files in the folder

        for fname in npz_files:
            fpath = os.path.join(scene_path, fname)
            try:
                data = np.load(fpath) # loading the data
                img = data["image"]  # shape: (H, W, 13) -- 13 bands
                mask = data["label"]  # shape: (H, W)

                # if fire is present if the pixel is 1
                fire_present = int(np.any(mask))
                # override if fire is actually present
                final_label = "yes" if fire_present else "no"

                # Extract RGB or RGB+NIR
                channels = [3, 2, 1]  # R, G, B
                if use_nir:
                    channels.append(7)  # NIR if present

                img_crop = img[:, :, channels]  # shape: (512, 512, 3 or 4)

                # Normalize to 0–255 for transferring into numpy img
                img_norm = (img_crop / img_crop.max()) * 255
                img_norm = img_norm.astype(np.uint8)
                # output file 
                out_file = os.path.join(output_dir, final_label, f"{scene}_{fname.replace('.npz', '.png')}") # goes into the dataset based on yes or no
                if use_nir:
                    np.save(out_file.replace(".png", ".npy"), img_norm)  # save 4-channel image as .npy or 3-channel for .png
                else:
                    Image.fromarray(img_norm).save(out_file)

            except Exception as e: # logging errors
                logger.warning(f"Failed to process {fpath}: {e}")