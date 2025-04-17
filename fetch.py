"""Utilities for fetching data for the model
- Selenium based flickr webscraper
- EONET wildfire cross reference tool
"""

import requests
import json
import time
import os


from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session

import pandas as pd
import pygeohash as pgh
from selenium import webdriver
from selenium.webdriver.common.by import By
from pyproj import Proj, transform


def nasa_firms_api():
    """
    Fetches data availability from NASA FIRMS API.

    This function uses the NASA FIRMS API to retrieve data availability in CSV format.
    The API key is retrieved from the environment variable `NASA_KEY`.

    https://firms.modaps.eosdis.nasa.gov/api/ Fire Information for Resource Management System
    /api/area/csv/[NASA_KEy]/[SOURCE]/[AREA_COORDINATES]/[DAY_RANGE]/[DATE]

    Returns:
        pd.DataFrame: A DataFrame containing the data availability information for NASA FIRMS.
    """
    # TODO: Implement AREA_COORDINATES and DAY_RANGE etc. paramters for API call
    NASA_KEY = os.getenv("NASA_KEY")
    data_url = 'https://firms.modaps.eosdis.nasa.gov/api/data_availability/csv/' + NASA_KEY + '/all'
    data_frame = pd.read_csv(data_url)
    print(data_frame)
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

def write_image(response, metadata, location):
    """Writes image data from an API response to a file.

    This function extracts image data (e.g., PNG) from the response object and writes it to a file.
    The filename is generated based on the metadata provided, such as the output format.

    Args:
        response (requests.Response): The response object containing the image data.
        metadata (dict): A dictionary containing metadata about the request, such as output format,
                         dimensions, and other relevant information.

    Returns:
        None

    Raises:
        IOError: If there is an error writing the image to a file.
    """
    try:
        # Extract the output format from metadata (default to PNG)
        output_format = metadata.get('output', {}).get('format', 'image/png').split('/')[-1]
        # TODO: write custom logic for filename to be populated by metadata satellite type and bands
        filename = f"./data/copernicus/{location.geohash}.{output_format}"

        # Write the reponse content data to a image file
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"Image successfully saved to {filename}")
    except Exception as e:
        print(f"Error writing image: {e}")

def convert_coords_to_bbox(longitude, latitude, buffer_distance=2):
    """
    Converts longitude and latitude to a bounding box (bbox).

    Args:
        longitude (float): Longitude of the central point.
        latitude (float): Latitude of the central point.
        buffer_distance (float): Buffer distance in meters for the bbox. Default is 2 meters.

    Returns:
        dict: A dictionary containing the bbox as a list of coordinates.
    """
    # Define the projection for WGS84 (lat/lon) and a local UTM projection
    wgs84 = Proj(proj="latlong", datum="WGS84")
    utm = Proj(proj="utm", zone=int((longitude + 180) / 6) + 1, datum="WGS84")

    # Transform the central point to UTM coordinates
    x_center, y_center = transform(wgs84, utm, longitude, latitude)

    # Calculate the bbox in UTM coordinates
    x_min = x_center - buffer_distance
    x_max = x_center + buffer_distance
    y_min = y_center - buffer_distance
    y_max = y_center + buffer_distance

    # Transform the bbox corners back to WGS84 if needed
    bbox = [x_min, y_min, x_max, y_max]
    return {"bbox": bbox}

class Location:
    def __init__(self, coordinates, time, geohash=None, bbox=None):
        self.coordinates = coordinates
        self.time = time
        self.geohash = self.create_geohash(coordinates)
        self.bbox = convert_coords_to_bbox(coordinates[0], coordinates[1])
    def create_geohash(self, coordinates):
        geohash = pgh.encode(latitude=coordinates[0], longitude=coordinates[1])
        return geohash

def create_locations(amount=5):
    """Creates a list of Location objects based on EONET data.

    This function extracts time ranges and coordinates from the EONET wildfire data
    and uses them to create Location objects. The number of locations created is
    determined by the `amount` parameter.

    Args:
        amount (int): The number of Location objects to create. Defaults to 5.

    Returns:
        list: A list of Location objects.
    """
    # list of dict entries in the form {'from': '2023-08-05T17:59:00Z', 'to': '2023-08-07T17:59:00Z'}
    locations = []
    time_ranges = extract_time_ranges_from_eonet()
    coordinates = extract_eonet_coordinates()
    # amount is a pre-determined parameter. the current value is arbitrary at this point
    for entry in range(amount):
        location = Location(coordinates[0][entry], time=time_ranges[entry])
        locations.append(location)
    return locations

def copernicus_sentiel_query():
    """Queries Sentinel-2 and Sentinel-1 data from the Copernicus Data Space Ecosystem.

    This function uses an inline evaluation script to process Sentinel-2 bands of interest
    and retrieves data for a specified bounding box and time range.

    Sentiel-2 bands of interest
    B2: Blue
    B3: Green
    B4: Red
    B8: Visible and Near Infared (VNIR)

    Returns:
        None
    """
    # Need a valid eval script, specified bands, specified data range
    ACCESS_TOKEN = setup_auth()
    headers={f"Authorization" : f"Bearer {ACCESS_TOKEN}"}

    locations = create_locations()

    # Example code how to query copernicus sentiel 2 data and do explcit image processing evals with inline script.
    # Currently reading from the eo_net wildfire json file.
    # TODO: need logic for long/lat to be converted to a proper bounding box

    evalscript = """
    //VERSION=3
    function setup() {
    return {
            input: ["B02", "B03", "B04"],
            output: {
            bands: 3
            },

        };
    }

    function evaluatePixel(sample) {
    return [sample.B04, sample.B03, sample.B02];
    }
    """

    request = {
        "input": {
            "bounds": {
                "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/32633"},
                "bbox": locations[0].bbox['bbox']

            },
            "data": [
                {
                    "type": "sentinel-2-l2a",
                    "dataFilter": {
                        "timeRange": {
                            "from": locations[0].time["from"],
                            "to": locations[0].time["to"],
                        }
                    },
                }
            ],
        },
        "output": {
            "width": 512,
            "height": 512,
            "format": "image/png"
        },
        "evalscript": evalscript,
    }
    url = "https://sh.dataspace.copernicus.eu/api/v1/process"
    response = requests.post(url, json=request, headers=headers)

    if response.status_code == 200:
        write_image(response, metadata=request, location=locations[0])
    else:
        print(f"{response.status_code}: error in request, outputting content for debugging {response.content}")

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

