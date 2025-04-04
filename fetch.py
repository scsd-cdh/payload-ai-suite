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
from selenium import webdriver
from selenium.webdriver.common.by import By

def nasa_firms_api():
    """Fetches data availability from NASA FIRMS API.

    This function uses the NASA FIRMS API to retrieve data availability in CSV format.
    The API key is retrieved from the environment variable `NASA_KEY`.

    Returns:
        None
    """
    # https://firms.modaps.eosdis.nasa.gov/api/ Fire Information for Resource Management System
    # /api/area/csv/[NASA_KEy]/[SOURCE]/[AREA_COORDINATES]/[DAY_RANGE]/[DATE]
    NASA_KEY = os.getenv("NASA_KEY")
    data_url = 'https://firms.modaps.eosdis.nasa.gov/api/data_availability/csv/' + NASA_KEY + '/all'
    data_frame = pd.read_csv(data_url)

def setup_auth():
    """Sets up OAuth2 authentication for Copernicus Data Space Ecosystem.

    Retrieves an access token using client credentials and returns it for use in API requests.

    Returns:
        dict: A dictionary containing the access token and related information.
    """
    # Your client credentials
    client_id = os.getenv("CLIENT_ID")
    client_secret = os.getenv("CLIENT_SECRET")

    # Create a session
    client = BackendApplicationClient(client_id=client_id)
    oauth = OAuth2Session(client=client)

    # Get token for the session
    token = oauth.fetch_token(token_url='https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token',
                            client_secret=client_secret, include_client_id=True)

    # All requests using this session will have an access token automatically added
    resp = oauth.get("https://sh.dataspace.copernicus.eu/configuration/v1/wms/instances")
    print(resp.content)
    return token

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
    destination = "../data/flickr"
    driver = webdriver.Chrome()  # Make sure you have chromedriver installed
    driver.get(url)
    downloaded = 0
    last_height = driver.execute_script("return document.body.scrollHeight")
    # TODO: figure out how to go past 100
    # either pagination or rate limit
    # might need to retrieve next page element
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

def copernicus_sentiel_query():
    """Queries Sentinel-2 and Sentinel-1 data from the Copernicus Data Space Ecosystem.

    This function uses an inline evaluation script to process Sentinel-2 bands of interest
    and retrieves data for a specified bounding box and time range.

    Sentiel-2 bands of interest
    B2: Blue
    B3: Green
    B4: Red
    B5-B8, B8a Visible and Near Infared (VNIR)

    Returns:
        None
    """
    # Need a valid eval script, specified bands, specified data range
    ACCESS_TOKEN = setup_auth()
    headers={f"Authorization" : "Bearer {ACCESS_TOKEN}"}
    # Example code how to query copernicus sentiel 2 data and do explcit image processing evals with inline script.

    # TODO: Add parameters to use FIRMS API long/lat
    request = {
    "input": {
        "bounds": {
            "bbox": [
                -59.75738525390625,
                -19.919130502461016,
                -58.7274169921875,
                -19.062117883514652,
            ],
            "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"},
        },
        "data": [
            {
                "type": "sentinel-2-l2a",
                "id": "l2a_t1",
                "dataFilter": {
                    "timeRange": {
                        "from": "2019-09-06T00:00:00Z",
                        "to": "2019-09-08T23:59:59Z",
                    }
                },
            },
            {
                "type": "sentinel-1-grd",
                "id": "s1_t1",
                "dataFilter": {
                    "timeRange": {
                        "from": "2019-09-06T00:00:00Z",
                        "to": "2019-09-08T23:59:59Z",
                    }
                },
            },
            {
                "type": "sentinel-1-grd",
                "id": "s1_t2",
                "dataFilter": {
                    "timeRange": {
                        "from": "2019-09-11T00:00:00Z",
                        "to": "2019-09-13T23:59:59Z",
                    }
                },
            },

        ]
    },
    "output": {
        "width": 1024,
        "height": 1024,
    },

    "evalscript": """
      //VERSION=3

        function setup() {
        return {
            input: ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B08a"],
            output: {
            bands: 8
            }
        };
        }

        function evaluatePixel(
        sample,
        scenes,
        inputMetadata,
        customData,
        outputMetadata
        ) {
        return [sample];
        }


    """
    }
    url = "https://sh.dataspace.copernicus.eu/api/v1/process"
    response = requests.post(url, json=request, headers=headers)
    print(response.json())
