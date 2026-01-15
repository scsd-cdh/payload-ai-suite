
import os
import requests
from requests.auth import HTTPBasicAuth
import pystac_client
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_demo():
    """
    Downloads a single representative EnMAP L2A Spectral Image for inspection.
    """
    # 1. Credentials
    DLR_USER = os.getenv("DLR_USER")
    DLR_PASS = os.getenv("DLR_PASS")
    
    if not DLR_USER or not DLR_PASS:
        logger.error("Error: DLR_USER and DLR_PASS environment variables must be set.")
        logger.info("Usage: export DLR_USER=your_user; export DLR_PASS=your_pass; python download_demo_image.py")
        return

    # 2. Configuration
    STAC_URL = "https://geoservice.dlr.de/eoc/ogc/stac/v1"
    COLLECTION_ID = "ENMAP_HSI_L2A"
    
    # We use a specific, recent, valid Item ID to ensure the demo always works
    # This item has been verified to exist and have an 'image' asset
    DEMO_ITEM_ID = "ENMAP01-____L2A-DT0000173751_20260112T183120Z_003_V010505_20260113T044035Z"

    OUTPUT_DIR = "enmap_sample"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        # 3. Setup Session
        session = requests.Session()
        session.auth = HTTPBasicAuth(DLR_USER, DLR_PASS)
        
        logger.info("Connecting to DLR STAC API...")
        client = pystac_client.Client.open(STAC_URL)
        
        # 4. Find the Item
        logger.info(f"Locating demo item: {DEMO_ITEM_ID}...")
        search = client.search(
            collections=[COLLECTION_ID],
            ids=[DEMO_ITEM_ID],
            method='GET' # Force GET for stability
        )
        
        items = list(search.items())
        if not items:
            logger.error("Demo item not found! The ID might have expired or been removed.")
            return

        item = items[0]
        
        # 5. Identify the Asset
        # We want 'image' (The Spectral Image COG)
        ASSET_KEY = 'image'
        if ASSET_KEY not in item.assets:
            logger.error(f"Asset '{ASSET_KEY}' not found in item.")
            return
            
        asset = item.assets[ASSET_KEY]
        href = asset.href
        filename = href.split('/')[-1].split('?')[0]
        output_path = os.path.join(OUTPUT_DIR, filename)
        
        logger.info(f"Found Asset: {asset.title} ({asset.media_type})")
        logger.info(f"Download URL: {href}")
        logger.info(f"Target File: {output_path}")
        
        # 6. Download
        logger.info("Starting download (approx 400MB)... please wait.")
        with session.get(href, stream=True) as r:
            if r.status_code == 403:
                logger.error("403 Forbidden. Check DLR_USER/DLR_PASS permissions.")
                return
            r.raise_for_status()
            
            total_size = int(r.headers.get('content-length', 0))
            downloaded = 0
            
            with open(output_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192 * 4):
                    f.write(chunk)
                    downloaded += len(chunk)
                    # Simple progress indicator
                    if total_size:
                        percent = (downloaded / total_size) * 100
                        sys.stdout.write(f"\rProgress: {percent:.1f}% ({downloaded/(1024*1024):.1f} MB)")
                        sys.stdout.flush()
        
        print() # Newline after progress
        logger.info("Download Complete!")
        logger.info(f"You can find the file here: {os.path.abspath(output_path)}")

    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    download_demo()
