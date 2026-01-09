import os
import shutil
import glob
import logging
import tempfile
import time
import json
from typing import List, Optional, Dict, Any
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)

from google import genai
from google.cloud import storage
from paths import get_gcs_credentials_path, resolve_path

# Uncomment to load environment variables from .env file
# from dotenv import load_dotenv
# load_dotenv()

GEMINI_AVAILABLE = False

try:
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    if client:
        GEMINI_AVAILABLE = True
except Exception as e:
    logging.warning(f"{e}: no google api key found in environment.")

class GCSHandler:
    """Handler for Google Cloud Storage operations with streaming support."""

    def __init__(self):
        # Use environment variables for configuration
        self.bucket_name = os.getenv('GCS_BUCKET_NAME')
        self.project_id = os.getenv('GCS_PROJECT_ID')
        self.credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', get_gcs_credentials_path())

        # Set up logging
        self.logger = logging.getLogger(__name__)

        # Initialize client with service account
        try:
            if not self.bucket_name:
                raise ValueError("GCS_BUCKET_NAME environment variable not set")
            if not self.project_id:
                raise ValueError("GCS_PROJECT_ID environment variable not set")
            if not self.credentials_path:
                raise ValueError("GOOGLE_APPLICATION_CREDENTIALS environment variable not set")

            self.client = storage.Client(project=self.project_id)
            self.bucket = self.client.bucket(self.bucket_name)
            self.logger.info(f"Successfully connected to GCS bucket: {self.bucket_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize GCS client: {str(e)}")
            raise

    def upload_bytes(self, data: bytes, gcs_path: str, content_type: str = None) -> bool:
        """Stream upload bytes data to GCS bucket with logging.

        Args:
            data: Bytes data to upload.
            gcs_path: Path in GCS bucket where data will be stored.
            content_type: MIME type of the data (optional).

        Returns:
            bool: True if upload successful, False otherwise.
        """
        try:
            self.logger.info(f"Starting stream upload to: gs://{self.bucket_name}/{gcs_path}")
            blob = self.bucket.blob(gcs_path)
            blob.upload_from_string(data, content_type=content_type)
            self.logger.info(f"Successfully uploaded {len(data)} bytes to {gcs_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to upload bytes to {gcs_path}: {str(e)}")
            return False

    def download_as_bytes(self, gcs_path: str) -> Optional[bytes]:
        """Stream download image as bytes.

        Args:
            gcs_path: Path in GCS bucket to download from.

        Returns:
            Optional[bytes]: Downloaded bytes data or None if failed.
        """
        try:
            self.logger.info(f"Streaming download: gs://{self.bucket_name}/{gcs_path}")
            blob = self.bucket.blob(gcs_path)
            data = blob.download_as_bytes()
            self.logger.info(f"Successfully streamed {len(data)} bytes from {gcs_path}")
            return data
        except Exception as e:
            self.logger.error(f"Failed to stream {gcs_path}: {str(e)}")
            return None

    def list_images(self, prefix: str = '') -> List[str]:
        """List images in bucket with optional prefix.

        Args:
            prefix: Path prefix to filter results (optional).

        Returns:
            List[str]: List of blob paths matching the prefix.
        """
        try:
            self.logger.info(f"Listing images with prefix: {prefix}")
            blobs = list(self.bucket.list_blobs(prefix=prefix))
            paths = [blob.name for blob in blobs if not blob.name.endswith('/')]
            self.logger.info(f"Found {len(paths)} files with prefix: {prefix}")
            return paths
        except Exception as e:
            self.logger.error(f"Failed to list images with prefix {prefix}: {str(e)}")
            return []

    def delete_blob(self, gcs_path: str) -> bool:
        """Delete a blob from GCS bucket.

        Args:
            gcs_path: Path in GCS bucket to delete.

        Returns:
            bool: True if deletion successful, False otherwise.
        """
        try:
            # Get blob metadata to verify it's an actual file
            blob = self.bucket.blob(gcs_path)

            # Check if blob exists
            if not blob.exists():
                self.logger.warning(f"Blob does not exist: {gcs_path}")
                return False

            # Reload to get metadata
            blob.reload()

            # Safety check: verify it has a size (actual file content)
            # Empty "directory" placeholders often have size 0
            if blob.size == 0:
                self.logger.warning(f"Refusing to delete zero-size blob (might be placeholder): {gcs_path}")
                return False

            # Additional safety: check for known image extensions
            valid_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp']
            if not any(gcs_path.lower().endswith(ext) for ext in valid_extensions):
                self.logger.warning(f"Refusing to delete non-image file: {gcs_path}")
                return False

            blob.delete()
            self.logger.info(f"Successfully deleted {gcs_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete {gcs_path}: {str(e)}")
            return False

def is_image_empty(image_data):
    """Check if an image is empty (all black pixels).

    Args:
        image_data: Either a file path (str) or bytes data.

    Returns:
        bool: True if image is empty (all black), False otherwise.
    """
    try:
        if isinstance(image_data, str):
            # Load image from file path
            img = cv2.imread(image_data, cv2.IMREAD_UNCHANGED)
        else:
            # Load image from bytes
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

        if img is None:
            logging.error("Failed to load image for empty check")
            return False

        # Check if all pixels are black (0)
        # This works for grayscale, RGB, and RGBA images
        is_empty = np.all(img == 0)

        if is_empty:
            logging.info("Image detected as empty (all black pixels)")

        return is_empty

    except Exception as e:
        logging.error(f"Error checking if image is empty: {str(e)}")
        # In case of error, assume not empty to avoid false positives
        return False

def multimodal_qc(file_input, file_name=None, use_gcs=False, gcs_handler=None):
    """Performs a multimodal quality control check on an image using Gemini.

    This function uploads an image to the Gemini API, asks the model to
    determine if there's a fire in the image, and then categorizes the image
    as 'fire' or 'no fire' based on the model's response. The processed image
    is then saved to a corresponding labeled directory (local or GCS).

    Args:
        file_input: Either a file path (str) for local files or bytes data
            for GCS streaming.
        file_name: The name of the file being processed (optional for local
            files).
        use_gcs: Whether to use GCS for storage operations.
        gcs_handler: GCS handler instance if use_gcs is True.

    Returns:
        str: "fire" if the model detects a fire, "no fire" otherwise.
    """
    # Determine file name
    if file_name is None:
        if isinstance(file_input, str):
            file_name = os.path.basename(file_input)
        else:
            raise ValueError("file_name must be provided when file_input is bytes")

    print(f"Processing file: {file_name}")

    # Check if image is empty before processing
    if is_image_empty(file_input):
        logging.warning(f"Skipping {file_name} - image is empty (all black pixels)")
        print(f"Skipped {file_name} - empty image detected")
        return "skipped_empty"

    # Upload to Gemini API - handle both file path and bytes
    if isinstance(file_input, str):
        # Local file path
        gemini_file = client.files.upload(file=file_input)
    else:
        # Bytes data from GCS needs temporary file for Gemini API
        with tempfile.NamedTemporaryFile(suffix=os.path.splitext(file_name)[1], delete=False) as tmp:
            tmp.write(file_input)
            tmp_path = tmp.name
        try:
            gemini_file = client.files.upload(file=tmp_path)
        finally:
            os.unlink(tmp_path)

    # Implement exponential backoff for API calls
    max_retries = 5
    base_delay = 60  # Start with 1 minute

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=[gemini_file, "Is there a fire in this image? Respond with 'yes' if there is a fire, or 'no' if there is no fire."],
            )
            raw_response_text = response.text.strip().lower()
            break  # Success, exit retry loop
        except Exception as e:
            if attempt == max_retries - 1:
                # Last attempt failed, re-raise the exception
                raise

            # Calculate delay with exponential backoff: 1min, 2min, 4min, 8min
            delay = base_delay * (2 ** attempt)
            delay_minutes = delay / 60
            logging.warning(f"API request failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
            logging.info(f"Retrying in {delay_minutes:.1f} minutes...")
            time.sleep(delay)
    logging.debug(f"Raw response for {file_name}: {raw_response_text}")

    if "yes" in raw_response_text and "no" not in raw_response_text:
        binary_output = "fire"
    elif "no" in raw_response_text:
        binary_output = "no fire"
    else:
        logging.warning(f"Ambiguous response for {file_name}: {raw_response_text}. Defaulting to 'no fire'.")
        binary_output = "no fire"

    logging.info(f"Binary output for {file_name}: {binary_output}")

    # Determine output path
    if binary_output == "fire":
        output_path = "labeled/yes" if use_gcs else resolve_path("data/labeled/yes")
    else:
        output_path = "labeled/no" if use_gcs else resolve_path("data/labeled/no")

    # Save to appropriate location
    if use_gcs and gcs_handler:
        # Save to GCS
        gcs_path = os.path.join(output_path, file_name)
        if isinstance(file_input, str):
            # Need to read file contents for GCS upload
            with open(file_input, 'rb') as f:
                data = f.read()
            success = gcs_handler.upload_bytes(data, gcs_path, content_type='image/tiff')
        else:
            # Bytes already in memory, direct upload
            success = gcs_handler.upload_bytes(file_input, gcs_path, content_type='image/tiff')

        if success:
            print(f"File saved to: gs://{gcs_handler.bucket_name}/{gcs_path}")
        else:
            logging.error(f"Failed to save file to GCS: {gcs_path}")
    else:
        # Save locally
        local_output_dir = os.path.join(".", output_path)
        os.makedirs(local_output_dir, exist_ok=True)
        destination_path = os.path.join(local_output_dir, file_name)

        if isinstance(file_input, str):
            # Move file instead of copy for local storage
            shutil.move(file_input, destination_path)
        else:
            # Write bytes data to local file
            with open(destination_path, 'wb') as f:
                f.write(file_input)

        print(f"File moved to: {destination_path}")

    return binary_output

def run_multimodal_qc(use_gcs=False, input_path=None):
    """Orchestrates the multimodal quality control process for a batch of images.

    This function identifies image files either locally or in GCS and processes
    each of them using the multimodal_qc function.

    Args:
        use_gcs: Whether to use Google Cloud Storage for reading/writing files.
        input_path: Path to the folder containing images to process.
                   Defaults to 'eonet_fire_events/to_process' if not specified.
    """
    if use_gcs:
        # Initialize GCS handler
        try:
            from fetch import Source

            gcs_handler = GCSHandler()
            logging.info("Using GCS for multimodal QC")

            for source in [ s.value for s in Source]:
                # List images from GCS
                prefix = f"raw_data/{source}/to_process/"
                image_files = gcs_handler.list_images(prefix=prefix)

                if not image_files:
                    print(f"No images found in GCS at gs://{gcs_handler.bucket_name}/{prefix}")
                    return

                print(f"Found {len(image_files)} images in GCS to process")

                # Process each image from GCS
                for gcs_path in image_files:
                    # GCS lists directories as paths ending with /
                    if gcs_path.endswith('/'):
                        continue

                    # Stream download from GCS
                    image_bytes = gcs_handler.download_as_bytes(gcs_path)
                    if image_bytes is None:
                        logging.error(f"Failed to download {gcs_path}, skipping")
                        continue

                    # Extract just the filename from full GCS path
                    file_name = os.path.basename(gcs_path)

                    # Run QC on streamed image
                    try:
                        result = multimodal_qc(image_bytes, file_name, use_gcs=True, gcs_handler=gcs_handler)
                        if result == "skipped_empty":
                            logging.info(f"Skipped empty image: {file_name}")
                            # Delete the empty image from GCS
                            gcs_handler.delete_blob(gcs_path)
                        elif result in ["fire", "no fire"]:
                            # Successfully processed and moved - delete original
                            gcs_handler.delete_blob(gcs_path)
                    except Exception as e:
                        logging.error(f"Error processing {file_name}: {str(e)}")
                        continue

        except Exception as e:
            logging.error(f"Failed to initialize GCS handler: {str(e)}")
            logging.info("Falling back to local processing due to GCS error")
            use_gcs = False

        if not use_gcs:
            # Standard local file processing
            if input_path:
                # Use provided path
                base_path = resolve_path(f"data/{input_path}")
            else:
                # Default to eonet path
                base_path = resolve_path("data/eonet_fire_events/to_process")

            image_files = glob.glob(f"{base_path}/*")

            if not image_files:
                print(f"No images found locally in {base_path}")
                return

            print(f"Found {len(image_files)} images locally to process in {base_path}")

            # Process each file from local directory
            if GEMINI_AVAILABLE:
                for file_path in image_files:
                    try:
                        result = multimodal_qc(file_path)
                        if result == "skipped_empty":
                            logging.info(f"Skipped empty image: {file_path}")
                    except Exception as e:
                        logging.error(f"Error processing {file_path}: {str(e)}")
                        continue
            else:
                logging.error(f"Multimodal qc not available. Genai client not set")


def upload_labeled_to_gcs():
    """Upload local labeled data to Google Cloud Storage after cleaning up files.

    This function:
    1. Runs the clean_up_files.py script to remove duplicates and empty files
    2. Uploads all files from ../data/labeled/yes and ../data/labeled/no to GCS
    3. Uses cloud directory structure: labeled/yes and labeled/no
    """
    import subprocess
    import os
    import glob

    print("=" * 80)
    print("Starting labeled data upload to GCS")
    print("=" * 80)

    # Step 1: Run cleanup script using uv
    print("\nStep 1: Running clean_up_files.py to remove duplicates and empty files...")
    cleanup_script = "clean_up_files.py"

    try:
        # Run the cleanup script - it will prompt for confirmation
        result = subprocess.run(
            ["uv", "run", cleanup_script],
            text=True,
            cwd=os.path.dirname(__file__)
        )

        if result.returncode == 0:
            logging.info("Cleanup completed successfully!")
            if result.stdout:
                logging.debug(result.stdout)
        else:
            logging.warning(f"Cleanup script returned non-zero exit code: {result.returncode}")
            if result.stderr:
                logging.error("Error output:", result.stderr)
            response = input("\nDo you want to continue with upload anyway? (yes/no): ").strip().lower()
            if response != 'yes':
                print("Upload cancelled.")
                return
    except Exception as e:
        logging.error(f"Error running cleanup script: {e}")
        response = input("\nDo you want to continue with upload anyway? (yes/no): ").strip().lower()
        if response != 'yes':
            print("Upload cancelled.")
            return

    # Step 2: Initialize GCS handler
    print("\nStep 2: Initializing Google Cloud Storage connection...")
    try:
        gcs_handler = GCSHandler()
    except Exception as e:
        logging.error(f"Failed to initialize GCS handler: {e}")
        logging.error("Make sure GCS environment variables are set correctly.")
        return

    # Step 3: Upload files
    print("\nStep 3: Uploading labeled data to GCS...")

    labels = ["yes", "no"]
    total_uploaded = 0
    failed_uploads = []

    for label in labels:
        local_path = resolve_path(f"data/labeled/{label}")
        # Cloud structure is different - no 'data' prefix
        gcs_base_path = f"labeled/{label}"

        # Find all files in the directory
        pattern = os.path.join(local_path, "*")
        files = glob.glob(pattern)

        print(f"\nUploading {len(files)} files from {local_path} to gs://{gcs_handler.bucket_name}/{gcs_base_path}/")

        for file_path in files:
            if os.path.isfile(file_path):
                filename = os.path.basename(file_path)
                gcs_path = f"{gcs_base_path}/{filename}"

                try:
                    with open(file_path, 'rb') as f:
                        data = f.read()

                    # Determine content type based on file extension
                    ext = os.path.splitext(filename)[1].lower()
                    content_type = {
                        '.png': 'image/png',
                        '.jpg': 'image/jpeg',
                        '.jpeg': 'image/jpeg',
                        '.tiff': 'image/tiff',
                        '.tif': 'image/tiff'
                    }.get(ext, 'application/octet-stream')

                    success = gcs_handler.upload_bytes(data, gcs_path, content_type=content_type)

                    if success:
                        total_uploaded += 1
                        if total_uploaded % 50 == 0:  # Progress indicator every 50 files
                            print(f"  Progress: {total_uploaded} files uploaded...")
                    else:
                        failed_uploads.append(file_path)

                except Exception as e:
                    logging.error(f"Error uploading {filename}: {e}")
                    failed_uploads.append(file_path)

    # Step 4: Summary
    print("\n" + "=" * 80)
    print("Upload Summary:")
    print(f"- Total files uploaded: {total_uploaded}")
    print(f"- Failed uploads: {len(failed_uploads)}")

    if failed_uploads:
        print("\nFailed files:")
        for f in failed_uploads[:10]:  # Show first 10
            print(f"  - {f}")
        if len(failed_uploads) > 10:
            print(f"  ... and {len(failed_uploads) - 10} more")

    print("\nUpload complete!")
    print(f"Data is now available at: gs://{gcs_handler.bucket_name}/labeled/")
    print("=" * 80)


def download_labeled_from_gcs():
    """Download labeled data from Google Cloud Storage to local filesystem.

    This function:
    1. Prompts user for confirmation before downloading
    2. Downloads all files from GCS labeled/yes and labeled/no
    3. Saves them to local data/labeled/yes and data/labeled/no directories
    4. Creates directories if they don't exist
    5. Shows progress and summary
    """
    import os

    # Set up logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    logger.info("=" * 80)
    logger.info("Download Labeled Data from Google Cloud Storage")
    logger.info("=" * 80)

    # Step 1: Get user confirmation
    logger.info("\nThis will download all labeled data from Google Cloud Storage.")
    logger.info("Files will be saved to:")
    logger.info("  - data/labeled/yes/")
    logger.info("  - data/labeled/no/")
    logger.info("\nExisting files with the same names will be overwritten.")

    response = input("\nDo you want to continue? (yes/no): ").strip().lower()
    if response != 'yes':
        logger.info("Download cancelled.")
        return

    # Step 2: Initialize GCS handler
    logger.info("Initializing Google Cloud Storage connection...")
    try:
        gcs_handler = GCSHandler()
    except Exception as e:
        logger.error(f"Failed to initialize GCS handler: {e}")
        logger.error("Make sure GCS environment variables are set correctly.")
        return

    # Step 3: Create local directories if they don't exist
    labels = ["yes", "no"]
    for label in labels:
        local_dir = resolve_path(f"data/labeled/{label}")
        os.makedirs(local_dir, exist_ok=True)
        logger.debug(f"Created/verified directory: {local_dir}")

    # Step 4: Download files
    logger.info("Downloading labeled data from GCS...")

    total_downloaded = 0
    failed_downloads = []

    for label in labels:
        gcs_prefix = f"labeled/{label}/"
        local_dir = resolve_path(f"data/labeled/{label}")

        # List all files in GCS with this prefix
        try:
            blobs = list(gcs_handler.bucket.list_blobs(prefix=gcs_prefix))
            files = [blob for blob in blobs if not blob.name.endswith('/')]

            logger.info(f"Found {len(files)} files in gs://{gcs_handler.bucket_name}/{gcs_prefix}")
            logger.info(f"Downloading to {local_dir}/...")

            for blob in files:
                filename = os.path.basename(blob.name)
                local_path = os.path.join(local_dir, filename)

                try:
                    # Download the file
                    logger.debug(f"Downloading {blob.name} to {local_path}")
                    data = blob.download_as_bytes()

                    # Save to local filesystem
                    with open(local_path, 'wb') as f:
                        f.write(data)

                    total_downloaded += 1
                    if total_downloaded % 50 == 0:  # Progress indicator every 50 files
                        logger.info(f"Progress: {total_downloaded} files downloaded...")

                except Exception as e:
                    logger.error(f"Failed to download {blob.name}: {e}")
                    failed_downloads.append((blob.name, str(e)))

        except Exception as e:
            logger.error(f"Error listing files with prefix {gcs_prefix}: {e}")
            continue

    # Step 5: Summary
    logger.info("=" * 80)
    logger.info("Download Summary:")
    logger.info(f"- Total files downloaded: {total_downloaded}")
    logger.info(f"- Failed downloads: {len(failed_downloads)}")

    if failed_downloads:
        logger.warning("Failed files:")
        for blob_name, error in failed_downloads[:10]:  # Show first 10
            logger.warning(f"  - {blob_name}: {error}")
        if len(failed_downloads) > 10:
            logger.warning(f"  ... and {len(failed_downloads) - 10} more")

    logger.info("Download complete!")
    logger.info("Data is now available at:")
    logger.info("  - data/labeled/yes/")
    logger.info("  - data/labeled/no/")
    logger.info("=" * 80)
