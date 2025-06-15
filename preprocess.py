"""All image preprocessing logic and algorithms should go here

Features
- Handles both 3-channel (RGB) and 4-channel (RGB+NIR from TIFF alpha)
- Supports streaming images directly from Google Cloud Storage
"""
import os
import cv2
import numpy as np
import logging
from typing import Optional

# Set up logging
logger = logging.getLogger(__name__)

def populate(X_array, y_array, path, use_nir=False, end=False, gcs_handler=None):
    """Populates the input arrays with preprocessed images and labels.

    Args:
        X_array (list): List to store the preprocessed images.
        y_array (list): List to store the labels corresponding to the images.
        path (str): Path to the directory containing image files.
        use_nir (bool): Whether to include the NIR channel.
        end (bool, optional): Flag to indicate whether to append default labels. Defaults to False.

    Returns:
        tuple: A tuple containing the updated X_array and y_array.
    """
    try:
        if gcs_handler:
            # Stream from GCS
            image_paths = gcs_handler.list_images(prefix=path)
            for image_path in image_paths:
                image_bytes = gcs_handler.download_as_bytes(image_path)
                if image_bytes is None:
                    print(f"Could not read {image_path}, skipping...")
                    continue
                rgb = stream_image_from_gcs(image_bytes)
                if rgb is None:
                    print(f"Could not decode {image_path}, skipping...")
                    continue

                rgb = cv2.resize(rgb, (224, 224))

                if use_nir:
                    nir = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
                    # Shape (224, 224, 1)
                    nir = np.expand_dims(nir, axis=-1)
                    # Shape (224, 224, 4)
                    rgb_nir = np.concatenate((rgb, nir), axis=-1)
                    X_array.append(rgb_nir)
                else:
                    X_array.append(rgb)

                if not end:
                    y_array.append(image_path[0:1])
        else:
            # Local files
            for image in os.listdir(path):
                image_path = os.path.join(path, image)

                rgb = cv2.imread(image_path)
                if rgb is None:
                    print(f"Could not read {image_path}, skipping...")
                    continue
                rgb = cv2.resize(rgb, (224, 224))

                if use_nir:
                    nir = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
                    # Shape (224, 224, 1)
                    nir = np.expand_dims(nir, axis=-1)
                    # Shape (224, 224, 4)
                    rgb_nir = np.concatenate((rgb, nir), axis=-1)
                    X_array.append(rgb_nir)
                else:
                    X_array.append(rgb)

                if not end:
                    y_array.append(image_path[0:1])

        if end:
            # Ensure y_array has the same length as X_array
            while len(y_array) < len(X_array):
                y_array.append("N")
    except cv2.error as e:
        print(f"CV2 error in preprocess: {e}")
        raise e

    return X_array, y_array

def stream_image_from_gcs(image_bytes: bytes) -> Optional[np.ndarray]:
    """Convert streamed image bytes from GCS to numpy array.

    Args:
        image_bytes: Raw image bytes from GCS

    Returns:
        Image as numpy array, or None if decoding fails
    """
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        # Match the behavior of for now cv2.imread() which defaults to IMREAD_COLOR
        # THis is to avoid non-homegenous shape issues. However, this means we cant support use-nir multichannel use.
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        logger.error(f"Failed to decode image: {str(e)}")
        return None