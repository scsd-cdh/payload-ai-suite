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
                    rgb_nir = dyn_zscore_normalize(rgb_nir) # normalization
                   # print("normalized rgb-nir":, rgb_nir.mean(), rgb_nir.std())
                    X_array.append(rgb_nir)
                else:
                    rgb = dyn_zscore_normalize(rgb) # normalization
                    #print("normalized rgb:", rgb.mean(), rgb_nir.std())
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
        return


# sassan ghazi - 2025/07/01
# issue: z-score preprocess for zetane

def dyn_zscore_normalize(img: np.ndarray, no_data_value: float = 0.0) -> np.ndarray:
    '''
    mimic the behaviour in omnicloudmask(pytorch) using the numpy/openCV for image arrays

    Requirements:
    -  per channel z-score normalization
    -  exclude no-data pixels from mean & standard deviation calculation (value: 0.0)
    -  set no-data pixels to 0 after normalization
    -  function should work for both 3- and 4-channel images

    '''
    logger.debug(f"Starting z-score normalization - Input shape: {img.shape}, dtype: {img.dtype}")

    img = img.astype(np.float32) # 32-bit float for images
    normal_img = np.zeros_like(img) # storing normalized values with an empty output image

    for c in range(img.shape[2]):
        channel = img[:,:, c] # by iterating through every channel in the images
        mask = channel != no_data_value # ignoring 0-values.... pixels are valid if they are not equal to the no_data_value
        
        valid_pixel_count = np.sum(mask)
        logger.debug(f"Channel {c}: Found {valid_pixel_count} valid pixels out of {channel.size} total")

        if np.any(mask):
            valid_pxl = channel[mask] # non-zero pixels for the channel

            mean = valid_pxl.mean() # computation for the mean
            standard_dev = valid_pxl.std() # computation for the standard deviation
            
            logger.debug(f"Channel {c}: Mean={mean:.4f}, Std={standard_dev:.4f}")

            standard_dev = standard_dev if standard_dev > 1e-8 else 1e-8 # checking for error by division of 0
            # set the number to a smaller number if too close to 0
            
            if standard_dev < 1e-8:
                logger.warning(f"Channel {c}: Very low standard deviation ({standard_dev}), using 1e-8 to avoid division by zero")

            normal_channel = (channel - mean) / standard_dev # applying normalization to z-score
            #TODO: all pixels here will be normalized. will look into this later on
            normal_channel[~mask] = 0.0 # 0.0 is assigned to pixels that were identified as 0

            normal_img[:,:,c] = normal_channel
            
            # Log statistics of normalized channel
            normalized_valid = normal_channel[mask]
            logger.debug(f"Channel {c} after normalization: Mean={normalized_valid.mean():.4f}, Std={normalized_valid.std():.4f}")

        else:
            normal_img[:,:,c] = 0.0 # filled with 0s for no-data
            logger.warning(f"Channel {c}: No valid pixels found, filling with zeros")

    logger.debug(f"Z-score normalization complete - Output shape: {normal_img.shape}")
    return normal_img # should return image with the same shape that was given by the input
