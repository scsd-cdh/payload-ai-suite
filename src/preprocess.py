"""All image preprocessing logic and algorithms should go here

Features
- Handles both 3-channel (RGB) and 4-channel (RGB+NIR from TIFF alpha)
- Supports streaming images directly from Google Cloud Storage
"""
import os
import cv2
import numpy as np
import logging
from typing import Optional, Any, Type
from pydantic import BaseModel, Field, validator

# Set up logging
logger = logging.getLogger(__name__)


class FusionInput(BaseModel):
    """Validation model for RGB-NIR fusion input."""
    image_shape: tuple = Field(..., description="Shape of input image")

    @validator('image_shape')
    def validate_shape(cls, v):
        if len(v) != 3:
            raise ValueError(f"Image must be 3-dimensional, got {len(v)} dimensions")
        if v[2] != 4:
            raise ValueError(f"Image must have 4 channels (RGB+NIR), got {v[2]} channels")
        return v


class FusionOutput(BaseModel):
    """Validation model for RGB-NIR fusion output."""
    image_shape: tuple = Field(..., description="Shape of output image")

    @validator('image_shape')
    def validate_shape(cls, v):
        if len(v) != 3:
            raise ValueError(f"Output must be 3-dimensional, got {len(v)} dimensions")
        if v[2] != 3:
            raise ValueError(f"Output must have 3 channels (RGB), got {v[2]} channels")
        return v

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
    logger.info(f"populate() called with path='{path}', use_nir={use_nir}, end={end}, gcs_handler={gcs_handler is not None}")
    logger.info(f"Initial array lengths: X_array={len(X_array)}, y_array={len(y_array)}")
    
    images_processed = 0
    images_skipped = 0
    
    try:
        if gcs_handler:
            # Stream from GCS
            logger.info(f"Listing images from GCS with prefix: {path}")
            image_paths = gcs_handler.list_images(prefix=path)
            logger.info(f"Found {len(image_paths)} images in GCS bucket with prefix '{path}'")
            
            for i, image_path in enumerate(image_paths):
                logger.debug(f"Processing GCS image {i+1}/{len(image_paths)}: {image_path}")
                
                image_bytes = gcs_handler.download_as_bytes(image_path)
                if image_bytes is None:
                    logger.warning(f"Could not read {image_path}, skipping...")
                    images_skipped += 1
                    continue
                    
                rgb = stream_image_from_gcs(image_bytes)
                if rgb is None:
                    logger.warning(f"Could not decode {image_path}, skipping...")
                    images_skipped += 1
                    continue

                logger.debug(f"Original image shape: {rgb.shape}")
                rgb = cv2.resize(rgb, (224, 224))
                logger.debug(f"Resized image shape: {rgb.shape}")

                if use_nir:
                    nir = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
                    # Shape (224, 224, 1)
                    nir = np.expand_dims(nir, axis=-1)
                    # Shape (224, 224, 4)
                    rgb_nir = np.concatenate((rgb, nir), axis=-1)
                    logger.debug(f"NIR processing - RGB shape: {rgb.shape}, NIR shape: {nir.shape}, Combined: {rgb_nir.shape}")
                    rgb_nir = dyn_zscore_normalize(rgb_nir)
                    X_array.append(rgb_nir)
                    logger.debug(f"Added 4-channel image, X_array length now: {len(X_array)}")
                else:
                    try:
                        fused_result = rgb_nir_fusion(rgb, use_enhanced_red=True)
                        logger.debug(f"RGB-NIR fusion successful, result shape: {fused_result.shape}")
                    except Exception as e:
                        logger.warning(f"{e}: issue with rgb nir fusion technique, skipping for {image_path}")
                        fused_result = rgb
                    # Normalization still applied after fallback
                    rgb = dyn_zscore_normalize(fused_result)
                    X_array.append(rgb)
                    logger.debug(f"Added 3-channel image, X_array length now: {len(X_array)}")

                if not end:
                    # Extract label from path - this might be the issue!
                    label = image_path.split('/')[-2] if '/' in image_path else 'unknown'  # Get parent directory name
                    y_array.append(label)
                    logger.debug(f"Added label '{label}', y_array length now: {len(y_array)}")
                
                images_processed += 1
                
        else:
            # Local files
            logger.info(f"Processing local directory: {path}")
            if not os.path.exists(path):
                logger.error(f"Local path does not exist: {path}")
                return X_array, y_array
                
            image_files = os.listdir(path)
            logger.info(f"Found {len(image_files)} files in local directory")
            
            for i, image in enumerate(image_files):
                image_path = os.path.join(path, image)
                logger.debug(f"Processing local image {i+1}/{len(image_files)}: {image_path}")

                rgb = cv2.imread(image_path)
                if rgb is None:
                    logger.warning(f"Could not read {image_path}, skipping...")
                    images_skipped += 1
                    continue
                    
                logger.debug(f"Original image shape: {rgb.shape}")
                rgb = cv2.resize(rgb, (224, 224))
                logger.debug(f"Resized image shape: {rgb.shape}")

                if use_nir:
                    nir = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
                    # Shape (224, 224, 1)
                    nir = np.expand_dims(nir, axis=-1)
                    # Shape (224, 224, 4)
                    rgb_nir = np.concatenate((rgb, nir), axis=-1)
                    logger.debug(f"NIR processing - RGB shape: {rgb.shape}, NIR shape: {nir.shape}, Combined: {rgb_nir.shape}")
                    rgb_nir = dyn_zscore_normalize(rgb_nir)
                    X_array.append(rgb_nir)
                    logger.debug(f"Added 4-channel image, X_array length now: {len(X_array)}")
                else:
                    # RGB NIR fusion algorthm still relevant if we have 4 channels
                    # the `use_nir` flag is soely for the AI model to take in a 4 channel input tensor
                    try:
                        fused_result = rgb_nir_fusion(rgb, use_enhanced_red=True)
                        logger.debug(f"RGB-NIR fusion successful, result shape: {fused_result.shape}")
                    except Exception as e:
                        logger.warning(f"{e}: issue with the rgb nir fusion technique. Skipping for {image_path}")
                        fused_result = rgb
                    # Normalization still applied after fallback 
                    rgb = dyn_zscore_normalize(fused_result)
                    X_array.append(rgb)
                    logger.debug(f"Added 3-channel image, X_array length now: {len(X_array)}")

                if not end:
                    # Extract label from parent directory name
                    label = os.path.basename(os.path.dirname(image_path))
                    y_array.append(label)
                    logger.debug(f"Added label '{label}', y_array length now: {len(y_array)}")
                
                images_processed += 1

        if end:
            # Ensure y_array has the same length as X_array
            initial_y_length = len(y_array)
            while len(y_array) < len(X_array):
                y_array.append("N")
            logger.info(f"Added {len(y_array) - initial_y_length} 'N' labels to balance arrays")
            
    except cv2.error as e:
        logger.error(f"CV2 error in preprocess: {e}")
        raise e

    logger.info(f"populate() complete: processed {images_processed} images, skipped {images_skipped} images")
    logger.info(f"Final array lengths: X_array={len(X_array)}, y_array={len(y_array)}")
    
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

def rgb_nir_fusion(image_data: np.ndarray[Any, np.dtype[np.integer[Any] | np.floating[Any]]],
                   use_enhanced_red: bool = False, use_hsv_fusion: bool = False):
    """
    Use nir band to "enhance" RGB image. Data fusion technique borrowed from robotics computer vision

    Two flag options:

    1. enhanced red
    alpha = 0.5
    enhanced_red = (1 - alpha) * red + alpha * nir
    fused = np.stack([enhanced_red, green, blue], axis=-1)

    2. We can try convert the image to a diferent colorspace (RGB -> HSV)
    and then we combine the brightness channel (V) with the NIR image.
    We fuse it back together using the inverse of RGB -> HSV matrix. Hue and saturation remain untouched.
    """
    logger.info(f"RGB-NIR fusion called with shape: {image_data.shape}")

    # Validate input
    try:
        input_validation = FusionInput(image_shape=image_data.shape)
    except ValueError as e:
        logger.error(f"Input validation failed: {e}")
        raise

    # Extract RGB and NIR from 4-channel input
    rgb = image_data[:, :, :3]
    nir = image_data[:, :, 3]

    if use_enhanced_red:
        logger.info("Using enhanced red fusion method")
        blue, green, red = cv2.split(rgb)
        alpha = 0.5
        enhanced_red = (1 - alpha) * red + alpha * nir
        fused = np.stack([blue, green, enhanced_red], axis=-1)
        logger.debug(f"Enhanced red fusion complete - output shape: {fused.shape}")

        # Validate output
        try:
            output_validation = FusionOutput(image_shape=fused.shape)
        except ValueError as e:
            logger.error(f"Output validation failed: {e}")
            raise

        return fused

    elif use_hsv_fusion:
        logger.info("Using HSV fusion method")
        hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        # Combine V channel with NIR
        enhanced_v = (v + nir) / 2
        fused_hsv = cv2.merge([h, s, enhanced_v])
        fused = cv2.cvtColor(fused_hsv, cv2.COLOR_HSV2BGR)
        logger.debug(f"HSV fusion complete - output shape: {fused.shape}")

        # Validate output
        try:
            output_validation = FusionOutput(image_shape=fused.shape)
        except ValueError as e:
            logger.error(f"Output validation failed: {e}")
            raise

        return fused

    else:
        logger.info("No fusion method specified, returning RGB channels only")
        return rgb
