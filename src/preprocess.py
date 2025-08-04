"""All image preprocessing logic and algorithms should go here

Features
- Handles both 3-channel (RGB) and 4-channel (RGB+NIR from TIFF alpha)
- Supports streaming images directly from Google Cloud Storage
- Includes mixed resolution operations for resolution-agnostic training
"""
import os
import cv2
import numpy as np
import random
import logging
from typing import Optional, List, Tuple, Union
from src.mixed_res_config import DEFAULT_MIXED_RES_CONFIG

# Set up logging
logger = logging.getLogger(__name__)

def random_resize_with_pad(image: np.ndarray, min_scale: float = 0.5, max_scale: float = 1.5) -> np.ndarray:
    """
    Randomly resize an image, then pad to maintain original dimensions.
    This simulates images of different resolutions during training.
    
    Args:
        image: Input image (HxWxC)
        min_scale: Minimum scaling factor
        max_scale: Maximum scaling factor
    
    Returns:
        Resized and padded image with original dimensions
    """
    logger.debug(f"Starting random_resize_with_pad - Input shape: {image.shape}")
    
    original_height, original_width = image.shape[:2]
    
    # Choose random scale factor
    scale = min_scale + random.random() * (max_scale - min_scale)
    
    # Calculate new dimensions
    new_height = int(original_height * scale)
    new_width = int(original_width * scale)
    
    # Ensure new dimensions are at least 1 pixel
    new_height = max(1, new_height)
    new_width = max(1, new_width)
    
    logger.debug(f"Resizing with scale {scale:.2f} to dimensions {new_width}x{new_height}")
    
    # If the scale would produce an image too large, cap it to the original size
    # This is done to prevent issues with padding calculations
    if new_height > original_height:
        new_height = original_height
    if new_width > original_width:
        new_width = original_width
    
    # Choose appropriate interpolation method based on scale
    # INTER_AREA is recommended for downsampling to avoid aliasing
    # INTER_LINEAR is better for upsampling
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        
    # Resize image with proper interpolation
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=interpolation)
    
    # Create canvas with original dimensions
    if len(image.shape) == 3:
        canvas = np.zeros((original_height, original_width, image.shape[2]), dtype=image.dtype)
    else:
        canvas = np.zeros((original_height, original_width), dtype=image.dtype)
    
    # Calculate padding
    y_offset = (original_height - new_height) // 2
    x_offset = (original_width - new_width) // 2
    
    # Place resized image in center of canvas - with safe bounds checking
    resized_h, resized_w = resized_image.shape[:2]
    
    # Calculate the valid regions for both the canvas and resized image
    canvas_y_end = min(y_offset + resized_h, original_height)
    canvas_x_end = min(x_offset + resized_w, original_width)
    
    # Calculate how much of the resized image can fit
    resized_y_end = canvas_y_end - y_offset
    resized_x_end = canvas_x_end - x_offset
    
    # Now do the assignment with properly matched dimensions
    canvas[y_offset:canvas_y_end, x_offset:canvas_x_end] = resized_image[:resized_y_end, :resized_x_end]
    
    logger.debug(f"Completed random_resize_with_pad - Output shape: {canvas.shape}")
    return canvas


def multi_resolution_batch(images: List[np.ndarray], resolution_scales: List[float] = None) -> List[np.ndarray]:
    """
    Apply different resolutions to a batch of images.
    Each image in the batch gets a randomly selected resolution scale.
    This helps train models to be resolution-agnostic by simulating 
    imagery captured at different resolutions.
    
    Args:
        images: List of input images
        resolution_scales: List of scale factors to apply. If None, defaults to [0.5, 0.75, 1.0, 1.25, 1.5]
    
    Returns:
        List of images with varied resolutions
    """
    if resolution_scales is None:
        resolution_scales = [0.5, 0.75, 1.0, 1.25, 1.5]
    
    logger.debug(f"Starting multi_resolution_batch with {len(images)} images")
    
    result = []
    for img in images:
        # Randomly select a scale
        scale = random.choice(resolution_scales)
        height, width = img.shape[:2]
        
        # Calculate new dimensions
        new_height = int(height * scale)
        new_width = int(width * scale)
        
        # Ensure new dimensions are at least 1 pixel
        new_height = max(1, new_height)
        new_width = max(1, new_width)
        
        logger.debug(f"Applying scale {scale:.2f} to image")
        
        # First downscale
        if scale < 1.0:
            # Downscale to lower resolution
            temp_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
            # Then upscale back to original size
            processed_img = cv2.resize(temp_img, (width, height), interpolation=cv2.INTER_LINEAR)
        elif scale > 1.0:
            # For upscaling, we'll simulate having a higher resolution image by upscaling
            # Then downscale back to original to simulate detail loss
            temp_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            processed_img = cv2.resize(temp_img, (width, height), interpolation=cv2.INTER_AREA)
        else:
            # No change for scale = 1.0
            processed_img = img.copy()
        
        result.append(processed_img)
    
    logger.debug(f"Completed multi_resolution_batch - Processed {len(result)} images")
    return result


def resolution_mixup(image: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    """
    Apply mixup between different resolution versions of the same image.
    This is based on the mixup data augmentation technique (Zhang et al., 2017),
    but applied to resolution variations instead of class labels.
    
    Args:
        image: Input image
        alpha: Mixing weight parameter (controls randomness of mix ratio)
        
    Returns:
        Mixed resolution image
    """
    logger.debug(f"Starting resolution_mixup - Input shape: {image.shape}")
    
    # Choose a random downsampling factor
    down_factor = 0.4 + random.random() * 0.3  # Between 0.4 and 0.7
    
    height, width = image.shape[:2]
    
    # Calculate downsampled dimensions
    down_height = int(height * down_factor)
    down_width = int(width * down_factor)
    
    # Ensure dimensions are at least 1 pixel
    down_height = max(1, down_height)
    down_width = max(1, down_width)
    
    # Downsample and then upsample to get lower resolution version
    downsampled = cv2.resize(image, (down_width, down_height), interpolation=cv2.INTER_AREA)
    low_res = cv2.resize(downsampled, (width, height), interpolation=cv2.INTER_LINEAR)
    
    # Generate mixing weight from beta distribution or simplified random
    mix_ratio = alpha + random.random() * (1.0 - alpha)
    
    # Mix original and low-resolution images using cv2.addWeighted for better performance
    mixed_image = cv2.addWeighted(image, mix_ratio, low_res, 1.0 - mix_ratio, 0)
    
    logger.debug(f"Completed resolution_mixup with down_factor={down_factor:.2f}, mix_ratio={mix_ratio:.2f}")
    return mixed_image


def apply_mixed_resolution_ops(images: List[np.ndarray], 
                              use_random_resize: bool = True, 
                              use_multi_resolution: bool = True, 
                              use_resolution_mixup: bool = True,
                              **kwargs) -> List[np.ndarray]:
    """
    Apply a combination of mixed resolution operations to a batch of images.
    
    Args:
        images: List of input images
        use_random_resize: Whether to apply random resize with padding
        use_multi_resolution: Whether to apply multi-resolution batch processing
        use_resolution_mixup: Whether to apply resolution mixup
        **kwargs: Additional parameters for the operations
        
    Returns:
        List of processed images
    
    Note: This function makes a copy of the list but modifies images in-place.
    """
    logger.debug(f"Starting apply_mixed_resolution_ops on {len(images)} images")
    
    processed_images = images.copy()  # Makes a copy of the list, but not the images themselves
    
    if use_random_resize:
        min_scale = kwargs.get('min_scale', 0.5)
        max_scale = kwargs.get('max_scale', 1.5)
        processed_images = [random_resize_with_pad(img, min_scale, max_scale) for img in processed_images]
    
    if use_multi_resolution:
        resolution_scales = kwargs.get('resolution_scales', [0.5, 0.75, 1.0, 1.25, 1.5])
        processed_images = multi_resolution_batch(processed_images, resolution_scales)
    
    if use_resolution_mixup:
        alpha = kwargs.get('mixup_alpha', 0.2)
        processed_images = [resolution_mixup(img, alpha) for img in processed_images]
    
    logger.debug(f"Completed apply_mixed_resolution_ops")
    return processed_images


def populate(X_array, y_array, path, use_nir=False, end=False, gcs_handler=None, 
             use_mixed_res=False, mixed_res_config=None):
    """Populates the input arrays with preprocessed images and labels.

    Args:
        X_array (list): List to store the preprocessed images.
        y_array (list): List to store the labels corresponding to the images.
        path (str): Path to the directory containing image files.
        use_nir (bool): Whether to include the NIR channel.
        end (bool, optional): Flag to indicate whether to append default labels. Defaults to False.
        gcs_handler: Handler for Google Cloud Storage.
        use_mixed_res (bool): Whether to apply mixed resolution operations.
        mixed_res_config (dict): Configuration for mixed resolution operations.

    Returns:
        tuple: A tuple containing the updated X_array and y_array.
    """
    # Default configuration for mixed resolution operations
    if use_mixed_res and mixed_res_config is None:
        mixed_res_config = DEFAULT_MIXED_RES_CONFIG
    
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

                # IMPORTANT: Apply mixed resolution operations BEFORE resizing to fixed size
                if use_mixed_res:
                    rgb = apply_mixed_resolution_ops([rgb], **mixed_res_config)[0]
                
                # AFTER augmentation, resize to fixed size for the model
                rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)

                if use_nir:
                    nir = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
                    # Shape (224, 224, 1)
                    nir = np.expand_dims(nir, axis=-1)
                    # Shape (224, 224, 4)
                    rgb_nir = np.concatenate((rgb, nir), axis=-1)
                    rgb_nir = dyn_zscore_normalize(rgb_nir) # normalization
                    X_array.append(rgb_nir)
                else:
                    rgb = dyn_zscore_normalize(rgb) # normalization
                    X_array.append(rgb)

                if not end:
                    y_array.append(image_path[0:1])
        else:
            # Local files
            for image in os.listdir(path):
                # Skip Windows metadata files
                if image.endswith(':Zone.Identifier'):
                    continue
                    
                image_path = os.path.join(path, image)
                rgb = cv2.imread(image_path)
                
                if rgb is None:
                    print(f"Could not read {image_path}, skipping...")
                    continue
                    
                # IMPORTANT: Apply mixed resolution operations BEFORE resizing to fixed size
                if use_mixed_res:
                    rgb = apply_mixed_resolution_ops([rgb], **mixed_res_config)[0]
                
                # AFTER augmentation, resize to fixed size for the model
                rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)

                if use_nir:
                    nir = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
                    # Shape (224, 224, 1)
                    nir = np.expand_dims(nir, axis=-1)
                    # Shape (224, 224, 4)
                    rgb_nir = np.concatenate((rgb, nir), axis=-1)
                    rgb_nir = dyn_zscore_normalize(rgb_nir) # normalization
                    X_array.append(rgb_nir)
                else:
                    rgb = dyn_zscore_normalize(rgb) # normalization
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

    # Log data loaded
    logger.info(f"Loaded {len(X_array)} images from {path}")
    
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


def clean_zone_identifiers(data_dir='data'):
    """
    Removes Windows Zone.Identifier files from the data directory.
    This is useful to run once to clean up the dataset.
    
    Args:
        data_dir: Base directory containing image data
    """
    import glob
    
    # Find all Zone.Identifier files
    zone_files = glob.glob(f'{data_dir}/**/*:Zone.Identifier', recursive=True)
    
    if zone_files:
        logger.info(f"Found {len(zone_files)} Zone.Identifier files to clean")
        
        for file in zone_files:
            try:
                # Get the original filename without the Zone.Identifier suffix
                original_file = file.split(':Zone.Identifier')[0]
                
                # Remove the Zone.Identifier file
                os.remove(file)
                logger.debug(f"Removed {file}")
                
            except Exception as e:
                logger.error(f"Failed to clean {file}: {str(e)}")
        
        logger.info(f"Cleaned {len(zone_files)} Zone.Identifier files")
    else:
        logger.info("No Zone.Identifier files found to clean")