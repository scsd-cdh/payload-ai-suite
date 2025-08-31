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
from typing import Optional, List, Tuple, Union, Any
from pydantic import BaseModel, Field, validator
from mixed_res_config import DEFAULT_MIXED_RES_CONFIG

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
        Resized and padded image with same shape as input
    """
    h, w = image.shape[:2]
    
    # Random scale
    scale = np.random.uniform(min_scale, max_scale)
    new_h = int(h * scale)
    new_w = int(w * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Pad to original size
    if scale < 1.0:
        # Image was shrunk, need to pad
        pad_h = h - new_h
        pad_w = w - new_w
        
        # Calculate padding for each side
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        
        # Pad with zeros (black)
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, 
                                   cv2.BORDER_CONSTANT, value=0)
    else:
        # Image was enlarged, need to crop to center
        start_h = (new_h - h) // 2
        start_w = (new_w - w) // 2
        padded = resized[start_h:start_h+h, start_w:start_w+w]
    
    return padded


def create_multi_resolution_batch(images: List[np.ndarray], 
                                scales: List[float] = [0.5, 0.75, 1.0, 1.25, 1.5]) -> List[np.ndarray]:
    """
    Create a batch with images at multiple resolutions.
    This helps the model learn scale-invariant features.
    
    Args:
        images: List of input images
        scales: List of scale factors to apply
    
    Returns:
        List of images at various scales (padded to original size)
    """
    multi_res_batch = []
    
    for img in images:
        # Randomly select a scale for this image
        scale = random.choice(scales)
        
        # Apply the scale
        h, w = img.shape[:2]
        new_h = int(h * scale)
        new_w = int(w * scale)
        
        # Resize
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Pad or crop to original size
        if scale < 1.0:
            # Pad
            pad_h = h - new_h
            pad_w = w - new_w
            top = pad_h // 2
            bottom = pad_h - top
            left = pad_w // 2
            right = pad_w - left
            final_img = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                         cv2.BORDER_CONSTANT, value=0)
        else:
            # Crop center
            start_h = (new_h - h) // 2
            start_w = (new_w - w) // 2
            final_img = resized[start_h:start_h+h, start_w:start_w+w]
        
        multi_res_batch.append(final_img)
    
    return multi_res_batch


def resolution_mixup(image1: np.ndarray, image2: np.ndarray, 
                    alpha: float = 0.2) -> np.ndarray:
    """
    Mix two images at different resolutions.
    This creates training samples that are combinations of different scales.
    
    Args:
        image1: First image
        image2: Second image  
        alpha: Mixing parameter (0 = all image1, 1 = all image2)
    
    Returns:
        Mixed image
    """
    # Ensure images have same shape
    assert image1.shape == image2.shape, "Images must have same shape for mixup"
    
    # Simple weighted average
    mixed = (1 - alpha) * image1 + alpha * image2
    
    # Ensure output is in valid range
    mixed = np.clip(mixed, 0, 255).astype(image1.dtype)
    
    return mixed


def apply_mixed_resolution_ops(images: List[np.ndarray], 
                             use_random_resize: bool = True,
                             use_multi_resolution: bool = True, 
                             use_resolution_mixup: bool = True,
                             min_scale: float = 0.5,
                             max_scale: float = 1.5,
                             resolution_scales: List[float] = [0.5, 0.75, 1.0, 1.25, 1.5],
                             mixup_alpha: float = 0.2) -> List[np.ndarray]:
    """
    Apply mixed resolution operations to a batch of images.
    
    Args:
        images: List of input images
        use_random_resize: Whether to apply random resize with padding
        use_multi_resolution: Whether to create multi-resolution batch
        use_resolution_mixup: Whether to apply resolution mixup
        min_scale: Minimum scale for random resize
        max_scale: Maximum scale for random resize
        resolution_scales: Scales for multi-resolution batch
        mixup_alpha: Alpha parameter for mixup
    
    Returns:
        List of processed images
    """
    logger.debug(f"Applying mixed resolution ops to {len(images)} images")
    processed_images = images.copy()
    
    # Random resize with padding
    if use_random_resize:
        processed_images = [random_resize_with_pad(img, min_scale, max_scale) 
                          for img in processed_images]
    
    # Multi-resolution batch
    if use_multi_resolution:
        processed_images = create_multi_resolution_batch(processed_images, resolution_scales)
    
    # Resolution mixup (only if we have at least 2 images)
    if use_resolution_mixup and len(processed_images) >= 2:
        # Randomly select pairs and mix them
        for i in range(0, len(processed_images) - 1, 2):
            if i + 1 < len(processed_images):
                alpha = np.random.beta(mixup_alpha, mixup_alpha)
                processed_images[i] = resolution_mixup(processed_images[i], 
                                                     processed_images[i + 1], 
                                                     alpha)
    
    logger.debug(f"Completed apply_mixed_resolution_ops")
    return processed_images


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
                    logger.warning(f"Could not read {image_path}, skipping...")
                    continue
                rgb = stream_image_from_gcs(image_bytes)
                if rgb is None:
                    logger.warning(f"Could not decode {image_path}, skipping...")
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
                    logger.debug(f"NIR processing - RGB shape: {rgb.shape}, NIR shape: {nir.shape}, Combined: {rgb_nir.shape}")
                    rgb_nir = dyn_zscore_normalize(rgb_nir)
                    X_array.append(rgb_nir)
                else:
                    try:
                        fused_result = rgb_nir_fusion(rgb, use_enhanced_red=True)
                    except Exception as e:
                        logger.warning(f"{e}: issue with rgb nir fusion technique, skipping for {image_path}")
                        fused_result = rgb
                    # Normalization still applied after fallback
                    rgb = dyn_zscore_normalize(fused_result)
                    X_array.append(rgb)

                if not end:
                    y_array.append(image_path[0:1])
        else:
            # Local files
            for image in os.listdir(path):
                image_path = os.path.join(path, image)

                rgb = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                if rgb is None:
                    logger.warning(f"Could not read {image_path}, skipping...")
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
                    logger.debug(f"NIR processing - RGB shape: {rgb.shape}, NIR shape: {nir.shape}, Combined: {rgb_nir.shape}")
                    rgb_nir = dyn_zscore_normalize(rgb_nir)
                    X_array.append(rgb_nir)
                else:
                    # RGB NIR fusion algorthm still relevant if we have 4 channels
                    # the `use_nir` flag is soely for the AI model to take in a 4 channel input tensor
                    try:
                        fused_result = rgb_nir_fusion(rgb, use_enhanced_red=True)
                    except Exception as e:
                        logger.warning(f"{e}: issue with the rgb nir fusion technique. Skipping for {image_path}")
                        fused_result = rgb
                    # Normalization still applied after fallback 
                    rgb = dyn_zscore_normalize(fused_result)
                    X_array.append(rgb)

                if not end:
                    y_array.append(image_path[0:1])

        if end:
            # Ensure y_array has the same length as X_array
            while len(y_array) < len(X_array):
                y_array.append("N")
    except cv2.error as e:
        logger.error(f"CV2 error in preprocess: {e}")
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
        # Use IMREAD_UNCHANGED to preserve all channels (including 4-channel TIFFs)
        img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
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