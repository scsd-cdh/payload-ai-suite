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

                try:
                    processed = prepare_model_input(rgb, use_nir=use_nir)
                except Exception as e:
                    logger.warning(f"{e}: failed to prepare model input for {image_path}, skipping...")
                    continue

                X_array.append(processed)

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

                try:
                    processed = prepare_model_input(rgb, use_nir=use_nir)
                except Exception as e:
                    logger.warning(f"{e}: failed to prepare model input for {image_path}, skipping...")
                    continue

                X_array.append(processed)

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
    """
    Perform per-channel z-score normalization excluding no-data pixels.

    Mimics the behavior in omnicloudmask (pytorch) using numpy for image arrays.
    Computes mean and standard deviation only from valid (non-zero) pixels,
    then applies z-score normalization to valid pixels while preserving no-data values.

    "Dynamic" means each image gets its own normalization statistics calculated on-the-fly
    rather than using pre-computed global statistics. This adapts to varying lighting
    conditions, sun angles, atmospheric conditions, and sensor characteristics between
    different satellite captures.

    Args:
        img: Input image array of shape (H, W, C)
        no_data_value: Pixel value to exclude from statistics (default: 0.0)

    Returns:
        Normalized image array with same shape as input. No-data pixels remain 0.0.
    """
    # 32 bit only for training, in the wild optical system will be raw 8 bit
    img = img.astype(np.float32)
    epsilon = 1e-8

    # Compute mask once
    mask = img != no_data_value

    # Calculate per-channel statistics using masked operations
    normal_img = np.zeros_like(img)

    for c in range(img.shape[2]):
        channel_mask = mask[:, :, c]
        if not channel_mask.any():
            continue

        channel = img[:, :, c]
        valid_count = channel_mask.sum()

        # Mean of valid pixels only
        mean = (channel * channel_mask).sum() / valid_count

        # Std of valid pixels only
        diff_sq = (channel - mean) ** 2 * channel_mask
        std = np.sqrt(diff_sq.sum() / valid_count + epsilon)

        # Apply normalization only to valid pixels
        normal_img[:, :, c] = np.where(channel_mask, (channel - mean) / std, 0.0)

    return normal_img


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
        logger.error(f"Input validation failed: {e} ")
        return image_data


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
            logger.error(f"Output validation failed: {e} {output_validation}")
            raise

        return fused

    else:
        logger.info("No fusion method specified, returning RGB channels only")
        return rgb


def prepare_model_input(image: np.ndarray, use_nir: bool) -> np.ndarray:
    """Normalize channel layout for model consumption."""

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if image.ndim != 3:
        raise ValueError(f"Unsupported image ndim: {image.ndim}")

    channels = image.shape[2]
    if channels < 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        channels = 3

    base_rgb = image[:, :, :3]

    if use_nir:
        if channels >= 4:
            nir = image[:, :, 3]
        else:
            nir = cv2.cvtColor(base_rgb, cv2.COLOR_BGR2GRAY)
        nir = np.expand_dims(nir, axis=-1)
        rgb_nir = np.concatenate((base_rgb, nir), axis=-1)
        return dyn_zscore_normalize(rgb_nir)

    if channels >= 4:
        fusion_input = image[:, :, :4]
    else:
        nir = cv2.cvtColor(base_rgb, cv2.COLOR_BGR2GRAY)
        fusion_input = np.concatenate((base_rgb, np.expand_dims(nir, axis=-1)), axis=-1)

    try:
        fused = rgb_nir_fusion(fusion_input, use_enhanced_red=True)
    except Exception as e:
        logger.warning(f"{e}: falling back to base RGB for fusion")
        fused = base_rgb

    return dyn_zscore_normalize(fused)