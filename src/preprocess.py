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

def degrade_to_cubesat_gsd(image: np.ndarray) -> np.ndarray:
    """
    Degrade Sentinel-2 imagery (10m GSD) to match CubeSat expected GSD (~85m).
    This simulates the lower resolution data that will be captured on-orbit.

    Degradation factor: 10m -> 85m = 8.5x downsampling

    Args:
        image: Input image at Sentinel-2 resolution (HxWxC)

    Returns:
        Degraded image at CubeSat GSD, upsampled back to original dimensions
    """
    h, w = image.shape[:2]

    # Downsample by 8.5x using INTER_AREA (best for downsampling)
    degraded = cv2.resize(image, (w//8, h//8), interpolation=cv2.INTER_AREA)

    # Upsample back to original size using INTER_LINEAR (simulates lower quality data)
    degraded = cv2.resize(degraded, (w, h), interpolation=cv2.INTER_LINEAR)

    return degraded

def random_resize_with_pad(image: np.ndarray, min_scale: float = 0.5, max_scale: float = 1.5) -> np.ndarray:
    """
    Randomly resize an image without cropping.
    This simulates images of different resolutions during training.

    Args:
        image: Input image (HxWxC)
        min_scale: Minimum scaling factor
        max_scale: Maximum scaling factor

    Returns:
        Resized image (dimensions may vary from input, no cropping applied)
    """
    h, w = image.shape[:2]

    # Random scale
    scale = np.random.uniform(min_scale, max_scale)
    new_h = int(h * scale)
    new_w = int(w * scale)

    # Resize image - no padding or cropping
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    return resized


def create_multi_resolution_batch(images: List[np.ndarray],
                                scales: List[float] = [0.5, 0.75, 1.0, 1.25, 1.5]) -> List[np.ndarray]:
    """
    Create a batch with images at multiple resolutions without cropping.
    This helps the model learn scale-invariant features.

    Args:
        images: List of input images
        scales: List of scale factors to apply

    Returns:
        List of images at various scales (no cropping or padding applied)
    """
    multi_res_batch = []

    for img in images:
        # Randomly select a scale for this image
        scale = random.choice(scales)

        # Apply the scale
        h, w = img.shape[:2]
        new_h = int(h * scale)
        new_w = int(w * scale)

        # Resize - no padding or cropping
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        multi_res_batch.append(resized)

    return multi_res_batch


def apply_mixed_resolution_ops(images: List[np.ndarray],
                             use_random_resize: bool = True,
                             use_multi_resolution: bool = True,
                             use_flip: bool = True,
                             min_scale: float = 0.5,
                             max_scale: float = 1.5,
                             resolution_scales: List[float] = [0.5, 0.75, 1.0, 1.25, 1.5]) -> List[np.ndarray]:
    """
    Apply mixed resolution operations to a batch of images.
    Only performs rescaling and flipping - no cropping or padding.

    Args:
        images: List of input images
        use_random_resize: Whether to apply random resize
        use_multi_resolution: Whether to create multi-resolution batch
        use_flip: Whether to apply random flipping
        min_scale: Minimum scale for random resize
        max_scale: Maximum scale for random resize
        resolution_scales: Scales for multi-resolution batch

    Returns:
        List of processed images (dimensions may vary, no cropping applied)
    """
    logger.debug(f"Applying mixed resolution ops to {len(images)} images")
    processed_images = images.copy()

    # Random resize (no padding or cropping)
    if use_random_resize:
        processed_images = [random_resize_with_pad(img, min_scale, max_scale)
                          for img in processed_images]

    # Multi-resolution batch
    if use_multi_resolution:
        processed_images = create_multi_resolution_batch(processed_images, resolution_scales)

    # Random flipping
    if use_flip:
        flipped_images = []
        for img in processed_images:
            # Random horizontal flip
            if np.random.random() < 0.5:
                img = cv2.flip(img, 1)
            # Random vertical flip
            if np.random.random() < 0.5:
                img = cv2.flip(img, 0)
            flipped_images.append(img)
        processed_images = flipped_images

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
             use_mixed_res=False, mixed_res_config=None, fusion_technique='enhanced_red',
             fusion_alpha=0.5, degrade_gsd=False):
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
        fusion_technique (str): RGB-NIR fusion technique: 'enhanced_red', 'hsv', or 'none'.
        fusion_alpha (float): Alpha value for enhanced_red fusion (default 0.5).
        degrade_gsd (bool): Whether to degrade imagery to CubeSat GSD (~85m from 10m).

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

                # Apply GSD degradation if requested (simulates CubeSat resolution)
                if degrade_gsd:
                    rgb = degrade_to_cubesat_gsd(rgb)

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
                        fused_result = rgb_nir_fusion(
                            rgb,
                            technique=fusion_technique,
                            alpha=fusion_alpha
                        )
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

                # Apply GSD degradation if requested (simulates CubeSat resolution)
                if degrade_gsd:
                    rgb = degrade_to_cubesat_gsd(rgb)

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
                        fused_result = rgb_nir_fusion(
                            rgb,
                            technique=fusion_technique,
                            alpha=fusion_alpha
                        )
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
    logger.info(f"X_array preprocess {X_array}")
    logger.info(f"y_array preprocess {y_array}")

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

        logger.debug(f"Channel {c} normalization - mean: {mean:.2f}, std: {std:.2f}, valid pixels: {valid_count}")

        # Apply normalization only to valid pixels
        normal_img[:, :, c] = np.where(channel_mask, (channel - mean) / std, 0.0)

    return normal_img


def rgb_nir_fusion(image_data: np.ndarray[Any, np.dtype[np.integer[Any] | np.floating[Any]]],
                   technique: str = 'enhanced_red', alpha: float = 0.5):
    """
    Use nir band to "enhance" RGB image. Data fusion technique borrowed from robotics computer vision

    Args:
        image_data: 4-channel input (RGB+NIR)
        technique: Fusion method - 'enhanced_red', 'hsv', or 'none'
        alpha: Blending factor for enhanced_red technique (default 0.5)

    Available techniques:

    1. 'enhanced_red': Blend NIR with red channel
       enhanced_red = (1 - alpha) * red + alpha * nir
       fused = np.stack([enhanced_red, green, blue], axis=-1)

    2. 'hsv': HSV colorspace fusion
       Convert RGB -> HSV, combine V channel with NIR, convert back to RGB.
       Hue and saturation remain untouched.

    3. 'none': Return RGB channels only (no fusion)
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

    if technique == 'enhanced_red':
        logger.info(f"Using enhanced red fusion method with alpha={alpha}")
        blue, green, red = cv2.split(rgb)
        enhanced_red = ((1 - alpha) * red + alpha * nir).astype(np.uint8)
        fused = np.stack([blue, green, enhanced_red], axis=-1)
        logger.debug(f"Enhanced red fusion complete - output shape: {fused.shape}")

        # Validate output
        try:
            output_validation = FusionOutput(image_shape=fused.shape)
        except ValueError as e:
            logger.error(f"Output validation failed: {e}")
            raise

        return fused

    elif technique == 'hsv':
        logger.info("Using HSV fusion method")
        hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        # Combine V channel with NIR
        enhanced_v = ((v + nir) / 2).astype(np.uint8)
        logger.warning(f"HSV merge inputs - h: {h.dtype}/{h.shape}, s: {s.dtype}/{s.shape}, enhanced_v: {enhanced_v.dtype}/{enhanced_v.shape}")
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

    elif technique == 'none':
        logger.info("No fusion - returning RGB channels only")
        return rgb

    else:
        logger.warning(f"Unknown fusion technique '{technique}', returning RGB only")
        return rgb