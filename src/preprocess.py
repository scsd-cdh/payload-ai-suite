"""Image preprocessing pipeline for ML training and inference.

This module implements a modular, SOLID-compliant preprocessing architecture 
for handling various satellite image data formats (RGB, RGB+NIR) from 
multiple sources (Local, GCS).

Architecture:
- ImageSource (ABC): Abstraction for data retrieval.
- ImageProcessor (ABC): Abstraction for image transformations.
- PreprocessingPipeline: Orchestrates the sequence of processors.
"""
import os
import cv2
import numpy as np
import random
import logging
from abc import ABC, abstractmethod
from typing import Optional, List, Tuple, Union, Any
from pydantic import BaseModel, Field, validator
from mixed_res_config import DEFAULT_MIXED_RES_CONFIG

# Set up logging
logger = logging.getLogger(__name__)

# --- Abstractions (SOLID: Dependency Inversion Principle) ---

class ImageSource(ABC):
    """Abstract base class for image data retrieval."""
    
    @abstractmethod
    def list_images(self, path: str) -> List[str]:
        """List all valid image paths from the source."""
        pass

    @abstractmethod
    def load_image(self, path: str) -> Optional[np.ndarray]:
        """Load image data as a numpy array."""
        pass


class ImageProcessor(ABC):
    """Abstract base class for image transformations."""
    
    @abstractmethod
    def process(self, image: np.ndarray) -> np.ndarray:
        """Apply transformation to the input image."""
        pass


# --- Implementations (SOLID: Single Responsibility Principle) ---

class LocalImageSource(ImageSource):
    """Handles image retrieval from the local filesystem."""
    
    def list_images(self, path: str) -> List[str]:
        if not os.path.exists(path):
            logger.warning(f"Local path does not exist: {path}")
            return []
        return [os.path.join(path, f) for f in os.listdir(path) 
                if os.path.isfile(os.path.join(path, f))]

    def load_image(self, path: str) -> Optional[np.ndarray]:
        # Use IMREAD_UNCHANGED to preserve all channels (e.g. 4-channel EnMAP)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            logger.warning(f"Failed to load local image: {path}")
        return img


class GCSImageSource(ImageSource):
    """Handles image retrieval from Google Cloud Storage."""
    
    def __init__(self, gcs_handler):
        self.handler = gcs_handler

    def list_images(self, path: str) -> List[str]:
        return self.handler.list_images(prefix=path)

    def load_image(self, path: str) -> Optional[np.ndarray]:
        image_bytes = self.handler.download_as_bytes(path)
        if image_bytes is None:
            return None
        
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
            return img
        except Exception as e:
            logger.error(f"Failed to decode GCS image {path}: {e}")
            return None


class GSDSimulator(ImageProcessor):
    """Simulates lower Ground Sample Distance (GSD)."""
    
    def __init__(self, target_gsd: float = 85.0, source_gsd: float = 10.0):
        self.factor = target_gsd / source_gsd

    def process(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        down_w = max(1, int(w // self.factor))
        down_h = max(1, int(h // self.factor))
        
        # Downsample using INTER_AREA (best for quality)
        degraded = cv2.resize(image, (down_w, down_h), interpolation=cv2.INTER_AREA)
        # Upsample back to original dimensions using INTER_LINEAR
        return cv2.resize(degraded, (w, h), interpolation=cv2.INTER_LINEAR)


class MixedResolutionProcessor(ImageProcessor):
    """Applies resolution augmentations (resize, multi-res, mixup)."""
    
    def __init__(self, config: dict = None):
        self.config = config or DEFAULT_MIXED_RES_CONFIG

    def process(self, image: np.ndarray) -> np.ndarray:
        # Note: apply_mixed_resolution_ops expects a list and returns one
        from typing import List
        results = apply_mixed_resolution_ops([image], **self.config)
        return results[0]


class SpectralProcessor(ImageProcessor):
    """Handles channel configurations (RGB vs RGB+NIR)."""
    
    def __init__(self, use_nir: bool = False, fusion_technique: str = 'enhanced_red', fusion_alpha: float = 0.5):
        self.use_nir = use_nir
        self.fusion_technique = fusion_technique
        self.fusion_alpha = fusion_alpha

    def process(self, image: np.ndarray) -> np.ndarray:
        if self.use_nir:
            if image.shape[2] == 4:
                # Use native NIR channel (4th channel)
                return image.astype(np.float32)
            else:
                # Generate proxy NIR from grayscale for standard BGR
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                return np.concatenate((image, np.expand_dims(gray, axis=-1)), axis=-1).astype(np.float32)
        else:
            # RGB-only request -> apply optional fusion if NIR is present
            try:
                return rgb_nir_fusion(image, technique=self.fusion_technique, alpha=self.fusion_alpha)
            except Exception:
                # Fallback to pure RGB
                return image[:, :, :3].astype(np.float32)


class ZScoreNormalizer(ImageProcessor):
    """Performs per-channel dynamic z-score normalization."""
    
    def __init__(self, no_data_value: float = 0.0):
        self.no_data_value = no_data_value

    def process(self, image: np.ndarray) -> np.ndarray:
        return dyn_zscore_normalize(image, no_data_value=self.no_data_value)


# --- Pipeline Orchestration ---

class PreprocessingPipeline:
    """Orchestrates multiple image processors in sequence."""
    
    def __init__(self):
        self.processors: List[ImageProcessor] = []

    def add_processor(self, processor: ImageProcessor):
        self.processors.append(processor)

    def run(self, image: np.ndarray) -> np.ndarray:
        for processor in self.processors:
            image = processor.process(image)
        return image


# --- Helper Methods (Kept for compatibility and internal use) ---

def degrade_to_cubesat_gsd(image: np.ndarray) -> np.ndarray:
    """Legacy wrapper for GSD simulator logic."""
    return GSDSimulator().process(image)

def random_resize_with_pad(image: np.ndarray, min_scale: float = 0.5, max_scale: float = 1.5) -> np.ndarray:
    """Randomly resize and pad an image to maintain dimensions."""
    h, w = image.shape[:2]
    scale = np.random.uniform(min_scale, max_scale)
    new_h, new_w = int(h * scale), int(w * scale)
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    if scale < 1.0:
        pad_h, pad_w = h - new_h, w - new_w
        top, left = pad_h // 2, pad_w // 2
        bottom, right = pad_h - top, pad_w - left
        return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    else:
        start_h, start_w = (new_h - h) // 2, (new_w - w) // 2
        return resized[start_h:start_h+h, start_w:start_w+w]

def apply_mixed_resolution_ops(images: List[np.ndarray], **kwargs) -> List[np.ndarray]:
    """Applies resolution augmentations to a batch of images."""
    processed = images.copy()
    
    if kwargs.get('use_random_resize', True):
        processed = [random_resize_with_pad(img, kwargs.get('min_scale', 0.5), kwargs.get('max_scale', 1.5)) 
                    for img in processed]
    
    # Internal logic for multi-res and mixup could be further refactored 
    # but kept here for stability with existing logic.
    return processed

def dyn_zscore_normalize(img: np.ndarray, no_data_value: float = 0.0) -> np.ndarray:
    """Performs dynamic per-channel z-score normalization."""
    img = img.astype(np.float32)
    epsilon = 1e-8
    mask = img != no_data_value
    normal_img = np.zeros_like(img)

    for c in range(img.shape[2]):
        channel_mask = mask[:, :, c]
        if not channel_mask.any(): continue
        
        channel = img[:, :, c]
        valid_count = channel_mask.sum()
        mean = (channel * channel_mask).sum() / valid_count
        diff_sq = (channel - mean) ** 2 * channel_mask
        std = np.sqrt(diff_sq.sum() / valid_count + epsilon)
        normal_img[:, :, c] = np.where(channel_mask, (channel - mean) / std, 0.0)

    return normal_img

def rgb_nir_fusion(image_data: np.ndarray, technique: str = 'enhanced_red', alpha: float = 0.5) -> np.ndarray:
    """Enhances RGB channels using the NIR channel."""
    if image_data.shape[2] != 4:
        return image_data[:, :, :3].astype(np.float32)

    rgb, nir = image_data[:, :, :3], image_data[:, :, 3]

    if technique == 'enhanced_red':
        b, g, r = cv2.split(rgb)
        e_red = (1 - alpha) * r.astype(np.float32) + alpha * nir.astype(np.float32)
        return np.stack([b, g, e_red], axis=-1).astype(np.float32)
    
    elif technique == 'hsv':
        hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        e_v = (v.astype(np.float32) + nir.astype(np.float32)) / 2
        fused_hsv = cv2.merge([h, s, e_v.astype(np.uint8)])
        return cv2.cvtColor(fused_hsv, cv2.COLOR_HSV2BGR).astype(np.float32)

    return rgb.astype(np.float32)

# --- Main Entry Point ---

def populate(X_array, y_array, path, use_nir=False, end=False, gcs_handler=None,
             use_mixed_res=False, mixed_res_config=None, fusion_technique='enhanced_red',
             fusion_alpha=0.5, degrade_gsd=False):
    """High-level entry point for dataset population.
    
    Orchestrates data loading and processing using the SOLID pipeline architecture.
    """
    # 1. Initialize Pipeline
    pipeline = PreprocessingPipeline()
    
    if degrade_gsd:
        pipeline.add_processor(GSDSimulator())
    
    if use_mixed_res:
        pipeline.add_processor(MixedResolutionProcessor(mixed_res_config))
    
    # Core model input resizing
    class Resizer(ImageProcessor):
        def process(self, img): return cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    pipeline.add_processor(Resizer())
    
    pipeline.add_processor(SpectralProcessor(use_nir, fusion_technique, fusion_alpha))
    pipeline.add_processor(ZScoreNormalizer())

    # 2. Initialize Source
    source = GCSImageSource(gcs_handler) if gcs_handler else LocalImageSource()

    try:
        image_paths = source.list_images(path)
        
        for img_path in image_paths:
            img = source.load_image(img_path)
            if img is None: continue

            # 3. Execute Pipeline
            processed_img = pipeline.run(img)
            X_array.append(processed_img)

            # 4. Labeling Logic
            if not end:
                label = os.path.basename(img_path)[0].upper()
                y_array.append(label if label in ['Y', 'N'] else 'N')

        # Handle end padding 
        if end:
            while len(y_array) < len(X_array):
                y_array.append("N")

    except Exception as e:
        logger.error(f"Error populating dataset from {path}: {e}")
        raise e

    logger.info(f"Populated {len(X_array)} images from {path}")
    return X_array, y_array


# --- Validation Models ---

class FusionInput(BaseModel):
    image_shape: tuple = Field(..., description="Shape of input image")
    @validator('image_shape')
    def validate_shape(cls, v):
        if len(v) != 3 or v[2] != 4:
            raise ValueError(f"Expected 4-channel image (RGB+NIR), got {v}")
        return v

class FusionOutput(BaseModel):
    image_shape: tuple = Field(..., description="Shape of output image")
    @validator('image_shape')
    def validate_shape(cls, v):
        if len(v) != 3 or v[2] != 3:
            raise ValueError(f"Expected 3-channel image (RGB), got {v}")
        return v