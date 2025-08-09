#!/usr/bin/env python3
"""
Export OmniCloudMask models to ONNX format.

This script loads the OmniCloudMask pre-trained models and exports them to ONNX format
for efficient inference.
"""

import torch
import numpy as np
from pathlib import Path
from functools import partial
import timm
from fastai.vision.learner import create_unet_model
import onnx
import onnxruntime as ort
import cv2
import logging
from preprocess import dyn_zscore_normalize
from paths import resolve_path

# Set up logging
logger = logging.getLogger(__name__)

def load_omnicloudmask_model(weights_path, model_name="regnety_004", device="cpu"):
    """
    Load OmniCloudMask model from weights file.

    Args:
        weights_path: Path to the .pth weights file
        model_name: Either "regnety_004" or "convnextv2_nano"
        device: Device to load model on

    Returns:
        Loaded PyTorch model
    """
    # Create timm model backbone
    timm_model = partial(
        timm.create_model,
        model_name=model_name,
        pretrained=True,
        in_chans=3,  # RGB channels
    )

    # Create FastAI DynamicUnet model
    model = create_unet_model(
        arch=timm_model,
        n_out=4,  # 4 output classes for cloud mask
        img_size=(512, 512),
        act_cls=torch.nn.Mish,
        pretrained=True,
    )

    # Load the state dict
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    return model.to(device)


def export_to_onnx(model, output_path, dummy_input_shape=(1, 3, 512, 512)):
    """
    Export PyTorch model to ONNX format.

    Args:
        model: PyTorch model to export
        output_path: Path to save ONNX file
        dummy_input_shape: Shape of dummy input for tracing
    """
    # Create dummy input
    dummy_input = torch.randn(dummy_input_shape)

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    logger.info(f"Model exported to {output_path}")

    # Verify the exported model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX model verified successfully")

    return onnx_model


def test_onnx_inference(onnx_path, test_shape=(1, 3, 509, 509)):
    """
    Test ONNX model inference.

    Args:
        onnx_path: Path to ONNX model
        test_shape: Shape of test input
    """
    # Create ONNX runtime session
    ort_session = ort.InferenceSession(onnx_path)

    # Create test input
    test_input = np.random.randn(*test_shape).astype(np.float32)

    # Run inference
    ort_inputs = {ort_session.get_inputs()[0].name: test_input}
    ort_outputs = ort_session.run(None, ort_inputs)

    logger.info(f"ONNX inference successful!")
    logger.info(f"Input shape: {test_input.shape}")
    logger.info(f"Output shape: {ort_outputs[0].shape}")
    logger.info(f"Output range: [{ort_outputs[0].min():.4f}, {ort_outputs[0].max():.4f}]")

    return ort_outputs[0]


def test_on_labeled_images(onnx_path, data_dir=None, max_images=5):
    """
    Test OmniCloudMask ONNX model on labeled wildfire images.

    Args:
        onnx_path: Path to ONNX model
        data_dir: Base directory containing 'yes' and 'no' subdirectories
        max_images: Maximum number of images to test per category
    """
    if data_dir is None:
        data_dir = resolve_path("data/labeled")
    # Create ONNX runtime session
    ort_session = ort.InferenceSession(onnx_path)

    # Class names for OmniCloudMask
    class_names = ['Clear', 'Thin Cloud', 'Thick Cloud', 'Cloud Shadow']

    logger.info(f"\n{'='*60}")
    logger.info("TESTING ON LABELED WILDFIRE DATA")
    logger.info(f"{'='*60}")

    for category in ['no', 'yes']:
        category_dir = Path(data_dir) / category
        if not category_dir.exists():
            logger.warning(f"Directory {category_dir} not found")
            continue

        logger.info(f"\n--- Testing on {category.upper()} (wildfire={category}) samples ---")

        image_files = list(category_dir.glob("*.tiff")) + list(category_dir.glob("*.tif"))

        for i, img_path in enumerate(image_files[:max_images]):
            try:
                logger.debug(f"\n{i+1}. Processing: {img_path.name}")

                # Load image using OpenCV
                img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
                if img is None:
                    logger.error(f"   Error: Could not read {img_path}")
                    continue

                # Check number of channels
                if len(img.shape) == 2:
                    # Grayscale - convert to RGB
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    nir = img[:, :, 0]  # Use red channel as NIR
                    logger.debug("   Grayscale image converted to RGB, using RED as NIR")
                elif img.shape[2] == 3:
                    # RGB only
                    nir = img[:, :, 0]  # Use red channel as NIR
                    logger.debug("   RGB image, using RED band as NIR substitute")
                elif img.shape[2] >= 4:
                    # Has alpha/NIR channel
                    nir = img[:, :, 3]
                    img = img[:, :, :3]  # Keep only RGB
                    logger.debug("   Using 4th channel as NIR band")
                else:
                    logger.warning(f"   Unexpected number of channels: {img.shape[2]}")
                    continue

                # Get RGB channels (OpenCV uses BGR order)
                blue = img[:, :, 0].astype(np.float32)
                green = img[:, :, 1].astype(np.float32)
                red = img[:, :, 2].astype(np.float32)
                nir = nir.astype(np.float32)

                # Stack into (H, W, 3) format for normalization
                # OmniCloudMask expects Red, Green, NIR order
                rgb_nir_stack = np.stack([red, green, nir], axis=-1)

                # Apply z-score normalization
                normalized = dyn_zscore_normalize(rgb_nir_stack)

                # Reshape to (3, H, W) for OmniCloudMask
                input_array = np.transpose(normalized, (2, 0, 1))

                # Get original dimensions
                _, h, w = input_array.shape

                # Resize to 509x509 using OpenCV
                # First transpose back to (H, W, C) for cv2.resize
                input_hwc = np.transpose(input_array, (1, 2, 0))
                input_resized = cv2.resize(input_hwc, (509, 509), interpolation=cv2.INTER_LINEAR)

                # Transpose back to (C, H, W)
                input_resized = np.transpose(input_resized, (2, 0, 1))

                # Add batch dimension
                input_batch = np.expand_dims(input_resized, axis=0).astype(np.float32)

                # Run inference
                ort_inputs = {ort_session.get_inputs()[0].name: input_batch}
                ort_outputs = ort_session.run(None, ort_inputs)

                # Get predictions (shape: [1, 4, 512, 512])
                predictions = ort_outputs[0][0]  # Remove batch dimension

                # Get class predictions
                predicted_classes = np.argmax(predictions, axis=0)

                # Calculate percentage of each class
                total_pixels = predicted_classes.size
                class_percentages = {}
                for cls_idx, cls_name in enumerate(class_names):
                    pixel_count = np.sum(predicted_classes == cls_idx)
                    percentage = (pixel_count / total_pixels) * 100
                    class_percentages[cls_name] = percentage

                logger.info(f"   Original size: {w}x{h}")
                logger.info("   Cloud mask predictions:")
                for cls_name, percentage in class_percentages.items():
                    logger.info(f"     - {cls_name}: {percentage:.1f}%")

                # Calculate overall cloud coverage
                cloud_coverage = class_percentages['Thin Cloud'] + class_percentages['Thick Cloud'] + class_percentages['Cloud Shadow']
                logger.info(f"   Total cloud/shadow coverage: {cloud_coverage:.1f}%")

            except Exception as e:
                logger.error(f"   Error processing {img_path.name}: {e}")
                import traceback
                traceback.print_exc()


def main():
    """Main function to export OmniCloudMask models to ONNX."""

    # Define model paths
    models_dir = Path(resolve_path("models/omnicloudmask"))

    # Model configurations
    model_configs = [
        {
            "weights_path": models_dir / "PM_model_2.2.10_RG_NIR_509_regnety_004.pycls_in1k_PT_state.pth",
            "model_name": "regnety_004",
            "output_path": models_dir / "omnicloudmask_regnety_004.onnx"
        },
        {
            "weights_path": models_dir / "PM_model_2.2.10_RG_NIR_509_convnextv2_nano.fcmae_ft_in1k_PT_state.pth",
            "model_name": "convnextv2_nano",
            "output_path": models_dir / "omnicloudmask_convnextv2_nano.onnx"
        }
    ]

    # Process each model
    for config in model_configs:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {config['model_name']} model")
        logger.info(f"{'='*60}")

        try:
            # Check if ONNX already exists
            if config["output_path"].exists():
                logger.info(f"ONNX model already exists at {config['output_path']}")
                logger.info("Skipping export, testing on labeled data...")

                # Test on labeled wildfire data
                test_on_labeled_images(config["output_path"])
                continue

            # Check if weights file exists
            if not config["weights_path"].exists():
                logger.warning(f"Weights file not found at {config['weights_path']}")
                logger.info("Skipping this model...")
                continue

            # Load model
            logger.info(f"\n1. Loading model from {config['weights_path']}")
            model = load_omnicloudmask_model(
                weights_path=config["weights_path"],
                model_name=config["model_name"],
                device="cpu"
            )
            logger.info("Model loaded successfully")

            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(f"Total parameters: {total_params:,}")

            # Export to ONNX
            logger.info(f"\n2. Exporting to ONNX format")
            export_to_onnx(model, config["output_path"])

            # Test ONNX inference
            logger.info(f"\n3. Testing ONNX inference")
            test_onnx_inference(config["output_path"])

            # Test on labeled wildfire data
            logger.info(f"\n4. Testing on labeled wildfire images")
            test_on_labeled_images(config["output_path"])

            # Clean up
            del model
            torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Error processing {config['model_name']}: {e}")
            import traceback
            traceback.print_exc()

    logger.info(f"\n{'='*60}")
    logger.info("Export complete!")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()