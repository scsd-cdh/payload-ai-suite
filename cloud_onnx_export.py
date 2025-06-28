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
        pretrained=False,
        in_chans=3,  # RGB channels
    )
    
    # Create FastAI DynamicUnet model
    model = create_unet_model(
        arch=timm_model,
        n_out=4,  # 4 output classes for cloud mask
        img_size=(509, 509),  # OmniCloudMask uses 509x509 patches
        act_cls=torch.nn.Mish,
        pretrained=False,
    )
    
    # Load the state dict
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    
    return model.to(device)


def export_to_onnx(model, output_path, dummy_input_shape=(1, 3, 509, 509)):
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
    
    print(f"Model exported to {output_path}")
    
    # Verify the exported model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model verified successfully")
    
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
    
    print(f"ONNX inference successful!")
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {ort_outputs[0].shape}")
    print(f"Output range: [{ort_outputs[0].min():.4f}, {ort_outputs[0].max():.4f}]")
    
    return ort_outputs[0]


def main():
    """Main function to export OmniCloudMask models to ONNX."""
    
    # Define model paths
    models_dir = Path("models/omnicloudmask")
    
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
        print(f"\n{'='*60}")
        print(f"Processing {config['model_name']} model")
        print(f"{'='*60}")
        
        try:
            # Check if weights file exists
            if not config["weights_path"].exists():
                print(f"Warning: Weights file not found at {config['weights_path']}")
                print("Skipping this model...")
                continue
            
            # Load model
            print(f"\n1. Loading model from {config['weights_path']}")
            model = load_omnicloudmask_model(
                weights_path=config["weights_path"],
                model_name=config["model_name"],
                device="cpu"
            )
            print("Model loaded successfully")
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Total parameters: {total_params:,}")
            
            # Export to ONNX
            print(f"\n2. Exporting to ONNX format")
            export_to_onnx(model, config["output_path"])
            
            # Test ONNX inference
            print(f"\n3. Testing ONNX inference")
            test_onnx_inference(config["output_path"])
            
            # Clean up
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error processing {config['model_name']}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("Export complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()