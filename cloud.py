import torch
import torch.nn as nn
import numpy as np

# Load the checkpoint
model_path = "models/omnicloudmask/PM_model_2.2.10_RG_NIR_509_convnextv2_nano.fcmae_ft_in1k_PT_state.pth"
checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

# What you can do with the checkpoint:

# 1. Analyze the model structure
print("1. Analyzing model structure from checkpoint:")
print(f"Total parameters: {len(checkpoint)}")
print(f"First layer: {list(checkpoint.keys())[0]}, shape: {checkpoint[list(checkpoint.keys())[0]].shape}")
print(f"Last layer: {list(checkpoint.keys())[-1]}, shape: {checkpoint[list(checkpoint.keys())[-1]].shape}")

# 2. Get input/output dimensions
first_conv = checkpoint['layers.0.0.0.weight']
last_layer = checkpoint['layers.9.0.weight']
print(f"\nInput channels: {first_conv.shape[1]}")
print(f"Output classes: {last_layer.shape[0]}")

# 3. Check total model size
total_params = sum(p.numel() for p in checkpoint.values() if isinstance(p, torch.Tensor))
print(f"\nTotal parameters: {total_params:,}")
model_size_mb = sum(p.numel() * p.element_size() for p in checkpoint.values() if isinstance(p, torch.Tensor)) / 1024 / 1024
print(f"Model size: {model_size_mb:.2f} MB")

# 4. To export to ONNX, you need the model architecture
# Since this is OmniCloudMask, let's try using their API
try:
    import omnicloudmask as ocm

    # Method 1: Use omnicloudmask's predict function which loads the model internally
    print("\n4. Attempting to use OmniCloudMask API...")

    # Create dummy input
    dummy_input = np.random.rand(3, 512, 512).astype(np.float32)

    # This will use the model internally
    result = ocm.predict_from_array(
        input_array=dummy_input,
        export_confidence=True,
        softmax_output=True,
        destination_model_dir="models/omnicloudmask"
    )
    print(result)
    print("Successfully loaded model through OmniCloudMask API")

except Exception as e:
    print(f"OmniCloudMask API error: {e}")

# 5. Alternative: Reconstruct the model architecture
# Based on the checkpoint keys, this appears to be a custom architecture
# with the following structure:
print("\n5. Model architecture based on checkpoint keys:")
print("- layers.0: Initial convolution blocks")
print("- layers.0.1.x: Multiple stages with ConvNeXt-like blocks")
print("- layers.1: Batch normalization")
print("- layers.3: Additional convolutions")
print("- layers.4: Feature fusion block")
print("- layers.5: More convolutions")
print("- layers.8: Skip connection path")
print("- layers.9: Final classification layer")

# To export to ONNX, you would need to:
# 1. Define a model class that matches this architecture
# 2. Load the checkpoint into the model
# 3. Export using torch.onnx.export()

print("\nTo export this checkpoint to ONNX, you need the model definition.")
print("Options:")
print("1. Use omnicloudmask library's model loading function")
print("2. Find the model definition in the omnicloudmask source code")
print("3. Reconstruct the architecture based on the checkpoint structure")