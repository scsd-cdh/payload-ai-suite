import numpy as np
import torch
import onnx
import onnxruntime as ort

class CloudModel(torch.nn.Module):
    def __init__(self, input_channels = 1, out_filters = 128, kernel = 5):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(input_channels, out_filters, kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.conv1(x))
    


# export the torch model to ONNX and verify it
def export_pytorch_to_onnx(model: torch.nn.Module,
                            onnx_path: str = "cloud_mask.onnx",
                            input_shape=(1,1,128,128),
                            opset_version=13):
    
    """
    Export the PyTorch CloudModel to ONNX format.

    Args:
        model (torch.nn.Module): The PyTorch model to export.
        onnx_path (str): Where to save the .onnx file.
        input_shape (tuple): Dummy input shape.
    """

    model.eval()
    dummy = torch.randn(*input_shape, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        export_params = True,
        opset_version = opset_version,
        input_names=['input'],
        output_names=['mask'],
        dynamic_axes={"input":{0:"batch"}, "mask":{0:"batch"}}
    )
    print(f"onnx model to {onnx_path}")

    onx = onnx.load(onnx_path)
    onnx.checker.check_model(onx)
    print("onnx model check passed")

def run_pytorch_onnx_inference(onnx_path="cloud_mask.onnx",
                                input_np: np.ndarray=None):
    
    if input_np is None:
        input_np = np.random.rand(1,1,128,128).astype(np.float32) #dummy data

    sess = ort.InferenceSession(onnx_path)
    name = sess.get_inputs()[0].name
    out = sess.run(None, {name: input_np})[0]
    print("output shape: ", out.shape)
    return out