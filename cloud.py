from cloud_model import CloudModel, export_pytorch_to_onnx, run_pytorch_onnx_inference

def main():
    model = CloudModel(input_channels=1, out_filters=128, kernel=5)
    export_pytorch_to_onnx(model, onnx_path="cloud_mask.onnx")

    run_pytorch_onnx_inference("cloud_mask.onnx")

if __name__ == "__main__":
    main()

'''
import numpy as np
import torch

from omnicloudmask import predict_from_array
from model import export_to_onnx

# Logic to do a basic prediction from omnicloudmask
# Example input array, in practice this should be Red, Green and NIR bands
input_array = np.random.rand(3, 1024, 1024)

# Predict cloud and cloud shadow masks
# pred_mask = predict_from_array(input_array)
# print(pred_mask)


# TODO: try to get omnicloudmask torch model in this class 
class CloudModel(torch.nn.Module):
    def __init__(self):
        super(CloudModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 128, 5)

    def forward(self, x):
        return torch.relu(self.conv1(x))

input_tensor = torch.rand((1, 1, 128, 128), dtype=torch.float32)

model = CloudModel()
# make a separate function for export to onnx using PyTorch, with the same features
#export_to_onnx(model) 
# find a way to support omnicloudmask with keras.model.model


torch.onnx.export(
    model,                  # model to export
    (input_tensor,),        # inputs of the model,
    "cloud_mask.onnx",        # filename of the ONNX model
    input_names=["input"],  # Rename inputs for the ONNX model
    dynamo=True             # True or False to select the exporter to use
)
'''