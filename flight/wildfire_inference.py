
import numpy as np
import onnxruntime as rt


def load_model(model_path):
    """
    Load the ONNX wildfire model once.
    """
    session = rt.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"]
    )
    return session


def run_inference(session, image_tensor):
    """
    Run wildfire inference on a preprocessed image.

    Args:
        session: Loaded ONNX inference session.
        image_tensor: Float32 image tensor, shape (1, H, W, C).

    Returns:
        Array of class probabilities, shape (1, 2).
    """
    image_tensor = np.asarray(
        image_tensor, dtype=np.float32
    )

    if image_tensor.ndim != 4:
        raise ValueError(
            f"Expected a 4D tensor, got {image_tensor.shape}"
        )

    inputs = session.get_inputs()

    if len(inputs) != 1:
        raise ValueError("Expected exactly one model input")

    model_shape = inputs[0].shape

    for actual, expected in zip(image_tensor.shape, model_shape):
        if isinstance(expected, int) and actual != expected:
            raise ValueError(
                f"Expected input shape {model_shape}, got {image_tensor.shape}"
            )

    input_name = inputs[0].name

    prediction = session.run(
        None,
        {input_name: np.ascontiguousarray(image_tensor)}
    )[0]

    prediction = np.asarray(prediction, dtype=np.float32)

    if prediction.shape != (1, 2):
        raise ValueError(
            f"Expected output shape (1, 2), got {prediction.shape}"
        )

    return prediction
