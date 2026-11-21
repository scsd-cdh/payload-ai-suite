
import numpy as np

# Must match the labels used during model training.
CLASS_NAMES = ["no_fire", "fire"]


def decode_prediction(prediction):
    probabilities = np.asarray(prediction, dtype=np.float32)

    if probabilities.shape != (1, 2):
        raise ValueError(
            f"Expected prediction shape (1, 2), got {probabilities.shape}"
        )

    probabilities = probabilities[0]
    class_index = int(np.argmax(probabilities))
    confidence = float(probabilities[class_index])

    return {
        "class_index": class_index,
        "class_name": CLASS_NAMES[class_index],
        "confidence": confidence,
        "probabilities": probabilities.tolist()
    }


def detect_fire(probabilities, threshold=0.70):
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Threshold must be between 0 and 1")

    probabilities = np.asarray(probabilities, dtype=np.float32)

    if probabilities.shape == (1, 2):
        probabilities = probabilities[0]

    if probabilities.shape != (2,):
        raise ValueError("Expected two class probabilities")

    fire_probability = probabilities[1]

    return bool(fire_probability >= np.float32(threshold))
