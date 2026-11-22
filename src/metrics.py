""" TODO:  - save misclassfied samples. could write out the numpy
- precision recall score/curve
- classficiation report
"""

import logging
import numpy as np
from sklearn.metrics import confusion_matrix
from typing import Dict, Any, List, Optional
from preprocess import image_group

class ModelEvaluator:
    """Handles comprehensive model evaluation and metrics generation for experiments."""

    def __init__(self, experiment_id: str, experiment_dir: str):
        """Initialize evaluator with experiment context.

        Args:
            experiment_id: Unique experiment identifier
            experiment_dir: Path to experiment output directory
        """
        self.experiment_id = experiment_id
        self.experiment_dir = experiment_dir
        self.logger = logging.getLogger(__name__)

    def evaluate_and_save_all(self, X_test: np.ndarray, y_test: np.ndarray,
                            model, sources: Optional[List[str]] = None,
                            class_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Evaluate the model on the test set.

        Args:
            X_test: Test images.
            y_test: One-hot test labels.
            model: Trained model with a predict() method.
            sources: File path of each test image (same order as X_test). When given,
                results are also broken down per image group (filename prefix).
            class_names: Name of each output index, used to label the group report.
        """
        self.logger.info(f"Starting evaluation for {self.experiment_id}")
        y_prediction = model.predict(X_test)

        confusion_matrix_results = self.generate_confusion_matrix(y_test, y_prediction)

        evaluation_metrics = {
            "confusion_matrix": confusion_matrix_results.tolist(),
            "raw_predictions_sample": y_prediction[:10].tolist()
        }

        if sources is not None:
            names = class_names or [str(i) for i in range(y_test.shape[1])]
            evaluation_metrics["per_group"] = self.generate_group_report(
                y_test, y_prediction, sources, names)

        self.logger.info(f"Sample raw predictions (first 10): {y_prediction[:10]}")

        return evaluation_metrics

    def generate_confusion_matrix(self, y_test, y_prediction):
        # TODO: generate beautiful figure
        y_test_labels = np.argmax(y_test, axis=1)
        y_prediction_labels = np.argmax(y_prediction, axis=1)
        return confusion_matrix(y_test_labels, y_prediction_labels)

    def generate_group_report(self, y_test, y_prediction, sources, class_names):
        """Break test results down by true class and filename prefix.

        Shows, for example, how many haze_* or cloud_* images were mistaken for smoke.

        Returns:
            dict: {"<true class>/<group>": {"count", "correct", "accuracy",
                   "predicted": {class: count}}}
        """
        y_true = np.argmax(y_test, axis=1)
        y_pred = np.argmax(y_prediction, axis=1)

        report = {}
        for true_idx, pred_idx, path in zip(y_true, y_pred, sources):
            key = f"{class_names[true_idx]}/{image_group(path)}"
            entry = report.setdefault(key, {"count": 0, "correct": 0,
                                            "predicted": {name: 0 for name in class_names}})
            entry["count"] += 1
            entry["correct"] += int(true_idx == pred_idx)
            entry["predicted"][class_names[pred_idx]] += 1

        for entry in report.values():
            entry["accuracy"] = entry["correct"] / entry["count"]

        header = f"{'true class / group':<24}{'n':>5}{'correct':>9}" + "".join(
            f"{'->' + name:>10}" for name in class_names)
        lines = [header]
        for key in sorted(report):
            e = report[key]
            lines.append(f"{key:<24}{e['count']:>5}{e['accuracy']:>9.0%}" + "".join(
                f"{e['predicted'][name]:>10}" for name in class_names))
        self.logger.info("Test results per image group:\n" + "\n".join(lines))

        return report
