""" TODO:  - save misclassfied samples. could write out the numpy
- precision recall score/curve
- classficiation report
"""

import logging
import numpy as np
from sklearn.metrics import confusion_matrix
from typing import Dict, Any

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
                            model) -> Dict[str, Any]:
        self.logger.info(f"Starting evaluation for {self.experiment_id}")
        y_prediction = model.predict(X_test)

        confusion_matrix_results = self.generate_confusion_matrix(y_test, y_prediction)

        evaluation_metrics = {
            "confusion_matrix": confusion_matrix_results.tolist(),
            "raw_predictions_sample": y_prediction[:10].tolist()
        }

        self.logger.info(f"Sample raw predictions (first 10): {y_prediction[:10]}")

        return evaluation_metrics

    def generate_confusion_matrix(self, y_test, y_prediction):
        # TODO: generate beautiful figure
        y_test_labels = np.argmax(y_test, axis=1)
        y_prediction_labels = np.argmax(y_prediction, axis=1)
        return confusion_matrix(y_test_labels, y_prediction_labels)
