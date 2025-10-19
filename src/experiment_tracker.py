"""Experiment tracking for model training runs."""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from paths import resolve_path

logger = logging.getLogger(__name__)


def generate_experiment_id() -> str:
    """Generate unique experiment ID based on timestamp.

    Returns:
        str: Experiment ID in format 'exp_YYYYMMDD_HHMMSS'
    """
    return f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def create_experiment_directory(experiment_id: str) -> str:
    """Create directory for experiment outputs.

    Args:
        experiment_id: Unique experiment identifier

    Returns:
        str: Path to experiment directory
    """
    exp_dir = resolve_path(f"experiments/{experiment_id}")
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir


def save_experiment_config(
    experiment_id: str,
    preprocessing_config: Dict[str, Any],
    model_config: Dict[str, Any],
    training_config: Dict[str, Any]
) -> str:
    """Save experiment configuration to JSON file.

    Args:
        experiment_id: Unique experiment identifier
        preprocessing_config: Preprocessing parameters
        model_config: Model architecture parameters
        training_config: Training hyperparameters

    Returns:
        str: Path to saved config file
    """
    exp_dir = create_experiment_directory(experiment_id)

    config = {
        "experiment_id": experiment_id,
        "timestamp": datetime.now().isoformat() + "Z",
        "preprocessing": preprocessing_config,
        "model": model_config,
        "training": training_config
    }

    config_path = os.path.join(exp_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    logger.info(f"Saved experiment config to {config_path}")
    return config_path


def save_experiment_results(
    experiment_id: str,
    results: Dict[str, Any],
    model_filename: str,
    plot_filename: str
) -> str:
    """Save experiment results and output file paths.

    Args:
        experiment_id: Unique experiment identifier
        results: Training results (accuracy, loss metrics)
        model_filename: Path to saved ONNX model
        plot_filename: Path to saved training plot

    Returns:
        str: Path to saved results file
    """
    exp_dir = create_experiment_directory(experiment_id)

    results_data = {
        "experiment_id": experiment_id,
        "results": results,
        "outputs": {
            "model_file": model_filename,
            "plot_file": plot_filename
        }
    }

    results_path = os.path.join(exp_dir, "results.json")
    with open(results_path, 'w') as f:
        json.dump(results_data, f, indent=2)

    logger.info(f"Saved experiment results to {results_path}")
    return results_path


def load_experiment_config(experiment_id: str) -> Optional[Dict[str, Any]]:
    """Load experiment configuration from JSON file.

    Args:
        experiment_id: Unique experiment identifier

    Returns:
        dict: Experiment configuration or None if not found
    """
    config_path = resolve_path(f"experiments/{experiment_id}/config.json")

    if not os.path.exists(config_path):
        logger.error(f"Experiment config not found: {config_path}")
        return None

    with open(config_path, 'r') as f:
        return json.load(f)


def load_experiment_results(experiment_id: str) -> Optional[Dict[str, Any]]:
    """Load experiment results from JSON file.

    Args:
        experiment_id: Unique experiment identifier

    Returns:
        dict: Experiment results or None if not found
    """
    results_path = resolve_path(f"experiments/{experiment_id}/results.json")

    if not os.path.exists(results_path):
        logger.error(f"Experiment results not found: {results_path}")
        return None

    with open(results_path, 'r') as f:
        return json.load(f)
