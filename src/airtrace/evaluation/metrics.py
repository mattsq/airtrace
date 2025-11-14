"""Evaluation metrics for timeseries forecasting."""

from typing import Dict

import numpy as np
import torch


def rmse(preds: np.ndarray, targets: np.ndarray) -> float:
    """Root mean squared error.

    Args:
        preds: Predictions [N, ...]
        targets: Ground truth [N, ...]

    Returns:
        RMSE value
    """
    return np.sqrt(np.mean((preds - targets) ** 2))


def mae(preds: np.ndarray, targets: np.ndarray) -> float:
    """Mean absolute error.

    Args:
        preds: Predictions
        targets: Ground truth

    Returns:
        MAE value
    """
    return np.mean(np.abs(preds - targets))


def mape(preds: np.ndarray, targets: np.ndarray, epsilon: float = 1e-8) -> float:
    """Mean absolute percentage error.

    Args:
        preds: Predictions
        targets: Ground truth
        epsilon: Small constant to avoid division by zero

    Returns:
        MAPE value (as percentage)
    """
    return np.mean(np.abs((targets - preds) / (targets + epsilon))) * 100


def mse(preds: np.ndarray, targets: np.ndarray) -> float:
    """Mean squared error.

    Args:
        preds: Predictions
        targets: Ground truth

    Returns:
        MSE value
    """
    return np.mean((preds - targets) ** 2)


def r2_score(preds: np.ndarray, targets: np.ndarray) -> float:
    """R² (coefficient of determination) score.

    Args:
        preds: Predictions
        targets: Ground truth

    Returns:
        R² value
    """
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    return 1 - (ss_res / (ss_tot + 1e-8))


def compute_all_metrics(
    preds: np.ndarray,
    targets: np.ndarray
) -> Dict[str, float]:
    """Compute all standard metrics.

    Args:
        preds: Predictions
        targets: Ground truth

    Returns:
        Dictionary of metric values
    """
    return {
        "rmse": rmse(preds, targets),
        "mae": mae(preds, targets),
        "mape": mape(preds, targets),
        "mse": mse(preds, targets),
        "r2": r2_score(preds, targets)
    }


def per_sensor_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    sensor_names: list
) -> Dict[str, Dict[str, float]]:
    """Compute metrics per sensor.

    Args:
        preds: Predictions [..., D]
        targets: Ground truth [..., D]
        sensor_names: List of sensor names

    Returns:
        Dictionary mapping sensor names to metric dictionaries
    """
    results = {}

    for i, sensor_name in enumerate(sensor_names):
        sensor_preds = preds[..., i]
        sensor_targets = targets[..., i]

        results[sensor_name] = compute_all_metrics(
            sensor_preds.flatten(),
            sensor_targets.flatten()
        )

    return results
