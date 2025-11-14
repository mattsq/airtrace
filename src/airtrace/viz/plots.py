"""Visualization utilities for timeseries and predictions."""

from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_timeseries(
    data: np.ndarray,
    sensor_names: Optional[List[str]] = None,
    title: str = "Timeseries",
    figsize: tuple = (12, 6)
):
    """Plot multivariate timeseries.

    Args:
        data: Timeseries data [T, D]
        sensor_names: Optional sensor names for legend
        title: Plot title
        figsize: Figure size
    """
    T, D = data.shape

    fig, axes = plt.subplots(D, 1, figsize=figsize, sharex=True)
    if D == 1:
        axes = [axes]

    for i in range(D):
        sensor_name = sensor_names[i] if sensor_names else f"Sensor {i}"
        axes[i].plot(data[:, i], label=sensor_name)
        axes[i].set_ylabel(sensor_name)
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()

    axes[-1].set_xlabel("Time")
    fig.suptitle(title)
    plt.tight_layout()

    return fig


def plot_predictions(
    targets: np.ndarray,
    predictions: np.ndarray,
    sensor_names: Optional[List[str]] = None,
    title: str = "Predictions vs Targets",
    figsize: tuple = (12, 8)
):
    """Plot predictions vs ground truth.

    Args:
        targets: Ground truth [T, D]
        predictions: Predictions [T, D]
        sensor_names: Optional sensor names
        title: Plot title
        figsize: Figure size
    """
    T, D = targets.shape

    fig, axes = plt.subplots(D, 1, figsize=figsize, sharex=True)
    if D == 1:
        axes = [axes]

    for i in range(D):
        sensor_name = sensor_names[i] if sensor_names else f"Sensor {i}"

        axes[i].plot(targets[:, i], label="Ground Truth", alpha=0.7)
        axes[i].plot(predictions[:, i], label="Prediction", alpha=0.7)
        axes[i].set_ylabel(sensor_name)
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()

    axes[-1].set_xlabel("Time")
    fig.suptitle(title)
    plt.tight_layout()

    return fig


def plot_error_distribution(
    errors: np.ndarray,
    sensor_names: Optional[List[str]] = None,
    title: str = "Error Distribution",
    figsize: tuple = (10, 6)
):
    """Plot error distributions.

    Args:
        errors: Prediction errors [T, D]
        sensor_names: Optional sensor names
        title: Plot title
        figsize: Figure size
    """
    _, D = errors.shape

    fig, axes = plt.subplots(1, D, figsize=figsize, sharey=True)
    if D == 1:
        axes = [axes]

    for i in range(D):
        sensor_name = sensor_names[i] if sensor_names else f"Sensor {i}"

        axes[i].hist(errors[:, i], bins=50, alpha=0.7, edgecolor='black')
        axes[i].axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.7)
        axes[i].set_xlabel("Error")
        axes[i].set_title(sensor_name)
        axes[i].grid(True, alpha=0.3)

    axes[0].set_ylabel("Frequency")
    fig.suptitle(title)
    plt.tight_layout()

    return fig


def plot_scatter_predictions(
    targets: np.ndarray,
    predictions: np.ndarray,
    sensor_names: Optional[List[str]] = None,
    title: str = "Prediction Scatter",
    figsize: tuple = (12, 4)
):
    """Plot scatter of predictions vs targets.

    Args:
        targets: Ground truth [N, D]
        predictions: Predictions [N, D]
        sensor_names: Optional sensor names
        title: Plot title
        figsize: Figure size
    """
    _, D = targets.shape

    fig, axes = plt.subplots(1, D, figsize=figsize)
    if D == 1:
        axes = [axes]

    for i in range(D):
        sensor_name = sensor_names[i] if sensor_names else f"Sensor {i}"

        axes[i].scatter(targets[:, i], predictions[:, i], alpha=0.3, s=10)

        # Add diagonal line (perfect predictions)
        min_val = min(targets[:, i].min(), predictions[:, i].min())
        max_val = max(targets[:, i].max(), predictions[:, i].max())
        axes[i].plot([min_val, max_val], [min_val, max_val],
                    'r--', linewidth=2, alpha=0.7)

        axes[i].set_xlabel("Ground Truth")
        axes[i].set_ylabel("Prediction")
        axes[i].set_title(sensor_name)
        axes[i].grid(True, alpha=0.3)

    fig.suptitle(title)
    plt.tight_layout()

    return fig


def plot_metrics_comparison(
    metrics_dict: dict,
    title: str = "Model Comparison",
    figsize: tuple = (10, 6)
):
    """Plot comparison of metrics across models.

    Args:
        metrics_dict: Dict mapping model names to metric dictionaries
        title: Plot title
        figsize: Figure size
    """
    model_names = list(metrics_dict.keys())
    metric_names = list(metrics_dict[model_names[0]].keys())

    fig, axes = plt.subplots(1, len(metric_names), figsize=figsize)
    if len(metric_names) == 1:
        axes = [axes]

    for i, metric_name in enumerate(metric_names):
        values = [metrics_dict[model][metric_name] for model in model_names]

        axes[i].bar(model_names, values)
        axes[i].set_title(metric_name.upper())
        axes[i].set_ylabel("Value")
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].grid(True, alpha=0.3, axis='y')

    fig.suptitle(title)
    plt.tight_layout()

    return fig
