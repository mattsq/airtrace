"""Base classes for prediction tasks."""

from abc import ABC, abstractmethod
from typing import Any, Dict

import torch


class Task(ABC):
    """Base class for all prediction tasks.

    Tasks define how to:
    - Slice (x, y) batches for the model
    - Compute losses
    - Aggregate metrics
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize task.

        Args:
            config: Task configuration dictionary
        """
        self.config = config
        self.loss_fn = self._build_loss_fn(config.get("loss", "mse"))
        self.metric_names = config.get("metrics", ["rmse", "mae"])

    @abstractmethod
    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        model: torch.nn.Module
    ) -> Dict[str, torch.Tensor]:
        """Execute one training step.

        Args:
            batch: Batch dictionary with 'x', 'y', 'meta' keys
            model: The model to train

        Returns:
            Dictionary with 'loss' and metric values
        """
        raise NotImplementedError

    @abstractmethod
    def validation_step(
        self,
        batch: Dict[str, torch.Tensor],
        model: torch.nn.Module
    ) -> Dict[str, torch.Tensor]:
        """Execute one validation step.

        Args:
            batch: Batch dictionary with 'x', 'y', 'meta' keys
            model: The model to evaluate

        Returns:
            Dictionary with 'loss' and metric values
        """
        raise NotImplementedError

    def _build_loss_fn(self, loss_name: str):
        """Build loss function from name.

        Args:
            loss_name: Name of loss function ('mse', 'mae', 'nll', etc.)

        Returns:
            Loss function

        Raises:
            ValueError: If loss name not recognized
        """
        loss_map = {
            "mse": torch.nn.MSELoss(),
            "mae": torch.nn.L1Loss(),
            "huber": torch.nn.HuberLoss(),
            "smooth_l1": torch.nn.SmoothL1Loss(),
        }

        if loss_name not in loss_map:
            available = ", ".join(loss_map.keys())
            raise ValueError(
                f"Loss '{loss_name}' not recognized. "
                f"Available: {available}"
            )

        return loss_map[loss_name]

    def compute_metrics(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """Compute evaluation metrics.

        Args:
            preds: Model predictions [B, T, D]
            targets: Ground truth targets [B, T, D]

        Returns:
            Dictionary of metric values
        """
        metrics = {}

        if "rmse" in self.metric_names:
            metrics["rmse"] = torch.sqrt(torch.mean((preds - targets) ** 2)).item()

        if "mae" in self.metric_names:
            metrics["mae"] = torch.mean(torch.abs(preds - targets)).item()

        if "mape" in self.metric_names:
            eps = 1e-8
            mape = torch.mean(torch.abs((targets - preds) / (targets + eps))) * 100
            metrics["mape"] = mape.item()

        if "mse" in self.metric_names:
            metrics["mse"] = torch.mean((preds - targets) ** 2).item()

        return metrics

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  loss={self.config.get('loss', 'mse')},\n"
            f"  metrics={self.metric_names}\n"
            f")"
        )
