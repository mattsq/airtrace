"""Base classes for prediction tasks."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import torch
from omegaconf import DictConfig, OmegaConf


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
            config: Task configuration dictionary (can be DictConfig or regular dict)
        """
        # Convert OmegaConf DictConfig to regular Python dict to ensure
        # proper handling of lists and .get() method behavior
        if isinstance(config, DictConfig):
            config = OmegaConf.to_container(config, resolve=True)

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
            # The anomaly task config uses "nll" to indicate probabilistic scoring, but
            # the models currently emit mean predictions only. Map "nll" to MSE so the
            # task can still be instantiated while treating the squared error as an NLL
            # proxy (AnomalyTask computes the proper metric values separately).
            "nll": torch.nn.MSELoss(),
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

    def _apply_extras(
        self,
        output: Dict[str, Any],
        loss: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Augment loss/metrics with model extras (pondering, auxiliaries)."""

        extras = output.get("extras", {}) if isinstance(output, dict) else {}
        metrics: Dict[str, float] = {}

        if not extras:
            return loss, metrics

        ponder_loss = extras.get("ponder_loss")
        if ponder_loss is not None:
            loss = loss + ponder_loss
            if isinstance(ponder_loss, torch.Tensor):
                metrics["ponder_loss"] = float(ponder_loss.detach())

        ponder_cost = extras.get("ponder_cost")
        if ponder_cost is not None:
            if isinstance(ponder_cost, torch.Tensor):
                ponder_cost = float(ponder_cost.detach())
            metrics["ponder_cost"] = float(ponder_cost)

        mean_steps = extras.get("mean_ponder_steps")
        if mean_steps is not None:
            if isinstance(mean_steps, torch.Tensor):
                mean_steps = float(mean_steps.detach())
            metrics["ponder_steps"] = float(mean_steps)

        halt_distribution = extras.get("halt_distribution")
        if isinstance(halt_distribution, torch.Tensor):
            metrics["halt_prob_mean"] = float(halt_distribution.mean().detach())

        aux_preds = extras.get("aux_preds")
        aux_weight = float(extras.get("aux_weight", 0.0))
        if aux_preds is not None and aux_weight > 0:
            target_expanded = targets.unsqueeze(1).expand_as(aux_preds)
            aux_loss = self.loss_fn(aux_preds, target_expanded)
            if isinstance(aux_loss, torch.Tensor):
                aux_loss = aux_loss.mean()
            loss = loss + aux_weight * aux_loss
            metrics["aux_loss"] = float(aux_loss.detach())

        max_steps_used = extras.get("max_steps_used")
        if max_steps_used is not None:
            metrics["max_steps_used"] = float(max_steps_used)

        return loss, metrics

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  loss={self.config.get('loss', 'mse')},\n"
            f"  metrics={self.metric_names}\n"
            f")"
        )
