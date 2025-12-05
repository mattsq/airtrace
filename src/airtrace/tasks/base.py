"""Base classes for prediction tasks."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from airtrace.models.halting_losses import pondernet_loss, trm_halting_loss


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
        self.loss_name = config.get("loss", "mse")
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

    def _elementwise_loss(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute loss with no reduction over batch or sequence dimensions."""

        if self.loss_name in {"mae"}:
            losses = F.l1_loss(preds, targets, reduction="none")
        elif self.loss_name in {"huber"}:
            losses = F.huber_loss(preds, targets, reduction="none")
        elif self.loss_name in {"smooth_l1"}:
            losses = F.smooth_l1_loss(preds, targets, reduction="none")
        else:
            # Default to MSE (covers "mse" and "nll" alias)
            losses = F.mse_loss(preds, targets, reduction="none")

        return losses

    def _per_example_loss(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute element-wise loss averaged per example.

        This mirrors the configured loss_fn but returns a loss per batch
        element rather than a single scalar.
        """

        losses = self._elementwise_loss(preds, targets)

        if losses.ndim > 1:
            reduce_dims = tuple(range(1, losses.ndim))
            losses = losses.mean(dim=reduce_dims)
        return losses

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

        halting_mode = extras.get("halting_mode")
        halt_logits = extras.get("halt_logits")
        step_preds = extras.get("step_preds")
        ponder_penalty = extras.get("ponder_penalty")
        halting_weight = float(extras.get("halting_weight", 1.0))

        if halting_mode in {None, "none"}:
            ponder_loss = extras.get("ponder_loss")
            if ponder_loss is not None:
                loss = loss + ponder_loss
                if isinstance(ponder_loss, torch.Tensor):
                    metrics["ponder_loss"] = float(ponder_loss.detach())

        elif halting_mode == "trm" and halt_logits is not None and step_preds is not None:
            halting_loss = trm_halting_loss(halt_logits, step_preds, targets)
            total_halting_loss = halting_weight * halting_loss
            loss = loss + total_halting_loss
            metrics["halting_loss"] = float(total_halting_loss.detach())

        elif halting_mode == "pondernet" and halt_logits is not None and step_preds is not None:
            ponder_penalty_value = float(ponder_penalty if ponder_penalty is not None else 0.0)
            loss = pondernet_loss(
                halt_logits=halt_logits,
                step_preds=step_preds,
                targets=targets,
                base_loss_fn=self._elementwise_loss,
                ponder_penalty=ponder_penalty_value,
            )
            metrics["pondernet_loss"] = float(loss.detach())

            q = torch.sigmoid(halt_logits)
            c = 1 - q
            c_prefix = torch.cumprod(
                torch.cat(
                    [torch.ones_like(c[:, :1]), c[:, :-1]],
                    dim=1,
                ),
                dim=1,
            )
            p_halt = q * c_prefix
            p_rest = c_prefix[:, -1]

            steps = torch.arange(
                1, halt_logits.shape[1] + 1, device=halt_logits.device, dtype=halt_logits.dtype
            )
            exp_steps = (p_halt * steps).sum(dim=1) + p_rest * steps[-1]
            metrics["ponder_exp_steps"] = float(exp_steps.mean().detach())

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
