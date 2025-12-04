"""One-step ahead prediction task."""

from typing import Any, Dict

import torch

from .base import Task
from .registry import register


@register("one_step")
class OneStepTask(Task):
    """One-step ahead prediction task.

    Predicts x[t+1] from x[:t].
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize one-step task.

        Args:
            config: Task configuration
        """
        super().__init__(config)
        self.horizon = config.get("horizon", 1)

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        model: torch.nn.Module
    ) -> Dict[str, torch.Tensor]:
        """Execute one training step.

        Args:
            batch: Batch with 'x' [B, T_in, D], 'y' [B, T_out, D], 'meta' (optional)
            model: Model to train

        Returns:
            Dictionary with 'loss' and metrics
        """
        x = batch["x"]  # [B, T_in, D]
        y = batch["y"]  # [B, T_out, D]

        # Get model predictions (pass metadata if available)
        output = model(x, meta=batch.get("meta", {}))
        preds = output["preds"]  # [B, 1, D_out]

        # For one-step, we predict the first timestep of y
        targets = y[:, 0:1, :]  # [B, 1, D_out]

        # Compute loss
        loss = self.loss_fn(preds, targets)

        # Compute metrics
        metrics = self.compute_metrics(preds, targets)
        loss, extra_metrics = self._apply_extras(output, loss, targets)
        metrics.update(extra_metrics)

        return {
            "loss": loss,  # Tensor for backprop
            **{k: v for k, v in metrics.items()}  # Metrics as floats/scalars
        }

    def validation_step(
        self,
        batch: Dict[str, torch.Tensor],
        model: torch.nn.Module
    ) -> Dict[str, torch.Tensor]:
        """Execute one validation step.

        Args:
            batch: Batch with 'x', 'y', 'meta' (optional)
            model: Model to evaluate

        Returns:
            Dictionary with 'loss' and metrics
        """
        # Same as training step but without gradient tracking
        with torch.no_grad():
            return self.training_step(batch, model)
