"""Multi-step ahead prediction task."""

import random
from typing import Any, Dict

import torch

from .base import Task
from .registry import register


@register("multi_step")
class MultiStepTask(Task):
    """Multi-step ahead prediction task.

    Predicts x[t+1:t+K] from x[:t] using autoregressive decoding.
    Supports teacher forcing during training.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize multi-step task.

        Args:
            config: Task configuration with 'horizon' and 'teacher_forcing_ratio'
        """
        super().__init__(config)
        self.horizon = config.get("horizon", 32)
        self.teacher_forcing_ratio = config.get("teacher_forcing_ratio", 0.5)

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        model: torch.nn.Module
    ) -> Dict[str, torch.Tensor]:
        """Execute one training step with teacher forcing.

        Args:
            batch: Batch with 'x' [B, T_in, D], 'y' [B, T_out, D]
            model: Model to train

        Returns:
            Dictionary with 'loss' and metrics
        """
        x = batch["x"]  # [B, T_in, D]
        y = batch["y"]  # [B, T_out, D]

        B, T_in, D = x.shape
        T_out = min(self.horizon, y.shape[1])

        # Initialize predictions list
        all_preds = []

        # Current input for autoregressive prediction
        current_input = x

        for t in range(T_out):
            # Get model prediction for next step
            output = model(current_input)
            pred = output["preds"][:, -1:, :]  # [B, 1, D]
            all_preds.append(pred)

            # Teacher forcing: randomly use ground truth instead of prediction
            use_teacher_forcing = random.random() < self.teacher_forcing_ratio

            if use_teacher_forcing and t < T_out - 1:
                # Use ground truth for next input
                next_input = y[:, t:t+1, :]
            else:
                # Use model prediction for next input
                next_input = pred

            # Update current input (sliding window)
            current_input = torch.cat([current_input[:, 1:, :], next_input], dim=1)

        # Stack all predictions
        preds = torch.cat(all_preds, dim=1)  # [B, T_out, D]

        # Targets are the ground truth sequence
        targets = y[:, :T_out, :]

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
        """Execute one validation step (no teacher forcing).

        Args:
            batch: Batch with 'x', 'y'
            model: Model to evaluate

        Returns:
            Dictionary with 'loss' and metrics
        """
        with torch.no_grad():
            x = batch["x"]
            y = batch["y"]

            B, T_in, D = x.shape
            T_out = min(self.horizon, y.shape[1])

            # Autoregressive prediction without teacher forcing
            all_preds = []
            current_input = x

            for t in range(T_out):
                output = model(current_input)
                pred = output["preds"][:, -1:, :]
                all_preds.append(pred)

                # Always use prediction (no teacher forcing in validation)
                current_input = torch.cat([current_input[:, 1:, :], pred], dim=1)

            preds = torch.cat(all_preds, dim=1)
            targets = y[:, :T_out, :]

            loss = self.loss_fn(preds, targets)
            metrics = self.compute_metrics(preds, targets)
            loss, extra_metrics = self._apply_extras(output, loss, targets)
            metrics.update(extra_metrics)

            return {
                "loss": loss,  # Tensor for backprop
                **{k: v for k, v in metrics.items()}  # Metrics as floats/scalars
            }
