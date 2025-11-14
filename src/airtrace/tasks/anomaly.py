"""Anomaly detection task."""

from typing import Any, Dict

import torch
import torch.nn.functional as F

from .base import Task
from .registry import register


@register("anomaly")
class AnomalyTask(Task):
    """Anomaly detection via likelihood-based scoring.

    Scores sequences by their log-likelihood under the model.
    Low likelihood indicates anomalous behavior.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize anomaly task.

        Args:
            config: Task configuration
        """
        super().__init__(config)
        self.score_type = config.get("score_type", "log_likelihood")
        self.threshold_percentile = config.get("threshold_percentile", 99.0)

        # For likelihood-based scoring, we need a probabilistic loss
        # Override loss_fn if NLL was specified
        if config.get("loss") == "nll":
            # Will compute Gaussian NLL
            self.use_probabilistic = True
        else:
            self.use_probabilistic = False

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        model: torch.nn.Module
    ) -> Dict[str, torch.Tensor]:
        """Execute one training step.

        Args:
            batch: Batch with 'x', 'y'
            model: Model to train

        Returns:
            Dictionary with 'loss' and metrics
        """
        x = batch["x"]
        y = batch["y"]

        # Get model predictions
        output = model(x)
        preds = output["preds"][:, 0:1, :]  # [B, 1, D]

        # Target is first step
        targets = y[:, 0:1, :]

        if self.use_probabilistic:
            # Compute negative log-likelihood (assuming Gaussian)
            # Simplified: use MSE as proxy for -log p(y|x)
            mse = F.mse_loss(preds, targets, reduction='none')
            # NLL = 0.5 * log(2π) + 0.5 * log(σ²) + 0.5 * (y-μ)²/σ²
            # For simplicity, assume σ=1 and compute reconstruction error
            loss = mse.mean()
            nll = loss  # Proxy for NLL
        else:
            loss = self.loss_fn(preds, targets)
            nll = loss

        metrics = {
            "loss": loss.item(),
            "nll": nll.item() if isinstance(nll, torch.Tensor) else nll
        }

        # Compute standard metrics
        pred_metrics = self.compute_metrics(preds, targets)
        metrics.update(pred_metrics)

        return {
            "loss": loss,
            **metrics
        }

    def validation_step(
        self,
        batch: Dict[str, torch.Tensor],
        model: torch.nn.Module
    ) -> Dict[str, torch.Tensor]:
        """Execute one validation step.

        Args:
            batch: Batch with 'x', 'y'
            model: Model to evaluate

        Returns:
            Dictionary with 'loss', 'nll', and anomaly scores
        """
        with torch.no_grad():
            x = batch["x"]
            y = batch["y"]

            output = model(x)
            preds = output["preds"][:, 0:1, :]
            targets = y[:, 0:1, :]

            # Compute anomaly scores (reconstruction error)
            reconstruction_errors = F.mse_loss(
                preds, targets, reduction='none'
            ).mean(dim=(1, 2))  # [B]

            # Compute loss
            if self.use_probabilistic:
                loss = reconstruction_errors.mean()
                nll = loss
            else:
                loss = self.loss_fn(preds, targets)
                nll = reconstruction_errors.mean()

            metrics = {
                "loss": loss.item(),
                "nll": nll.item() if isinstance(nll, torch.Tensor) else nll,
                "anomaly_scores": reconstruction_errors.cpu().numpy()
            }

            # Standard metrics
            pred_metrics = self.compute_metrics(preds, targets)
            metrics.update(pred_metrics)

            return {
                "loss": loss,
                **metrics
            }
