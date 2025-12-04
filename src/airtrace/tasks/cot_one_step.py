"""Chain-of-Thought aware one-step prediction task.

Extends the standard one-step task to handle models that use latent
chain-of-thought reasoning with Adaptive Computation Time (ACT).

The task computes both:
1. Primary prediction loss (e.g., MSE between predictions and targets)
2. Auxiliary ACT loss (regularizing expected computation steps)

Total loss = prediction_loss + act_weight * act_loss
"""

from typing import Any, Dict

import torch

from .base import Task
from .registry import register


@register("cot_one_step")
class COTOneStepTask(Task):
    """Chain-of-Thought aware one-step ahead prediction task.

    Predicts x[t+1] from x[:t] while handling auxiliary losses from
    models that use latent chain-of-thought reasoning.

    The model is expected to return:
        {
            "preds": [B, 1, D_out],
            "extras": {
                "act_loss": scalar tensor (optional),
                "num_steps": [B] (optional, for logging),
                ...
            }
        }
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize COT one-step task.

        Args:
            config: Task configuration with optional:
                - horizon: Prediction horizon (default: 1)
                - act_loss_weight: Weight for ACT regularization (default: 0.01)
                - act_loss_schedule: How to schedule act_loss_weight
                    Options: "constant", "linear_decay", "cosine_decay"
                - act_warmup_steps: Steps before applying full ACT loss (default: 0)
        """
        super().__init__(config)
        self.horizon = config.get("horizon", 1)
        self.act_loss_weight = config.get("act_loss_weight", 0.01)
        self.act_loss_schedule = config.get("act_loss_schedule", "constant")
        self.act_warmup_steps = config.get("act_warmup_steps", 0)

        # Track global training step for scheduling
        self.global_step = 0

    def get_act_loss_weight(self) -> float:
        """Compute current ACT loss weight based on schedule.

        Returns:
            Current ACT loss weight
        """
        if self.global_step < self.act_warmup_steps:
            # Linear warmup
            alpha = self.global_step / max(1, self.act_warmup_steps)
            return self.act_loss_weight * alpha

        if self.act_loss_schedule == "constant":
            return self.act_loss_weight

        elif self.act_loss_schedule == "linear_decay":
            # Linearly decay to 0 over training
            # (Would need max_steps from config for proper implementation)
            return self.act_loss_weight

        elif self.act_loss_schedule == "cosine_decay":
            # Cosine decay to 0 over training
            # (Would need max_steps from config for proper implementation)
            return self.act_loss_weight

        else:
            return self.act_loss_weight

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
            Dictionary with 'loss' (total), 'pred_loss', 'act_loss', and metrics
        """
        x = batch["x"]  # [B, T_in, D]
        y = batch["y"]  # [B, T_out, D]

        # Get model predictions (pass metadata if available)
        output = model(x, meta=batch.get("meta", {}))
        preds = output["preds"]  # [B, 1, D_out]

        # For one-step, we predict the first timestep of y
        targets = y[:, 0:1, :]  # [B, 1, D_out]

        # === PRIMARY LOSS: Prediction accuracy ===
        pred_loss = self.loss_fn(preds, targets)

        # === AUXILIARY LOSS: ACT regularization ===
        act_loss = torch.tensor(0.0, device=preds.device)
        if "extras" in output and "act_loss" in output["extras"]:
            act_loss = output["extras"]["act_loss"]

        # === TOTAL LOSS ===
        act_weight = self.get_act_loss_weight()
        total_loss = pred_loss + act_weight * act_loss

        # === COMPUTE METRICS ===
        metrics = self.compute_metrics(preds, targets)

        # === ADDITIONAL LOGGING INFO ===
        result = {
            "loss": total_loss,  # Used for backprop
            "pred_loss": pred_loss.item(),  # For logging
            "act_loss": act_loss.item() if isinstance(act_loss, torch.Tensor) else act_loss,
            "act_weight": act_weight,
            **metrics
        }

        # Log pondering statistics if available
        if "extras" in output:
            extras = output["extras"]
            if "mean_steps" in extras:
                result["mean_steps"] = extras["mean_steps"]
            if "max_steps" in extras:
                result["max_steps"] = extras["max_steps"]
            if "num_steps" in extras:
                # Log distribution statistics
                num_steps = extras["num_steps"]
                if isinstance(num_steps, torch.Tensor):
                    result["steps_std"] = num_steps.float().std().item()

        # Increment global step counter
        self.global_step += 1

        return result

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
        x = batch["x"]  # [B, T_in, D]
        y = batch["y"]  # [B, T_out, D]

        # Get model predictions
        with torch.no_grad():
            output = model(x, meta=batch.get("meta", {}))
            preds = output["preds"]  # [B, 1, D_out]

        # For one-step, we predict the first timestep of y
        targets = y[:, 0:1, :]  # [B, 1, D_out]

        # === PRIMARY LOSS ===
        pred_loss = self.loss_fn(preds, targets)

        # === AUXILIARY LOSS ===
        act_loss = torch.tensor(0.0, device=preds.device)
        if "extras" in output and "act_loss" in output["extras"]:
            act_loss = output["extras"]["act_loss"]

        # === TOTAL LOSS ===
        act_weight = self.act_loss_weight  # Use base weight for validation
        total_loss = pred_loss + act_weight * act_loss

        # === COMPUTE METRICS ===
        metrics = self.compute_metrics(preds, targets)

        # === RESULT ===
        result = {
            "loss": total_loss,
            "pred_loss": pred_loss.item(),
            "act_loss": act_loss.item() if isinstance(act_loss, torch.Tensor) else act_loss,
            **metrics
        }

        # Log pondering statistics
        if "extras" in output:
            extras = output["extras"]
            if "mean_steps" in extras:
                result["mean_steps"] = extras["mean_steps"]
            if "max_steps" in extras:
                result["max_steps"] = extras["max_steps"]

        return result

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  loss={self.config.get('loss', 'mse')},\n"
            f"  metrics={self.metric_names},\n"
            f"  act_loss_weight={self.act_loss_weight},\n"
            f"  act_loss_schedule={self.act_loss_schedule}\n"
            f")"
        )
