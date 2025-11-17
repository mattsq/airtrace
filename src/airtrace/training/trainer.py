"""Training loop implementation."""

import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class Trainer:
    """Main training loop handler."""

    def __init__(
        self,
        model: nn.Module,
        task: Any,
        config: Dict[str, Any],
        train_loader: Any,
        val_loader: Any,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """Initialize trainer.

        Args:
            model: Model to train
            task: Task object defining training/validation steps
            config: Training configuration
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to train on
        """
        self.model = model.to(device)
        self.task = task
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Training config (must be set before building scheduler)
        train_config = config.get("train", {})
        self.epochs = train_config.get("epochs", 50)
        self.log_every_n_steps = train_config.get("log_every_n_steps", 50)

        # Build optimizer and scheduler
        self.optimizer = self._build_optimizer(config.get("train", {}).get("optimizer", {}))
        self.scheduler = self._build_scheduler(config.get("train", {}).get("scheduler", {}))

        # Gradient clipping
        grad_clip_config = train_config.get("grad_clip", {})
        self.use_grad_clip = grad_clip_config.get("enabled", True)
        self.grad_clip_max_norm = grad_clip_config.get("max_norm", 1.0)

        # Early stopping
        early_stop_config = train_config.get("early_stopping", {})
        self.early_stop_patience = early_stop_config.get("patience", 10)
        self.early_stop_min_delta = early_stop_config.get("min_delta", 1e-4)
        self.early_stop_counter = 0
        self.best_val_loss = float('inf')

        # Checkpointing
        checkpoint_config = train_config.get("checkpoint", {})
        self.checkpoint_dir = Path(config.get("log_dir", "runs/debug")) / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_top_k = checkpoint_config.get("save_top_k", 3)
        self.saved_checkpoints = []

        # Logging
        self.log_dir = Path(config.get("log_dir", "runs/debug"))
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

        # Tracking
        self.current_epoch = 0
        self.global_step = 0

    def _build_optimizer(self, optim_config: Dict[str, Any]):
        """Build optimizer from config.

        Args:
            optim_config: Optimizer configuration

        Returns:
            Optimizer instance
        """
        name = optim_config.get("name", "adam").lower()
        lr = optim_config.get("lr", 1e-3)
        weight_decay = optim_config.get("weight_decay", 0.0)

        if name == "adam":
            betas = optim_config.get("betas", [0.9, 0.999])
            return Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
        elif name == "adamw":
            betas = optim_config.get("betas", [0.9, 0.999])
            return AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
        elif name == "sgd":
            momentum = optim_config.get("momentum", 0.9)
            return SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
        else:
            raise ValueError(f"Unknown optimizer: {name}")

    def _build_scheduler(self, sched_config: Dict[str, Any]):
        """Build learning rate scheduler.

        Args:
            sched_config: Scheduler configuration

        Returns:
            Scheduler instance or None
        """
        name = sched_config.get("name", "cosine").lower()

        if name == "cosine":
            T_max = self.epochs - sched_config.get("warmup_epochs", 0)
            min_lr = sched_config.get("min_lr", 1e-6)
            return CosineAnnealingLR(self.optimizer, T_max=T_max, eta_min=min_lr)
        elif name == "step":
            step_size = sched_config.get("step_size", 10)
            gamma = sched_config.get("gamma", 0.1)
            return StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        elif name == "none":
            return None
        else:
            raise ValueError(f"Unknown scheduler: {name}")

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        epoch_metrics = {}
        metric_sums = {}
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}", disable=True)

        for batch in pbar:
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.task.training_step(batch, self.model)
            loss = outputs["loss"]

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip_max_norm
                )

            # Optimizer step
            self.optimizer.step()

            # Accumulate metrics
            for k, v in outputs.items():
                if k != "loss":
                    metric_sums[k] = metric_sums.get(k, 0.0) + v
                else:
                    metric_sums[k] = metric_sums.get(k, 0.0) + v.item()

            num_batches += 1
            self.global_step += 1

            # Logging
            if self.global_step % self.log_every_n_steps == 0:
                for k, v in outputs.items():
                    val = v.item() if isinstance(v, torch.Tensor) else v
                    self.writer.add_scalar(f"train/{k}", val, self.global_step)

            # Update progress bar
            pbar.set_postfix(loss=outputs["loss"].item())

        # Compute epoch averages
        for k, v in metric_sums.items():
            epoch_metrics[k] = v / num_batches

        return epoch_metrics

    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch.

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        metric_sums = {}
        num_batches = 0

        for batch in tqdm(self.val_loader, desc="Validation", disable=True):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            outputs = self.task.validation_step(batch, self.model)

            # Accumulate metrics
            for k, v in outputs.items():
                if k != "loss":
                    metric_sums[k] = metric_sums.get(k, 0.0) + v
                else:
                    metric_sums[k] = metric_sums.get(k, 0.0) + v.item()

            num_batches += 1

        # Compute averages
        val_metrics = {}
        for k, v in metric_sums.items():
            val_metrics[k] = v / num_batches

        # Log to tensorboard
        for k, v in val_metrics.items():
            self.writer.add_scalar(f"val/{k}", v, self.current_epoch)

        return val_metrics

    def save_checkpoint(self, val_loss: float, is_best: bool = False):
        """Save model checkpoint.

        Args:
            val_loss: Validation loss for this checkpoint
            is_best: Whether this is the best checkpoint so far
        """
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "config": self.config
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        # Save checkpoint
        if is_best:
            checkpoint_path = self.checkpoint_dir / "best.ckpt"
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved best checkpoint: {checkpoint_path}")

        # Save epoch checkpoint
        checkpoint_path = self.checkpoint_dir / f"epoch_{self.current_epoch}.ckpt"
        torch.save(checkpoint, checkpoint_path)

        # Manage top-k checkpoints
        self.saved_checkpoints.append((val_loss, checkpoint_path))
        self.saved_checkpoints.sort(key=lambda x: x[0])

        # Remove worst checkpoints
        while len(self.saved_checkpoints) > self.save_top_k:
            _, path_to_remove = self.saved_checkpoints.pop()
            if path_to_remove.exists() and "best" not in path_to_remove.name:
                path_to_remove.unlink()

    def train(self):
        """Run full training loop."""
        print(f"Starting training for {self.epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Model parameters: {self.model.get_num_params():,}")

        for epoch in range(self.epochs):
            self.current_epoch = epoch

            # Train epoch
            train_metrics = self.train_epoch()
            print(f"\nEpoch {epoch} - Train metrics: {train_metrics}")

            # Validate epoch
            val_metrics = self.validate_epoch()
            print(f"Epoch {epoch} - Val metrics: {val_metrics}")

            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar("lr", current_lr, epoch)

            # Checkpointing
            val_loss = val_metrics.get("loss", float('inf'))
            is_best = val_loss < self.best_val_loss

            if is_best:
                self.best_val_loss = val_loss
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1

            self.save_checkpoint(val_loss, is_best=is_best)

            # Early stopping
            if self.early_stop_counter >= self.early_stop_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

        print(f"\nTraining complete. Best val loss: {self.best_val_loss:.4f}")
        self.writer.close()


def set_seed(seed: int):
    """Set random seed for reproducibility.

    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
