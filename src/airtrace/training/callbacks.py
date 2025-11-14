"""Training callbacks."""

from abc import ABC, abstractmethod
from typing import Any, Dict


class Callback(ABC):
    """Base class for training callbacks."""

    def on_train_start(self, trainer):
        """Called at the start of training."""
        pass

    def on_train_end(self, trainer):
        """Called at the end of training."""
        pass

    def on_epoch_start(self, trainer, epoch: int):
        """Called at the start of each epoch."""
        pass

    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float]):
        """Called at the end of each epoch."""
        pass

    def on_batch_start(self, trainer, batch: Any):
        """Called at the start of each batch."""
        pass

    def on_batch_end(self, trainer, batch: Any, outputs: Dict[str, Any]):
        """Called at the end of each batch."""
        pass


class EarlyStoppingCallback(Callback):
    """Early stopping callback.

    Stops training when a monitored metric stops improving.
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = "min"
    ):
        """Initialize early stopping callback.

        Args:
            monitor: Metric to monitor
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max' - whether lower or higher is better
        """
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self.counter = 0
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.should_stop = False

    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float]):
        """Check if training should stop."""
        current_value = metrics.get(self.monitor)

        if current_value is None:
            return

        # Check if improved
        if self.mode == 'min':
            improved = current_value < (self.best_value - self.min_delta)
        else:
            improved = current_value > (self.best_value + self.min_delta)

        if improved:
            self.best_value = current_value
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            print(f"Early stopping triggered: {self.monitor} has not improved for {self.patience} epochs")
            self.should_stop = True


class LearningRateMonitor(Callback):
    """Monitor and log learning rate."""

    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float]):
        """Log current learning rate."""
        lr = trainer.optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch} - Learning rate: {lr:.6e}")
