"""Tests for training callbacks."""

from unittest.mock import MagicMock

import pytest

from airtrace.training.callbacks import (
    Callback,
    EarlyStoppingCallback,
    LearningRateMonitor,
)


class TestCallback:
    """Tests for base Callback class."""

    def test_callback_methods_do_nothing(self):
        """Test that base callback methods are no-ops."""
        callback = Callback()
        trainer = MagicMock()
        
        # All methods should execute without error
        callback.on_train_start(trainer)
        callback.on_train_end(trainer)
        callback.on_epoch_start(trainer, epoch=0)
        callback.on_epoch_end(trainer, epoch=0, metrics={})
        callback.on_batch_start(trainer, batch={})
        callback.on_batch_end(trainer, batch={}, outputs={})


class TestEarlyStoppingCallback:
    """Tests for EarlyStoppingCallback."""

    def test_early_stopping_triggers(self, capsys):
        """Test that early stopping triggers after patience epochs."""
        callback = EarlyStoppingCallback(
            monitor="val_loss",
            patience=3,
            mode="min"
        )
        
        trainer = MagicMock()
        
        # Epochs with no improvement
        callback.on_epoch_end(trainer, epoch=0, metrics={"val_loss": 1.0})
        assert not callback.should_stop
        
        callback.on_epoch_end(trainer, epoch=1, metrics={"val_loss": 1.1})
        assert not callback.should_stop
        
        callback.on_epoch_end(trainer, epoch=2, metrics={"val_loss": 1.2})
        assert not callback.should_stop
        
        callback.on_epoch_end(trainer, epoch=3, metrics={"val_loss": 1.3})
        # Should trigger early stopping after patience=3 epochs
        assert callback.should_stop
        
        captured = capsys.readouterr()
        assert "Early stopping triggered" in captured.out

    def test_early_stopping_resets_on_improvement(self):
        """Test that early stopping counter resets on improvement."""
        callback = EarlyStoppingCallback(
            monitor="val_loss",
            patience=2,
            mode="min"
        )
        
        trainer = MagicMock()
        
        callback.on_epoch_end(trainer, epoch=0, metrics={"val_loss": 1.0})
        callback.on_epoch_end(trainer, epoch=1, metrics={"val_loss": 1.1})
        
        # Improvement - should reset counter
        callback.on_epoch_end(trainer, epoch=2, metrics={"val_loss": 0.5})
        assert callback.counter == 0
        
        # More epochs without improvement
        callback.on_epoch_end(trainer, epoch=3, metrics={"val_loss": 0.6})
        callback.on_epoch_end(trainer, epoch=4, metrics={"val_loss": 0.7})
        
        # Should trigger now
        assert callback.should_stop

    def test_early_stopping_mode_max(self):
        """Test early stopping with mode='max'."""
        callback = EarlyStoppingCallback(
            monitor="val_acc",
            patience=2,
            mode="max"
        )
        
        trainer = MagicMock()
        
        callback.on_epoch_end(trainer, epoch=0, metrics={"val_acc": 0.8})
        callback.on_epoch_end(trainer, epoch=1, metrics={"val_acc": 0.7})  # Worse
        callback.on_epoch_end(trainer, epoch=2, metrics={"val_acc": 0.6})  # Worse
        
        # Should trigger
        assert callback.should_stop

    def test_early_stopping_missing_metric(self):
        """Test early stopping when monitored metric is missing."""
        callback = EarlyStoppingCallback(
            monitor="val_loss",
            patience=2,
            mode="min"
        )
        
        trainer = MagicMock()
        
        # Missing metric should not cause error
        callback.on_epoch_end(trainer, epoch=0, metrics={})
        assert not callback.should_stop


class TestLearningRateMonitor:
    """Tests for LearningRateMonitor."""

    def test_lr_monitor_logs(self, capsys):
        """Test that LR monitor logs learning rate."""
        callback = LearningRateMonitor()
        
        trainer = MagicMock()
        trainer.optimizer.param_groups = [{'lr': 0.001}]
        
        callback.on_epoch_end(trainer, epoch=5, metrics={})
        
        captured = capsys.readouterr()
        assert "Epoch 5" in captured.out
        assert "Learning rate" in captured.out
        assert "1.000000e-03" in captured.out
