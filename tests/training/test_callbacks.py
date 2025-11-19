import torch

from airtrace.training.callbacks import EarlyStoppingCallback, LearningRateMonitor


class DummyTrainer:
    def __init__(self, lr: float):
        self.optimizer = type("Opt", (), {"param_groups": [{"lr": lr}]})()


def test_early_stopping_callback_triggers_after_patience():
    callback = EarlyStoppingCallback(monitor="val_loss", patience=2, min_delta=0.01, mode="min")
    metrics = [1.0, 0.95, 0.951, 0.952]
    for epoch, value in enumerate(metrics):
        callback.on_epoch_end(None, epoch, {"val_loss": value})
    assert callback.should_stop is True


def test_early_stopping_callback_supports_max_mode():
    callback = EarlyStoppingCallback(monitor="accuracy", patience=1, min_delta=0.05, mode="max")
    callback.on_epoch_end(None, 0, {"accuracy": 0.7})
    # Improvement resets the counter
    callback.on_epoch_end(None, 1, {"accuracy": 0.78})
    assert callback.should_stop is False
    # Lack of improvement triggers stop after patience
    callback.on_epoch_end(None, 2, {"accuracy": 0.79})
    callback.on_epoch_end(None, 3, {"accuracy": 0.80})
    assert callback.should_stop is True


def test_learning_rate_monitor_prints_current_value(capsys):
    monitor = LearningRateMonitor()
    trainer = DummyTrainer(lr=5e-4)
    monitor.on_epoch_end(trainer, epoch=3, metrics={})
    captured = capsys.readouterr()
    assert "5.000000e-04" in captured.out
