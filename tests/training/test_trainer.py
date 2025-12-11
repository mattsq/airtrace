import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from airtrace.training import trainer as trainer_module
from airtrace.training.trainer import Trainer, set_seed


class _WriterStub:
    def __init__(self, *args, **kwargs):
        self.scalars: List[tuple[str, float, int]] = []
        self.closed = False

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        if isinstance(value, torch.Tensor):
            value = float(value.item())
        self.scalars.append((tag, float(value), int(step)))

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def writer_stub(monkeypatch) -> List[_WriterStub]:
    instances: List[_WriterStub] = []

    class _Factory(_WriterStub):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            instances.append(self)

    monkeypatch.setattr(trainer_module, "SummaryWriter", _Factory)
    return instances


class TinyModel(nn.Module):
    def __init__(self, input_dim: int = 2, output_dim: int = 1):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - tiny passthrough
        return self.linear(x)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class ConstantModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - deterministic path
        batch = x.shape[0]
        return torch.zeros(batch, 1)

    def get_num_params(self) -> int:
        return 0


class DummyTask:
    def training_step(self, batch: Dict[str, torch.Tensor], model: nn.Module) -> Dict[str, torch.Tensor]:
        preds = model(batch["inputs"])
        loss = F.mse_loss(preds, batch["targets"])
        return {"loss": loss, "mae": torch.abs(preds - batch["targets"]).mean()}

    def validation_step(self, batch: Dict[str, torch.Tensor], model: nn.Module) -> Dict[str, torch.Tensor]:
        preds = model(batch["inputs"])
        loss = F.mse_loss(preds, batch["targets"])
        return {"loss": loss, "mae": torch.abs(preds - batch["targets"]).mean()}


class ConstantValidationTask(DummyTask):
    def validation_step(self, batch: Dict[str, torch.Tensor], model: nn.Module) -> Dict[str, torch.Tensor]:
        _ = model(batch["inputs"])
        value = torch.tensor(1.0)
        return {"loss": value, "mae": value}


def _make_batches(num_batches: int = 2, batch_size: int = 4) -> Iterable[Dict[str, torch.Tensor]]:
    base = {
        "inputs": torch.ones(batch_size, 2),
        "targets": torch.ones(batch_size, 1),
    }
    return [
        {key: value.clone() for key, value in base.items()}
        for _ in range(num_batches)
    ]


def _build_config(
    tmp_path: Path, overrides: Optional[Dict[str, Dict[str, object]]] = None
) -> Dict[str, object]:
    config: Dict[str, object] = {
        "log_dir": str(tmp_path / "logs"),
        "train": {
            "epochs": 2,
            "log_every_n_steps": 1,
            "verbose_progress": False,
            "optimizer": {"name": "sgd", "lr": 0.1},
            "scheduler": {"name": "none"},
            "grad_clip": {"enabled": True, "max_norm": 0.5},
            "early_stopping": {"patience": 3, "min_delta": 0.0},
            "checkpoint": {"save_top_k": 2},
        },
    }

    if overrides is not None:
        config = {**config, **{k: v for k, v in overrides.items() if k != "train"}}
        if "train" in overrides:
            config["train"] = {**config["train"], **overrides["train"]}

    return config


def _make_trainer(
    tmp_path: Path,
    writer_stub: List[_WriterStub],
    *,
    model: Optional[nn.Module] = None,
    task: Optional[DummyTask] = None,
    train_loader: Optional[Iterable[Dict[str, torch.Tensor]]] = None,
    val_loader: Optional[Iterable[Dict[str, torch.Tensor]]] = None,
    overrides: Optional[Dict[str, Dict[str, object]]] = None,
) -> Trainer:
    model = model or TinyModel()
    task = task or DummyTask()
    train_loader = train_loader or _make_batches()
    val_loader = val_loader or _make_batches()
    config = _build_config(tmp_path, overrides)
    trainer = Trainer(model, task, config, train_loader, val_loader, device="cpu")
    assert writer_stub, "SummaryWriter stub should record instance"
    return trainer


def test_set_seed_makes_random_generators_deterministic():
    set_seed(123)
    first = (
        random.random(),
        np.random.rand(),
        torch.rand(2),
    )
    set_seed(123)
    second = (
        random.random(),
        np.random.rand(),
        torch.rand(2),
    )
    assert first[0] == second[0]
    assert np.isclose(first[1], second[1])
    assert torch.allclose(first[2], second[2])


def test_trainer_handles_models_without_trainable_parameters(tmp_path, writer_stub):
    trainer = _make_trainer(
        tmp_path,
        writer_stub,
        model=ConstantModel(),
        task=DummyTask(),
        overrides={
            "train": {
                "epochs": 1,
            }
        },
    )
    assert trainer.has_trainable_params is False
    assert trainer.optimizer is None
    trainer.train()
    assert trainer.global_step == 0
    assert writer_stub[0].closed is True


def test_train_epoch_updates_weights_and_logs(tmp_path, writer_stub):
    trainer = _make_trainer(tmp_path, writer_stub)
    initial_weight = trainer.model.linear.weight.detach().clone()
    metrics = trainer.train_epoch()
    assert "loss" in metrics and metrics["loss"] > 0
    assert trainer.global_step == len(trainer.train_loader)
    assert not torch.allclose(initial_weight, trainer.model.linear.weight.detach())
    train_logs = [tag for tag, _, _ in writer_stub[0].scalars if tag.startswith("train/")]
    assert train_logs, "expected train metrics to be logged"


def test_validate_epoch_logs_metrics(tmp_path, writer_stub):
    trainer = _make_trainer(tmp_path, writer_stub)
    trainer.current_epoch = 1
    val_metrics = trainer.validate_epoch()
    assert "loss" in val_metrics
    val_logs = [tag for tag, _, _ in writer_stub[0].scalars if tag.startswith("val/")]
    assert val_logs, "expected validation metrics to be logged"


def test_save_checkpoint_keeps_top_k(tmp_path, writer_stub):
    trainer = _make_trainer(
        tmp_path,
        writer_stub,
        overrides={"train": {"checkpoint": {"save_top_k": 2}}},
    )
    losses = [0.9, 0.7, 1.1, 0.6]
    best_so_far = float("inf")
    for epoch, loss in enumerate(losses):
        trainer.current_epoch = epoch
        is_best = loss < best_so_far
        best_so_far = min(best_so_far, loss)
        trainer.save_checkpoint(loss, is_best=is_best)

    checkpoint_dir = Path(trainer.checkpoint_dir)
    epoch_files = sorted(checkpoint_dir.glob("epoch_*.ckpt"))
    assert len(epoch_files) == 2
    assert (checkpoint_dir / "best.ckpt").exists()


def test_trainer_train_stops_with_early_stopping(tmp_path, writer_stub):
    trainer = _make_trainer(
        tmp_path,
        writer_stub,
        task=ConstantValidationTask(),
        overrides={
            "train": {
                "epochs": 5,
                "early_stopping": {"patience": 1, "min_delta": 0.0},
                "checkpoint": {"save_top_k": 1},
                "scheduler": {"name": "none"},
            }
        },
    )
    trainer.train()
    assert trainer.current_epoch == 1
    assert trainer.early_stop_counter == 1
    checkpoint_dir = Path(trainer.checkpoint_dir)
    assert len(list(checkpoint_dir.glob("epoch_*.ckpt"))) <= 1


def test_validate_epoch_with_multiple_batches_returns_average(tmp_path, writer_stub):
    trainer = _make_trainer(tmp_path, writer_stub, val_loader=_make_batches(num_batches=3))
    val_metrics = trainer.validate_epoch()
    assert val_metrics["loss"] > 0
    val_logs = [entry for entry in writer_stub[0].scalars if entry[0].startswith("val/")]
    assert val_logs, "validation metrics should be logged once per epoch"


def test_save_checkpoint_includes_transform_stats(tmp_path, writer_stub):
    """Test that checkpoints include transform statistics when available."""
    # Create a mock transform pipeline with stats
    class MockTransform:
        def __init__(self):
            self.is_fitted = True

        def get_stats(self):
            return {
                "mock_mean": torch.tensor([1.0, 2.0]).numpy(),
                "mock_std": torch.tensor([0.5, 1.0]).numpy(),
            }

    mock_transforms = MockTransform()

    trainer = _make_trainer(
        tmp_path,
        writer_stub,
        overrides={"train": {"checkpoint": {"save_top_k": 1}}},
    )

    # Manually set transforms
    trainer.transforms = mock_transforms

    # Save checkpoint
    trainer.save_checkpoint(val_loss=0.5, is_best=True)

    # Load and verify
    checkpoint_path = trainer.checkpoint_dir / "best.ckpt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    assert "transform_stats" in checkpoint
    assert checkpoint["transform_stats"] is not None
    assert "mock_mean" in checkpoint["transform_stats"]


def test_save_checkpoint_handles_missing_transforms(tmp_path, writer_stub):
    """Test that checkpoints save successfully when transforms are None."""
    trainer = _make_trainer(
        tmp_path,
        writer_stub,
        overrides={"train": {"checkpoint": {"save_top_k": 1}}},
    )

    # transforms should be None by default
    assert trainer.transforms is None

    # Save checkpoint
    trainer.save_checkpoint(val_loss=0.5, is_best=True)

    # Load and verify
    checkpoint_path = trainer.checkpoint_dir / "best.ckpt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    assert "transform_stats" in checkpoint
    assert checkpoint["transform_stats"] is None


def test_save_checkpoint_handles_unfitted_transforms(tmp_path, writer_stub):
    """Test that checkpoints save successfully when transforms raise errors."""
    class UnfittedTransform:
        def get_stats(self):
            raise RuntimeError("Transform not fitted. Call fit() first.")

    trainer = _make_trainer(
        tmp_path,
        writer_stub,
        overrides={"train": {"checkpoint": {"save_top_k": 1}}},
    )

    trainer.transforms = UnfittedTransform()

    # Save checkpoint - should not fail
    trainer.save_checkpoint(val_loss=0.5, is_best=True)

    # Load and verify
    checkpoint_path = trainer.checkpoint_dir / "best.ckpt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    assert "transform_stats" in checkpoint
    assert checkpoint["transform_stats"] is None  # Set to None due to error


def test_baseline_model_saves_checkpoint(tmp_path, writer_stub):
    """Test that baseline models (non-trainable) save checkpoints correctly."""
    trainer = _make_trainer(
        tmp_path,
        writer_stub,
        model=ConstantModel(),  # Model with no trainable parameters
        task=DummyTask(),
        overrides={"train": {"epochs": 1}},
    )

    # Verify model has no trainable params
    assert trainer.has_trainable_params is False
    assert trainer.optimizer is None
    assert trainer.scheduler is None

    # Run training (should exit early but save checkpoint)
    trainer.train()

    # Verify checkpoint was saved
    checkpoint_path = trainer.checkpoint_dir / "best.ckpt"
    assert checkpoint_path.exists(), "Baseline model should save checkpoint"

    # Load and verify checkpoint contents
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Required fields should be present
    assert "model_state_dict" in checkpoint
    assert "config" in checkpoint
    assert "val_loss" in checkpoint
    assert "epoch" in checkpoint

    # Epoch should be 0 for baseline models (no training epochs)
    assert checkpoint["epoch"] == 0

    # Optimizer/scheduler should NOT be in baseline checkpoint
    assert "optimizer_state_dict" not in checkpoint, "Baseline models should not save optimizer state"
    assert "scheduler_state_dict" not in checkpoint, "Baseline models should not save scheduler state"

    # Transform stats should be present (even if None)
    assert "transform_stats" in checkpoint


def test_baseline_model_checkpoint_can_be_loaded(tmp_path, writer_stub):
    """Test that baseline model checkpoints can be loaded successfully."""
    # Create a mock transform to verify it's included
    class MockTransform:
        def get_stats(self):
            return {"test_param": torch.tensor([1.0, 2.0]).numpy()}

    trainer = _make_trainer(
        tmp_path,
        writer_stub,
        model=ConstantModel(),
        task=DummyTask(),
        overrides={"train": {"epochs": 1}},
    )
    trainer.transforms = MockTransform()

    # Train and save
    trainer.train()

    # Load checkpoint
    checkpoint_path = trainer.checkpoint_dir / "best.ckpt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Verify model state can be loaded
    new_model = ConstantModel()
    new_model.load_state_dict(checkpoint["model_state_dict"])

    # Verify transform stats were saved
    assert checkpoint["transform_stats"] is not None
    assert "test_param" in checkpoint["transform_stats"]

    # Verify val_loss is a valid number
    assert isinstance(checkpoint["val_loss"], (int, float))
    assert checkpoint["val_loss"] >= 0


def test_trainable_model_checkpoint_still_includes_optimizer(tmp_path, writer_stub):
    """Test that trainable models still save optimizer/scheduler state (regression test)."""
    trainer = _make_trainer(
        tmp_path,
        writer_stub,
        model=TinyModel(),  # Model WITH trainable parameters
        overrides={"train": {"epochs": 1, "checkpoint": {"save_top_k": 1}}},
    )

    # Verify model has trainable params and optimizer
    assert trainer.has_trainable_params is True
    assert trainer.optimizer is not None

    # Run training
    trainer.train()

    # Verify checkpoint was saved
    checkpoint_path = trainer.checkpoint_dir / "best.ckpt"
    assert checkpoint_path.exists()

    # Load and verify checkpoint contents
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Trainable models SHOULD have optimizer state
    assert "optimizer_state_dict" in checkpoint, "Trainable models should save optimizer state"
    assert checkpoint["optimizer_state_dict"] is not None
