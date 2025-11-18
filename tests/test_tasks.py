import importlib
from pathlib import Path
from typing import Dict

import pandas as pd
import pytest
import torch

from airtrace.tasks.base import Task
from airtrace.tasks.one_step import OneStepTask
from airtrace.tasks import registry


class _DummyTask(Task):
    def training_step(self, batch: Dict[str, torch.Tensor], model: torch.nn.Module):
        return {"loss": torch.tensor(0.0)}

    def validation_step(self, batch: Dict[str, torch.Tensor], model: torch.nn.Module):
        return {"loss": torch.tensor(0.0)}


def test_build_loss_fn_and_metrics():
    task = _DummyTask({"loss": "mae", "metrics": ["rmse", "mape", "mse", "mae"]})

    preds = torch.tensor([[[1.0, 2.0]]])
    targets = torch.tensor([[[2.0, 1.0]]])
    metrics = task.compute_metrics(preds, targets)

    expected_rmse = torch.sqrt(torch.mean((preds - targets) ** 2)).item()
    assert metrics["rmse"] == pytest.approx(expected_rmse)
    assert metrics["mae"] == pytest.approx(1.0)
    assert metrics["mape"] == pytest.approx(75.0)
    assert metrics["mse"] == pytest.approx(1.0)


def test_build_loss_fn_invalid_name():
    with pytest.raises(ValueError):
        _DummyTask({"loss": "does_not_exist"})


def test_task_repr_includes_config():
    task = _DummyTask({"loss": "mse", "metrics": ["rmse"]})
    text = repr(task)
    assert "loss=mse" in text
    assert "metrics=['rmse']" in text


@pytest.fixture()
def registry_restore():
    original = registry.TASK_REGISTRY.copy()
    registry.TASK_REGISTRY.clear()
    yield
    registry.TASK_REGISTRY.clear()
    registry.TASK_REGISTRY.update(original)


def test_task_registry_round_trip(registry_restore):
    @registry.register("dummy")
    class AnotherDummy(_DummyTask):
        pass

    built = registry.build_task({"name": "dummy", "loss": "mse"})
    assert isinstance(built, AnotherDummy)
    assert "dummy" in registry.list_tasks()


def test_task_registry_errors(registry_restore):
    class NotATask:
        pass

    with pytest.raises(TypeError):
        registry.register("invalid")(NotATask)

    @registry.register("unique")
    class UniqueTask(_DummyTask):
        pass

    with pytest.raises(ValueError):
        registry.register("unique")(UniqueTask)

    with pytest.raises(ValueError):
        registry.build_task({"name": "missing"})


def test_one_step_task_training_and_validation(monkeypatch):
    class SimpleModel(torch.nn.Module):
        def forward(self, x, meta=None):
            return {"preds": torch.ones_like(x[:, :1, :]) * 0.5}

    batch = {
        "x": torch.zeros(2, 3, 1),
        "y": torch.tensor([[[1.0], [2.0]], [[0.0], [2.0]]]),
    }

    task = OneStepTask({"loss": "mse", "metrics": ["rmse", "mae"]})
    output = task.training_step(batch, SimpleModel())

    expected_targets = batch["y"][:, :1, :]
    assert torch.isclose(task.loss_fn(torch.full_like(expected_targets, 0.5), expected_targets), output["loss"])
    assert set(output.keys()) == {"loss", "rmse", "mae"}
    assert output["rmse"] == pytest.approx(torch.sqrt(torch.tensor(0.25)).item())
    assert output["mae"] == pytest.approx(0.5)

    validation = task.validation_step(batch, SimpleModel())
    assert validation["loss"] == output["loss"]
    assert validation["rmse"] == output["rmse"]
    assert validation["mae"] == output["mae"]
