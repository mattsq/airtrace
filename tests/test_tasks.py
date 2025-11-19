import importlib
from pathlib import Path
from typing import Dict

import pandas as pd
import pytest
import torch

from airtrace.tasks.anomaly import AnomalyTask
from airtrace.tasks.base import Task
from airtrace.tasks.multi_step import MultiStepTask
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


def test_multi_step_task_training_without_teacher_forcing():
    class CounterModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, x):
            self.calls += 1
            value = torch.full_like(x, float(self.calls))
            return {"preds": value}

    batch = {
        "x": torch.zeros(1, 3, 1),
        "y": torch.tensor([[[0.0], [10.0], [20.0]]]),
    }

    task = MultiStepTask(
        {"loss": "mse", "metrics": ["rmse"], "horizon": 2, "teacher_forcing_ratio": 0.0}
    )
    model = CounterModel()
    result = task.training_step(batch, model)

    expected_preds = torch.tensor([[[1.0], [2.0]]])
    expected_targets = batch["y"][:, :2, :]
    expected_mse = torch.mean((expected_preds - expected_targets) ** 2)

    assert model.calls == 2
    assert torch.isclose(result["loss"], expected_mse)
    assert result["rmse"] == pytest.approx(torch.sqrt(expected_mse).item())


def test_multi_step_task_teacher_forcing_uses_ground_truth():
    class EchoModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inputs = []

        def forward(self, x):
            self.inputs.append(x[:, -1:, :].detach())
            return {"preds": x}

    batch = {
        "x": torch.tensor([[[1.0], [2.0], [3.0]]]),
        "y": torch.tensor([[[10.0], [20.0], [30.0]]]),
    }

    task = MultiStepTask(
        {"loss": "mse", "metrics": [], "horizon": 3, "teacher_forcing_ratio": 1.0}
    )
    model = EchoModel()
    result = task.training_step(batch, model)

    recorded = torch.cat(model.inputs, dim=1)
    assert torch.allclose(recorded[0, :, 0], torch.tensor([3.0, 10.0, 20.0]))
    assert isinstance(result["loss"], torch.Tensor)


def test_multi_step_task_validation_autoregressive():
    class CountingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, x):
            self.calls += 1
            return {"preds": torch.full_like(x, float(self.calls))}

    batch = {
        "x": torch.zeros(1, 2, 1),
        "y": torch.tensor([[[0.0], [1.0], [2.0], [3.0]]]),
    }

    task = MultiStepTask({"loss": "mse", "metrics": ["rmse", "mae"], "horizon": 3})
    model = CountingModel()
    result = task.validation_step(batch, model)

    expected_preds = torch.tensor([[[1.0], [2.0], [3.0]]])
    expected_targets = batch["y"][:, :3, :]
    expected_mse = torch.mean((expected_preds - expected_targets) ** 2)

    assert model.calls == 3
    assert torch.isclose(result["loss"], expected_mse)
    assert result["rmse"] == pytest.approx(torch.sqrt(expected_mse).item())
    assert result["mae"] == pytest.approx(
        torch.mean(torch.abs(expected_preds - expected_targets)).item()
    )
    assert not result["loss"].requires_grad


def test_anomaly_task_probabilistic_outputs():
    class IdentityModel(torch.nn.Module):
        def forward(self, x):
            return {"preds": x}

    batch = {
        "x": torch.tensor(
            [
                [[0.0], [1.0]],
                [[2.0], [3.0]],
            ]
        ),
        "y": torch.tensor(
            [
                [[1.0], [0.0]],
                [[2.5], [3.5]],
            ]
        ),
    }

    task = AnomalyTask({"loss": "nll", "metrics": ["rmse"]})
    train_result = task.training_step(batch, IdentityModel())

    assert isinstance(train_result["loss"], torch.Tensor)
    assert train_result["loss_value"] == pytest.approx(train_result["loss"].item())
    assert train_result["nll"] == pytest.approx(train_result["loss"].item())
    assert "rmse" in train_result

    validation = task.validation_step(batch, IdentityModel())
    assert isinstance(validation["loss"], torch.Tensor)
    assert not validation["loss"].requires_grad
    expected_errors = torch.mean(
        (batch["x"][:, 0:1, :] - batch["y"][:, 0:1, :]) ** 2, dim=(1, 2)
    )
    assert validation["nll"] == pytest.approx(expected_errors.mean().item())
    assert validation["loss_value"] == pytest.approx(validation["loss"].item())
    assert torch.allclose(
        torch.from_numpy(validation["anomaly_scores"]),
        expected_errors,
    )
