import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Optional

from airtrace.evaluation import metrics
from airtrace.evaluation.eval_runner import EvaluationRunner
from airtrace.tasks.base import Task


def test_metric_functions_compute_expected_values():
    preds = np.array([1.0, 2.0, 3.0])
    targets = np.array([1.0, 1.0, 4.0])

    results = metrics.compute_all_metrics(preds, targets)

    assert np.isclose(results["rmse"], np.sqrt(2 / 3))
    assert np.isclose(results["mae"], 2 / 3)
    assert np.isclose(results["mse"], 2 / 3)
    assert results["r2"] < 1.0

    per_sensor = metrics.per_sensor_metrics(preds.reshape(1, -1, 1), targets.reshape(1, -1, 1), ["sensor"])
    assert set(per_sensor.keys()) == {"sensor"}
    assert np.isclose(per_sensor["sensor"]["mae"], 2 / 3)


class _TinyDataset(Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, idx):
        x = torch.full((3, 2), float(idx))
        y = torch.full((2, 2), float(idx))
        return {"x": x, "y": y, "meta": {}}


class _TinyModel(torch.nn.Module):
    def forward(self, x):
        # Predict constant increment over input
        preds = x[:, -2:, :] + 1.0
        return {"preds": preds}


class _TinyTask(Task):
    def training_step(self, batch, model):
        raise NotImplementedError

    def validation_step(self, batch, model):
        output = model(batch["x"])
        loss = torch.mean((output["preds"] - batch["y"]) ** 2)
        return {"loss": loss}


def test_evaluation_runner_aggregates_losses_and_metrics():
    dataset = _TinyDataset()
    loader = DataLoader(dataset, batch_size=1)
    runner = EvaluationRunner(_TinyModel(), _TinyTask({"loss": "mse"}), loader, device="cpu")

    results = runner.evaluate(return_predictions=True)

    assert "metrics" in results
    assert results["metrics"]["avg_loss"] > 0
    assert results["predictions"].shape[0] == len(dataset)

    per_sensor = runner.evaluate_per_sensor(["s1", "s2"])
    assert set(per_sensor.keys()) == {"s1", "s2"}


def test_evaluation_runner_loads_from_checkpoint(tmp_path):
    torch.manual_seed(0)

    class _LinearModel(torch.nn.Module):
        def __init__(self, input_dim: int = 2, output_dim: int = 2):
            super().__init__()
            self.linear = torch.nn.Linear(input_dim, output_dim)

        def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None):
            preds = self.linear(x)
            return {"preds": preds}

    original_model = _LinearModel()
    checkpoint_path = tmp_path / "checkpoint.pt"

    torch.save(
        {
            "model_state_dict": original_model.state_dict(),
            "config": {"model": {"params": {"input_dim": 2, "output_dim": 2}}},
        },
        checkpoint_path,
    )

    dataset = _TinyDataset()
    loader = DataLoader(dataset, batch_size=1)
    runner = EvaluationRunner.from_checkpoint(
        checkpoint_path, _LinearModel, _TinyTask({"loss": "mse"}), loader, device="cpu"
    )

    sample_input = torch.ones(1, 3, 2)
    with torch.no_grad():
        expected = original_model(sample_input)["preds"]
        loaded = runner.model(sample_input)["preds"]

    assert isinstance(runner, EvaluationRunner)
    assert torch.allclose(loaded, expected)
