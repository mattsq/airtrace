"""Additional core coverage tests for base utilities and tasks."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pytest
import torch

from airtrace.models.base import ARBaseModel
from airtrace.tasks.anomaly import AnomalyTask
from airtrace.transforms.base import Compose, Transform


class _NoInverseTransform(Transform):
    """Transform implementation that relies on the base inverse."""

    def fit(self, dataset) -> "_NoInverseTransform":
        self.is_fitted = True
        return self

    def __call__(
        self, x: np.ndarray, y: np.ndarray, meta: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        return x, y, meta


class _StatsTransform(Transform):
    """Transform that exposes cached statistics for Compose helpers."""

    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.offset = 0.0

    def fit(self, dataset) -> "_StatsTransform":  # pragma: no cover - unused
        self.is_fitted = True
        return self

    def __call__(
        self, x: np.ndarray, y: np.ndarray, meta: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        return x + self.offset, y + self.offset, {**meta, "used": True}

    def inverse(
        self, x: np.ndarray, y: np.ndarray | None = None
    ) -> Tuple[np.ndarray, np.ndarray | None]:
        return x - self.offset, None if y is None else y - self.offset

    def get_stats(self) -> Dict[str, float]:
        return {"offset": self.offset}

    def set_stats(self, stats: Dict[str, float]) -> None:
        self.offset = stats.get("offset", 0.0)


class _LazyLinearModel(ARBaseModel):
    """ARBaseModel with lazy parameters for counting coverage."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__(input_dim=input_dim, output_dim=output_dim)
        self.proj = torch.nn.LazyLinear(output_dim)

    def forward(
        self, x: torch.Tensor, context: torch.Tensor | None = None, **kwargs
    ) -> Dict[str, torch.Tensor]:
        preds = self.proj(x)
        return {"preds": preds}


def test_transform_inverse_not_implemented():
    transform = _NoInverseTransform()
    array = np.ones((2, 2))

    assert transform.is_fitted is False
    transform.fit(dataset=None)
    assert transform.is_fitted is True

    with pytest.raises(
        NotImplementedError, match="_NoInverseTransform does not support inverse"
    ):
        transform.inverse(array, array)


def test_compose_propagates_stats_and_repr():
    t1 = _StatsTransform("first")
    t2 = _StatsTransform("second")
    pipeline = Compose([t1, t2])

    pipeline.set_stats({"_StatsTransform_0": {"offset": 1.5}})
    x, y, meta = pipeline(np.array([[0.0]]), np.array([[1.0]]), {})

    assert float(x.item()) == 1.5
    assert float(y.item()) == 2.5
    assert meta["used"] is True
    assert pipeline.get_stats()["_StatsTransform_0"]["offset"] == 1.5
    assert "_StatsTransform" in repr(pipeline)


def test_ar_base_model_skips_uninitialized_parameters_and_counts_after_init():
    model = _LazyLinearModel(input_dim=3, output_dim=2)

    # LazyLinear parameters should be uninitialized before the first forward pass
    assert model.get_num_params() == 0
    assert "num_params=0" in repr(model)

    batch = torch.ones(1, 2, 3)
    output = model(batch)["preds"]

    assert output.shape == (1, 2, 2)
    assert model.get_num_params() > 0


def test_anomaly_task_probabilistic_scoring():
    class _IdentityModel(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
            return {"preds": x}

    task = AnomalyTask({"loss": "nll", "metrics": ["mse"]})
    batch = {"x": torch.ones(2, 3, 4), "y": torch.ones(2, 3, 4)}
    model = _IdentityModel()

    train_out = task.training_step(batch, model)
    assert task.use_probabilistic is True
    assert torch.isclose(train_out["loss"], torch.tensor(0.0))
    assert train_out["nll"] == train_out["loss_value"]

    val_out = task.validation_step(batch, model)
    assert val_out["loss"].item() == 0.0
    assert val_out["nll"] == 0.0
    assert val_out["anomaly_scores"].shape == (2,)
