import json
from pathlib import Path
from typing import List

import numpy as np
import pytest
import torch
from omegaconf import DictConfig

from airtrace.export.onnx_exporter import ONNXExporter
from airtrace.export.end_to_end_model import EndToEndModel, ModelOnlyWrapper


class DummyModel(torch.nn.Module):
    def __init__(self, output_len: int = 1):
        super().__init__()
        self.input_dim = 2
        self.output_dim = 2
        self.output_len = output_len

    def forward(self, x: torch.Tensor, context=None):
        batch, seq, _ = x.shape
        preds = torch.ones(batch, self.output_len or seq, self.output_dim)
        return {"preds": preds}


TRANSFORM_STATS = {
    "ZScoreTransform_0": {
        "scaler_x_mean": np.array([0.0, 1.0], dtype=np.float32),
        "scaler_x_scale": np.array([1.0, 2.0], dtype=np.float32),
        "scaler_y_mean": np.array([0.0, 0.5], dtype=np.float32),
        "scaler_y_scale": np.array([1.0, 0.5], dtype=np.float32),
    }
}


def test_infer_dimensions_variants():
    config_dims = DictConfig({"model": {"input_dim": 4, "output_dim": 3}})
    assert ONNXExporter._infer_dimensions({}, config_dims) == (4, 3)

    config_sensors = DictConfig({"data": {"sensors": ["a", "b", "c"]}})
    assert ONNXExporter._infer_dimensions({}, config_sensors) == (3, 3)

    weights = {
        "encoder.weight_ih_l0": torch.randn(6, 7),
        "decoder.weight": torch.randn(2, 6),
    }
    inferred = ONNXExporter._infer_dimensions(weights, DictConfig({}))
    assert inferred == (7, 2)

    fallback = ONNXExporter._infer_dimensions({}, DictConfig({}))
    assert fallback == (15, 15)


def test_from_checkpoint_without_config(tmp_path: Path):
    checkpoint_path = tmp_path / "bad.pt"
    torch.save({"model_state_dict": {}}, checkpoint_path)

    with pytest.raises(ValueError):
        ONNXExporter.from_checkpoint(checkpoint_path)


def test_export_writes_metadata_and_stats(tmp_path: Path, monkeypatch):
    model = DummyModel()
    config = DictConfig({"data": {"window_size_in": 8}, "model": {"name": "dummy"}})
    exporter = ONNXExporter(model=model, config=config, transform_stats=TRANSFORM_STATS)

    exported_models: List[torch.nn.Module] = []

    def fake_export(model_arg, args, output_file, *_, **__):
        exported_models.append(model_arg)
        Path(output_file).write_bytes(b"onnx")

    monkeypatch.setattr(torch.onnx, "export", fake_export)

    output_path = tmp_path / "model.onnx"
    files = exporter.export(output_path, end_to_end=False, verbose=False)

    assert isinstance(exported_models[0], ModelOnlyWrapper)
    assert output_path.exists()
    assert files["onnx_model"].exists()

    stats_path = output_path.with_suffix(".transform_stats.json")
    assert stats_path.exists()
    with stats_path.open() as f:
        saved_stats = json.load(f)
    assert "ZScoreTransform_0" in saved_stats

    metadata_path = output_path.with_suffix(".metadata.json")
    with metadata_path.open() as f:
        metadata = json.load(f)
    assert metadata["input_shape"] == [1, 8, 2]
    assert metadata["end_to_end"] is False


def test_export_end_to_end_uses_wrapped_model(tmp_path: Path, monkeypatch):
    model = DummyModel(output_len=2)
    config = DictConfig({"data": {"window_size_in": 4}, "model": {"name": "dummy"}})
    exporter = ONNXExporter(model=model, config=config, transform_stats=TRANSFORM_STATS)

    exported_models: List[torch.nn.Module] = []

    def fake_export(model_arg, args, output_file, *_, **__):
        exported_models.append(model_arg)
        Path(output_file).write_bytes(b"onnx")

    monkeypatch.setattr(torch.onnx, "export", fake_export)

    output_path = tmp_path / "end_to_end.onnx"
    files = exporter.export(output_path, end_to_end=True, verbose=False)

    assert isinstance(exported_models[0], EndToEndModel)
    assert "transform_stats" not in files

    metadata_path = output_path.with_suffix(".metadata.json")
    with metadata_path.open() as f:
        metadata = json.load(f)
    assert metadata["end_to_end"] is True
    assert metadata["has_transforms"] is True
