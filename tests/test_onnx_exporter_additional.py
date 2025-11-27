import builtins
import json
import sys
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

    # New behavior: raises ValueError instead of silent fallback to (15, 15)
    with pytest.raises(ValueError, match="Could not infer dimensions"):
        ONNXExporter._infer_dimensions({}, DictConfig({}))


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


def test_verify_export_skips_without_onnxruntime(tmp_path: Path, monkeypatch):
    model = DummyModel()
    config = DictConfig({"data": {"window_size_in": 3}, "model": {"name": "dummy"}})
    exporter = ONNXExporter(model=model, config=config)

    onnx_path = tmp_path / "missing_ort.onnx"
    onnx_path.write_bytes(b"fake")

    original_import = builtins.__import__

    def raising_import(name, *args, **kwargs):
        if name == "onnxruntime":
            raise ImportError("onnxruntime not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", raising_import)

    assert exporter.verify_export(onnx_path, verbose=False) is False


def test_verify_export_with_stubbed_onnxruntime(tmp_path: Path, monkeypatch):
    model = DummyModel(output_len=2)
    config = DictConfig({"data": {"window_size_in": 5}, "model": {"name": "dummy"}})
    exporter = ONNXExporter(model=model, config=config)

    onnx_path = tmp_path / "stubbed.onnx"
    onnx_path.write_bytes(b"fake")

    class FakeSession:
        def __init__(self, _path: str):
            self.path = _path

        def run(self, _outputs, inputs):
            input_array = inputs["input"]
            batch, seq, _ = input_array.shape
            output_len = exporter.model.output_len or seq
            return [np.ones((batch, output_len, exporter.model.output_dim), dtype=np.float32)]

    class FakeOrt:
        InferenceSession = FakeSession

    monkeypatch.setitem(sys.modules, "onnxruntime", FakeOrt)

    assert exporter.verify_export(onnx_path, verbose=False, num_samples=2) is True


def test_save_transform_stats_serializes_numpy_scalars(tmp_path: Path):
    stats = {
        "CustomTransform": {
            "array": np.array([1.0, 2.0], dtype=np.float32),
            "int_value": np.int64(5),
            "float_value": np.float64(3.5),
            "label": "ok",
        }
    }
    exporter = ONNXExporter(
        model=DummyModel(),
        config=DictConfig({"model": {"name": "dummy"}}),
        transform_stats=stats,
    )

    stats_path = tmp_path / "stats.json"
    exporter._save_transform_stats(stats_path)

    with stats_path.open() as f:
        saved_stats = json.load(f)

    assert saved_stats["CustomTransform"]["array"] == [1.0, 2.0]
    assert saved_stats["CustomTransform"]["int_value"] == 5
    assert saved_stats["CustomTransform"]["float_value"] == pytest.approx(3.5)
    assert saved_stats["CustomTransform"]["label"] == "ok"


def test_export_with_fixed_sequence_length_metadata(tmp_path: Path, monkeypatch):
    """Test that fixed_sequence_length flag is recorded in metadata."""
    model = DummyModel(output_len=2)
    config = DictConfig({"data": {"window_size_in": 10}, "model": {"name": "dummy"}})
    exporter = ONNXExporter(model=model, config=config)

    def fake_export(model_arg, args, output_file, *_, **__):
        Path(output_file).write_bytes(b"onnx")

    monkeypatch.setattr(torch.onnx, "export", fake_export)

    output_path = tmp_path / "model.onnx"
    files = exporter.export(
        output_path=output_path,
        sequence_length=10,
        fixed_sequence_length=True,
        verbose=False,
    )

    # Verify metadata includes fixed_sequence_length
    metadata_path = output_path.with_suffix(".metadata.json")
    with metadata_path.open() as f:
        metadata = json.load(f)

    assert metadata["export"]["fixed_sequence_length"] is True


def test_export_uses_profile_dynamic_axes(tmp_path: Path, monkeypatch):
    """Test that export uses profile.dynamic_axes instead of hardcoding."""
    model = DummyModel(output_len=2)
    config = DictConfig({"data": {"window_size_in": 10}, "model": {"name": "dummy"}})
    exporter = ONNXExporter(model=model, config=config)

    # Get profile and verify it has dynamic_axes
    profile = exporter.get_export_profile()
    assert profile.dynamic_axes is not None
    assert 'input' in profile.dynamic_axes
    assert 'output' in profile.dynamic_axes

    # Track what dynamic_axes were passed to torch.onnx.export
    captured_dynamic_axes = None

    def fake_export(model_arg, args, output_file, *_, dynamic_axes=None, **__):
        nonlocal captured_dynamic_axes
        captured_dynamic_axes = dynamic_axes
        Path(output_file).write_bytes(b"onnx")

    monkeypatch.setattr(torch.onnx, "export", fake_export)

    output_path = tmp_path / "model.onnx"
    exporter.export(
        output_path=output_path,
        sequence_length=10,
        verbose=False,
    )

    # Verify dynamic_axes were passed and match profile structure
    assert captured_dynamic_axes is not None
    assert 'input' in captured_dynamic_axes
    assert 'output' in captured_dynamic_axes
    assert 0 in captured_dynamic_axes['input']  # batch_size
    assert 1 in captured_dynamic_axes['input']  # sequence_length


def test_default_export_unchanged(tmp_path: Path, monkeypatch):
    """Test that default export behavior is unchanged (both axes dynamic)."""
    model = DummyModel(output_len=2)
    config = DictConfig({"data": {"window_size_in": 10}, "model": {"name": "dummy"}})
    exporter = ONNXExporter(model=model, config=config)

    def fake_export(model_arg, args, output_file, *_, **__):
        Path(output_file).write_bytes(b"onnx")

    monkeypatch.setattr(torch.onnx, "export", fake_export)

    output_path = tmp_path / "model.onnx"
    # Export without fixed_sequence_length flag (default behavior)
    files = exporter.export(
        output_path=output_path,
        sequence_length=10,
        verbose=False,
    )

    # Verify metadata shows fixed_sequence_length=False by default
    metadata_path = output_path.with_suffix(".metadata.json")
    with metadata_path.open() as f:
        metadata = json.load(f)

    assert metadata["export"]["fixed_sequence_length"] is False


def test_fixed_sequence_removes_dynamic_axes(tmp_path: Path, monkeypatch):
    """Test that fixed_sequence_length removes sequence from dynamic axes."""
    model = DummyModel(output_len=2)
    config = DictConfig({"data": {"window_size_in": 10}, "model": {"name": "dummy"}})
    exporter = ONNXExporter(model=model, config=config)

    # Track what dynamic_axes were passed to torch.onnx.export
    captured_dynamic_axes = None

    def fake_export(model_arg, args, output_file, *_, dynamic_axes=None, **__):
        nonlocal captured_dynamic_axes
        captured_dynamic_axes = dynamic_axes
        Path(output_file).write_bytes(b"onnx")

    monkeypatch.setattr(torch.onnx, "export", fake_export)

    output_path = tmp_path / "model.onnx"
    exporter.export(
        output_path=output_path,
        sequence_length=10,
        fixed_sequence_length=True,
        verbose=False,
    )

    # Verify dynamic_axes only have axis 0 (batch_size), not axis 1 (sequence_length)
    assert captured_dynamic_axes is not None
    assert 'input' in captured_dynamic_axes
    assert 'output' in captured_dynamic_axes

    # Should only have batch_size (axis 0), not sequence_length (axis 1)
    assert 0 in captured_dynamic_axes['input']
    assert 1 not in captured_dynamic_axes['input']  # sequence_length removed
    assert 0 in captured_dynamic_axes['output']
    assert 1 not in captured_dynamic_axes['output']  # output_length removed
