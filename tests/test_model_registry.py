import inspect

import pytest
import torch
from omegaconf import OmegaConf
import numpy as np
import onnxruntime as ort

from airtrace.models.base import ARBaseModel
from airtrace.models import registry
from airtrace.transforms import registry as transform_registry
from airtrace.tasks import registry as task_registry, build_task
from airtrace.models import list_models


class _TinyModel(ARBaseModel):
    def forward(self, x: torch.Tensor, context=None, **kwargs):
        preds = torch.zeros(x.shape[0], x.shape[1], self.output_dim)
        return {"preds": preds}


class _MinimalDataset:
    def __init__(self, input_dim: int = 2, output_dim: int = 2, input_length: int = 16, target_length: int = 4):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_length = input_length
        self.target_length = target_length
        self._x = np.random.randn(input_length, input_dim).astype(np.float32)
        self._y = np.random.randn(target_length, output_dim).astype(np.float32)
        self._meta = {"aircraft_type": "Q400"}

    def __len__(self) -> int:
        return 3

    def __getitem__(self, index):
        return {"x": self._x, "y": self._y, "meta": self._meta}

    def as_batch(self, batch_size: int = 2):
        x = torch.from_numpy(np.stack([self._x] * batch_size))
        y = torch.from_numpy(np.stack([self._y] * batch_size))
        return {"x": x, "y": y, "meta": {}}


@pytest.fixture()
def model_registry_restore():
    original = registry.MODEL_REGISTRY.copy()
    registry.MODEL_REGISTRY.clear()
    yield
    registry.MODEL_REGISTRY.clear()
    registry.MODEL_REGISTRY.update(original)


def test_register_and_build_model(model_registry_restore):
    @registry.register("tiny")
    class Tiny(_TinyModel):
        pass

    built = registry.build_model({"name": "tiny", "params": {}}, input_dim=4, output_dim=2)

    assert isinstance(built, Tiny)
    output = built(torch.ones(2, 3, 4))
    assert output["preds"].shape == (2, 3, 2)
    assert torch.all(output["preds"] == 0)
    assert registry.list_models() == ["tiny"]


def test_model_registry_validation(model_registry_restore):
    class NotAModel:
        pass

    with pytest.raises(TypeError):
        registry.register("invalid")(NotAModel)

    @registry.register("tiny")
    class Tiny(_TinyModel):
        pass

    with pytest.raises(ValueError):
        registry.register("tiny")(Tiny)

    with pytest.raises(ValueError) as exc:
        registry.build_model({"name": "missing"}, input_dim=1, output_dim=1)

    assert "Available models" in str(exc.value)


@pytest.mark.parametrize("model_name", list_models())
def test_registered_models_support_core_interfaces(model_name, tmp_path):
    dataset = _MinimalDataset()

    # Build all transforms to ensure they fit and run on minimal data
    for transform_name in transform_registry.list_transforms():
        transform_cls = transform_registry.TRANSFORM_REGISTRY[transform_name]
        transform = transform_cls()
        transform.fit(dataset)
        x, y, meta = transform(dataset[0]["x"], dataset[0]["y"], dataset[0]["meta"])
        assert x.shape[1] == dataset.input_dim
        assert y.shape[1] == dataset.output_dim
        assert isinstance(meta, dict)

    model_cls = registry.MODEL_REGISTRY[model_name]
    init_sig = inspect.signature(model_cls.__init__)
    extra_kwargs = {}
    if "seq_len" in init_sig.parameters:
        extra_kwargs["seq_len"] = dataset.input_length
    if "pred_len" in init_sig.parameters:
        extra_kwargs["pred_len"] = dataset.target_length

    try:
        model = model_cls(
            input_dim=dataset.input_dim,
            output_dim=dataset.output_dim,
            **extra_kwargs,
        ).eval()
    except ImportError as exc:
        pytest.skip(f"Optional dependency missing for {model_name}: {exc}")

    batch = dataset.as_batch(batch_size=1)

    task_configs = {
        "one_step": {"name": "one_step", "loss": "mse", "metrics": ["rmse"], "horizon": 1},
        "multi_step": {
            "name": "multi_step",
            "loss": "mse",
            "metrics": ["rmse"],
            "horizon": dataset.target_length,
            "teacher_forcing_ratio": 0.0,
        },
        "anomaly": {"name": "anomaly", "loss": "nll", "metrics": ["rmse"]},
    }

    for task_name in task_registry.list_tasks():
        task = build_task(task_configs[task_name])
        output = task.training_step(batch, model)
        assert "loss" in output
        assert torch.isfinite(output["loss"]).all()

    export_config = OmegaConf.create(
        {
            "data": {"window_size_in": dataset.input_length, "sensors": ["s1", "s2"]},
            "model": {"name": model_name, "params": {}},
        }
    )

    # Export to ONNX and validate that the exported model runs
    from airtrace.export.onnx_exporter import ONNXExporter

    exporter = ONNXExporter(model, export_config)
    onnx_paths = exporter.export(
        tmp_path / f"{model_name}.onnx",
        end_to_end=False,
        sequence_length=dataset.input_length,
        batch_size=1,
        verbose=False,
        fixed_sequence_length=True,
    )

    session = ort.InferenceSession(str(onnx_paths["onnx_model"]))
    ort_inputs = {"input": np.random.randn(1, dataset.input_length, dataset.input_dim).astype(np.float32)}
    ort_outputs = session.run(None, ort_inputs)
    assert ort_outputs[0].shape[0] == 1
