import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from airtrace import cli


def test_prepare_hydra_overrides_export_requires_format():
    with pytest.raises(SystemExit):
        cli.prepare_hydra_overrides(["export"])


def test_prepare_hydra_overrides_export_options(tmp_path):
    checkpoint = tmp_path / "model.ckpt"
    args = [
        "export",
        "onnx",
        "--checkpoint",
        str(checkpoint),
        "--output",
        "exported.onnx",
        "--end-to-end",
        "--batch-size",
        "4",
        "--sequence-length",
        "32",
        "--no-verify",
        "--opset-version",
        "17",
        "extra.override=true",
    ]

    overrides = cli.prepare_hydra_overrides(args)

    assert overrides[:2] == ["mode=export", "cli.export_format=onnx"]
    assert f"checkpoint={checkpoint}" in overrides
    assert "cli.output=exported.onnx" in overrides
    assert "cli.end_to_end=true" in overrides
    assert "cli.batch_size=4" in overrides
    assert "cli.sequence_length=32" in overrides
    assert "cli.verify=false" in overrides
    assert "cli.opset_version=17" in overrides
    assert overrides[-1] == "extra.override=true"


def test_evaluate_runs_with_stub_components(monkeypatch, tmp_path, capsys):
    data_root = tmp_path / "data"
    metadata = data_root / "metadata"
    metadata.mkdir(parents=True)
    for name in ["train.parquet", "val.parquet", "test.parquet"]:
        (metadata / name).touch()

    checkpoint_path = tmp_path / "checkpoint.ckpt"
    model = torch.nn.Linear(2, 1)
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)

    class _TinyDataModule:
        def __init__(self, data_config, transforms, batch_size, num_workers, shuffle=None):
            self.data_config = data_config
            self.transforms = transforms
            self.batch_size = batch_size
            self.num_workers = num_workers
            self.shuffle = shuffle
            self.in_dim = 2
            self.out_dim = 1

        def setup(self):
            return None

        def test_dataloader(self):
            return [{"inputs": torch.zeros(2, 2), "targets": torch.zeros(2, 1)}]

        def val_dataloader(self):
            return []

    class _TinyTask:
        pass

    class _TinyEvaluationRunner:
        def __init__(self, model, task, test_loader):
            self.model = model
            self.task = task
            self.test_loader = test_loader

        def evaluate(self, return_predictions=False):
            return {"metrics": {"mae": 0.0}, "num_samples": 1}

    monkeypatch.setattr("airtrace.data.datamodule.SensorDataModule", _TinyDataModule)
    monkeypatch.setattr("airtrace.models.registry.build_model", lambda config, input_dim, output_dim: torch.nn.Linear(input_dim, output_dim))
    monkeypatch.setattr("airtrace.tasks.registry.build_task", lambda cfg: _TinyTask())
    monkeypatch.setattr("airtrace.evaluation.eval_runner.EvaluationRunner", _TinyEvaluationRunner)
    monkeypatch.setattr("airtrace.training.trainer.set_seed", lambda seed: None)
    monkeypatch.setattr("airtrace.transforms.registry.build_transforms", lambda pipeline: pipeline)

    cfg = OmegaConf.create(
        {
            "mode": "eval",
            "exp_name": "demo_eval",
            "seed": 0,
            "data": {
                "root": str(data_root),
                "dataset_name": "synthetic_cruise",
                "train_index": "metadata/train.parquet",
                "val_index": "metadata/val.parquet",
                "test_index": "metadata/test.parquet",
            },
            "train": {"batch_size": 1, "num_workers": 0},
            "task": {"name": "tiny_task"},
            "model": {"name": "tiny_model"},
            "transforms": {"pipeline": []},
            "cli": {"data_check": False},
            "checkpoint": str(checkpoint_path),
        }
    )

    cli.evaluate(cfg)
    output = capsys.readouterr().out
    assert "Evaluation Results" in output
    assert "MAE" in output


def test_export_onnx_uses_exporter(monkeypatch, tmp_path, capsys):
    checkpoint_path = tmp_path / "ckpt.ckpt"
    torch.save({"model_state_dict": {}}, checkpoint_path)

    called = {}

    class _FakeExporter:
        @classmethod
        def from_checkpoint(cls, checkpoint):
            called["from_checkpoint"] = checkpoint
            return cls()

        def export(self, output_path, end_to_end, batch_size, sequence_length, opset_version, verbose):
            called["export"] = {
                "output_path": output_path,
                "end_to_end": end_to_end,
                "batch_size": batch_size,
                "sequence_length": sequence_length,
                "opset_version": opset_version,
                "verbose": verbose,
            }
            return {"onnx_model": output_path}

        def verify_export(self, onnx_path, end_to_end, verbose):
            called["verify_export"] = {"onnx_path": onnx_path, "end_to_end": end_to_end, "verbose": verbose}

    monkeypatch.setattr("airtrace.export.ONNXExporter", _FakeExporter)

    cfg = OmegaConf.create(
        {
            "mode": "export",
            "cli": {
                "export_format": "onnx",
                "output": tmp_path / "model.onnx",
                "end_to_end": True,
                "batch_size": 2,
                "sequence_length": 16,
                "verify": True,
                "opset_version": 17,
            },
            "checkpoint": str(checkpoint_path),
        }
    )

    cli.export_onnx(cfg)

    assert called["from_checkpoint"] == checkpoint_path
    assert called["export"]["output_path"] == cfg.cli.output
    assert called["export"]["end_to_end"] is True
    assert called["export"]["batch_size"] == 2
    assert called["export"]["sequence_length"] == 16
    assert called["export"]["opset_version"] == 17
    assert called["verify_export"]["onnx_path"] == cfg.cli.output


def test_main_unknown_modes_exit(capsys):
    cfg = OmegaConf.create({"mode": "invalid", "cli": {}})
    with pytest.raises(SystemExit):
        cli.main.__wrapped__(cfg)
    assert "Unknown mode" in capsys.readouterr().out

    cfg_export = OmegaConf.create({"mode": "export", "cli": {"export_format": "xml"}})
    with pytest.raises(SystemExit):
        cli.main.__wrapped__(cfg_export)
    assert "Unknown export format" in capsys.readouterr().out
