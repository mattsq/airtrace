from pathlib import Path

import numpy as np
import pandas as pd
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


def test_format_evaluation_results_includes_metrics_and_counts():
    results = {"metrics": {"mae": 0.1234, "rmse": 0.9876}, "num_samples": 42}

    formatted = cli._format_evaluation_results(results)

    assert "Evaluation Results" in formatted
    assert "MAE" in formatted
    assert "0.1234" in formatted
    assert "Samples" in formatted
    assert "42" in formatted


def _create_minimal_data_root(base_dir: Path) -> Path:
    data_root = base_dir / "data"
    processed = data_root / "processed"
    metadata = data_root / "metadata"
    processed.mkdir(parents=True)
    metadata.mkdir(parents=True)

    # Minimal flight with two sensors.
    flight_id = "f1"
    flight_df = pd.DataFrame(
        {
            "s1": np.linspace(0.0, 1.0, num=5),
            "s2": np.linspace(1.0, 2.0, num=5),
        }
    )
    flight_df.to_parquet(processed / f"{flight_id}.parquet")

    index_df = pd.DataFrame({"flight_id": [flight_id], "start_idx": [0], "end_idx": [4]})
    index_df.to_parquet(metadata / "train.parquet")
    index_df.to_parquet(metadata / "val.parquet")

    return data_root


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


def test_train_dry_run_allows_missing_data(tmp_path, capsys):
    cfg = OmegaConf.create(
        {
            "mode": "train",
            "exp_name": "demo",
            "seed": 0,
            "data": {
                "root": str(tmp_path / "missing"),
                "dataset_name": "synthetic_cruise",
                "train_index": "metadata/train.parquet",
                "val_index": "metadata/val.parquet",
            },
            "train": {"batch_size": 1, "num_workers": 0},
            "task": {"name": "tiny_task"},
            "model": {"name": "tiny_model"},
            "transforms": {"pipeline": []},
            "cli": {"data_check": False, "dry_run": True},
            "checkpoint": "",
            "log_dir": str(tmp_path / "logs"),
        }
    )

    cli.train(cfg)
    output = capsys.readouterr().out
    assert "missing data assets" in output.lower()
    assert "Dry run requested" in output


def test_train_data_check_runs_setup_and_skips_training(tmp_path, capsys):
    data_root = _create_minimal_data_root(tmp_path)

    cfg = OmegaConf.create(
        {
            "mode": "train",
            "exp_name": "demo",
            "seed": 0,
            "data": {
                "root": str(data_root),
                "dataset_name": "tiny",
                "sensors": {"use": ["s1", "s2"]},
                "window": {"input_len": 3, "pred_len": 1, "stride": 1, "target_sensors": ["s2"]},
                "train_index": "metadata/train.parquet",
                "val_index": "metadata/val.parquet",
            },
            "train": {"batch_size": 1, "num_workers": 0},
            "task": {"name": "tiny_task"},
            "model": {"name": "tiny_model"},
            "transforms": {"pipeline": []},
            "cli": {"data_check": True, "dry_run": False},
            "checkpoint": "",
            "log_dir": str(tmp_path / "logs"),
        }
    )

    cli.train(cfg)
    output = capsys.readouterr().out
    assert "Data check complete" in output
    assert "Starting Training" not in output
