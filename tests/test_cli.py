from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

from airtrace import __version__
import airtrace.cli as cli


def test_prepare_hydra_overrides_defaults():
    overrides = cli.prepare_hydra_overrides(["train", "model=tcn"])

    assert overrides[0] == "mode=train"
    assert "model=tcn" in overrides
    assert "cli.data_check=true" not in overrides
    assert "cli.dry_run=true" not in overrides


def test_prepare_hydra_overrides_with_flags(tmp_path):
    checkpoint_path = tmp_path / "best.ckpt"
    overrides = cli.prepare_hydra_overrides(
        [
            "eval",
            "--checkpoint",
            str(checkpoint_path),
            "--data-check",
            "train.epochs=1",
        ]
    )

    assert overrides[0] == "mode=eval"
    assert f"checkpoint={checkpoint_path}" in overrides
    assert "cli.data_check=true" in overrides
    assert overrides[-1] == "train.epochs=1"


def test_missing_data_assets(tmp_path):
    data_cfg = {
        "root": str(tmp_path),
        "train_index": "metadata/train.parquet",
        "val_index": "metadata/val.parquet",
    }

    missing_before = cli._missing_data_assets(data_cfg, require_test=True)
    expected_missing = {
        tmp_path / "metadata/train.parquet",
        tmp_path / "metadata/val.parquet",
    }
    assert expected_missing.issubset(set(missing_before))

    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    (metadata_dir / "train.parquet").touch()
    (metadata_dir / "val.parquet").touch()

    missing_after = cli._missing_data_assets(data_cfg, require_test=False)
    assert missing_after == []


def test_require_checkpoint_validation(tmp_path):
    with pytest.raises(SystemExit):
        cli._require_checkpoint({})

    cfg = {"checkpoint": str(tmp_path / "missing.ckpt")}
    with pytest.raises(SystemExit):
        cli._require_checkpoint(cfg)


def test_load_checkpoint_if_present(tmp_path):
    model = torch.nn.Linear(2, 1)
    checkpoint_path = tmp_path / "model.ckpt"

    # Save initial weights and mutate model to ensure loading restores them
    saved_state = {k: v.clone() for k, v in model.state_dict().items()}
    torch.save({"model_state_dict": saved_state}, checkpoint_path)

    for param in model.parameters():
        param.data.add_(1.0)

    cli._load_checkpoint_if_present(model, checkpoint_path, preload_message="loading")
    for name, param in model.state_dict().items():
        assert torch.equal(param, saved_state[name])

    # Ensure function does not fail when no model is provided
    cli._load_checkpoint_if_present(None, checkpoint_path)


def test_missing_data_assets_includes_root_when_missing(tmp_path):
    data_root = tmp_path / "does-not-exist"
    data_cfg = {
        "root": str(data_root),
        "train_index": "train.parquet",
        "val_index": "val.parquet",
    }

    missing = cli._missing_data_assets(data_cfg, require_test=False)

    assert Path(data_cfg["root"]) in missing


def test_require_checkpoint_success(tmp_path):
    checkpoint_path = tmp_path / "model.ckpt"
    checkpoint_path.touch()
    cfg = {"checkpoint": str(checkpoint_path)}

    resolved = cli._require_checkpoint(cfg)

    assert resolved == checkpoint_path


def _create_minimal_data_root(base_dir: Path) -> Path:
    data_root = base_dir / "data"
    processed = data_root / "processed"
    metadata = data_root / "metadata"
    processed.mkdir(parents=True)
    metadata.mkdir(parents=True)

    flight_id = "f1"
    # Provide enough timesteps to create a single input/target split
    flight_df = pd.DataFrame(
        {
            "s1": np.linspace(0.0, 1.0, num=4, dtype=np.float32),
            "s2": np.linspace(1.0, 2.0, num=4, dtype=np.float32),
        }
    )
    flight_df.to_parquet(processed / f"{flight_id}.parquet")

    index_df = pd.DataFrame({"flight_id": [flight_id], "start_idx": [0], "end_idx": [4]})
    index_df.to_parquet(metadata / "train.parquet")
    index_df.to_parquet(metadata / "val.parquet")

    return data_root


def _build_train_cfg(data_root: Path, *, data_check: bool, dry_run: bool) -> OmegaConf:
    return OmegaConf.create(
        {
            "mode": "train",
            "exp_name": "tiny",
            "seed": 0,
            "data": {
                "root": str(data_root),
                "dataset_name": "tiny",
                "sensors": {"use": ["s1", "s2"]},
                "window": {
                    "input_len": 3,
                    "pred_len": 1,
                    "stride": 1,
                    "target_sensors": ["s2"],
                },
                "train_index": "metadata/train.parquet",
                "val_index": "metadata/val.parquet",
            },
            "train": {"batch_size": 1, "num_workers": 0, "epochs": 1, "verbose_progress": False},
            "task": {"name": "one_step"},
            "model": {"name": "persistence"},
            "transforms": {"pipeline": []},
            "cli": {"data_check": data_check, "dry_run": dry_run},
            "checkpoint": "",
            "log_dir": str(data_root / "logs"),
        }
    )


def test_resolve_version_prefers_metadata(monkeypatch):
    expected = "9.9.9"

    def fake_version(package_name: str) -> str:
        assert package_name == "airtrace"
        return expected

    monkeypatch.setattr(cli.metadata, "version", fake_version)

    assert cli._resolve_version() == expected


def test_resolve_version_falls_back_to_package_file(monkeypatch):
    def raise_not_found(_: str) -> str:
        raise cli.metadata.PackageNotFoundError

    monkeypatch.setattr(cli.metadata, "version", raise_not_found)

    assert cli._resolve_version() == __version__


def test_print_run_summary_lists_transforms_and_flags(tmp_path, capsys):
    cfg = OmegaConf.create(
        {
            "mode": "train",
            "exp_name": "demo",
            "data": {
                "root": str(tmp_path),
                "train_index": "train.parquet",
                "val_index": "val.parquet",
                "test_index": "test.parquet",
            },
            "transforms": {"pipeline": [{"name": "zscore"}, {"name": "difference"}]},
            "checkpoint": "runs/best.ckpt",
            "log_dir": "runs/demo",
            "cli": {"data_check": True, "dry_run": True},
        }
    )

    cli._print_run_summary(cfg, heading="Training")
    captured = capsys.readouterr().out

    assert "Training Configuration Summary" in captured
    assert "Transforms:  zscore, difference" in captured
    assert str(Path(tmp_path).resolve()) in captured
    assert "Data check:  enabled" in captured
    assert "Dry run:     enabled" in captured


def test_print_data_guidance_for_synthetic_dataset(capsys):
    cli._print_data_guidance({"dataset_name": "synthetic_cruise"}, [Path("/missing.parquet")])
    output = capsys.readouterr().out

    assert "Synthetic configs expect generated parquet index files." in output
    assert "airtrace-generate-synthetic data=synthetic_cruise" in output


def test_print_data_guidance_for_real_dataset(capsys):
    cli._print_data_guidance({"dataset_name": "real_flights"}, [Path("/missing.parquet")])
    output = capsys.readouterr().out

    assert "Populate the expected parquet index files" in output
    assert "airtrace-generate-synthetic" in output


def test_train_data_check_short_circuits_before_training(tmp_path, capsys):
    data_root = _create_minimal_data_root(tmp_path)
    cfg = _build_train_cfg(data_root, data_check=True, dry_run=False)

    cli.train(cfg)

    output = capsys.readouterr().out
    assert "Data check complete" in output
    assert "Building model" not in output
    assert not Path(cfg.log_dir).exists()


def test_train_runs_minimal_pipeline(tmp_path, capsys):
    data_root = _create_minimal_data_root(tmp_path)
    cfg = _build_train_cfg(data_root, data_check=False, dry_run=False)

    cli.train(cfg)

    output = capsys.readouterr().out
    assert "Starting Training" in output
    # Persistence model is non-trainable; training loop exits after validation
    assert "Model has no trainable parameters" in output
    assert Path(cfg.log_dir).exists()
