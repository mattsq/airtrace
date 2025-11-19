from pathlib import Path

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
