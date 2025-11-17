import sys
from pathlib import Path

from airtrace.cli import _missing_data_assets, prepare_hydra_overrides


def test_prepare_hydra_overrides_defaults():
    overrides = prepare_hydra_overrides(["train", "model=tcn"])

    assert overrides[0] == "mode=train"
    assert "model=tcn" in overrides
    assert "cli.data_check=true" not in overrides
    assert "cli.dry_run=true" not in overrides


def test_prepare_hydra_overrides_with_flags(tmp_path):
    checkpoint_path = tmp_path / "best.ckpt"
    overrides = prepare_hydra_overrides(
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

    missing_before = _missing_data_assets(data_cfg, require_test=True)
    expected_missing = {
        tmp_path / "metadata/train.parquet",
        tmp_path / "metadata/val.parquet",
    }
    assert expected_missing.issubset(set(missing_before))

    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    (metadata_dir / "train.parquet").touch()
    (metadata_dir / "val.parquet").touch()

    missing_after = _missing_data_assets(data_cfg, require_test=False)
    assert missing_after == []
