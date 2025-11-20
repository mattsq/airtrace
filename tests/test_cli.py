from pathlib import Path

import pytest
import pandas as pd
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


@pytest.mark.parametrize("data_check", [False, True])
@pytest.mark.parametrize("dry_run", [False, True])
@pytest.mark.parametrize("use_checkpoint", [False, True])
def test_train_accepts_all_flag_combinations(monkeypatch, tmp_path, data_check, dry_run, use_checkpoint):
    data_root = tmp_path / "tiny-data"
    processed_dir = data_root / "processed"
    metadata_dir = data_root / "metadata"
    processed_dir.mkdir(parents=True)
    metadata_dir.mkdir(parents=True)

    pd.DataFrame({"flight_id": ["f1"], "start_idx": [0], "end_idx": [3]}).to_parquet(
        metadata_dir / "train.parquet"
    )
    pd.DataFrame({"flight_id": ["f1"], "start_idx": [1], "end_idx": [4]}).to_parquet(
        metadata_dir / "val.parquet"
    )
    pd.DataFrame({"s1": [0.0, 0.1, 0.2, 0.3], "s2": [0.5, 0.4, 0.3, 0.2]}).to_parquet(
        processed_dir / "f1.parquet"
    )

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

        def train_dataloader(self):
            return [{"inputs": torch.zeros(2, 2), "targets": torch.zeros(2, 1)}]

        def val_dataloader(self):
            return [{"inputs": torch.zeros(2, 2), "targets": torch.zeros(2, 1)}]

    class _TinyTask:
        def training_step(self, batch, model):
            _ = model(batch["inputs"])
            return {"loss": torch.tensor(0.0)}

        def validation_step(self, batch, model):
            _ = model(batch["inputs"])
            return {"loss": torch.tensor(0.0)}

    built_models = []

    def _build_model_stub(config, input_dim, output_dim):
        model = torch.nn.Linear(input_dim, output_dim)
        built_models.append(model)
        return model

    trainers = []

    class _TinyTrainer:
        def __init__(self, model, task, config, train_loader, val_loader):
            trainers.append(self)
            self.model = model
            self.checkpoint_dir = Path(config.get("log_dir", "runs/debug")) / "checkpoints"
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.log_dir = Path(config.get("log_dir", "runs/debug"))

        def train(self):
            return None

    checkpoint_model = torch.nn.Linear(2, 1)
    with torch.no_grad():
        checkpoint_model.weight.fill_(0.5)
        checkpoint_model.bias.fill_(0.25)

    checkpoint_path = tmp_path / "checkpoint.ckpt"
    if use_checkpoint:
        torch.save({"model_state_dict": checkpoint_model.state_dict()}, checkpoint_path)

    monkeypatch.setattr("airtrace.data.datamodule.SensorDataModule", _TinyDataModule)
    monkeypatch.setattr("airtrace.models.registry.build_model", _build_model_stub)
    monkeypatch.setattr("airtrace.tasks.registry.build_task", lambda _: _TinyTask())
    monkeypatch.setattr("airtrace.training.trainer.Trainer", _TinyTrainer)

    cfg = OmegaConf.create(
        {
            "mode": "train",
            "exp_name": "tiny",
            "seed": 123,
            "data": {
                "root": str(data_root),
                "dataset_name": "tiny",
                "sensors": {"use": ["s1", "s2"]},
                "window": {"input_len": 2, "pred_len": 1, "stride": 1, "target_sensors": ["s2"]},
                "train_index": "metadata/train.parquet",
                "val_index": "metadata/val.parquet",
            },
            "train": {"batch_size": 2, "num_workers": 0, "epochs": 1},
            "task": {"name": "tiny_task"},
            "model": {"name": "tiny_model"},
            "transforms": {},
            "cli": {"data_check": data_check, "dry_run": dry_run},
            "checkpoint": str(checkpoint_path) if use_checkpoint else "",
            "log_dir": str(tmp_path / "logs"),
        }
    )

    cli.train(cfg)

    if data_check or dry_run:
        assert not trainers
        assert not built_models
    else:
        assert len(trainers) == 1
        assert len(built_models) == 1
        if use_checkpoint:
            assert torch.allclose(built_models[-1].weight, checkpoint_model.weight)
