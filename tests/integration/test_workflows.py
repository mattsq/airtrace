from pathlib import Path

import numpy as np
import torch

from airtrace.data.datamodule import SensorDataModule
from airtrace.evaluation.eval_runner import EvaluationRunner
from airtrace.models.dlinear import DLinearModel
from airtrace.tasks.one_step import OneStepTask
from airtrace.training.trainer import Trainer


def test_training_and_evaluation_pipeline(minimal_data_root: Path, tmp_path: Path) -> None:
    data_config = {
        "root": minimal_data_root,
        "sensors": {"use": ["s1", "s2"]},
        "window": {"input_len": 4, "pred_len": 1, "stride": 1, "target_sensors": ["s2"]},
        "train_index": "metadata/train.parquet",
        "val_index": "metadata/val.parquet",
        "test_index": "metadata/test.parquet",
    }

    module = SensorDataModule(data_config, transforms=None, batch_size=2, num_workers=0)
    module.setup()

    model = DLinearModel(input_dim=2, output_dim=1, seq_len=4, pred_len=1)
    task = OneStepTask({"loss": "mse", "metrics": ["rmse", "mae"]})

    train_config = {
        "train": {
            "epochs": 2,
            "log_every_n_steps": 1,
            "checkpoint": {"save_top_k": 1},
            "grad_clip": {"enabled": False},
        },
        "log_dir": tmp_path / "logs",
    }

    trainer = Trainer(
        model=model,
        task=task,
        config=train_config,
        train_loader=module.train_dataloader(),
        val_loader=module.val_dataloader(),
        device="cpu",
    )
    trainer.train()

    checkpoint_dir = Path(train_config["log_dir"]) / "checkpoints"
    best_checkpoint = checkpoint_dir / "best.ckpt"
    assert best_checkpoint.exists()

    reloaded = DLinearModel(input_dim=2, output_dim=1, seq_len=4, pred_len=1)
    state = torch.load(best_checkpoint, map_location="cpu", weights_only=False)
    reloaded.load_state_dict(state["model_state_dict"])

    runner = EvaluationRunner(reloaded, task, module.test_dataloader(), device="cpu")
    results = runner.evaluate()

    assert results["num_samples"] == len(module.test_dataloader())
    assert results["metrics"]["avg_loss"] >= 0.0
    assert all(np.isfinite(val) for val in results["metrics"].values())
