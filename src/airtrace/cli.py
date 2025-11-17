"""Command-line interface for AirTrace."""

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from airtrace.data.datamodule import SensorDataModule
from airtrace.evaluation.eval_runner import EvaluationRunner
from airtrace.models.registry import build_model
from airtrace.tasks.registry import build_task
from airtrace.training.trainer import Trainer, set_seed
from airtrace.transforms.registry import build_transforms


def train(cfg: DictConfig):
    """Train a model.

    Args:
        cfg: Hydra configuration
    """
    print("=" * 80)
    print("Training Configuration")
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    # Set random seed
    set_seed(cfg.seed)

    # Build transforms
    transform_pipeline = None
    if "transforms" in cfg and "pipeline" in cfg.transforms:
        transform_pipeline = build_transforms(cfg.transforms.pipeline)
        print(f"\nTransform pipeline: {transform_pipeline}")

    # Build data module
    print("\nSetting up data...")
    datamodule = SensorDataModule(
        data_config=cfg.data,
        transforms=transform_pipeline,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers
    )

    try:
        datamodule.setup()
        print(f"Data module: {datamodule}")
    except Exception as e:
        print(f"Warning: Could not set up data module: {e}")
        print("This is expected if data files don't exist yet.")
        print("Please prepare your data first using the data preparation scripts.")
        return

    # Build model
    print("\nBuilding model...")
    model = build_model(
        config=cfg.model,
        input_dim=datamodule.in_dim,
        output_dim=datamodule.out_dim
    )
    print(f"Model: {model}")

    # Build task
    print("\nBuilding task...")
    task = build_task(cfg.task)
    print(f"Task: {task}")

    # Create dataloaders
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    # Build trainer
    print("\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        task=task,
        config=cfg,
        train_loader=train_loader,
        val_loader=val_loader
    )

    # Train
    print("\n" + "=" * 80)
    print("Starting Training")
    print("=" * 80)
    trainer.train()

    print("\n" + "=" * 80)
    print("Training Complete!")
    print(f"Best model saved to: {trainer.checkpoint_dir / 'best.ckpt'}")
    print(f"Logs saved to: {trainer.log_dir}")
    print("=" * 80)


def evaluate(cfg: DictConfig):
    """Evaluate a trained model.

    Args:
        cfg: Hydra configuration
    """
    print("=" * 80)
    print("Evaluation Configuration")
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    # Set random seed
    set_seed(cfg.seed)

    # Build transforms
    transform_pipeline = None
    if "transforms" in cfg and "pipeline" in cfg.transforms:
        transform_pipeline = build_transforms(cfg.transforms.pipeline)

    # Build data module
    print("\nSetting up data...")
    datamodule = SensorDataModule(
        data_config=cfg.data,
        transforms=transform_pipeline,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers
    )

    try:
        datamodule.setup()
    except Exception as e:
        print(f"Error: Could not set up data module: {e}")
        return

    # Build task
    task = build_task(cfg.task)

    # Load checkpoint
    checkpoint_path = Path(cfg.get("checkpoint", "checkpoints/best.ckpt"))
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    print(f"\nLoading checkpoint from {checkpoint_path}...")

    # Build model from checkpoint
    model = build_model(
        config=cfg.model,
        input_dim=datamodule.in_dim,
        output_dim=datamodule.out_dim
    )

    # Load state dict
    import torch
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Create evaluation runner
    test_loader = datamodule.test_dataloader() or datamodule.val_dataloader()

    evaluator = EvaluationRunner(
        model=model,
        task=task,
        test_loader=test_loader
    )

    # Run evaluation
    print("\nEvaluating...")
    results = evaluator.evaluate(return_predictions=False)

    print("\n" + "=" * 80)
    print("Evaluation Results")
    print("=" * 80)
    for metric, value in results["metrics"].items():
        print(f"{metric.upper():12s}: {value:.4f}")
    print(f"{'Samples':12s}: {results['num_samples']}")
    print("=" * 80)


@hydra.main(version_base=None, config_path="pkg://airtrace.configs", config_name="config")
def main(cfg: DictConfig):
    """Main entry point.

    Args:
        cfg: Hydra configuration
    """
    mode = cfg.get("mode", "train")

    if mode == "train":
        train(cfg)
    elif mode == "eval":
        evaluate(cfg)
    else:
        print(f"Error: Unknown mode '{mode}'. Use 'train' or 'eval'.")
        sys.exit(1)


if __name__ == "__main__":
    main()
