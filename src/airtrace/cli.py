"""Command-line interface for AirTrace."""

from __future__ import annotations

import argparse
import re
import sys
import textwrap
from pathlib import Path
from typing import Iterable, List, TYPE_CHECKING

import hydra
from importlib import metadata
from omegaconf import DictConfig, OmegaConf

if TYPE_CHECKING:
    import torch


def _resolve_version() -> str:
    try:
        return metadata.version("airtrace")
    except metadata.PackageNotFoundError:
        init_path = Path(__file__).resolve().parent / "__init__.py"
        if init_path.exists():
            for line in init_path.read_text().splitlines():
                if line.startswith("__version__"):
                    return re.split(r"=\s*", line, maxsplit=1)[1].strip("\"' ")
    return "0.0.0"


CLI_VERSION = _resolve_version()


def train(cfg: DictConfig):
    """Train a model.

    Args:
        cfg: Hydra configuration
    """
    from airtrace.data.datamodule import SensorDataModule
    from airtrace.models.registry import build_model
    from airtrace.tasks.registry import build_task
    from airtrace.training.trainer import Trainer, set_seed
    from airtrace.transforms.registry import build_transforms
    _print_run_summary(cfg, heading="Training")

    # Set random seed
    set_seed(cfg.seed)

    cli_flags = cfg.get("cli", {})
    is_data_check = cli_flags.get("data_check")
    is_dry_run = cli_flags.get("dry_run")

    # Validate data presence before constructing the pipeline
    missing_assets = _missing_data_assets(cfg.data, require_test=False)
    if missing_assets:
        _print_data_guidance(cfg.data, missing_assets)
        if is_dry_run:
            print("\nDry run requested: missing data assets will be skipped; data setup and training will not run.")
            return
        sys.exit(1)

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
        print(f"Error: Could not set up data module: {e}")
        print("Check that the index parquet files match the configured sensors and window specs.")
        sys.exit(1)

    if is_data_check:
        print("\nData check complete. Training was not started because --data-check was supplied.")
        return

    if is_dry_run:
        print("\nDry run complete. Configuration and data checks passed; training skipped by request.")
        return

    # Build model
    print("\nBuilding model...")
    model = build_model(
        config=cfg.model,
        input_dim=datamodule.in_dim,
        output_dim=datamodule.out_dim
    )
    print(f"Model: {model}")

    if cfg.checkpoint:
        checkpoint_path = Path(cfg.checkpoint)
        if not checkpoint_path.exists():
            print(f"Error: Checkpoint not found at {checkpoint_path}")
            sys.exit(1)
        _load_checkpoint_if_present(
            model,
            checkpoint_path=checkpoint_path,
            preload_message="Loading checkpoint before training",
        )

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
    from airtrace.data.datamodule import SensorDataModule
    from airtrace.evaluation.eval_runner import EvaluationRunner
    from airtrace.models.registry import build_model
    from airtrace.tasks.registry import build_task
    from airtrace.training.trainer import set_seed
    from airtrace.transforms.registry import build_transforms
    _print_run_summary(cfg, heading="Evaluation")

    # Set random seed
    set_seed(cfg.seed)

    missing_assets = _missing_data_assets(cfg.data, require_test=True)
    if missing_assets:
        _print_data_guidance(cfg.data, missing_assets)
        sys.exit(1)

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
        sys.exit(1)

    if cfg.cli.get("data_check"):
        print("\nData check complete. Evaluation was not started because --data-check was supplied.")
        return

    # Build task
    task = build_task(cfg.task)

    checkpoint_path = _require_checkpoint(cfg)

    # Build model from checkpoint
    model = build_model(
        config=cfg.model,
        input_dim=datamodule.in_dim,
        output_dim=datamodule.out_dim
    )

    _load_checkpoint_if_present(model, checkpoint_path)

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


def _build_parser() -> argparse.ArgumentParser:
    description = textwrap.dedent(
        """AirTrace command-line interface. Subcommands map to Hydra overrides and
        accept additional overrides after the CLI flags.

        Examples:
          airtrace train --data-check data=synthetic
          airtrace train --dry-run exp=exp_001_gru_zscore
          airtrace eval --checkpoint runs/20240516/demo/checkpoints/best.ckpt data=synthetic
        """
    )

    epilog = textwrap.dedent(
        """Hydra overrides (e.g., model=tcn train.epochs=5) can be appended after
        any CLI flags. Use --config-name to switch experiments if needed.
        """
    )

    parser = argparse.ArgumentParser(
        prog="airtrace",
        description=description,
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"airtrace {CLI_VERSION}")
    subparsers = parser.add_subparsers(dest="command", title="commands", metavar="command")

    train_parser = subparsers.add_parser(
        "train",
        help="Train a model with the selected config overrides",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_train_flags(train_parser)

    eval_parser = subparsers.add_parser(
        "eval",
        help="Evaluate a trained checkpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_eval_flags(eval_parser)

    parser.set_defaults(command="train")
    return parser


def _add_train_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-check", action="store_true", help="Validate data paths and exit before training")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compose the config and run data checks without launching training",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Optional checkpoint to load before training (same semantics as checkpoint override)",
    )


def _add_eval_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Checkpoint path to evaluate. You can also set checkpoint=<path> as a Hydra override.",
    )
    parser.add_argument(
        "--data-check",
        action="store_true",
        help="Validate data paths for the evaluation dataset and exit",
    )


def prepare_hydra_overrides(argv: Iterable[str]) -> List[str]:
    """Convert CLI arguments into Hydra override strings."""

    parser = _build_parser()
    args, overrides = parser.parse_known_args(list(argv))

    command = args.command or "train"
    hydra_overrides = [f"mode={command}"]

    if getattr(args, "checkpoint", None):
        hydra_overrides.append(f"checkpoint={args.checkpoint}")

    if getattr(args, "data_check", False):
        hydra_overrides.append("cli.data_check=true")

    if getattr(args, "dry_run", False):
        hydra_overrides.append("cli.dry_run=true")

    hydra_overrides.extend(overrides)
    return hydra_overrides


def _print_run_summary(cfg: DictConfig, heading: str) -> None:
    data_cfg = cfg.data
    transforms = cfg.get("transforms", {})
    transform_names = [step.get("name", "?") for step in transforms.get("pipeline", [])]

    print("=" * 80)
    print(f"{heading} Configuration Summary")
    print("=" * 80)
    print(f"Mode:        {cfg.get('mode', heading.lower())}")
    print(f"Experiment:  {cfg.get('exp_name', 'debug')}")
    print(f"Data root:   {Path(data_cfg.root).resolve()}")
    print(f"Train index: {data_cfg.get('train_index')}")
    print(f"Val index:   {data_cfg.get('val_index')}")
    if data_cfg.get("test_index"):
        print(f"Test index:  {data_cfg.get('test_index')}")
    print(f"Transforms:  {', '.join(transform_names) if transform_names else 'none'}")
    if cfg.get("checkpoint"):
        print(f"Checkpoint:  {cfg.checkpoint}")
    print(f"Log dir:     {cfg.get('log_dir', 'runs/debug')}")
    if cfg.cli.get("data_check"):
        print("Data check:  enabled (will exit after validation)")
    if cfg.cli.get("dry_run"):
        print("Dry run:     enabled (data files optional; will skip training)")
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print("=" * 80)


def _missing_data_assets(data_config: DictConfig, require_test: bool) -> List[Path]:
    data_root = Path(data_config.get("root", "data"))
    missing: List[Path] = []

    if not data_root.exists():
        missing.append(data_root)

    required_keys = ["train_index", "val_index"]
    if require_test and data_config.get("test_index"):
        required_keys.append("test_index")

    for key in required_keys:
        candidate = data_root / data_config.get(key, "")
        if not candidate.exists():
            missing.append(candidate)

    return missing


def _print_data_guidance(data_config: DictConfig, missing_assets: List[Path]) -> None:
    print("\nData assets are missing:")
    for path in missing_assets:
        print(f"  - {path}")

    dataset_name = data_config.get("dataset_name", "unknown")

    if str(dataset_name).startswith("synthetic"):
        print(
            "\nSynthetic configs expect generated parquet index files. "
            "Create them with the helper script:\n"
            f"  airtrace-generate-synthetic data={dataset_name}\n"
            "  # or when running from a repo checkout:\n"
            f"  python -m airtrace.scripts.generate_synthetic_data data={dataset_name}"
        )
    else:
        print(
            "\nPopulate the expected parquet index files under your data root. "
            "If you don't have real data on disk, generate the bundled synthetic "
            "cruise dataset with:\n"
            "  airtrace-generate-synthetic\n"
            "  # or specify a specific synthetic config:\n"
            "  airtrace-generate-synthetic data=synthetic_cruise"
        )
    print("Run with --data-check after updating your data to verify the configuration.")


def _require_checkpoint(cfg: DictConfig) -> Path:
    checkpoint_value = cfg.get("checkpoint") if hasattr(cfg, "get") else getattr(cfg, "checkpoint", None)
    if not checkpoint_value:
        print("Error: No checkpoint provided. Use --checkpoint <path> or checkpoint=<path>.")
        sys.exit(1)

    checkpoint_path = Path(checkpoint_value)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)
    return checkpoint_path


def _load_checkpoint_if_present(model: torch.nn.Module | None, checkpoint_path: Path, preload_message: str | None = None) -> None:
    if preload_message:
        print(f"\n{preload_message} from {checkpoint_path}...")

    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if model is not None:
        model.load_state_dict(checkpoint["model_state_dict"])

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


def cli() -> None:
    """Entry point for the ``airtrace`` console script."""

    hydra_overrides = prepare_hydra_overrides(sys.argv[1:])
    sys.argv = [sys.argv[0], *hydra_overrides]
    main()


if __name__ == "__main__":
    cli()
