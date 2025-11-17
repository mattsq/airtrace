"""
Automated testing script for Q400 model zoo validation.

Systematically tests all registered models against Q400 dataset and tracks results.
"""

import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from airtrace.data.datamodule import SensorDataModule
from airtrace.evaluation.eval_runner import EvaluationRunner
from airtrace.models.registry import build_model
from airtrace.tasks.registry import build_task
from airtrace.training.trainer import set_seed
from airtrace.transforms.registry import build_transforms


# Model testing tiers
BASELINE_MODELS = [
    "persistence", "moving_average", "zero", "drift",
    "linear_trend", "mean", "median",
    "exponential_smoothing", "seasonal_naive",
    "holt_linear_trend", "holt_winters", "theta",
    "polynomial_trend", "sarima", "var"
]

SIMPLE_TRAINABLE_MODELS = [
    "linear_ar", "mlp_ar", "gru_ar", "lstm_ar",
    "gru_seq2seq", "lstm_seq2seq"
]

ADVANCED_DL_MODELS = [
    "tcn", "moderntcn", "transformer", "patchtst", "itransformer",
    "autoformer", "nonstationary_transformer", "crossformer",
    "tft", "timemixer", "cyclenet"
]

PRETRAINED_MODELS = [
    "chronos_bolt", "moirai", "lag_llama"
]

STATE_OF_ART_MODELS = [
    "mamba2"
]


class ModelTester:
    """Automated model testing framework."""

    def __init__(
        self,
        data_config_name: str = "q400",
        transform_config_name: str = "minimal",
        task_config_name: str = "one_step",
        batch_size: int = 64,
        num_workers: int = 4,
        max_samples: int = 1000,  # Limit samples for fast testing
        results_dir: str = "results/q400_model_zoo"
    ):
        """Initialize tester.

        Args:
            data_config_name: Data configuration to use
            transform_config_name: Transform pipeline to use
            task_config_name: Task configuration to use
            batch_size: Batch size for dataloaders
            num_workers: Number of data loading workers
            max_samples: Maximum samples to evaluate per split (for speed)
            results_dir: Directory to save results
        """
        self.data_config_name = data_config_name
        self.transform_config_name = transform_config_name
        self.task_config_name = task_config_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_samples = max_samples

        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.results = []

        # Load configurations
        self._load_configs()

    def _load_configs(self):
        """Load Hydra configurations."""
        from hydra import initialize, compose

        # Initialize Hydra
        with initialize(version_base=None, config_path="../airtrace/configs"):
            cfg = compose(
                config_name="config",
                overrides=[
                    f"data={self.data_config_name}",
                    f"transforms={self.transform_config_name}",
                    f"task={self.task_config_name}",
                ]
            )

            self.data_config = cfg.data
            self.transform_config = cfg.transforms
            self.task_config = cfg.task

        print(f"Loaded configurations:")
        print(f"  Data: {self.data_config_name}")
        print(f"  Transforms: {self.transform_config_name}")
        print(f"  Task: {self.task_config_name}")

    def setup_data(self):
        """Set up data module."""
        print("\nSetting up data module...")

        # Build transforms
        transform_pipeline = None
        if "pipeline" in self.transform_config:
            transform_pipeline = build_transforms(self.transform_config.pipeline)

        # Build data module
        self.datamodule = SensorDataModule(
            data_config=OmegaConf.to_container(self.data_config, resolve=True),
            transforms=transform_pipeline,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

        self.datamodule.setup()

        print(f"  Input dim: {self.datamodule.in_dim}")
        print(f"  Output dim: {self.datamodule.out_dim}")
        print(f"  Window spec: input_len={self.datamodule.window_spec.input_len}, "
              f"pred_len={self.datamodule.window_spec.pred_len}")

        # Build task
        self.task = build_task(OmegaConf.to_container(self.task_config, resolve=True))
        print(f"  Task: {self.task}")

    def test_model(
        self,
        model_name: str,
        model_config: Dict[str, Any] = None,
        is_trainable: bool = False,
        train_epochs: int = 5
    ) -> Dict[str, Any]:
        """Test a single model.

        Args:
            model_name: Name of the model to test
            model_config: Model configuration (optional)
            is_trainable: Whether model requires training
            train_epochs: Number of epochs to train (if trainable)

        Returns:
            Dictionary with test results
        """
        print(f"\n{'='*70}")
        print(f"Testing: {model_name}")
        print(f"{'='*70}")

        result = {
            "model_name": model_name,
            "is_trainable": is_trainable,
            "status": "pending",
            "error": None,
            "metrics": {},
            "runtime_seconds": 0,
            "num_parameters": 0,
            "timestamp": datetime.now().isoformat()
        }

        start_time = time.time()

        try:
            # Set seed
            set_seed(42)

            # Build model config
            if model_config is None:
                model_config = {"name": model_name, "params": {}}

            # Build model
            print(f"  Building model...")
            model = build_model(
                config=model_config,
                input_dim=self.datamodule.in_dim,
                output_dim=self.datamodule.out_dim
            )

            # Count parameters
            try:
                result["num_parameters"] = sum(p.numel() for p in model.parameters())
                print(f"  Parameters: {result['num_parameters']:,}")
            except:
                result["num_parameters"] = 0

            # Train if needed
            if is_trainable:
                print(f"  Training for {train_epochs} epochs...")
                from airtrace.training.trainer import Trainer

                train_loader = self.datamodule.train_dataloader()
                val_loader = self.datamodule.val_dataloader()

                # Create minimal training config
                train_cfg = OmegaConf.create({
                    "train": {
                        "epochs": train_epochs,
                        "log_every_n_steps": 100,
                        "optimizer": {"name": "adam", "lr": 1e-3},
                        "scheduler": {"name": "none"},  # No scheduler for quick tests
                        "grad_clip": {"enabled": True, "max_norm": 1.0},
                        "checkpoint": {"save_top_k": 0},  # Don't save checkpoints
                        "early_stopping": {"patience": 999}  # No early stopping
                    },
                    "exp_name": f"q400_{model_name}",
                    "seed": 42
                })

                trainer = Trainer(
                    model=model,
                    task=self.task,
                    config=train_cfg,
                    train_loader=train_loader,
                    val_loader=val_loader
                )

                trainer.train()

                # Load best model
                # (For this quick test, we'll just use the final model)

            # Evaluate
            print(f"  Evaluating...")
            test_loader = self.datamodule.test_dataloader() or self.datamodule.val_dataloader()

            evaluator = EvaluationRunner(
                model=model,
                task=self.task,
                test_loader=test_loader
            )

            # Evaluate with limited samples
            if self.max_samples is not None:
                # Manual evaluation with sample limit
                all_preds = []
                all_targets = []
                sample_count = 0

                model.eval()
                with torch.no_grad():
                    for batch in test_loader:
                        x = batch["x"]
                        y = batch["y"]
                        output = model(x)

                        # Extract predictions from model output dictionary
                        if isinstance(output, dict):
                            pred = output["preds"]
                        else:
                            pred = output

                        all_preds.append(pred.cpu())
                        all_targets.append(y.cpu())

                        sample_count += len(x)
                        if sample_count >= self.max_samples:
                            break

                # Compute metrics
                all_preds = torch.cat(all_preds, dim=0)
                all_targets = torch.cat(all_targets, dim=0)

                result["metrics"] = self.task.compute_metrics(all_preds, all_targets)
                result["num_samples_evaluated"] = len(all_preds)
            else:
                # Full evaluation
                eval_results = evaluator.evaluate(return_predictions=False)
                result["metrics"] = eval_results["metrics"]
                result["num_samples_evaluated"] = eval_results["num_samples"]

            result["status"] = "success"

            print(f"  Results:")
            for metric, value in result["metrics"].items():
                print(f"    {metric.upper()}: {value:.4f}")

        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)
            result["traceback"] = traceback.format_exc()
            print(f"  ERROR: {e}")
            print(f"  Traceback:\n{result['traceback']}")

        finally:
            result["runtime_seconds"] = time.time() - start_time
            print(f"  Runtime: {result['runtime_seconds']:.2f}s")

        return result

    def test_model_tier(
        self,
        tier_name: str,
        model_list: List[str],
        is_trainable: bool = False,
        train_epochs: int = 5
    ):
        """Test all models in a tier.

        Args:
            tier_name: Name of the tier for logging
            model_list: List of model names to test
            is_trainable: Whether models require training
            train_epochs: Number of training epochs
        """
        print(f"\n{'#'*70}")
        print(f"# {tier_name}")
        print(f"# Models: {len(model_list)}")
        print(f"{'#'*70}")

        for model_name in model_list:
            result = self.test_model(
                model_name=model_name,
                is_trainable=is_trainable,
                train_epochs=train_epochs
            )

            self.results.append(result)

            # Save intermediate results
            self.save_results()

    def save_results(self):
        """Save results to JSON and CSV."""
        # Save JSON
        json_path = self.results_dir / "model_zoo_results.json"
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        # Save CSV summary
        csv_path = self.results_dir / "model_zoo_summary.csv"

        # Flatten results for CSV
        csv_data = []
        for r in self.results:
            row = {
                "model_name": r["model_name"],
                "status": r["status"],
                "is_trainable": r["is_trainable"],
                "num_parameters": r["num_parameters"],
                "runtime_seconds": r["runtime_seconds"],
                "num_samples": r.get("num_samples_evaluated", 0)
            }

            # Add metrics
            for metric, value in r["metrics"].items():
                row[metric] = value

            # Add error if present
            if r["error"]:
                row["error"] = r["error"]

            csv_data.append(row)

        df = pd.DataFrame(csv_data)
        df.to_csv(csv_path, index=False)

        print(f"\n[Saved] Results to {self.results_dir}/")

    def run_all_tests(self):
        """Run complete model zoo testing."""
        print("\n" + "="*70)
        print("Q400 MODEL ZOO TESTING")
        print("="*70)

        # Setup data once
        self.setup_data()

        # Phase 2: Baseline models (non-trainable)
        self.test_model_tier(
            tier_name="PHASE 2: BASELINE MODELS",
            model_list=BASELINE_MODELS,
            is_trainable=False
        )

        # Phase 3: Simple trainable models
        self.test_model_tier(
            tier_name="PHASE 3: SIMPLE TRAINABLE MODELS",
            model_list=SIMPLE_TRAINABLE_MODELS,
            is_trainable=True,
            train_epochs=5
        )

        # Phase 4: Advanced DL models
        self.test_model_tier(
            tier_name="PHASE 4: ADVANCED DEEP LEARNING MODELS",
            model_list=ADVANCED_DL_MODELS,
            is_trainable=True,
            train_epochs=5
        )

        # Phase 5: Pretrained models
        # (Skip for now - requires downloads and special handling)
        # self.test_model_tier(
        #     tier_name="PHASE 5: PRETRAINED MODELS",
        #     model_list=PRETRAINED_MODELS,
        #     is_trainable=False
        # )

        # Phase 6: State-of-the-art
        self.test_model_tier(
            tier_name="PHASE 6: STATE-OF-THE-ART MODELS",
            model_list=STATE_OF_ART_MODELS,
            is_trainable=True,
            train_epochs=5
        )

        # Final summary
        self.print_summary()

    def print_summary(self):
        """Print test summary."""
        print("\n" + "="*70)
        print("TESTING SUMMARY")
        print("="*70)

        total = len(self.results)
        success = sum(1 for r in self.results if r["status"] == "success")
        failed = sum(1 for r in self.results if r["status"] == "failed")

        print(f"\nTotal models tested: {total}")
        print(f"  Success: {success} ({success/total*100:.1f}%)")
        print(f"  Failed:  {failed} ({failed/total*100:.1f}%)")

        # Best performing models
        successful_results = [r for r in self.results if r["status"] == "success"]
        if successful_results:
            print(f"\nTop 5 models by RMSE:")
            sorted_by_rmse = sorted(
                successful_results,
                key=lambda x: x["metrics"].get("rmse", float('inf'))
            )[:5]

            for i, r in enumerate(sorted_by_rmse, 1):
                rmse = r["metrics"].get("rmse", 0)
                mae = r["metrics"].get("mae", 0)
                print(f"  {i}. {r['model_name']:30s} - RMSE: {rmse:.4f}, MAE: {mae:.4f}")

        print(f"\nResults saved to: {self.results_dir}/")
        print("="*70)


def main():
    """Main entry point."""
    tester = ModelTester(
        data_config_name="q400",
        transform_config_name="minimal",
        task_config_name="one_step",
        batch_size=128,
        num_workers=4,
        max_samples=5000  # Evaluate 5000 samples for reasonable testing
    )

    tester.run_all_tests()


if __name__ == "__main__":
    main()
