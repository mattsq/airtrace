"""Evaluation runner for trained models."""

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from tqdm import tqdm

from .metrics import compute_all_metrics, per_sensor_metrics


class EvaluationRunner:
    """Runner for evaluating trained models."""

    def __init__(
        self,
        model: torch.nn.Module,
        task: Any,
        test_loader: Any,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """Initialize evaluation runner.

        Args:
            model: Trained model to evaluate
            task: Task object
            test_loader: Test data loader
            device: Device to run on
        """
        self.model = model.to(device)
        self.task = task
        self.test_loader = test_loader
        self.device = device

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path,
        model_class: type,
        task: Any,
        test_loader: Any,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """Load model from checkpoint and create runner.

        Args:
            checkpoint_path: Path to checkpoint file
            model_class: Model class to instantiate
            task: Task object
            test_loader: Test data loader
            device: Device to run on

        Returns:
            EvaluationRunner instance
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Extract model config from checkpoint
        config = checkpoint.get("config", {})
        model_config = config.get("model", {})

        # Instantiate model
        # Note: This is simplified - real implementation would need to determine
        # input_dim and output_dim from the checkpoint or config
        model = model_class(**model_config.get("params", {}))

        # Load state dict
        model.load_state_dict(checkpoint["model_state_dict"])

        return cls(model, task, test_loader, device)

    def evaluate(self, return_predictions: bool = False) -> Dict[str, Any]:
        """Run evaluation on test set.

        Args:
            return_predictions: Whether to return all predictions

        Returns:
            Dictionary with metrics and optionally predictions
        """
        self.model.eval()

        all_preds = []
        all_targets = []
        all_losses = []

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating"):
                # Move to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                # Get predictions
                outputs = self.task.validation_step(batch, self.model)

                # Store results
                x = batch["x"]
                y = batch["y"]

                model_output = self.model(x)
                preds = model_output["preds"].cpu().numpy()
                targets = y[:, :preds.shape[1], :].cpu().numpy()

                all_preds.append(preds)
                all_targets.append(targets)
                all_losses.append(outputs["loss"].item() if isinstance(outputs["loss"], torch.Tensor)
                                else outputs["loss"])

        # Concatenate all predictions
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        # Compute metrics
        metrics = compute_all_metrics(all_preds.flatten(), all_targets.flatten())
        metrics["avg_loss"] = np.mean(all_losses)

        results = {
            "metrics": metrics,
            "num_samples": len(all_preds)
        }

        if return_predictions:
            results["predictions"] = all_preds
            results["targets"] = all_targets

        return results

    def evaluate_per_sensor(
        self,
        sensor_names: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate metrics per sensor.

        Args:
            sensor_names: List of sensor names

        Returns:
            Dictionary mapping sensor names to metrics
        """
        results = self.evaluate(return_predictions=True)

        preds = results["predictions"]
        targets = results["targets"]

        return per_sensor_metrics(preds, targets, sensor_names)
