
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import torch
import numpy as np

from airtrace.evaluation.eval_runner import EvaluationRunner

class TestEvaluationRunnerCoverage(unittest.TestCase):
    def setUp(self):
        self.model = MagicMock(spec=torch.nn.Module)
        self.model.return_value = {"preds": torch.randn(10, 5, 2)}
        self.model.to.return_value = self.model
        
        self.task = MagicMock()
        self.task.validation_step.return_value = {"loss": torch.tensor(0.5)}
        
        # Mock test loader
        self.batch = {
            "x": torch.randn(10, 5, 3),
            "y": torch.randn(10, 5, 2)
        }
        self.test_loader = [self.batch]
        
        self.runner = EvaluationRunner(
            model=self.model,
            task=self.task,
            test_loader=self.test_loader,
            device="cpu"
        )

    def test_init(self):
        self.assertEqual(self.runner.model, self.model)
        self.assertEqual(self.runner.task, self.task)
        self.assertEqual(self.runner.device, "cpu")
        self.model.to.assert_called_with("cpu")

    @patch("torch.load")
    def test_from_checkpoint(self, mock_load):
        mock_load.return_value = {
            "config": {"model": {"params": {"a": 1}}},
            "model_state_dict": {}
        }
        model_class = MagicMock()
        model_instance = model_class.return_value
        model_instance.to.return_value = model_instance
        
        runner = EvaluationRunner.from_checkpoint(
            Path("ckpt.pt"),
            model_class,
            self.task,
            self.test_loader,
            device="cpu"
        )
        
        model_class.assert_called_with(a=1)
        model_instance.load_state_dict.assert_called()
        self.assertEqual(runner.model, model_instance)

    @patch("airtrace.evaluation.eval_runner.compute_all_metrics")
    def test_evaluate_basic(self, mock_metrics):
        mock_metrics.return_value = {"mse": 0.1}
        
        results = self.runner.evaluate(return_predictions=False)
        
        self.assertIn("metrics", results)
        self.assertEqual(results["metrics"]["mse"], 0.1)
        self.assertEqual(results["metrics"]["avg_loss"], 0.5)
        self.assertEqual(results["num_samples"], 10)
        self.assertNotIn("predictions", results)
        
        self.task.validation_step.assert_called()
        self.model.eval.assert_called()

    @patch("airtrace.evaluation.eval_runner.compute_all_metrics")
    def test_evaluate_with_predictions(self, mock_metrics):
        mock_metrics.return_value = {"mse": 0.1}
        
        results = self.runner.evaluate(return_predictions=True)
        
        self.assertIn("predictions", results)
        self.assertIn("targets", results)
        self.assertEqual(results["predictions"].shape, (10, 5, 2))
        self.assertEqual(results["targets"].shape, (10, 5, 2))

    @patch("airtrace.evaluation.eval_runner.compute_all_metrics")
    def test_evaluate_loss_float(self, mock_metrics):
        mock_metrics.return_value = {"mse": 0.1}
        self.task.validation_step.return_value = {"loss": 0.5} # Not tensor
        
        results = self.runner.evaluate()
        self.assertEqual(results["metrics"]["avg_loss"], 0.5)

    @patch("airtrace.evaluation.eval_runner.per_sensor_metrics")
    @patch("airtrace.evaluation.eval_runner.compute_all_metrics")
    def test_evaluate_per_sensor(self, mock_metrics, mock_per_sensor):
        mock_metrics.return_value = {"mse": 0.1}
        mock_per_sensor.return_value = {"s1": {"mse": 0.2}}
        
        results = self.runner.evaluate_per_sensor(["s1", "s2"])
        
        self.assertEqual(results["s1"]["mse"], 0.2)
        mock_per_sensor.assert_called()

