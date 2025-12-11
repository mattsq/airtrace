
import unittest
from unittest.mock import patch, MagicMock, mock_open
import sys
from pathlib import Path
from importlib import metadata
import pytest
from omegaconf import OmegaConf

from airtrace.cli import (
    _resolve_version,
    _format_metric_value,
    _format_evaluation_results,
    _missing_data_assets,
    _print_data_guidance,
    _require_checkpoint,
    train,
    evaluate,
    export_onnx,
    main,
    prepare_hydra_overrides
)

class TestCliUtils(unittest.TestCase):
    def test_resolve_version_installed(self):
        with patch("importlib.metadata.version", return_value="1.2.3"):
            self.assertEqual(_resolve_version(), "1.2.3")

    def test_resolve_version_not_installed_with_init(self):
        with patch("importlib.metadata.version", side_effect=metadata.PackageNotFoundError), \
             patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.read_text", return_value='__version__ = "0.9.0"\n'):
            self.assertEqual(_resolve_version(), "0.9.0")

    def test_resolve_version_not_installed_no_init(self):
        with patch("importlib.metadata.version", side_effect=metadata.PackageNotFoundError), \
             patch("pathlib.Path.exists", return_value=False):
            self.assertEqual(_resolve_version(), "0.0.0")

    def test_format_metric_value_float(self):
        self.assertEqual(_format_metric_value(0.12345), "0.1235")

    def test_format_metric_value_tensor_like(self):
        class TensorLike:
            def item(self): return 0.56789
        self.assertEqual(_format_metric_value(TensorLike()), "0.5679")

    def test_format_metric_value_tensor_error(self):
        class WeirdTensor:
            def item(self): return "not a number"
            
        self.assertEqual(_format_metric_value(WeirdTensor()), "not a number")

    def test_format_evaluation_results(self):
        results = {"metrics": {"mae": 0.1}, "num_samples": 100}
        output = _format_evaluation_results(results)
        self.assertIn("MAE", output)
        self.assertIn("0.1000", output)
        self.assertIn("Samples", output)
        self.assertIn("100", output)

    def test_format_evaluation_results_empty(self):
        output = _format_evaluation_results({})
        self.assertIn("Evaluation Results", output)

class TestCliTrain(unittest.TestCase):
    @patch("airtrace.training.trainer.set_seed")
    @patch("airtrace.cli._missing_data_assets")
    @patch("airtrace.cli._print_data_guidance")
    @patch("sys.exit")
    def test_train_dry_run_missing_assets(self, mock_exit, mock_print, mock_missing, mock_seed):
        cfg = OmegaConf.create({
            "seed": 42,
            "cli": {"dry_run": True},
            "data": {"root": "data"},
            "train": {"batch_size": 32, "num_workers": 0}
        })
        mock_missing.return_value = [Path("missing")]
        
        train(cfg)
        
        mock_print.assert_called_once()
        mock_exit.assert_not_called()

    @patch("airtrace.training.trainer.set_seed")
    @patch("airtrace.cli._missing_data_assets", return_value=[])
    @patch("airtrace.transforms.registry.build_transforms")
    @patch("airtrace.data.datamodule.SensorDataModule")
    @patch("sys.exit")
    def test_train_datamodule_setup_error(self, mock_exit, mock_dm_cls, mock_bt, mock_missing, mock_seed):
        cfg = OmegaConf.create({
            "seed": 42,
            "cli": {},
            "data": {"root": "data"},
            "transforms": {"pipeline": [{"name": "foo"}]},
            "train": {"batch_size": 32, "num_workers": 0}
        })
        
        mock_dm = mock_dm_cls.return_value
        mock_dm.setup.side_effect = Exception("Setup failed")
        mock_exit.side_effect = SystemExit
        
        with self.assertRaises(SystemExit):
            train(cfg)
        
        mock_exit.assert_called_with(1)
        mock_bt.assert_called()

    @patch("airtrace.training.trainer.set_seed")
    @patch("airtrace.cli._missing_data_assets", return_value=[])
    @patch("airtrace.data.datamodule.SensorDataModule")
    def test_train_data_check_only(self, mock_dm_cls, mock_missing, mock_seed):
        cfg = OmegaConf.create({
            "seed": 42,
            "cli": {"data_check": True},
            "data": {"root": "data"},
            "train": {"batch_size": 32, "num_workers": 0}
        })
        
        train(cfg)
        
        mock_dm_cls.return_value.setup.assert_called()
        # Should return before model building

    @patch("airtrace.training.trainer.set_seed")
    @patch("airtrace.cli._missing_data_assets", return_value=[])
    @patch("airtrace.data.datamodule.SensorDataModule")
    def test_train_dry_run_success(self, mock_dm_cls, mock_missing, mock_seed):
        cfg = OmegaConf.create({
            "seed": 42,
            "cli": {"dry_run": True},
            "data": {"root": "data"},
            "train": {"batch_size": 32, "num_workers": 0}
        })
        
        train(cfg)
        
        mock_dm_cls.return_value.setup.assert_called()
        # Should return before model building

    @patch("airtrace.training.trainer.set_seed")
    @patch("airtrace.cli._missing_data_assets", return_value=[])
    @patch("airtrace.data.datamodule.SensorDataModule")
    @patch("airtrace.models.registry.build_model")
    @patch("sys.exit")
    def test_train_checkpoint_not_found(self, mock_exit, mock_bm, mock_dm_cls, mock_missing, mock_seed):
        cfg = OmegaConf.create({
            "seed": 42,
            "cli": {},
            "data": {"root": "data"},
            "train": {"batch_size": 32, "num_workers": 0},
            "model": {},
            "checkpoint": "non_existent.ckpt"
        })
        
        mock_dm = mock_dm_cls.return_value
        mock_dm.in_dim = 10
        mock_dm.out_dim = 5
        
        mock_exit.side_effect = SystemExit

        with patch("pathlib.Path.exists", return_value=False):
            with self.assertRaises(SystemExit):
                train(cfg)
            
        mock_exit.assert_called_with(1)

    @patch("airtrace.training.trainer.set_seed")
    @patch("airtrace.cli._missing_data_assets", return_value=[])
    @patch("airtrace.data.datamodule.SensorDataModule")
    @patch("airtrace.models.registry.build_model")
    @patch("airtrace.tasks.registry.build_task")
    @patch("airtrace.training.trainer.Trainer")
    @patch("airtrace.cli._load_checkpoint_if_present")
    def test_train_full_flow(self, mock_load_ckpt, mock_trainer_cls, mock_bt, mock_bm, mock_dm_cls, mock_missing, mock_seed):
        cfg = OmegaConf.create({
            "seed": 42,
            "cli": {},
            "data": {"root": "data"},
            "train": {"batch_size": 32, "num_workers": 0},
            "model": {},
            "task": {},
            "checkpoint": "exists.ckpt"
        })
        
        mock_dm = mock_dm_cls.return_value
        mock_dm.in_dim = 10
        mock_dm.out_dim = 5
        
        with patch("pathlib.Path.exists", return_value=True):
            train(cfg)
            
        mock_load_ckpt.assert_called()
        mock_trainer_cls.return_value.train.assert_called()

class TestCliEvaluate(unittest.TestCase):
    @patch("airtrace.training.trainer.set_seed")
    @patch("airtrace.cli._missing_data_assets", return_value=[])
    @patch("airtrace.data.datamodule.SensorDataModule")
    @patch("sys.exit")
    def test_evaluate_setup_error(self, mock_exit, mock_dm_cls, mock_missing, mock_seed):
        cfg = OmegaConf.create({
            "seed": 42,
            "cli": {},
            "data": {"root": "data"},
            "train": {"batch_size": 32, "num_workers": 0},
            "transforms": {"pipeline": []}
        })
        
        mock_dm_cls.return_value.setup.side_effect = Exception("Setup failed")
        mock_exit.side_effect = SystemExit
        
        with self.assertRaises(SystemExit):
            evaluate(cfg)
        mock_exit.assert_called_with(1)

    @patch("airtrace.training.trainer.set_seed")
    @patch("airtrace.cli._missing_data_assets", return_value=[])
    @patch("airtrace.data.datamodule.SensorDataModule")
    def test_evaluate_data_check(self, mock_dm_cls, mock_missing, mock_seed):
        cfg = OmegaConf.create({
            "seed": 42,
            "cli": {"data_check": True},
            "data": {"root": "data"},
            "train": {"batch_size": 32, "num_workers": 0}
        })
        
        evaluate(cfg)
        mock_dm_cls.return_value.setup.assert_called()
        # Should return

    @patch("airtrace.training.trainer.set_seed")
    @patch("airtrace.cli._missing_data_assets", return_value=[])
    @patch("airtrace.data.datamodule.SensorDataModule")
    @patch("airtrace.cli._require_checkpoint")
    @patch("airtrace.models.registry.build_model")
    @patch("airtrace.tasks.registry.build_task")
    @patch("airtrace.cli._load_checkpoint_if_present")
    @patch("airtrace.evaluation.eval_runner.EvaluationRunner")
    def test_evaluate_full_flow(self, mock_runner, mock_load, mock_task, mock_model, mock_req_ckpt, mock_dm_cls, mock_missing, mock_seed):
        cfg = OmegaConf.create({
            "seed": 42,
            "cli": {},
            "data": {"root": "data"},
            "train": {"batch_size": 32, "num_workers": 0},
            "model": {},
            "task": {}
        })
        
        mock_req_ckpt.return_value = Path("ckpt")
        mock_dm = mock_dm_cls.return_value
        mock_dm.in_dim = 10
        mock_dm.out_dim = 5
        
        evaluate(cfg)
        
        mock_runner.return_value.evaluate.assert_called()

class TestCliExport(unittest.TestCase):
    @patch("airtrace.cli._require_checkpoint")
    @patch("airtrace.export.ONNXExporter")
    @patch("sys.exit")
    def test_export_onnx_load_error(self, mock_exit, mock_exporter, mock_req_ckpt):
        cfg = OmegaConf.create({
            "cli": {"output": "out.onnx"},
            "checkpoint": "ckpt"
        })
        
        mock_req_ckpt.return_value = Path("ckpt")
        mock_exporter.from_checkpoint.side_effect = Exception("Load failed")
        
        export_onnx(cfg)
        mock_exit.assert_called_with(1)

    @patch("airtrace.cli._require_checkpoint")
    @patch("airtrace.export.ONNXExporter")
    @patch("sys.exit")
    def test_export_onnx_validate_only_fail(self, mock_exit, mock_exporter, mock_req_ckpt):
        cfg = OmegaConf.create({
            "cli": {"validate_only": True},
            "checkpoint": "ckpt"
        })
        
        mock_req_ckpt.return_value = Path("ckpt")
        mock_exporter.from_checkpoint.return_value.validate.return_value = {"passed": False}
        
        export_onnx(cfg)
        mock_exit.assert_called_with(1)

    @patch("airtrace.cli._require_checkpoint")
    @patch("airtrace.export.ONNXExporter")
    @patch("sys.exit")
    def test_export_onnx_validate_only_pass(self, mock_exit, mock_exporter, mock_req_ckpt):
        cfg = OmegaConf.create({
            "cli": {"validate_only": True},
            "checkpoint": "ckpt"
        })
        
        mock_req_ckpt.return_value = Path("ckpt")
        mock_exporter.from_checkpoint.return_value.validate.return_value = {"passed": True}
        
        export_onnx(cfg)
        mock_exit.assert_called_with(0)

    @patch("airtrace.cli._require_checkpoint")
    @patch("airtrace.export.ONNXExporter")
    @patch("sys.exit")
    def test_export_onnx_dry_run_fail(self, mock_exit, mock_exporter, mock_req_ckpt):
        cfg = OmegaConf.create({
            "cli": {"dry_run": True},
            "checkpoint": "ckpt"
        })
        
        mock_req_ckpt.return_value = Path("ckpt")
        mock_exporter.from_checkpoint.return_value.dry_run.return_value = False
        
        export_onnx(cfg)
        mock_exit.assert_called_with(1)

    @patch("airtrace.cli._require_checkpoint")
    @patch("airtrace.export.ONNXExporter")
    @patch("sys.exit")
    def test_export_onnx_dry_run_pass(self, mock_exit, mock_exporter, mock_req_ckpt):
        cfg = OmegaConf.create({
            "cli": {"dry_run": True},
            "checkpoint": "ckpt"
        })
        
        mock_req_ckpt.return_value = Path("ckpt")
        mock_exporter.from_checkpoint.return_value.dry_run.return_value = True
        
        export_onnx(cfg)
        mock_exit.assert_called_with(0)

    @patch("airtrace.cli._require_checkpoint")
    @patch("airtrace.export.ONNXExporter")
    @patch("sys.exit")
    def test_export_onnx_export_error(self, mock_exit, mock_exporter, mock_req_ckpt):
        cfg = OmegaConf.create({
            "cli": {},
            "checkpoint": "ckpt"
        })
        
        mock_req_ckpt.return_value = Path("ckpt")
        mock_exporter.from_checkpoint.return_value.export.side_effect = Exception("Export failed")
        
        export_onnx(cfg)
        mock_exit.assert_called_with(1)

    @patch("airtrace.cli._require_checkpoint")
    @patch("airtrace.export.ONNXExporter")
    def test_export_onnx_verify_warning(self, mock_exporter, mock_req_ckpt):
        cfg = OmegaConf.create({
            "cli": {"verify": True},
            "checkpoint": "ckpt"
        })
        
        mock_req_ckpt.return_value = Path("ckpt")
        mock_exporter.from_checkpoint.return_value.verify_export.side_effect = Exception("Verify failed")
        
        # Should not raise
        export_onnx(cfg)

class TestCliHelpers(unittest.TestCase):
    def test_missing_data_assets_test_required(self):
        cfg = OmegaConf.create({
            "root": "data",
            "train_index": "train",
            "val_index": "val",
            "test_index": "test"
        })
        
        with patch("pathlib.Path.exists", return_value=False):
            missing = _missing_data_assets(cfg, require_test=True)
            self.assertEqual(len(missing), 4) # root, train, val, test

    def test_print_data_guidance_synthetic(self):
        cfg = OmegaConf.create({"dataset_name": "synthetic_foo"})
        # Just check it runs without error
        _print_data_guidance(cfg, [])

    def test_print_data_guidance_real(self):
        cfg = OmegaConf.create({"dataset_name": "real_data"})
        _print_data_guidance(cfg, [])

    @patch("sys.exit")
    def test_require_checkpoint_missing(self, mock_exit):
        cfg = OmegaConf.create({})
        mock_exit.side_effect = SystemExit
        with self.assertRaises(SystemExit):
            _require_checkpoint(cfg)
        mock_exit.assert_called_with(1)

    @patch("sys.exit")
    def test_require_checkpoint_not_found(self, mock_exit):
        cfg = OmegaConf.create({"checkpoint": "missing.ckpt"})
        mock_exit.side_effect = SystemExit
        with patch("pathlib.Path.exists", return_value=False):
            with self.assertRaises(SystemExit):
                _require_checkpoint(cfg)
        mock_exit.assert_called_with(1)

    def test_prepare_hydra_overrides(self):
        # Test basic train
        overrides = prepare_hydra_overrides(["train", "exp=foo"])
        self.assertIn("mode=train", overrides)
        self.assertIn("exp=foo", overrides)
        
        # Test export onnx
        overrides = prepare_hydra_overrides([
            "export", "onnx", 
            "--checkpoint", "ckpt",
            "--output", "out",
            "--end-to-end",
            "--batch-size", "2",
            "--sequence-length", "10",
            "--no-verify",
            "--opset-version", "15",
            "--validate-only",
            "--dry-run",
            "--fixed-sequence-length",
            "--single-batch"
        ])
        
        self.assertIn("mode=export", overrides)
        self.assertIn("+cli.export_format=onnx", overrides)
        self.assertIn("checkpoint=ckpt", overrides)
        self.assertIn("+cli.output=out", overrides)
        self.assertIn("+cli.end_to_end=true", overrides)
        self.assertIn("+cli.batch_size=2", overrides)
        self.assertIn("+cli.sequence_length=10", overrides)
        self.assertIn("+cli.verify=false", overrides)
        self.assertIn("+cli.opset_version=15", overrides)
        self.assertIn("+cli.validate_only=true", overrides)
        self.assertIn("cli.dry_run=true", overrides)
        self.assertIn("+cli.fixed_sequence_length=true", overrides)
        self.assertIn("+cli.single_batch=true", overrides)

    @patch("sys.exit")
    def test_prepare_hydra_overrides_export_no_format(self, mock_exit):
        mock_exit.side_effect = SystemExit
        with self.assertRaises(SystemExit):
            prepare_hydra_overrides(["export"])
        mock_exit.assert_called_with(1)

    @patch("airtrace.cli.train")
    def test_main_train(self, mock_train):
        cfg = OmegaConf.create({"mode": "train"})
        main(cfg)
        mock_train.assert_called_with(cfg)

    @patch("airtrace.cli.evaluate")
    def test_main_eval(self, mock_eval):
        cfg = OmegaConf.create({"mode": "eval"})
        main(cfg)
        mock_eval.assert_called_with(cfg)

    @patch("airtrace.cli.export_onnx")
    def test_main_export(self, mock_export):
        cfg = OmegaConf.create({
            "mode": "export",
            "cli": {"export_format": "onnx"}
        })
        main(cfg)
        mock_export.assert_called_with(cfg)

    @patch("sys.exit")
    def test_main_unknown_export_format(self, mock_exit):
        cfg = OmegaConf.create({
            "mode": "export",
            "cli": {"export_format": "xml"}
        })
        mock_exit.side_effect = SystemExit
        with self.assertRaises(SystemExit):
            main(cfg)
        mock_exit.assert_called_with(1)

    @patch("sys.exit")
    def test_main_unknown_mode(self, mock_exit):
        cfg = OmegaConf.create({"mode": "dance"})
        mock_exit.side_effect = SystemExit
        with self.assertRaises(SystemExit):
            main(cfg)
        mock_exit.assert_called_with(1)
