
import unittest
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path
import torch
from omegaconf import OmegaConf, DictConfig, ListConfig
import numpy as np

from airtrace.export.onnx_exporter import ONNXExporter

class TestONNXExporterCoverage(unittest.TestCase):
    def setUp(self):
        self.model = MagicMock(spec=torch.nn.Module)
        self.model.input_dim = 10
        self.model.output_dim = 5
        self.model.training = False
        self.config = OmegaConf.create({
            "data": {"window_size_in": 100, "sensors": ["a"] * 10},
            "model": {"input_dim": 10, "output_dim": 5}
        })
        self.exporter = ONNXExporter(self.model, self.config)

    def test_init(self):
        self.assertEqual(self.exporter.model, self.model)
        self.assertEqual(self.exporter.config, self.config)
        self.assertIsNone(self.exporter.transform_stats)

    @patch("torch.load")
    def test_from_checkpoint_no_config(self, mock_load):
        mock_load.return_value = {"model_state_dict": {}}
        with self.assertRaises(ValueError):
            ONNXExporter.from_checkpoint("ckpt.pt")

    @patch("torch.load")
    @patch("airtrace.models.registry.build_model")
    @patch("airtrace.export.onnx_exporter.ONNXExporter._infer_dimensions", return_value=(10, 5))
    def test_from_checkpoint_success(self, mock_infer, mock_build, mock_load):
        mock_load.return_value = {
            "config": self.config,
            "model_state_dict": {},
            "transform_stats": {"foo": "bar"}
        }
        mock_build.return_value = self.model
        
        exporter = ONNXExporter.from_checkpoint("ckpt.pt")
        self.assertIsNotNone(exporter.transform_stats)
        mock_build.assert_called()

    @patch("torch.load")
    @patch("airtrace.models.registry.build_model")
    @patch("airtrace.export.onnx_exporter.ONNXExporter._infer_dimensions", return_value=(10, 5))
    def test_from_checkpoint_no_stats(self, mock_infer, mock_build, mock_load):
        mock_load.return_value = {
            "config": self.config,
            "model_state_dict": {}
        }
        mock_build.return_value = self.model
        
        exporter = ONNXExporter.from_checkpoint("ckpt.pt")
        self.assertIsNone(exporter.transform_stats)

    def test_compute_input_dim_with_transforms(self):
        pipeline = [
            {"name": "context", "use_static": ["a", "b"]},
            {"name": "temporal_features"},
            {"name": "other"}
        ]
        dim = ONNXExporter._compute_input_dim_with_transforms(10, pipeline)
        self.assertEqual(dim, 12)

    def test_infer_from_model_weights_informer(self):
        state = {
            "value_embedding.proj.weight": torch.randn(10, 20), # input=20
            "projection.weight": torch.randn(5, 10) # output=5
        }
        dims = ONNXExporter._infer_from_model_weights(state)
        self.assertEqual(dims, (20, 5))

    def test_infer_from_model_weights_gru(self):
        state = {
            "gru.weight_ih_l0": torch.randn(30, 10), # input=10
            "decoder.bias": torch.randn(5) # output=5
        }
        dims = ONNXExporter._infer_from_model_weights(state)
        self.assertEqual(dims, (10, 5))

    def test_infer_from_model_weights_generic(self):
        state = {
            "encoder.weight": torch.randn(10, 20), # input=20
            "decoder.weight": torch.randn(5, 10) # output=5
        }
        dims = ONNXExporter._infer_from_model_weights(state)
        self.assertEqual(dims, (20, 5))

    def test_infer_from_model_weights_fail(self):
        state = {"foo": torch.randn(1)}
        dims = ONNXExporter._infer_from_model_weights(state)
        self.assertIsNone(dims)

    def test_infer_dimensions_explicit(self):
        dims = ONNXExporter._infer_dimensions({}, self.config)
        self.assertEqual(dims, (10, 5))

    def test_infer_dimensions_sensors_list(self):
        cfg = OmegaConf.create({"data": {"sensors": ["a"] * 8}})
        with patch.object(ONNXExporter, "_infer_from_model_weights", return_value=None):
            dims = ONNXExporter._infer_dimensions({}, cfg)
            self.assertEqual(dims, (8, 8))

    def test_infer_dimensions_sensors_dict(self):
        cfg = OmegaConf.create({"data": {"sensors": {"use": ["a"] * 8}}})
        with patch.object(ONNXExporter, "_infer_from_model_weights", return_value=None):
            dims = ONNXExporter._infer_dimensions({}, cfg)
            self.assertEqual(dims, (8, 8))

    def test_infer_dimensions_weight_mismatch(self):
        cfg = OmegaConf.create({"data": {"sensors": ["a"] * 8}})
        with patch.object(ONNXExporter, "_infer_from_model_weights", return_value=(10, 5)):
            dims = ONNXExporter._infer_dimensions({}, cfg)
            self.assertEqual(dims, (10, 5)) # Weights override

    def test_infer_dimensions_fallback_weights(self):
        cfg = OmegaConf.create({})
        with patch.object(ONNXExporter, "_infer_from_model_weights", return_value=(10, 5)):
            dims = ONNXExporter._infer_dimensions({}, cfg)
            self.assertEqual(dims, (10, 5))

    def test_infer_dimensions_fail(self):
        cfg = OmegaConf.create({})
        with patch.object(ONNXExporter, "_infer_from_model_weights", return_value=None):
            with self.assertRaises(ValueError):
                ONNXExporter._infer_dimensions({}, cfg)

    def test_recommend_opset_version(self):
        class InformerModel(torch.nn.Module): pass
        self.assertEqual(ONNXExporter._recommend_opset_version(InformerModel()), 17)
        
        class GRUModel(torch.nn.Module): pass
        self.assertEqual(ONNXExporter._recommend_opset_version(GRUModel()), 14)
        
        class LinearModel(torch.nn.Module): pass
        self.assertEqual(ONNXExporter._recommend_opset_version(LinearModel()), 14)
        
        class OtherModel(torch.nn.Module): pass
        self.assertEqual(ONNXExporter._recommend_opset_version(OtherModel()), 17)

    def test_validate_missing_attrs(self):
        model = MagicMock(spec=torch.nn.Module)
        del model.input_dim
        # output_dim might be missing too if we strictly follow spec
        model.training = False # Ensure training attribute exists
        
        exporter = ONNXExporter(model, self.config)
        res = exporter.validate(verbose=False)
        self.assertFalse(res['passed'])
        self.assertIn("Model missing input_dim attribute", res['errors'])

    def test_validate_missing_window_size(self):
        cfg = OmegaConf.create({"data": {}})
        exporter = ONNXExporter(self.model, cfg)
        res = exporter.validate(verbose=False)
        self.assertIn("Missing window_size_in in config", res['warnings'])

    def test_validate_end_to_end_no_stats(self):
        exporter = ONNXExporter(self.model, self.config, transform_stats=None)
        res = exporter.validate(end_to_end=True, verbose=False)
        self.assertFalse(res['passed'])
        self.assertIn("End-to-end export requires transform_stats", res['errors'])

    def test_validate_training_mode(self):
        self.model.training = True
        res = self.exporter.validate(verbose=False)
        self.assertIn("Model should be in eval mode", res['warnings'])

    def test_validate_runtime_resolvers(self):
        cfg = OmegaConf.create({"foo": "${now:%H-%M-%S}"})
        exporter = ONNXExporter(self.model, cfg)
        res = exporter.validate(verbose=False)
        self.assertIn("Config has runtime resolvers (${now:...}, ${oc.env:...})", res['warnings'])

    def test_validate_complex_model(self):
        class TransformerModel(torch.nn.Module): 
            input_dim=10
            output_dim=5
            def __init__(self):
                super().__init__()
                self.training = False # Explicitly set instance attribute
        
        exporter = ONNXExporter(TransformerModel(), self.config)
        res = exporter.validate(verbose=False)
        self.assertTrue(any("TransformerModel may have attention-related" in w for w in res['warnings']))

    def test_dry_run_validation_fail(self):
        with patch.object(ONNXExporter, "validate", return_value={"passed": False, "errors": ["err"]}):
            self.assertFalse(self.exporter.dry_run(verbose=True))

    def test_dry_run_input_fail(self):
        with patch.object(ONNXExporter, "validate", return_value={"passed": True}):
            with patch("torch.randn", side_effect=Exception("Input fail")):
                self.assertFalse(self.exporter.dry_run(verbose=False))

    def test_dry_run_wrapping_fail(self):
        with patch.object(ONNXExporter, "validate", return_value={"passed": True}):
            with patch("airtrace.export.onnx_exporter.ModelOnlyWrapper", side_effect=Exception("Wrap fail")):
                self.assertFalse(self.exporter.dry_run(verbose=False))

    def test_dry_run_forward_fail(self):
        with patch.object(ONNXExporter, "validate", return_value={"passed": True}):
            with patch("airtrace.export.onnx_exporter.ModelOnlyWrapper"):
                # Mock the wrapped model instance to fail on forward
                instance = MagicMock()
                instance.side_effect = Exception("Forward fail")
                with patch("airtrace.export.onnx_exporter.ModelOnlyWrapper", return_value=instance):
                    self.assertFalse(self.exporter.dry_run(verbose=False))

    def test_dry_run_success_end_to_end(self):
        exporter = ONNXExporter(self.model, self.config, transform_stats={"a": {}})
        with patch.object(exporter, "validate", return_value={"passed": True}):
            with patch("airtrace.export.transform_wrappers.create_forward_transform_pipeline"):
                with patch("airtrace.export.transform_wrappers.create_inverse_transform_pipeline"):
                    with patch("airtrace.export.onnx_exporter.EndToEndModel") as mock_e2e:
                        mock_e2e.return_value.return_value = torch.randn(1, 5)
                        self.assertTrue(exporter.dry_run(end_to_end=True, verbose=False))

    @patch("torch.onnx.export")
    def test_export_options(self, mock_export):
        # Test various options
        self.exporter.export(
            "model.onnx",
            single_batch_mode=True,
            fixed_sequence_length=True,
            batch_size=4, # Should be ignored for single batch
            verbose=False
        )
        # Check if single_batch_mode override worked
        # batch_size arg to torch.onnx.export is not passed directly, but used for dummy input
        # Check dummy input shape
        args, kwargs = mock_export.call_args
        dummy_input = args[1]
        self.assertEqual(dummy_input.dim(), 2) # [seq, feats]

    @patch("torch.onnx.export", side_effect=Exception("Dynamic axis error"))
    def test_export_dynamic_error(self, mock_export):
        with self.assertRaises(Exception):
            self.exporter.export("model.onnx", verbose=False)

    @patch("torch.onnx.export", side_effect=Exception("opset version error"))
    def test_export_opset_error(self, mock_export):
        with self.assertRaises(Exception):
            self.exporter.export("model.onnx", verbose=False)

    @patch("torch.onnx.export", side_effect=Exception("prim:: error"))
    def test_export_unsupported_error(self, mock_export):
        with self.assertRaises(Exception):
            self.exporter.export("model.onnx", verbose=False)

    @patch("torch.onnx.export", side_effect=Exception("Unknown error"))
    def test_export_generic_error(self, mock_export):
        with self.assertRaises(Exception):
            self.exporter.export("model.onnx", verbose=False)

    @patch("torch.onnx.export")
    def test_save_config_resolver_fail(self, mock_export):
        cfg = OmegaConf.create({"foo": "${now:%H}"})
        exporter = ONNXExporter(self.model, cfg)
        with patch("builtins.open", mock_open()):
            exporter.export("model.onnx", verbose=False)
            # Should print warning but succeed

    @patch("builtins.print")
    def test_verify_export_no_ort(self, mock_print):
        with patch.dict("sys.modules", {"onnxruntime": None}):
            self.assertFalse(self.exporter.verify_export("model.onnx", verbose=False))

    def test_verify_export_fail(self):
        with patch.dict("sys.modules", {"onnxruntime": MagicMock()}):
            import onnxruntime
            session = onnxruntime.InferenceSession.return_value
            # Return something very different
            session.run.return_value = [np.ones((1, 5)) * 100]
            
            with patch("airtrace.export.onnx_exporter.ModelOnlyWrapper") as mock_wrapper:
                mock_wrapper.return_value.return_value = torch.zeros(1, 5)
                
                self.assertFalse(self.exporter.verify_export("model.onnx", verbose=False))

    def test_verify_export_success(self):
        with patch.dict("sys.modules", {"onnxruntime": MagicMock()}):
            import onnxruntime
            session = onnxruntime.InferenceSession.return_value
            # Return same as torch
            session.run.return_value = [np.zeros((1, 5))]
            
            with patch("airtrace.export.onnx_exporter.ModelOnlyWrapper") as mock_wrapper:
                mock_wrapper.return_value.return_value = torch.zeros(1, 5)
                
                self.assertTrue(self.exporter.verify_export("model.onnx", verbose=False))

    def test_save_transform_stats(self):
        stats = {
            "t1": {"mean": np.array([1.0]), "std": 2.0, "name": "foo"}
        }
        self.exporter.transform_stats = stats
        with patch("builtins.open", mock_open()) as mock_file:
            with patch("json.dump") as mock_json:
                self.exporter._save_transform_stats(Path("stats.json"))
                args, _ = mock_json.call_args
                saved_dict = args[0]
                self.assertEqual(saved_dict["t1"]["mean"], [1.0])
                self.assertEqual(saved_dict["t1"]["std"], 2.0)

    def test_get_onnx_unsupported_reason_found(self):
        """Test _get_onnx_unsupported_reason for unsupported models"""
        reason = self.exporter._get_onnx_unsupported_reason("autoformer")
        self.assertIn("fft_rfft", reason)

        reason = self.exporter._get_onnx_unsupported_reason("timer")
        self.assertIn("unfold", reason)

        reason = self.exporter._get_onnx_unsupported_reason("median")
        self.assertIn("median operator", reason)

    def test_get_onnx_unsupported_reason_not_found(self):
        """Test _get_onnx_unsupported_reason for supported models"""
        reason = self.exporter._get_onnx_unsupported_reason("gru")
        self.assertIsNone(reason)

        reason = self.exporter._get_onnx_unsupported_reason("dlinear")
        self.assertIsNone(reason)

    def test_get_model_name_fallback(self):
        """Test _get_model_name fallback when config has no proper model name"""
        # Config with no model section
        cfg = OmegaConf.create({})
        exporter = ONNXExporter(self.model, cfg)
        name = exporter._get_model_name()
        # Should fall back to class name
        self.assertIsInstance(name, str)

    def test_get_model_name_with_valid_config(self):
        """Test _get_model_name with proper model.name in config"""
        cfg = OmegaConf.create({"model": {"name": "gru_ar"}})
        exporter = ONNXExporter(self.model, cfg)
        name = exporter._get_model_name()
        self.assertEqual(name, "gru_ar")

    def test_compute_input_dim_with_temporal_features(self):
        """Test _compute_input_dim_with_transforms with temporal features"""
        pipeline = [
            {"name": "context", "use_static": ["a", "b", "c"]},
            {"name": "temporal_features"},  # This branch is uncovered
            {"name": "zscore"}
        ]
        dim = ONNXExporter._compute_input_dim_with_transforms(10, pipeline)
        # Context adds 3, temporal_features adds 0 (pass through)
        self.assertEqual(dim, 13)

    def test_infer_from_model_weights_decoder_weight(self):
        """Test _infer_from_model_weights with decoder.weight pattern"""
        state = {
            "encoder.weight": torch.randn(10, 20),  # input=20
            "decoder.weight": torch.randn(5, 10)    # output=5
        }
        dims = ONNXExporter._infer_from_model_weights(state)
        self.assertEqual(dims, (20, 5))

    def test_infer_from_model_weights_only_decoder_bias(self):
        """Test _infer_from_model_weights finding only decoder.bias"""
        state = {
            "gru.weight_ih_l0": torch.randn(30, 10),  # input=10
            "decoder.bias": torch.randn(5)             # output=5 (fallback)
        }
        dims = ONNXExporter._infer_from_model_weights(state)
        self.assertEqual(dims, (10, 5))

    def test_export_unsupported_model(self):
        """Test export raises error for ONNX unsupported models"""
        cfg = OmegaConf.create({"model": {"name": "autoformer"}, "data": {"window_size_in": 100}})
        exporter = ONNXExporter(self.model, cfg)

        with self.assertRaises(ValueError) as context:
            exporter.export(Path("model.onnx"), verbose=False)

        self.assertIn("autoformer", str(context.exception))
        self.assertIn("fft_rfft", str(context.exception))

    def test_get_export_profile(self):
        """Test get_export_profile returns a valid profile"""
        profile = self.exporter.get_export_profile()
        # Profile should have expected attributes
        self.assertTrue(hasattr(profile, 'name'))
        self.assertTrue(hasattr(profile, 'verification_tolerance'))

