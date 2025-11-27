"""Tests for ONNX export functionality."""

import json
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
import torch
import torch.nn as nn

from airtrace.export.transform_wrappers import (
    ZScoreWrapper,
    RobustScalerWrapper,
    TransformCompose,
    create_transform_wrapper,
    create_inverse_transform_pipeline,
)
from airtrace.export.end_to_end_model import (
    EndToEndModel,
    ModelOnlyWrapper,
)


class DummyModel(nn.Module):
    """Dummy model for testing."""

    def __init__(self, input_dim: int = 10, output_dim: int = 10):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x, context=None):
        """Forward pass."""
        # Return dict to match ARBaseModel interface
        batch_size, seq_len, _ = x.shape
        preds = self.linear(x)
        return {"preds": preds}


class TestZScoreWrapper:
    """Tests for ZScoreWrapper."""

    def test_forward_transform(self):
        """Test forward z-score normalization."""
        mean = np.array([1.0, 2.0, 3.0])
        std = np.array([0.5, 1.0, 1.5])

        wrapper = ZScoreWrapper(mean=mean, std=std)

        x = torch.tensor([[4.0, 5.0, 6.0]])  # [1, 3]
        y = wrapper(x)

        # Expected: (x - mean) / std
        expected = torch.tensor([[(4.0-1.0)/0.5, (5.0-2.0)/1.0, (6.0-3.0)/1.5]])

        assert torch.allclose(y, expected, atol=1e-5)

    def test_inverse_transform(self):
        """Test inverse z-score normalization."""
        mean = np.array([1.0, 2.0, 3.0])
        std = np.array([0.5, 1.0, 1.5])

        wrapper = ZScoreWrapper(mean=mean, std=std)

        # Start with normalized data
        x_normalized = torch.tensor([[6.0, 3.0, 2.0]])  # [1, 3]

        # Apply inverse
        x_original = wrapper.inverse(x_normalized)

        # Expected: x * std + mean
        expected = torch.tensor([[6.0*0.5+1.0, 3.0*1.0+2.0, 2.0*1.5+3.0]])

        assert torch.allclose(x_original, expected, atol=1e-5)

    def test_forward_inverse_identity(self):
        """Test that forward and inverse are inverse operations."""
        mean = np.array([1.0, 2.0, 3.0])
        std = np.array([0.5, 1.0, 1.5])

        wrapper = ZScoreWrapper(mean=mean, std=std)

        x = torch.randn(2, 5, 3)
        x_normalized = wrapper(x)
        x_reconstructed = wrapper.inverse(x_normalized)

        assert torch.allclose(x, x_reconstructed, atol=1e-5)


class TestRobustScalerWrapper:
    """Tests for RobustScalerWrapper."""

    def test_forward_transform(self):
        """Test forward robust scaling."""
        center = np.array([1.0, 2.0, 3.0])
        scale = np.array([0.5, 1.0, 1.5])

        wrapper = RobustScalerWrapper(center=center, scale=scale)

        x = torch.tensor([[4.0, 5.0, 6.0]])  # [1, 3]
        y = wrapper(x)

        # Expected: (x - center) / scale
        expected = torch.tensor([[(4.0-1.0)/0.5, (5.0-2.0)/1.0, (6.0-3.0)/1.5]])

        assert torch.allclose(y, expected, atol=1e-5)

    def test_inverse_transform(self):
        """Test inverse robust scaling."""
        center = np.array([1.0, 2.0, 3.0])
        scale = np.array([0.5, 1.0, 1.5])

        wrapper = RobustScalerWrapper(center=center, scale=scale)

        x_scaled = torch.tensor([[6.0, 3.0, 2.0]])
        x_original = wrapper.inverse(x_scaled)

        expected = torch.tensor([[6.0*0.5+1.0, 3.0*1.0+2.0, 2.0*1.5+3.0]])

        assert torch.allclose(x_original, expected, atol=1e-5)


class TestTransformCompose:
    """Tests for TransformCompose."""

    def test_compose_multiple_transforms(self):
        """Test composing multiple transforms."""
        mean1 = np.array([1.0, 2.0])
        std1 = np.array([0.5, 1.0])
        mean2 = np.array([0.0, 0.0])
        std2 = np.array([2.0, 2.0])

        transform1 = ZScoreWrapper(mean=mean1, std=std1)
        transform2 = ZScoreWrapper(mean=mean2, std=std2)

        compose = TransformCompose([transform1, transform2])

        x = torch.tensor([[4.0, 5.0]])

        # Apply composed transforms
        y = compose(x)

        # Manual application
        y_manual = transform1(x)
        y_manual = transform2(y_manual)

        assert torch.allclose(y, y_manual, atol=1e-5)

    def test_inverse_compose(self):
        """Test inverse of composed transforms."""
        mean1 = np.array([1.0, 2.0])
        std1 = np.array([0.5, 1.0])
        mean2 = np.array([0.0, 0.0])
        std2 = np.array([2.0, 2.0])

        transform1 = ZScoreWrapper(mean=mean1, std=std1)
        transform2 = ZScoreWrapper(mean=mean2, std=std2)

        compose = TransformCompose([transform1, transform2])

        x = torch.randn(2, 3, 2)

        # Apply forward and inverse
        y = compose(x)
        x_reconstructed = compose.inverse(y)

        assert torch.allclose(x, x_reconstructed, atol=1e-5)


class TestCreateTransformWrapper:
    """Tests for create_transform_wrapper function."""

    def test_create_zscore_wrapper(self):
        """Test creating a ZScore wrapper from stats."""
        stats = {
            'scaler_y_mean': np.array([1.0, 2.0, 3.0]),
            'scaler_y_scale': np.array([0.5, 1.0, 1.5]),
        }

        wrapper = create_transform_wrapper(stats, 'ZScoreTransform_0', use_x_scaler=False)

        assert isinstance(wrapper, ZScoreWrapper)
        assert np.array_equal(wrapper.mean, stats['scaler_y_mean'])
        assert np.array_equal(wrapper.std, stats['scaler_y_scale'])

    def test_create_zscore_wrapper_for_preprocessing(self):
        """Test creating a ZScore wrapper for preprocessing (x scaler)."""
        stats = {
            'scaler_x_mean': np.array([1.0, 2.0, 3.0]),
            'scaler_x_scale': np.array([0.5, 1.0, 1.5]),
        }

        wrapper = create_transform_wrapper(stats, 'ZScoreTransform_0', use_x_scaler=True)

        assert isinstance(wrapper, ZScoreWrapper)
        assert np.array_equal(wrapper.mean, stats['scaler_x_mean'])
        assert np.array_equal(wrapper.std, stats['scaler_x_scale'])

        # Verify it uses the x scaler
        x = torch.tensor([[1.0, 2.0, 3.0]])
        result = wrapper(x)
        # Should apply (x - mean) / scale, which should be all zeros
        expected = torch.zeros_like(x)
        assert torch.allclose(result, expected, atol=1e-5)

    def test_unsupported_transform_returns_none(self):
        """Test that unsupported transforms return None."""
        stats = {}
        wrapper = create_transform_wrapper(stats, 'UnsupportedTransform_0')

        assert wrapper is None


class TestTransformPipelineOrder:
    """Tests for transform pipeline ordering."""

    def test_inverse_pipeline_preserves_order(self):
        """Test that inverse pipeline preserves transform order (not alphabetical)."""
        from airtrace.export import create_inverse_transform_pipeline

        # Create stats that would be sorted differently alphabetically
        # ZScoreTransform_0 should come before ZScoreTransform_1
        transform_stats = {
            'ZScoreTransform_0': {
                'scaler_y_mean': np.array([1.0]),
                'scaler_y_scale': np.array([1.0]),
            },
            'ZScoreTransform_1': {
                'scaler_y_mean': np.array([2.0]),
                'scaler_y_scale': np.array([2.0]),
            },
        }

        pipeline = create_inverse_transform_pipeline(transform_stats)

        # Verify it's a TransformCompose with 2 transforms
        assert isinstance(pipeline, TransformCompose)
        assert len(pipeline.transforms) == 2

        # Verify the order by checking the mean values
        # First transform should have mean=1.0, second should have mean=2.0
        assert torch.allclose(pipeline.transforms[0].mean, torch.tensor([1.0]))
        assert torch.allclose(pipeline.transforms[1].mean, torch.tensor([2.0]))

    def test_forward_pipeline_preserves_order(self):
        """Test that forward pipeline preserves transform order (not alphabetical)."""
        from airtrace.export import create_forward_transform_pipeline

        # Create stats with numeric ordering
        transform_stats = {
            'ZScoreTransform_0': {
                'scaler_x_mean': np.array([1.0]),
                'scaler_x_scale': np.array([1.0]),
            },
            'ZScoreTransform_1': {
                'scaler_x_mean': np.array([2.0]),
                'scaler_x_scale': np.array([2.0]),
            },
        }

        pipeline = create_forward_transform_pipeline(transform_stats)

        # Verify it's a TransformCompose with 2 transforms
        assert isinstance(pipeline, TransformCompose)
        assert len(pipeline.transforms) == 2

        # Verify the order
        assert torch.allclose(pipeline.transforms[0].mean, torch.tensor([1.0]))
        assert torch.allclose(pipeline.transforms[1].mean, torch.tensor([2.0]))

    def test_pipeline_order_with_different_names(self):
        """Test that pipeline order works correctly with different transform types."""
        from airtrace.export import create_inverse_transform_pipeline

        # Mix different transform types to ensure ordering is robust
        transform_stats = {
            'ZScoreTransform_0': {
                'scaler_y_mean': np.array([1.0]),
                'scaler_y_scale': np.array([1.0]),
            },
            'AnotherTransform_1': {
                # This would be unsupported and skipped
            },
            'ZScoreTransform_2': {
                'scaler_y_mean': np.array([3.0]),
                'scaler_y_scale': np.array([1.0]),
            },
        }

        pipeline = create_inverse_transform_pipeline(transform_stats)

        # Should have 2 transforms (AnotherTransform_1 is unsupported)
        assert isinstance(pipeline, TransformCompose)
        assert len(pipeline.transforms) == 2

        # Verify order: transform_0 then transform_2
        assert torch.allclose(pipeline.transforms[0].mean, torch.tensor([1.0]))
        assert torch.allclose(pipeline.transforms[1].mean, torch.tensor([3.0]))


class TestEndToEndModel:
    """Tests for EndToEndModel."""

    def test_end_to_end_forward(self):
        """Test end-to-end model forward pass."""
        model = DummyModel(input_dim=3, output_dim=3)

        # Create simple preprocessing (identity in this case)
        mean = np.zeros(3)
        std = np.ones(3)
        preprocess = ZScoreWrapper(mean=mean, std=std)

        # Create postprocessing
        postprocess = ZScoreWrapper(mean=mean, std=std)

        end_to_end = EndToEndModel(
            model=model,
            preprocess=preprocess,
            postprocess=postprocess,
        )

        x = torch.randn(2, 5, 3)
        output = end_to_end(x)

        assert output.shape == (2, 5, 3)
        assert isinstance(output, torch.Tensor)

    def test_end_to_end_with_inverse_transform(self):
        """Test that postprocessing correctly applies inverse transform."""
        model = DummyModel(input_dim=3, output_dim=3)

        # Create a non-identity transform
        mean = np.array([1.0, 2.0, 3.0])
        std = np.array([0.5, 1.0, 1.5])

        # Postprocessing should invert the transform
        postprocess = ZScoreWrapper(mean=mean, std=std)

        end_to_end = EndToEndModel(
            model=model,
            preprocess=None,
            postprocess=postprocess,
        )

        x = torch.randn(2, 5, 3)

        # Get model output directly
        model.eval()
        with torch.no_grad():
            model_output = model(x)["preds"]

        # Get end-to-end output (with postprocessing)
        end_to_end.eval()
        with torch.no_grad():
            e2e_output = end_to_end(x)

        # End-to-end output should be different due to postprocessing
        assert not torch.allclose(model_output, e2e_output, atol=1e-5)

    def test_end_to_end_with_preprocessing_and_postprocessing(self):
        """Test that preprocessing and postprocessing are both applied."""
        model = DummyModel(input_dim=3, output_dim=3)

        # Create preprocessing (forward transform)
        input_mean = np.array([10.0, 20.0, 30.0])
        input_std = np.array([5.0, 10.0, 15.0])
        preprocess = ZScoreWrapper(mean=input_mean, std=input_std)

        # Create postprocessing (inverse transform)
        output_mean = np.array([1.0, 2.0, 3.0])
        output_std = np.array([0.5, 1.0, 1.5])
        postprocess = ZScoreWrapper(mean=output_mean, std=output_std)

        end_to_end = EndToEndModel(
            model=model,
            preprocess=preprocess,
            postprocess=postprocess,
        )

        # Create raw input data
        x_raw = torch.tensor([[[10.0, 20.0, 30.0]]])  # [1, 1, 3]

        # Manual processing to verify
        # 1. Preprocessing: (x - input_mean) / input_std = all zeros
        x_preprocessed = (x_raw - torch.from_numpy(input_mean).float()) / torch.from_numpy(input_std).float()

        # 2. Model forward
        model.eval()
        with torch.no_grad():
            model_output = model(x_preprocessed, None)["preds"]

        # 3. Postprocessing: y * output_std + output_mean
        expected_output = model_output * torch.from_numpy(output_std).float() + torch.from_numpy(output_mean).float()

        # Get end-to-end output
        end_to_end.eval()
        with torch.no_grad():
            e2e_output = end_to_end(x_raw, None)

        # Should match the manually computed output
        assert torch.allclose(e2e_output, expected_output, atol=1e-5)

        # Verify preprocessing was actually applied
        # (output should differ from what we'd get without preprocessing)
        with torch.no_grad():
            no_preprocess_output = model(x_raw, None)["preds"]
            no_preprocess_output = no_preprocess_output * torch.from_numpy(output_std).float() + torch.from_numpy(output_mean).float()

        assert not torch.allclose(e2e_output, no_preprocess_output, atol=1e-5)

    def test_postprocessing_calls_inverse_not_forward(self):
        """Test that postprocessing uses inverse() method for denormalization, not forward()."""
        model = DummyModel(input_dim=3, output_dim=3)

        # Create postprocessing with specific mean/std
        output_mean = np.array([10.0, 20.0, 30.0])
        output_std = np.array([2.0, 4.0, 6.0])
        postprocess = ZScoreWrapper(mean=output_mean, std=output_std)

        end_to_end = EndToEndModel(
            model=model,
            preprocess=None,
            postprocess=postprocess,
        )

        # Use a simple input
        x = torch.zeros(1, 1, 3)  # All zeros

        # Get model output (with all-zero input, the linear layer will produce some output)
        model.eval()
        with torch.no_grad():
            model_preds = model(x, None)["preds"]

        # Get end-to-end output
        end_to_end.eval()
        with torch.no_grad():
            e2e_output = end_to_end(x, None)

        # The end-to-end output should be: model_preds * std + mean (inverse transform)
        # NOT: (model_preds - mean) / std (forward transform)
        expected_inverse = model_preds * torch.from_numpy(output_std).float() + torch.from_numpy(output_mean).float()
        wrong_forward = (model_preds - torch.from_numpy(output_mean).float()) / (torch.from_numpy(output_std).float() + 1e-8)

        # Verify it matches inverse transform
        assert torch.allclose(e2e_output, expected_inverse, atol=1e-5), \
            "Postprocessing should apply inverse transform (denormalization)"

        # Verify it does NOT match forward transform (would be the bug)
        assert not torch.allclose(e2e_output, wrong_forward, atol=1e-5), \
            "Postprocessing should NOT apply forward transform (normalization)"

    def test_end_to_end_handles_tensor_output_without_inverse(self):
        """EndToEndModel should postprocess tensor outputs even without inverse support."""

        class _TensorModel(torch.nn.Module):
            def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
                return x + 1.0

        class _AddOne(torch.nn.Module):
            def forward(self, preds: torch.Tensor) -> torch.Tensor:
                return preds + 1.0

        x = torch.randn(2, 3, 4)
        end_to_end = EndToEndModel(model=_TensorModel(), preprocess=None, postprocess=_AddOne())

        output = end_to_end(x)

        assert torch.allclose(output, x + 2.0)
        assert output.shape == x.shape


class TestModelOnlyWrapper:
    """Tests for ModelOnlyWrapper."""

    def test_wrapper_extracts_preds(self):
        """Test that wrapper extracts predictions from dict."""
        model = DummyModel(input_dim=3, output_dim=3)
        wrapper = ModelOnlyWrapper(model)

        x = torch.randn(2, 5, 3)
        output = wrapper(x)

        # Should return tensor, not dict
        assert isinstance(output, torch.Tensor)
        assert output.shape == (2, 5, 3)

    def test_wrapper_with_context(self):
        """Test wrapper with context parameter."""
        model = DummyModel(input_dim=3, output_dim=3)
        wrapper = ModelOnlyWrapper(model)

        x = torch.randn(2, 5, 3)
        context = torch.randn(2, 5, 2)

        output = wrapper(x, context)

        assert isinstance(output, torch.Tensor)

    def test_wrapper_passes_through_tensor_output(self):
        """ModelOnlyWrapper should return raw tensor outputs unchanged."""

        class _TensorOutputModel(torch.nn.Module):
            def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
                return x * 2

        model = _TensorOutputModel()
        wrapper = ModelOnlyWrapper(model)

        x = torch.randn(1, 2, 3)
        output = wrapper(x)

        assert torch.allclose(output, x * 2)
        assert output.shape == x.shape


class TestDimensionInference:
    """Tests for dimension inference from checkpoints and configs."""

    def test_infer_with_context_transform(self):
        """Test dimension inference when context transform adds static features."""
        from airtrace.export.onnx_exporter import ONNXExporter
        from omegaconf import OmegaConf

        # Simulate a checkpoint with context transform
        # 11 base sensors + 2 static features = 13 input_dim
        model_state = {
            "value_embedding.proj.weight": torch.randn(96, 13),  # input_dim=13
            "projection.weight": torch.randn(11, 96),  # output_dim=11
            "projection.bias": torch.randn(11),
        }

        config = OmegaConf.create({
            "model": {"name": "informer"},
            "data": {
                "sensors": {"use": ["sensor_" + str(i) for i in range(11)]},
            },
            "transforms": {
                "pipeline": [
                    {"name": "zscore"},
                    {"name": "context", "use_static": ["aircraft_type", "route_id"]},
                ]
            },
        })

        input_dim, output_dim = ONNXExporter._infer_dimensions(model_state, config)

        # Should correctly infer 13 (11 sensors + 2 static) and 11
        assert input_dim == 13, f"Expected input_dim=13, got {input_dim}"
        assert output_dim == 11, f"Expected output_dim=11, got {output_dim}"

    def test_infer_without_context_transform(self):
        """Test dimension inference when no transforms modify dimensions."""
        from airtrace.export.onnx_exporter import ONNXExporter
        from omegaconf import OmegaConf

        model_state = {
            "value_embedding.proj.weight": torch.randn(96, 11),
            "projection.weight": torch.randn(11, 96),
            "projection.bias": torch.randn(11),
        }

        config = OmegaConf.create({
            "model": {"name": "informer"},
            "data": {
                "sensors": {"use": ["sensor_" + str(i) for i in range(11)]},
            },
            "transforms": {
                "pipeline": [
                    {"name": "zscore"},
                ]
            },
        })

        input_dim, output_dim = ONNXExporter._infer_dimensions(model_state, config)

        assert input_dim == 11
        assert output_dim == 11

    def test_infer_from_informer_weights(self):
        """Test dimension inference from Informer model weights."""
        from airtrace.export.onnx_exporter import ONNXExporter
        from omegaconf import OmegaConf

        model_state = {
            "value_embedding.proj.weight": torch.randn(96, 15),
            "projection.weight": torch.randn(12, 96),
            "projection.bias": torch.randn(12),
        }

        # Empty config - should infer from weights
        config = OmegaConf.create({
            "model": {},
            "data": {},
            "transforms": {"pipeline": []},
        })

        input_dim, output_dim = ONNXExporter._infer_dimensions(model_state, config)

        assert input_dim == 15
        assert output_dim == 12

    def test_infer_from_gru_weights(self):
        """Test dimension inference from GRU model weights."""
        from airtrace.export.onnx_exporter import ONNXExporter
        from omegaconf import OmegaConf

        # GRU weight_ih_l0 has shape [3*hidden_size, input_size]
        model_state = {
            "encoder.weight_ih_l0": torch.randn(384, 10),  # 3*128, input_dim=10
            "decoder.weight": torch.randn(8, 128),  # output_dim=8
            "decoder.bias": torch.randn(8),
        }

        config = OmegaConf.create({
            "model": {},
            "data": {},
            "transforms": {"pipeline": []},
        })

        input_dim, output_dim = ONNXExporter._infer_dimensions(model_state, config)

        assert input_dim == 10
        assert output_dim == 8

    def test_infer_raises_on_missing_info(self):
        """Test that inference raises clear error when information is missing."""
        from airtrace.export.onnx_exporter import ONNXExporter
        from omegaconf import OmegaConf

        # Empty model state and config
        model_state = {}
        config = OmegaConf.create({
            "model": {},
            "data": {},
            "transforms": {"pipeline": []},
        })

        with pytest.raises(ValueError, match="Could not infer dimensions"):
            ONNXExporter._infer_dimensions(model_state, config)

    def test_config_weight_mismatch_uses_weights(self):
        """Test that when config and weights disagree, weights are preferred."""
        from airtrace.export.onnx_exporter import ONNXExporter
        from omegaconf import OmegaConf

        # Config suggests 11, but weights show 13 (missing context transform in config)
        model_state = {
            "value_embedding.proj.weight": torch.randn(96, 13),
            "projection.weight": torch.randn(11, 96),
        }

        config = OmegaConf.create({
            "model": {},
            "data": {
                "sensors": {"use": ["sensor_" + str(i) for i in range(11)]},
            },
            "transforms": {
                "pipeline": [
                    {"name": "zscore"},
                ]
            },
        })

        input_dim, output_dim = ONNXExporter._infer_dimensions(model_state, config)

        # Should use weight-based dimension when there's a mismatch
        # (This is the critical behavior - weights are ground truth)
        assert input_dim == 13
        assert output_dim == 11


class TestValidation:
    """Tests for ONNX export validation and dry-run."""

    def test_validate_passes_for_good_model(self):
        """Test that validation passes for a properly configured model."""
        from airtrace.export import ONNXExporter
        from omegaconf import OmegaConf

        model = DummyModel(input_dim=3, output_dim=3)
        model.eval()

        config = OmegaConf.create({
            "model": {"name": "dummy"},
            "data": {"window_size_in": 10},
        })

        exporter = ONNXExporter(model=model, config=config, transform_stats=None)
        results = exporter.validate(end_to_end=False, verbose=False)

        assert results['passed'] is True
        assert len(results['errors']) == 0

    def test_validate_fails_without_dimensions(self):
        """Test that validation fails when model lacks dimensions."""
        from airtrace.export import ONNXExporter
        from omegaconf import OmegaConf

        # Model without input_dim/output_dim attributes
        model = nn.Linear(3, 3)

        config = OmegaConf.create({
            "model": {"name": "dummy"},
            "data": {"window_size_in": 10},
        })

        exporter = ONNXExporter(model=model, config=config, transform_stats=None)
        results = exporter.validate(end_to_end=False, verbose=False)

        assert results['passed'] is False
        assert len(results['errors']) > 0

    def test_validate_fails_for_end_to_end_without_stats(self):
        """Test that validation fails for end-to-end without transform stats."""
        from airtrace.export import ONNXExporter
        from omegaconf import OmegaConf

        model = DummyModel(input_dim=3, output_dim=3)
        model.eval()

        config = OmegaConf.create({
            "model": {"name": "dummy"},
            "data": {"window_size_in": 10},
        })

        exporter = ONNXExporter(model=model, config=config, transform_stats=None)
        results = exporter.validate(end_to_end=True, verbose=False)

        assert results['passed'] is False
        assert any("transform" in str(e).lower() for e in results['errors'])

    def test_dry_run_succeeds(self):
        """Test that dry-run succeeds for a good model."""
        from airtrace.export import ONNXExporter
        from omegaconf import OmegaConf

        model = DummyModel(input_dim=3, output_dim=3)
        model.eval()

        config = OmegaConf.create({
            "model": {"name": "dummy"},
            "data": {"window_size_in": 10},
        })

        exporter = ONNXExporter(model=model, config=config, transform_stats=None)
        success = exporter.dry_run(end_to_end=False, verbose=False)

        assert success is True

    def test_dry_run_fails_for_bad_model(self):
        """Test that dry-run fails when model is misconfigured."""
        from airtrace.export import ONNXExporter
        from omegaconf import OmegaConf

        # Model without required attributes
        model = nn.Linear(3, 3)

        config = OmegaConf.create({
            "model": {"name": "dummy"},
            "data": {"window_size_in": 10},
        })

        exporter = ONNXExporter(model=model, config=config, transform_stats=None)
        success = exporter.dry_run(end_to_end=False, verbose=False)

        assert success is False


class TestONNXExport:
    """Tests for ONNX export functionality."""

    def test_export_model_only(self):
        """Test exporting model without transforms."""
        from airtrace.export import ONNXExporter
        from omegaconf import OmegaConf

        model = DummyModel(input_dim=3, output_dim=3)

        # Create minimal config
        config = OmegaConf.create({
            "model": {"name": "dummy", "input_dim": 3, "output_dim": 3},
            "data": {"window_size_in": 10},
        })

        exporter = ONNXExporter(model=model, config=config, transform_stats=None)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model.onnx"

            exported_files = exporter.export(
                output_path=output_path,
                end_to_end=False,
                batch_size=1,
                sequence_length=10,
                verbose=False,
            )

            assert output_path.exists()
            assert "onnx_model" in exported_files
            assert "metadata" in exported_files
            assert "config" in exported_files

    def test_export_with_transforms(self):
        """Test exporting model with transforms."""
        from airtrace.export import ONNXExporter
        from omegaconf import OmegaConf

        model = DummyModel(input_dim=3, output_dim=3)

        # Create transform statistics
        transform_stats = {
            'ZScoreTransform_0': {
                'scaler_y_mean': np.array([1.0, 2.0, 3.0]),
                'scaler_y_scale': np.array([0.5, 1.0, 1.5]),
            }
        }

        config = OmegaConf.create({
            "model": {"name": "dummy", "input_dim": 3, "output_dim": 3},
            "data": {"window_size_in": 10},
        })

        exporter = ONNXExporter(
            model=model,
            config=config,
            transform_stats=transform_stats,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model.onnx"

            exported_files = exporter.export(
                output_path=output_path,
                end_to_end=True,
                batch_size=1,
                sequence_length=10,
                verbose=False,
            )

            assert output_path.exists()
            assert "onnx_model" in exported_files

    def test_export_metadata(self):
        """Test that metadata is correctly saved."""
        from airtrace.export import ONNXExporter
        from omegaconf import OmegaConf

        model = DummyModel(input_dim=3, output_dim=3)

        config = OmegaConf.create({
            "model": {"name": "dummy", "input_dim": 3, "output_dim": 3},
            "data": {"window_size_in": 10},
        })

        exporter = ONNXExporter(model=model, config=config)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model.onnx"

            exported_files = exporter.export(
                output_path=output_path,
                batch_size=2,
                sequence_length=15,
                opset_version=13,
                verbose=False,
            )

            metadata_path = exported_files["metadata"]
            assert metadata_path.exists()

            with open(metadata_path) as f:
                metadata = json.load(f)

            assert metadata["input_shape"] == [2, 15, 3]
            assert metadata["output_dim"] == 3
            assert metadata["opset_version"] == 13


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
