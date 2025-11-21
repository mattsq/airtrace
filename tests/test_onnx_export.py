"""Tests for ONNX export functionality."""

import json
import tempfile
from pathlib import Path

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

        wrapper = create_transform_wrapper(stats, 'ZScoreTransform_0')

        assert wrapper is not None
        assert isinstance(wrapper, ZScoreWrapper)

    def test_unsupported_transform_returns_none(self):
        """Test that unsupported transforms return None."""
        stats = {}
        wrapper = create_transform_wrapper(stats, 'UnsupportedTransform_0')

        assert wrapper is None


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


@pytest.mark.skipif(
    not torch.onnx.is_onnx_available(),
    reason="ONNX export not available"
)
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
