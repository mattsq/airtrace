"""Tests for the Timer foundation model.

Timer is a pre-trained GPT-style Transformer for time series forecasting.
These tests verify the model's integration with AirTrace's interface and
correct handling of various input scenarios.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from airtrace.models.timer import TimerModel, TimerInputNormalizer


# Mock the HuggingFace model to avoid downloading during tests
@pytest.fixture
def mock_timer_backbone():
    """Mock Timer backbone for testing without downloading checkpoints."""
    mock_model = MagicMock()

    # Mock the generate method to return plausible predictions
    def mock_generate(inputs, max_new_tokens=24):
        B, T = inputs.shape
        # Return input sequence + generated tokens
        generated = torch.randn(B, T + max_new_tokens)
        return generated

    mock_model.generate = MagicMock(side_effect=mock_generate)
    mock_model.named_parameters = MagicMock(return_value=[
        ("layer1.weight", torch.nn.Parameter(torch.randn(10, 10))),
        ("layer2.weight", torch.nn.Parameter(torch.randn(10, 10))),
    ])
    mock_model.to = MagicMock(return_value=mock_model)

    return mock_model


@pytest.fixture
def mock_automodel(mock_timer_backbone):
    """Mock AutoModelForCausalLM to return our mock model."""
    with patch("transformers.AutoModelForCausalLM") as mock_cls:
        mock_cls.from_pretrained = MagicMock(return_value=mock_timer_backbone)
        yield mock_cls


def _build_model(mock_automodel, **overrides) -> TimerModel:
    """Build a Timer model for testing.

    Args:
        mock_automodel: Mocked AutoModelForCausalLM fixture
        **overrides: Parameters to override defaults

    Returns:
        TimerModel instance
    """
    params = dict(
        input_dim=6,
        output_dim=3,
        pred_len=24,
        checkpoint="thuml/timer-base-84m",
        lookback_length=128,
        normalize_inputs=True,
        freeze_backbone=False,
        device="cpu",
    )
    params.update(overrides)
    return TimerModel(**params)


def test_timer_normalizer_forward():
    """Normalizer should compute correct z-score statistics."""
    normalizer = TimerInputNormalizer(eps=1e-8)

    # Create input with known statistics
    x = torch.tensor([
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],  # Batch 0
        [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]],  # Batch 1
    ])  # [2, 3, 2]

    normalized, mean, std = normalizer(x)

    # Check shapes
    assert normalized.shape == (2, 3, 2)
    assert mean.shape == (2, 1, 2)
    assert std.shape == (2, 1, 2)

    # Check normalization: normalized values should have mean≈0, std≈1
    assert torch.allclose(normalized.mean(dim=1), torch.zeros(2, 2), atol=1e-6)
    assert torch.allclose(normalized.std(dim=1), torch.ones(2, 2), atol=1e-6)


def test_timer_normalizer_inverse():
    """Normalizer inverse should correctly denormalize."""
    normalizer = TimerInputNormalizer()

    x = torch.randn(2, 10, 3)
    normalized, mean, std = normalizer(x)

    # Denormalize
    reconstructed = normalizer.inverse(normalized, mean, std)

    # Should recover original input
    assert torch.allclose(x, reconstructed, atol=1e-5)


def test_timer_forward_shapes(mock_automodel):
    """Timer should produce correct output shapes."""
    model = _build_model(mock_automodel)

    x = torch.randn(4, 128, 6)  # [B, T, D]
    output = model(x)

    # Check output structure
    assert "preds" in output
    assert "extras" in output

    # Check prediction shape
    preds = output["preds"]
    assert preds.shape == (4, 24, 3), f"Expected (4, 24, 3), got {preds.shape}"

    # Check extras
    extras = output["extras"]
    assert "raw_preds" in extras
    assert extras["raw_preds"].shape == (4, 24, 3)


def test_timer_checkpoint_loading(mock_automodel):
    """Timer should load checkpoint from HuggingFace."""
    model = _build_model(
        mock_automodel,
        checkpoint="thuml/timer-base-84m"
    )

    # Verify AutoModelForCausalLM was called correctly
    mock_automodel.from_pretrained.assert_called_once()
    call_kwargs = mock_automodel.from_pretrained.call_args[1]

    assert call_kwargs["trust_remote_code"] is True
    assert "device_map" not in call_kwargs
    mock_automodel.from_pretrained.return_value.to.assert_called_once_with(
        torch.device("cpu")
    )

    # Model should have backbone
    assert model.timer_backbone is not None


def test_timer_normalization_applied(mock_automodel):
    """Timer should normalize inputs when enabled."""
    model = _build_model(mock_automodel, normalize_inputs=True)

    # Large unnormalized values
    x = torch.randn(2, 64, 6) * 1000
    output = model(x)

    # Should produce finite predictions
    assert torch.isfinite(output["preds"]).all()

    # Should have normalization stats in extras
    assert "normalization_mean" in output["extras"]
    assert "normalization_std" in output["extras"]


def test_timer_normalization_disabled(mock_automodel):
    """Timer should skip normalization when disabled."""
    model = _build_model(mock_automodel, normalize_inputs=False)

    x = torch.randn(2, 64, 6)
    output = model(x)

    # Should not have normalization stats
    assert "normalization_mean" not in output["extras"]
    assert "normalization_std" not in output["extras"]


def test_timer_multivariate_handling(mock_automodel):
    """Timer should handle multivariate inputs correctly."""
    # 10 input dimensions -> 10 output dimensions
    model = _build_model(
        mock_automodel,
        input_dim=10,
        output_dim=10,
        pred_len=12
    )

    x = torch.randn(3, 256, 10)
    output = model(x)

    # Each dimension should have predictions
    assert output["preds"].shape == (3, 12, 10)


def test_timer_different_output_dim(mock_automodel):
    """Timer should handle different input/output dimensions."""
    # 8 input dims, 4 output dims
    model = _build_model(
        mock_automodel,
        input_dim=8,
        output_dim=4,
        pred_len=16
    )

    x = torch.randn(2, 128, 8)
    output = model(x)

    # Should output only first 4 dimensions
    assert output["preds"].shape == (2, 16, 4)


def test_timer_freeze_backbone(mock_automodel):
    """Timer should freeze backbone parameters when requested."""
    model = _build_model(mock_automodel, freeze_backbone=True)

    # Count frozen parameters
    frozen_params = [
        name for name, param in model.named_parameters()
        if not param.requires_grad
    ]

    # Should have frozen parameters
    assert len(frozen_params) > 0, "Expected frozen parameters in backbone"

    # Trainable param count should be less than total
    assert model._count_trainable_params() < model.get_num_params()


def test_timer_input_truncation(mock_automodel):
    """Timer should truncate long inputs to lookback_length."""
    model = _build_model(mock_automodel, lookback_length=64)

    # Input longer than lookback_length
    x = torch.randn(2, 200, 6)

    # Should not raise
    output = model(x)

    # Should produce correct output
    assert output["preds"].shape == (2, 24, 3)


def test_timer_input_padding(mock_automodel):
    """Timer should pad short inputs to lookback_length."""
    model = _build_model(mock_automodel, lookback_length=128)

    # Input shorter than lookback_length
    x = torch.randn(2, 32, 6)

    # Should not raise
    output = model(x)

    # Should produce correct output
    assert output["preds"].shape == (2, 24, 3)


def test_timer_input_dimension_mismatch(mock_automodel):
    """Timer should raise error on input dimension mismatch."""
    model = _build_model(mock_automodel, input_dim=6)

    # Wrong input dimension
    x = torch.randn(2, 64, 10)  # Expected 6, got 10

    with pytest.raises(ValueError, match="Input dimension mismatch"):
        model(x)


def test_timer_variable_pred_len(mock_automodel):
    """Timer should support different prediction horizons."""
    # Short horizon
    model_short = _build_model(mock_automodel, pred_len=5)
    x = torch.randn(2, 64, 6)
    output_short = model_short(x)
    assert output_short["preds"].shape == (2, 5, 3)

    # Long horizon
    model_long = _build_model(mock_automodel, pred_len=48)
    output_long = model_long(x)
    assert output_long["preds"].shape == (2, 48, 3)


def test_timer_trust_remote_code_warning(mock_automodel):
    """Timer should warn if trust_remote_code is disabled."""
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        model = _build_model(
            mock_automodel,
            trust_remote_code=False
        )

        # Should have issued a warning
        assert len(w) > 0
        assert "trust_remote_code" in str(w[0].message)


def test_timer_repr(mock_automodel):
    """Timer __repr__ should display key parameters."""
    model = _build_model(mock_automodel)

    repr_str = repr(model)

    # Should contain key information
    assert "TimerModel" in repr_str
    assert "checkpoint" in repr_str
    assert "input_dim=6" in repr_str
    assert "output_dim=3" in repr_str
    assert "pred_len=24" in repr_str


def test_timer_context_ignored(mock_automodel):
    """Timer should ignore context parameter (not used)."""
    model = _build_model(mock_automodel)

    x = torch.randn(2, 64, 6)
    context = torch.randn(2, 10)  # Some context

    # Should not raise
    output = model(x, context=context)

    # Should produce normal output
    assert output["preds"].shape == (2, 24, 3)


def test_timer_batch_size_one(mock_automodel):
    """Timer should handle single-sample batches."""
    model = _build_model(mock_automodel)

    x = torch.randn(1, 128, 6)  # Batch size 1
    output = model(x)

    assert output["preds"].shape == (1, 24, 3)


def test_timer_large_batch(mock_automodel):
    """Timer should handle large batches."""
    model = _build_model(mock_automodel)

    x = torch.randn(32, 128, 6)  # Large batch
    output = model(x)

    assert output["preds"].shape == (32, 24, 3)


def test_timer_device_parameter(mock_automodel):
    """Timer should respect device parameter."""
    model = _build_model(mock_automodel, device="cpu")

    # Verify device was passed to HuggingFace loader
    call_kwargs = mock_automodel.from_pretrained.call_args[1]
    assert "device_map" not in call_kwargs
    mock_automodel.from_pretrained.return_value.to.assert_called_once_with(
        torch.device("cpu")
    )


def test_timer_from_config():
    """Timer should build from config dictionary."""
    from airtrace.models.registry import build_model

    config = {
        "name": "timer",
        "params": {
            "pred_len": 24,
            "checkpoint": "thuml/timer-base-84m",
            "lookback_length": 256,
            "normalize_inputs": True,
        }
    }

    with patch("transformers.AutoModelForCausalLM"):
        model = build_model(config, input_dim=6, output_dim=3)

        assert isinstance(model, TimerModel)
        assert model.pred_len == 24
        assert model.input_dim == 6
        assert model.output_dim == 3


def test_timer_lora_warning(mock_automodel):
    """Timer should warn about unimplemented LoRA."""
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        model = _build_model(
            mock_automodel,
            lora_rank=8  # Enable LoRA
        )

        # Should have issued a warning about LoRA not being implemented
        lora_warnings = [warning for warning in w if "LoRA" in str(warning.message)]
        assert len(lora_warnings) > 0


def test_timer_imports_error():
    """Timer should raise helpful error if transformers not installed."""
    # Mock the import to fail
    import sys
    with patch.dict(sys.modules, {'transformers': None}):
        with pytest.raises(ImportError, match="transformers"):
            model = TimerModel(
                input_dim=6,
                output_dim=3,
                checkpoint="thuml/timer-base-84m"
            )


def test_timer_parameter_count(mock_automodel):
    """Timer should correctly count parameters."""
    model = _build_model(mock_automodel)

    total_params = model.get_num_params()
    trainable_params = model._count_trainable_params()

    # Should have positive parameter counts
    assert total_params > 0
    assert trainable_params > 0

    # When not frozen, should be equal
    assert total_params == trainable_params


def test_timer_frozen_parameter_count(mock_automodel):
    """Timer should have fewer trainable params when frozen."""
    model = _build_model(mock_automodel, freeze_backbone=True)

    total_params = model.get_num_params()
    trainable_params = model._count_trainable_params()

    # Trainable should be less than total when frozen
    assert trainable_params < total_params
