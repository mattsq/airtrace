"""Tests for the MOMENT foundation model.

MOMENT is a pre-trained patch-based Transformer for time series forecasting.
These tests verify the model's integration with AirTrace's interface and
correct handling of various input scenarios.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from airtrace.models.moment import MomentInputNormalizer, MomentModel


# Mock the MOMENTPipeline to avoid downloading during tests
@pytest.fixture
def mock_moment_pipeline():
    """Mock MOMENT pipeline for testing without downloading checkpoints."""
    mock_pipeline = MagicMock()

    # Mock the __call__ method to return plausible predictions
    def mock_forward(inputs):
        B, T, D = inputs.shape
        # Return predictions of shape [B, pred_len, D]
        # Default pred_len=96 (will be parameterized in model init)
        pred_len = 96
        predictions = torch.randn(B, pred_len, D)
        return predictions

    mock_pipeline.__call__ = MagicMock(side_effect=mock_forward)
    mock_pipeline.named_parameters = MagicMock(
        return_value=[
            ("encoder.layer1.weight", torch.nn.Parameter(torch.randn(128, 128))),
            ("encoder.layer2.weight", torch.nn.Parameter(torch.randn(128, 128))),
            ("forecast_head.weight", torch.nn.Parameter(torch.randn(96, 128))),
        ]
    )
    mock_pipeline.to = MagicMock(return_value=mock_pipeline)
    mock_pipeline.init = MagicMock()

    return mock_pipeline


@pytest.fixture
def mock_moment_from_pretrained(mock_moment_pipeline):
    """Mock MOMENTPipeline.from_pretrained to return our mock pipeline."""
    # Mock the import inside _load_moment_pipeline method
    # We need to mock the momentfm module itself since it's imported dynamically
    mock_momentfm = MagicMock()
    mock_cls = MagicMock()
    mock_cls.from_pretrained = MagicMock(return_value=mock_moment_pipeline)
    mock_momentfm.MOMENTPipeline = mock_cls

    with patch.dict("sys.modules", {"momentfm": mock_momentfm}):
        yield mock_cls


def _build_model(mock_moment_from_pretrained, **overrides) -> MomentModel:
    """Build a MOMENT model for testing.

    Args:
        mock_moment_from_pretrained: Mocked MOMENTPipeline.from_pretrained fixture
        **overrides: Parameters to override defaults

    Returns:
        MomentModel instance
    """
    params = dict(
        input_dim=6,
        output_dim=3,
        pred_len=96,
        checkpoint="AutonLab/MOMENT-1-large",
        patch_size=16,
        context_length=512,
        normalize_inputs=True,
        freeze_backbone=False,
        device="cpu",
    )
    params.update(overrides)
    return MomentModel(**params)


def test_moment_normalizer_forward():
    """Normalizer should compute correct z-score statistics."""
    normalizer = MomentInputNormalizer(eps=1e-8)

    # Create input with known statistics
    x = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],  # Batch 0
            [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]],  # Batch 1
        ]
    )  # [2, 3, 2]

    normalized, mean, std = normalizer(x)

    # Check shapes
    assert normalized.shape == (2, 3, 2)
    assert mean.shape == (2, 1, 2)
    assert std.shape == (2, 1, 2)

    # Check normalization: normalized values should have mean≈0, std≈1
    assert torch.allclose(normalized.mean(dim=1), torch.zeros(2, 2), atol=1e-6)
    assert torch.allclose(normalized.std(dim=1), torch.ones(2, 2), atol=1e-6)


def test_moment_normalizer_inverse():
    """Normalizer inverse should correctly denormalize."""
    normalizer = MomentInputNormalizer()

    x = torch.randn(2, 10, 3)
    normalized, mean, std = normalizer(x)

    # Denormalize
    reconstructed = normalizer.inverse(normalized, mean, std)

    # Should recover original input
    assert torch.allclose(x, reconstructed, atol=1e-5)


def test_moment_forward_shapes(mock_moment_from_pretrained):
    """MOMENT should produce correct output shapes."""
    model = _build_model(mock_moment_from_pretrained)

    x = torch.randn(4, 512, 6)  # [B, T, D]
    output = model(x)

    # Check output structure
    assert "preds" in output
    assert "extras" in output

    # Check prediction shape
    preds = output["preds"]
    assert preds.shape == (4, 96, 3), f"Expected (4, 96, 3), got {preds.shape}"

    # Check extras
    extras = output["extras"]
    assert "representation" in extras


def test_moment_checkpoint_loading(mock_moment_from_pretrained):
    """MOMENT should load checkpoint from HuggingFace."""
    model = _build_model(mock_moment_from_pretrained, checkpoint="AutonLab/MOMENT-1-large")

    # Verify from_pretrained was called with correct arguments
    mock_moment_from_pretrained.from_pretrained.assert_called_once()
    call_args = mock_moment_from_pretrained.from_pretrained.call_args

    assert call_args[0][0] == "AutonLab/MOMENT-1-large"
    assert "model_kwargs" in call_args[1]
    assert call_args[1]["model_kwargs"]["task_name"] == "forecasting"

    # Verify model was created successfully
    assert model.checkpoint == "AutonLab/MOMENT-1-large"


def test_moment_multivariate_handling(mock_moment_from_pretrained):
    """MOMENT should handle multivariate inputs."""
    model = _build_model(mock_moment_from_pretrained, input_dim=10, output_dim=10)

    x = torch.randn(2, 512, 10)
    output = model(x)

    assert output["preds"].shape == (2, 96, 10)


def test_moment_input_padding(mock_moment_from_pretrained):
    """MOMENT should pad short inputs to context_length."""
    model = _build_model(mock_moment_from_pretrained, context_length=512)

    # Input shorter than context_length
    x = torch.randn(2, 50, 6)
    output = model(x)

    # Should not raise and should produce valid predictions
    assert output["preds"].shape == (2, 96, 3)


def test_moment_input_truncation(mock_moment_from_pretrained):
    """MOMENT should truncate long inputs to context_length."""
    model = _build_model(mock_moment_from_pretrained, context_length=512)

    # Input longer than context_length
    x = torch.randn(2, 1000, 6)
    output = model(x)

    # Should not raise and should produce valid predictions
    assert output["preds"].shape == (2, 96, 3)


def test_moment_normalization_applied(mock_moment_from_pretrained):
    """MOMENT should normalize inputs when enabled."""
    model = _build_model(mock_moment_from_pretrained, normalize_inputs=True)

    # Large unnormalized values
    x = torch.randn(2, 512, 6) * 1000
    output = model(x)

    # Should produce finite predictions
    assert torch.isfinite(output["preds"]).all()

    # Should include normalization stats in extras
    assert "normalization_mean" in output["extras"]
    assert "normalization_std" in output["extras"]


def test_moment_no_normalization(mock_moment_from_pretrained):
    """MOMENT should skip normalization when disabled."""
    model = _build_model(mock_moment_from_pretrained, normalize_inputs=False)

    x = torch.randn(2, 512, 6)
    output = model(x)

    # Should produce valid predictions
    assert output["preds"].shape == (2, 96, 3)

    # Should not include normalization stats
    assert "normalization_mean" not in output["extras"]
    assert "normalization_std" not in output["extras"]


def test_moment_freeze_backbone(mock_moment_from_pretrained):
    """MOMENT should freeze backbone parameters when requested."""
    model = _build_model(mock_moment_from_pretrained, freeze_backbone=True)

    # Count frozen parameters (excluding forecast head if train_forecast_head=True)
    frozen_params = [
        name for name, param in model.moment_pipeline.named_parameters() if not param.requires_grad
    ]

    # Should have some frozen parameters
    assert len(frozen_params) > 0


def test_moment_freeze_but_train_head(mock_moment_from_pretrained):
    """MOMENT should keep forecast head trainable when train_forecast_head=True."""
    model = _build_model(
        mock_moment_from_pretrained, freeze_backbone=True, train_forecast_head=True
    )

    # Forecast head parameters should be trainable
    head_params = [
        (name, param)
        for name, param in model.moment_pipeline.named_parameters()
        if "forecast" in name.lower()
    ]

    # If there are forecast head parameters, they should be trainable
    if head_params:
        for name, param in head_params:
            assert param.requires_grad, f"Forecast head param {name} should be trainable"


def test_moment_encode_decode_split(mock_moment_from_pretrained):
    """MOMENT should support encode/decode split for ResidualWrapperCompatible."""
    model = _build_model(mock_moment_from_pretrained)

    x = torch.randn(2, 512, 6)

    # Test encode
    latent, extras = model.encode(x)
    assert latent.shape[0] == 2  # Batch dimension preserved
    assert isinstance(extras, dict)

    # Test decode
    preds = model.decode(latent, pred_len=48, extras=extras)
    assert preds.shape == (2, 48, 3)


def test_moment_variable_pred_len(mock_moment_from_pretrained):
    """MOMENT should support variable prediction lengths via kwargs."""
    model = _build_model(mock_moment_from_pretrained, pred_len=96)

    x = torch.randn(2, 512, 6)

    # Override pred_len in forward
    output = model(x, pred_len=48)
    assert output["preds"].shape == (2, 48, 3)


def test_moment_input_dimension_mismatch(mock_moment_from_pretrained):
    """MOMENT should raise error on input dimension mismatch."""
    model = _build_model(mock_moment_from_pretrained, input_dim=6)

    # Input with wrong dimension
    x = torch.randn(2, 512, 10)  # Expected 6, got 10

    with pytest.raises(ValueError, match="Input dimension mismatch"):
        model(x)


def test_moment_device_placement(mock_moment_from_pretrained):
    """MOMENT should support device placement."""
    # Test CPU placement
    model = _build_model(mock_moment_from_pretrained, device="cpu")
    assert model.device_str == "cpu"

    # Test CUDA placement (device_str is set even if actual placement may fail)
    model_cuda = _build_model(mock_moment_from_pretrained, device="cuda")
    assert model_cuda.device_str == "cuda"


def test_moment_parameter_counting(mock_moment_from_pretrained):
    """MOMENT should correctly count total and trainable parameters."""
    model = _build_model(mock_moment_from_pretrained, freeze_backbone=False)

    total_params = model.get_num_params()
    trainable_params = model._count_trainable_params()

    # All params should be trainable when freeze_backbone=False
    assert total_params > 0
    assert trainable_params > 0

    # When freeze_backbone=True, trainable should be less
    model_frozen = _build_model(mock_moment_from_pretrained, freeze_backbone=True)
    frozen_trainable = model_frozen._count_trainable_params()
    assert frozen_trainable < total_params


def test_moment_repr(mock_moment_from_pretrained):
    """MOMENT should have informative string representation."""
    model = _build_model(mock_moment_from_pretrained)

    repr_str = repr(model)

    # Should include key parameters
    assert "MomentModel" in repr_str
    assert "input_dim=6" in repr_str
    assert "output_dim=3" in repr_str
    assert "pred_len=96" in repr_str
    assert "AutonLab/MOMENT-1-large" in repr_str


def test_moment_zero_pred_len_raises(mock_moment_from_pretrained):
    """MOMENT should raise error for non-positive pred_len."""
    model = _build_model(mock_moment_from_pretrained)
    x = torch.randn(2, 512, 6)
    latent, extras = model.encode(x)

    with pytest.raises(ValueError, match="pred_len must be positive"):
        model.decode(latent, pred_len=0, extras=extras)


def test_moment_different_output_dim(mock_moment_from_pretrained):
    """MOMENT should handle output_dim different from input_dim."""
    # Mock pipeline that returns input_dim features
    model = _build_model(mock_moment_from_pretrained, input_dim=10, output_dim=5, pred_len=48)

    x = torch.randn(2, 512, 10)
    output = model(x)

    # Should trim to output_dim
    assert output["preds"].shape == (2, 48, 5)


def test_moment_batch_size_one(mock_moment_from_pretrained):
    """MOMENT should handle batch size of 1."""
    model = _build_model(mock_moment_from_pretrained)

    x = torch.randn(1, 512, 6)
    output = model(x)

    assert output["preds"].shape == (1, 96, 3)


def test_moment_large_batch(mock_moment_from_pretrained):
    """MOMENT should handle large batch sizes."""
    model = _build_model(mock_moment_from_pretrained)

    x = torch.randn(64, 512, 6)
    output = model(x)

    assert output["preds"].shape == (64, 96, 3)


def test_moment_import_error():
    """MOMENT should raise informative error if momentfm not installed."""
    # Patch the import to raise ImportError
    with patch("airtrace.models.moment.MomentModel._load_moment_pipeline") as mock_load:
        mock_load.side_effect = ImportError(
            "MOMENT requires the 'momentfm' package. " "Install it with: pip install momentfm"
        )

        with pytest.raises(ImportError, match="momentfm"):
            MomentModel(input_dim=6, output_dim=3)


def test_moment_patch_size_divisibility(mock_moment_from_pretrained):
    """MOMENT should pad inputs to be divisible by patch_size."""
    model = _build_model(mock_moment_from_pretrained, patch_size=16, context_length=100)

    x = torch.randn(2, 100, 6)
    output = model(x)

    # Should not raise despite 100 not being divisible by 16
    assert output["preds"].shape == (2, 96, 3)


def test_moment_extras_representation(mock_moment_from_pretrained):
    """MOMENT should include representation in extras."""
    model = _build_model(mock_moment_from_pretrained)

    x = torch.randn(2, 512, 6)
    output = model(x)

    assert "representation" in output["extras"]
    assert output["extras"]["representation"].shape[0] == 2  # Batch size


def test_moment_normalization_inverse_with_different_dims(mock_moment_from_pretrained):
    """MOMENT should handle denormalization when output_dim != input_dim."""
    model = _build_model(
        mock_moment_from_pretrained,
        input_dim=6,
        output_dim=3,
        normalize_inputs=True,
    )

    x = torch.randn(2, 512, 6) * 100
    output = model(x)

    # Should produce finite predictions even with dimension mismatch
    assert torch.isfinite(output["preds"]).all()
    assert output["preds"].shape == (2, 96, 3)


def test_moment_context_parameter_ignored(mock_moment_from_pretrained):
    """MOMENT should ignore the context parameter (not used)."""
    model = _build_model(mock_moment_from_pretrained)

    x = torch.randn(2, 512, 6)
    context = torch.randn(2, 512, 4)  # Dummy context

    # Should not raise even though context is provided
    output = model(x, context=context)
    assert output["preds"].shape == (2, 96, 3)
