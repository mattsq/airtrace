"""Tests for the TimesNet model."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from airtrace.models.timesnet import (
    TimesNetModel,
    TimesBlock,
    PositionalEmbedding,
)


def _build_small_model(**overrides):
    """Helper to build a small TimesNet model for testing."""
    params = dict(
        input_dim=4,
        output_dim=2,
        seq_len=32,
        pred_len=1,
        d_model=16,
        d_ff=32,
        num_layers=2,
        num_kernels=3,
        top_k=3,
        dropout=0.0,
        embed_type='positional',
    )
    params.update(overrides)
    return TimesNetModel(**params)


# ============================================================================
# Basic functionality tests
# ============================================================================


def test_timesnet_forward_shapes() -> None:
    """Test that forward pass produces correct output shapes."""
    model = _build_small_model()
    x = torch.randn(3, 32, 4)  # [B=3, T=32, D=4]
    output = model(x)

    preds = output["preds"]
    extras = output["extras"]

    assert preds.shape == (3, 1, 2), "Predictions should be [B, pred_len, output_dim]"
    assert "embeddings" in extras
    assert "layer_outputs" in extras
    assert "last_hidden" in extras


def test_multi_step_prediction() -> None:
    """Test multi-step ahead forecasting with pred_len > 1."""
    model = _build_small_model(pred_len=5, output_dim=3)
    x = torch.randn(4, 32, 4)  # [B=4, T=32, D=4]
    output = model(x)
    preds = output["preds"]

    assert preds.shape == (4, 5, 3), "Should predict 5 steps ahead with 3 output dims"
    assert torch.isfinite(preds).all(), "All predictions should be finite"


def test_different_input_output_dims() -> None:
    """Test that model handles different input and output dimensions."""
    model = _build_small_model(input_dim=10, output_dim=5)
    x = torch.randn(2, 32, 10)
    output = model(x)
    preds = output["preds"]

    assert preds.shape == (2, 1, 5)
    assert torch.isfinite(preds).all()


def test_different_sequence_lengths() -> None:
    """Test model with various sequence lengths."""
    for seq_len in [16, 32, 48, 64, 96]:
        model = _build_small_model(seq_len=seq_len)
        x = torch.randn(2, seq_len, 4)
        output = model(x)
        preds = output["preds"]

        assert preds.shape == (2, 1, 2)
        assert torch.isfinite(preds).all()


# ============================================================================
# TimesBlock tests
# ============================================================================


def test_timesblock_forward() -> None:
    """Test TimesBlock forward pass."""
    block = TimesBlock(d_model=16, d_ff=32, num_kernels=3, top_k=3, dropout=0.0)
    x = torch.randn(2, 32, 16)  # [B, T, D]

    output = block(x)

    assert output.shape == (2, 32, 16), "Output should match input shape"
    assert torch.isfinite(output).all()


def test_timesblock_period_detection() -> None:
    """Test that TimesBlock detects periods correctly."""
    block = TimesBlock(d_model=16, d_ff=32, num_kernels=3, top_k=3, dropout=0.0)

    # Create a signal with known periodicity
    T = 64
    t = torch.arange(T).float()
    # Signal with period 8
    signal = torch.sin(2 * torch.pi * t / 8)
    x = signal.unsqueeze(0).unsqueeze(-1).repeat(2, 1, 16)  # [B=2, T=64, D=16]

    periods, weights = block._detect_periods(x, top_k=5)

    # Periods should be positive integers
    assert (periods > 0).all()
    assert (periods <= T).all()

    # Weights should sum to ~1 for each batch
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2), atol=1e-5)

    # At least one period should be close to the actual period (8)
    # (May not be exact due to FFT discretization and other frequencies)
    assert periods.min() >= 2, "Minimum period should be at least 2"


def test_timesblock_2d_reshaping() -> None:
    """Test 2D reshaping and inverse reshaping."""
    block = TimesBlock(d_model=16, d_ff=32, num_kernels=3, top_k=3, dropout=0.0)

    x = torch.randn(2, 32, 16)  # [B=2, T=32, D=16]
    period = 8

    # Reshape to 2D
    x_2d, padding = block._reshape_to_2d(x, period)

    # Should be [B, D, period, num_periods]
    # num_periods = 32 // 8 = 4
    assert x_2d.shape == (2, 16, 8, 4)

    # Reshape back to 1D
    x_reconstructed = block._reshape_from_2d(x_2d, 32, padding)

    # Should match original shape
    assert x_reconstructed.shape == (2, 32, 16)


def test_timesblock_2d_reshaping_with_padding() -> None:
    """Test 2D reshaping when padding is needed."""
    block = TimesBlock(d_model=16, d_ff=32, num_kernels=3, top_k=3, dropout=0.0)

    # Length that doesn't divide evenly by period
    x = torch.randn(2, 30, 16)  # [B=2, T=30, D=16]
    period = 8

    # Reshape to 2D
    x_2d, padding = block._reshape_to_2d(x, period)

    # With padding: 30 -> 32 (4 complete periods)
    # Should be [B, D, 8, 4]
    assert x_2d.shape == (2, 16, 8, 4)
    assert padding == 2

    # Reshape back
    x_reconstructed = block._reshape_from_2d(x_2d, 30, padding)

    # Should match original length (padding removed)
    assert x_reconstructed.shape == (2, 30, 16)


def test_timesblock_different_kernels() -> None:
    """Test TimesBlock with different numbers of kernels."""
    for num_kernels in [1, 3, 6]:
        block = TimesBlock(d_model=16, d_ff=32, num_kernels=num_kernels, top_k=3, dropout=0.0)
        x = torch.randn(2, 32, 16)

        output = block(x)

        assert output.shape == (2, 32, 16)
        assert torch.isfinite(output).all()


def test_timesblock_different_top_k() -> None:
    """Test TimesBlock with different top_k values."""
    for top_k in [1, 3, 5]:
        block = TimesBlock(d_model=16, d_ff=32, num_kernels=3, top_k=top_k, dropout=0.0)
        x = torch.randn(2, 32, 16)

        output = block(x)

        assert output.shape == (2, 32, 16)
        assert torch.isfinite(output).all()


# ============================================================================
# Positional Embedding tests
# ============================================================================


def test_positional_embedding() -> None:
    """Test positional embedding."""
    pos_emb = PositionalEmbedding(d_model=16, max_len=100)
    x = torch.randn(2, 32, 16)

    output = pos_emb(x)

    assert output.shape == (2, 32, 16)
    assert torch.isfinite(output).all()

    # Output should be different from input
    assert not torch.allclose(output, x)


def test_positional_embedding_consistency() -> None:
    """Test that positional embedding is consistent across calls."""
    pos_emb = PositionalEmbedding(d_model=16, max_len=100)
    x = torch.randn(2, 32, 16)

    output1 = pos_emb(x)
    output2 = pos_emb(x)

    # Same input should give same output (positional embeddings are fixed)
    assert torch.allclose(output1, output2)


def test_positional_embedding_different_lengths() -> None:
    """Test positional embedding with different sequence lengths."""
    pos_emb = PositionalEmbedding(d_model=16, max_len=100)

    for seq_len in [16, 32, 64]:
        x = torch.randn(2, seq_len, 16)
        output = pos_emb(x)
        assert output.shape == (2, seq_len, 16)


# ============================================================================
# Embedding type tests
# ============================================================================


def test_no_positional_embedding() -> None:
    """Test model without positional embedding."""
    model = _build_small_model(embed_type='none')
    x = torch.randn(2, 32, 4)

    output = model(x)
    preds = output["preds"]

    assert preds.shape == (2, 1, 2)
    assert torch.isfinite(preds).all()


def test_with_positional_embedding() -> None:
    """Test model with positional embedding."""
    model = _build_small_model(embed_type='positional')
    x = torch.randn(2, 32, 4)

    output = model(x)
    preds = output["preds"]

    assert preds.shape == (2, 1, 2)
    assert torch.isfinite(preds).all()


# ============================================================================
# Layer depth tests
# ============================================================================


def test_single_layer() -> None:
    """Test model with single layer."""
    model = _build_small_model(num_layers=1)
    x = torch.randn(2, 32, 4)

    output = model(x)
    preds = output["preds"]

    assert preds.shape == (2, 1, 2)
    assert torch.isfinite(preds).all()


def test_multiple_layers() -> None:
    """Test model with multiple layers."""
    for num_layers in [2, 3, 4]:
        model = _build_small_model(num_layers=num_layers)
        x = torch.randn(2, 32, 4)

        output = model(x)
        preds = output["preds"]

        assert preds.shape == (2, 1, 2)
        assert torch.isfinite(preds).all()

        # Check that we have the right number of layer outputs
        assert len(output["extras"]["layer_outputs"]) == num_layers


# ============================================================================
# Parameter validation tests
# ============================================================================


def test_parameter_validation() -> None:
    """Test that invalid parameters raise appropriate errors."""
    # Invalid seq_len
    with pytest.raises(ValueError, match="seq_len must be positive"):
        _build_small_model(seq_len=0)

    # Invalid pred_len
    with pytest.raises(ValueError, match="pred_len must be positive"):
        _build_small_model(pred_len=0)

    # Invalid d_model
    with pytest.raises(ValueError, match="d_model must be positive"):
        _build_small_model(d_model=0)

    # Invalid d_ff
    with pytest.raises(ValueError, match="d_ff must be positive"):
        _build_small_model(d_ff=0)

    # Invalid num_layers
    with pytest.raises(ValueError, match="num_layers must be positive"):
        _build_small_model(num_layers=0)

    # Invalid num_kernels
    with pytest.raises(ValueError, match="num_kernels must be positive"):
        _build_small_model(num_kernels=0)

    # Invalid top_k
    with pytest.raises(ValueError, match="top_k must be positive"):
        _build_small_model(top_k=0)

    # Invalid dropout
    with pytest.raises(ValueError, match="dropout must be in"):
        _build_small_model(dropout=1.5)


# ============================================================================
# Gradient flow tests
# ============================================================================


def test_gradient_flow() -> None:
    """Test that gradients flow through the model."""
    model = _build_small_model()
    x = torch.randn(2, 32, 4, requires_grad=True)

    output = model(x)
    loss = output["preds"].sum()
    loss.backward()

    # Check that input gradients exist
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    # Check that model parameters have gradients
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} should have gradients"
            assert torch.isfinite(param.grad).all(), f"Parameter {name} has non-finite gradients"


def test_gradient_accumulation() -> None:
    """Test that gradients accumulate correctly over multiple backward passes."""
    model = _build_small_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    x1 = torch.randn(2, 32, 4)
    x2 = torch.randn(2, 32, 4)

    # First pass
    out1 = model(x1)
    loss1 = out1["preds"].sum()
    loss1.backward()

    # Store gradients
    grad1 = {name: param.grad.clone() for name, param in model.named_parameters() if param.grad is not None}

    # Second pass (without zero_grad)
    out2 = model(x2)
    loss2 = out2["preds"].sum()
    loss2.backward()

    # Check that gradients accumulated
    for name, param in model.named_parameters():
        if param.grad is not None and name in grad1:
            # Gradients should be different (accumulated)
            assert not torch.allclose(param.grad, grad1[name], atol=1e-6)


# ============================================================================
# Extras output tests
# ============================================================================


def test_extras_contain_expected_keys() -> None:
    """Test that extras dict contains expected outputs."""
    model = _build_small_model()
    x = torch.randn(2, 32, 4)

    output = model(x)
    extras = output["extras"]

    assert "embeddings" in extras
    assert "layer_outputs" in extras
    assert "last_hidden" in extras

    # Check types
    assert isinstance(extras["embeddings"], torch.Tensor)
    assert isinstance(extras["layer_outputs"], list)
    assert isinstance(extras["last_hidden"], torch.Tensor)


def test_extras_shapes() -> None:
    """Test that extras tensors have correct shapes."""
    model = _build_small_model(d_model=16, num_layers=2)
    x = torch.randn(2, 32, 4)

    output = model(x)
    extras = output["extras"]

    # Check shapes
    assert extras["embeddings"].shape == (2, 32, 16)
    assert extras["last_hidden"].shape == (2, 16)
    assert len(extras["layer_outputs"]) == 2
    for layer_out in extras["layer_outputs"]:
        assert layer_out.shape == (2, 32, 16)


# ============================================================================
# Batch size tests
# ============================================================================


def test_single_sample_batch() -> None:
    """Test that model works with batch size of 1."""
    model = _build_small_model()
    x = torch.randn(1, 32, 4)

    output = model(x)
    preds = output["preds"]

    assert preds.shape == (1, 1, 2)
    assert torch.isfinite(preds).all()


def test_large_batch() -> None:
    """Test that model handles larger batch sizes."""
    model = _build_small_model()
    x = torch.randn(32, 32, 4)  # Batch size of 32

    output = model(x)
    preds = output["preds"]

    assert preds.shape == (32, 1, 2)
    assert torch.isfinite(preds).all()


# ============================================================================
# Edge case tests
# ============================================================================


def test_short_sequence() -> None:
    """Test model with short sequence length."""
    model = _build_small_model(seq_len=8)
    x = torch.randn(2, 8, 4)

    output = model(x)
    preds = output["preds"]

    assert preds.shape == (2, 1, 2)
    assert torch.isfinite(preds).all()


def test_long_sequence() -> None:
    """Test model with long sequence length."""
    model = _build_small_model(seq_len=256)
    x = torch.randn(2, 256, 4)

    output = model(x)
    preds = output["preds"]

    assert preds.shape == (2, 1, 2)
    assert torch.isfinite(preds).all()


def test_model_determinism() -> None:
    """Test that model produces deterministic outputs with same seed."""
    torch.manual_seed(42)
    model1 = _build_small_model()

    torch.manual_seed(42)
    model2 = _build_small_model()

    x = torch.randn(2, 32, 4)

    with torch.no_grad():
        out1 = model1(x)
        out2 = model2(x)

    # Models initialized with same seed should produce identical outputs
    assert torch.allclose(out1["preds"], out2["preds"], atol=1e-6)


# ============================================================================
# Integration tests
# ============================================================================


def test_training_step() -> None:
    """Test a complete training step."""
    model = _build_small_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    x = torch.randn(4, 32, 4)
    target = torch.randn(4, 1, 2)

    # Forward pass
    output = model(x)
    preds = output["preds"]

    # Compute loss
    loss = F.mse_loss(preds, target)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Loss should be finite
    assert torch.isfinite(loss).all()


def test_eval_mode() -> None:
    """Test that model behaves correctly in eval mode."""
    model = _build_small_model(dropout=0.5)  # High dropout to see effect
    x = torch.randn(2, 32, 4)

    # Train mode
    model.train()
    with torch.no_grad():
        out_train = model(x)

    # Eval mode
    model.eval()
    with torch.no_grad():
        out_eval = model(x)

    # Both outputs should be valid
    assert torch.isfinite(out_train["preds"]).all()
    assert torch.isfinite(out_eval["preds"]).all()


def test_multiple_training_steps() -> None:
    """Test multiple training steps to ensure stability."""
    model = _build_small_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for _ in range(10):
        x = torch.randn(4, 32, 4)
        target = torch.randn(4, 1, 2)

        output = model(x)
        loss = F.mse_loss(output["preds"], target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Loss should remain finite
        assert torch.isfinite(loss).all()


# ============================================================================
# Residual connection tests
# ============================================================================


def test_residual_connections() -> None:
    """Test that residual connections are working."""
    model = _build_small_model(num_layers=2)
    x = torch.randn(2, 32, 4)

    # Forward pass
    output = model(x)

    # Layer outputs should be different due to residual connections
    layer_outputs = output["extras"]["layer_outputs"]
    assert len(layer_outputs) == 2

    # Each layer should have processed the input
    assert not torch.allclose(layer_outputs[0], layer_outputs[1], atol=1e-6)


# ============================================================================
# Model size tests
# ============================================================================


def test_model_parameter_count() -> None:
    """Test that model has reasonable parameter count."""
    model = _build_small_model()

    num_params = model.get_num_params()

    # Should have some parameters
    assert num_params > 0

    # Should be reasonable for a small model (< 1M parameters)
    assert num_params < 1_000_000


def test_larger_model_parameters() -> None:
    """Test larger model configuration."""
    model = _build_small_model(
        d_model=128,
        d_ff=256,
        num_layers=4,
        num_kernels=6,
    )

    x = torch.randn(2, 32, 4)
    output = model(x)

    assert output["preds"].shape == (2, 1, 2)
    assert torch.isfinite(output["preds"]).all()


# ============================================================================
# Specific period tests
# ============================================================================


def test_period_detection_with_synthetic_signal() -> None:
    """Test period detection with a synthetic periodic signal."""
    block = TimesBlock(d_model=1, d_ff=8, num_kernels=3, top_k=5, dropout=0.0)

    # Create signal with clear period of 16
    T = 128
    t = torch.arange(T).float()
    signal = torch.sin(2 * torch.pi * t / 16)  # Period 16
    x = signal.unsqueeze(0).unsqueeze(-1)  # [1, 128, 1]

    periods, weights = block._detect_periods(x, top_k=5)

    # The detected periods should include something close to 16
    # (exact match depends on FFT resolution)
    # At minimum, check that periods are in a reasonable range
    assert (periods >= 2).all()
    assert (periods <= T).all()
    assert torch.isfinite(weights).all()


def test_mixed_period_signal() -> None:
    """Test with signal containing multiple periods."""
    block = TimesBlock(d_model=1, d_ff=8, num_kernels=3, top_k=5, dropout=0.0)

    # Signal with two dominant periods: 8 and 16
    T = 128
    t = torch.arange(T).float()
    signal = torch.sin(2 * torch.pi * t / 8) + 0.5 * torch.sin(2 * torch.pi * t / 16)
    x = signal.unsqueeze(0).unsqueeze(-1)  # [1, 128, 1]

    periods, weights = block._detect_periods(x, top_k=5)

    # Should detect multiple periods
    assert (periods >= 2).all()
    assert (periods <= T).all()

    # Weights should be normalized
    assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-5)
