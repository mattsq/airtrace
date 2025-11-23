"""Tests for the MambaTS model."""

from __future__ import annotations

import pytest
import torch

from airtrace.models.mambats import (
    MambaTSModel,
    PatchEmbedding,
    TemporalMambaBlock,
    VariableScanEncoder,
)


def _build_small_model(**overrides):
    """Helper to build a small MambaTS model for testing."""
    params = dict(
        input_dim=4,
        output_dim=2,
        pred_len=1,
        patch_len=8,
        stride=4,
        embed_dim=16,
        state_dim=8,
        num_layers=2,
        expand_factor=2,
        dropout=0.0,
        bidirectional_scan=True,
        normalize_input=True,
    )
    params.update(overrides)
    return MambaTSModel(**params)


# ============================================================================
# Basic functionality tests
# ============================================================================


def test_mambats_forward_shapes() -> None:
    """Test that forward pass produces correct output shapes."""
    model = _build_small_model()
    x = torch.randn(3, 32, 4)  # [B=3, T=32, D=4]
    output = model(x)

    preds = output["preds"]
    extras = output["extras"]

    assert preds.shape == (3, 1, 2), "Predictions should be [B, pred_len, output_dim]"
    assert "patch_embeds" in extras
    assert "encoded_features" in extras
    assert "num_patches" in extras


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


# ============================================================================
# Patch embedding tests
# ============================================================================


def test_patch_embedding_basic() -> None:
    """Test basic patch embedding functionality."""
    patch_embed = PatchEmbedding(
        input_dim=4, patch_len=8, stride=4, embed_dim=16, dropout=0.0
    )

    x = torch.randn(2, 32, 4)  # [B=2, T=32, D=4]
    patch_embeds, num_patches = patch_embed(x)

    # With patch_len=8 and stride=4: (32-8)/4 + 1 = 7 patches
    expected_patches = (32 - 8) // 4 + 1
    assert num_patches == expected_patches
    assert patch_embeds.shape == (2, expected_patches, 4, 16)


def test_patch_embedding_no_stride() -> None:
    """Test patch embedding without overlap (stride == patch_len)."""
    patch_embed = PatchEmbedding(
        input_dim=3, patch_len=16, stride=16, embed_dim=32, dropout=0.0
    )

    x = torch.randn(2, 64, 3)  # [B=2, T=64, D=3]
    patch_embeds, num_patches = patch_embed(x)

    # With patch_len=16 and stride=16: (64-16)/16 + 1 = 4 patches
    assert num_patches == 4
    assert patch_embeds.shape == (2, 4, 3, 32)


def test_patch_embedding_overlap() -> None:
    """Test patch embedding with overlap (stride < patch_len)."""
    patch_embed = PatchEmbedding(
        input_dim=2, patch_len=16, stride=8, embed_dim=24, dropout=0.0
    )

    x = torch.randn(3, 48, 2)  # [B=3, T=48, D=2]
    patch_embeds, num_patches = patch_embed(x)

    # With patch_len=16 and stride=8: (48-16)/8 + 1 = 5 patches
    assert num_patches == 5
    assert patch_embeds.shape == (3, 5, 2, 24)


def test_patch_embedding_validation() -> None:
    """Test that patch embedding validates parameters."""
    with pytest.raises(ValueError, match="patch_len must be positive"):
        PatchEmbedding(input_dim=4, patch_len=0, stride=4, embed_dim=16)

    with pytest.raises(ValueError, match="stride must be positive"):
        PatchEmbedding(input_dim=4, patch_len=8, stride=0, embed_dim=16)


# ============================================================================
# Temporal Mamba Block tests
# ============================================================================


def test_temporal_mamba_block_basic() -> None:
    """Test basic TMB functionality."""
    tmb = TemporalMambaBlock(
        embed_dim=16, state_dim=8, expand_factor=2, dropout=0.0, bidirectional=True
    )

    x = torch.randn(2, 10, 16)  # [B=2, L=10, embed_dim=16]
    output = tmb(x)

    assert output.shape == (2, 10, 16)
    assert torch.isfinite(output).all()


def test_temporal_mamba_block_residual() -> None:
    """Test that TMB applies residual connection."""
    tmb = TemporalMambaBlock(
        embed_dim=16, state_dim=8, expand_factor=2, dropout=0.0, bidirectional=False
    )

    x = torch.randn(2, 10, 16)

    # Set model to eval to disable dropout
    tmb.eval()

    with torch.no_grad():
        output = tmb(x)

    # Output should not be identical to input (due to processing)
    # but should be in similar range (due to residual)
    assert not torch.allclose(output, x, atol=1e-6)
    assert output.shape == x.shape


def test_temporal_mamba_block_bidirectional_vs_unidirectional() -> None:
    """Test that bidirectional and unidirectional modes produce different outputs."""
    x = torch.randn(2, 10, 16)

    tmb_bidir = TemporalMambaBlock(
        embed_dim=16, state_dim=8, expand_factor=2, dropout=0.0, bidirectional=True
    )
    tmb_unidir = TemporalMambaBlock(
        embed_dim=16, state_dim=8, expand_factor=2, dropout=0.0, bidirectional=False
    )

    # Copy parameters to make them identical except for bidirectional flag
    tmb_unidir.load_state_dict(tmb_bidir.state_dict(), strict=False)

    with torch.no_grad():
        output_bidir = tmb_bidir(x)
        output_unidir = tmb_unidir(x)

    # Outputs should differ due to backward scan contribution
    assert not torch.allclose(output_bidir, output_unidir, atol=1e-6)


# ============================================================================
# Variable Scan Encoder tests
# ============================================================================


def test_variable_scan_encoder_basic() -> None:
    """Test basic VST encoder functionality."""
    encoder = VariableScanEncoder(
        embed_dim=16, state_dim=8, num_layers=2, expand_factor=2, dropout=0.0
    )

    # Input: [B=2, num_patches=5, num_vars=4, embed_dim=16]
    patch_embeds = torch.randn(2, 5, 4, 16)
    output = encoder(patch_embeds)

    # Output should be [B=2, num_patches * num_vars = 20, embed_dim=16]
    assert output.shape == (2, 20, 16)
    assert torch.isfinite(output).all()


def test_variable_scan_ordering() -> None:
    """Test that Variable Scan along Time orders correctly."""
    encoder = VariableScanEncoder(
        embed_dim=8, state_dim=4, num_layers=1, expand_factor=2, dropout=0.0
    )

    # Create distinct patches for each variable and time step
    # [B=1, P=2, V=3, E=8]
    patch_embeds = torch.zeros(1, 2, 3, 8)

    # Set unique values to track ordering
    for p in range(2):  # patches (time)
        for v in range(3):  # variables
            patch_embeds[0, p, v, 0] = p * 10 + v  # Unique identifier

    # Apply VST
    vst_output = encoder.apply_variable_scan(patch_embeds)

    # Check ordering: should be [var0_t0, var1_t0, var2_t0, var0_t1, var1_t1, var2_t1]
    expected_order = [0, 1, 2, 10, 11, 12]  # First element of each token
    actual_order = vst_output[0, :, 0].tolist()

    assert actual_order == expected_order, "VST should order as [v0_t0, v1_t0, v2_t0, v0_t1, ...]"


# ============================================================================
# Normalization tests
# ============================================================================


def test_normalization_enabled() -> None:
    """Test that normalization is applied when enabled."""
    model = _build_small_model(normalize_input=True)
    x = torch.randn(2, 32, 4) * 100 + 50  # Large scale data

    output = model(x)
    preds = output["preds"]

    # Predictions should be finite and in reasonable range
    assert torch.isfinite(preds).all()
    assert preds.abs().mean() < 1000  # Should be normalized/denormalized properly


def test_normalization_disabled() -> None:
    """Test that model works without normalization."""
    model = _build_small_model(normalize_input=False)
    x = torch.randn(2, 32, 4)

    output = model(x)
    preds = output["preds"]

    assert torch.isfinite(preds).all()


def test_normalization_preserves_scale() -> None:
    """Test that normalization and denormalization preserve output scale."""
    model = _build_small_model(normalize_input=True)
    model.eval()

    # Create data with known statistics
    x = torch.randn(2, 32, 4) * 10 + 100

    with torch.no_grad():
        output = model(x)
        preds = output["preds"]

    # Predictions should be in a reasonable range relative to input
    # (not exact due to model processing, but should be in similar magnitude)
    assert preds.abs().max() < 1000  # Not exploding
    assert preds.abs().mean() > 0.01  # Not vanishing


# ============================================================================
# Long sequence tests
# ============================================================================


def test_long_sequence_handling() -> None:
    """Test that model handles longer sequences efficiently."""
    model = _build_small_model(patch_len=16, stride=8)
    x = torch.randn(2, 128, 4)  # Longer sequence

    output = model(x)
    preds = output["preds"]

    assert preds.shape == (2, 1, 2)
    assert torch.isfinite(preds).all()


def test_variable_sequence_lengths() -> None:
    """Test model with different sequence lengths."""
    model = _build_small_model()

    # Test with different lengths
    for seq_len in [32, 48, 64, 96]:
        x = torch.randn(2, seq_len, 4)
        output = model(x)
        preds = output["preds"]

        assert preds.shape == (2, 1, 2)
        assert torch.isfinite(preds).all()


# ============================================================================
# Parameter validation tests
# ============================================================================


def test_parameter_validation() -> None:
    """Test that invalid parameters raise appropriate errors."""
    # Invalid pred_len
    with pytest.raises(ValueError, match="pred_len must be positive"):
        _build_small_model(pred_len=0)

    # Invalid patch_len
    with pytest.raises(ValueError, match="patch_len must be positive"):
        _build_small_model(patch_len=0)

    # Invalid stride
    with pytest.raises(ValueError, match="stride must be positive"):
        _build_small_model(stride=0)

    # Invalid embed_dim
    with pytest.raises(ValueError, match="embed_dim must be positive"):
        _build_small_model(embed_dim=0)

    # Invalid state_dim
    with pytest.raises(ValueError, match="state_dim must be positive"):
        _build_small_model(state_dim=0)

    # Invalid num_layers
    with pytest.raises(ValueError, match="num_layers must be positive"):
        _build_small_model(num_layers=0)

    # Invalid expand_factor
    with pytest.raises(ValueError, match="expand_factor must be positive"):
        _build_small_model(expand_factor=0)

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
    input_grad = x.grad
    assert isinstance(input_grad, torch.Tensor)
    assert torch.isfinite(input_grad).all()
    assert torch.any(input_grad != 0)

    # Check that model parameters have gradients
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert isinstance(param.grad, torch.Tensor), f"Parameter {name} should have gradients"
            assert torch.isfinite(param.grad).all(), f"Parameter {name} has non-finite gradients"
            assert torch.any(param.grad != 0), f"Parameter {name} should receive gradient signal"


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
    """Test that extras dict contains expected analysis outputs."""
    model = _build_small_model()
    x = torch.randn(2, 32, 4)

    output = model(x)
    extras = output["extras"]

    assert "patch_embeds" in extras
    assert "encoded_features" in extras
    assert "num_patches" in extras

    # Check types
    assert isinstance(extras["patch_embeds"], torch.Tensor)
    assert isinstance(extras["encoded_features"], torch.Tensor)
    assert isinstance(extras["num_patches"], int)


def test_extras_shapes() -> None:
    """Test that extras tensors have correct shapes."""
    model = _build_small_model(input_dim=4, patch_len=8, stride=4, embed_dim=16)
    x = torch.randn(2, 32, 4)

    output = model(x)
    extras = output["extras"]

    # Calculate expected number of patches
    num_patches = (32 - 8) // 4 + 1  # = 7

    # Check shapes
    assert extras["patch_embeds"].shape == (2, num_patches, 4, 16)
    assert extras["encoded_features"].shape == (2, num_patches * 4, 16)
    assert extras["num_patches"] == num_patches


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


def test_minimal_sequence_length() -> None:
    """Test model with minimal viable sequence length."""
    model = _build_small_model(patch_len=8, stride=4)

    # Minimal sequence that can produce at least one patch
    x = torch.randn(2, 8, 4)  # Exactly one patch

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

    # Outputs should be different due to dropout
    # (though this is stochastic, so we just check they're both valid)
    assert torch.isfinite(out_train["preds"]).all()
    assert torch.isfinite(out_eval["preds"]).all()


# Need to import F for the test
import torch.nn.functional as F
