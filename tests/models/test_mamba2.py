"""Tests for the Temporal Mamba-2 model."""

from __future__ import annotations

import torch

from airtrace.models.mamba2 import Mamba2Model


def _build_small_model(**overrides):
    params = dict(
        input_dim=4,
        output_dim=2,
        pred_len=1,
        embed_dim=16,
        state_dim=8,
        num_layers=2,
        conv_kernel_size=3,
        chunk_length=4,
        bidirectional_scan=True,
        dropout=0.0,
        ff_expansion=2,
    )
    params.update(overrides)
    return Mamba2Model(**params)


def test_mamba2_forward_shapes() -> None:
    model = _build_small_model()
    x = torch.randn(3, 12, 4)
    output = model(x)
    preds = output["preds"]
    extras = output["extras"]
    assert preds.shape == (3, 1, 2)
    assert len(extras["selective_states"]) == 2
    for state in extras["selective_states"]:
        assert state.shape == (3, 8)


def test_chunked_scan_handles_long_contexts() -> None:
    model = _build_small_model(chunk_length=2)
    x = torch.randn(2, 33, 4)
    preds = model(x)["preds"]
    assert torch.isfinite(preds).all()


def test_lora_only_fine_tuning_path() -> None:
    model = _build_small_model(
        adapter_rank=2,
        adapter_alpha=4.0,
        freeze_backbone=True,
        train_head=False,
    )
    trainable = {name for name, param in model.named_parameters() if param.requires_grad}
    assert any("lora" in name for name in trainable)
    assert not any(
        name.startswith("input_proj") for name in trainable
    ), "Input projection should be frozen when freeze_backbone=True"


def test_bidirectional_vs_unidirectional() -> None:
    """Verify bidirectional and unidirectional modes produce different outputs."""
    x = torch.randn(2, 10, 4)
    model_bidir = _build_small_model(bidirectional_scan=True)
    model_unidir = _build_small_model(bidirectional_scan=False)

    # Copy weights to make models identical except for bidirectional flag
    model_unidir.load_state_dict(model_bidir.state_dict(), strict=False)

    with torch.no_grad():
        preds_bidir = model_bidir(x)["preds"]
        preds_unidir = model_unidir(x)["preds"]

    # Outputs should differ due to backward scan contribution
    assert not torch.allclose(preds_bidir, preds_unidir, atol=1e-6)


def test_gradient_flow_with_frozen_backbone() -> None:
    """Verify gradients only flow to LoRA params when backbone is frozen."""
    model = _build_small_model(
        adapter_rank=4,
        freeze_backbone=True,
        train_head=False,
    )
    x = torch.randn(2, 8, 4)
    output = model(x)
    loss = output["preds"].sum()
    loss.backward()

    # Check that only LoRA params have gradients
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert "lora" in name, f"Only LoRA params should have gradients, got {name}"
            assert isinstance(param.grad, torch.Tensor), f"LoRA param {name} should have gradient"
            assert torch.isfinite(param.grad).all(), f"LoRA param {name} gradient should be finite"
            assert torch.any(param.grad != 0), f"LoRA param {name} gradient should be non-zero"
        else:
            # Frozen params might still accumulate gradients in some PyTorch versions
            # but they shouldn't be updated
            pass


def test_multi_step_prediction() -> None:
    """Test multi-step ahead forecasting with pred_len > 1."""
    model = _build_small_model(pred_len=5, output_dim=3)
    x = torch.randn(4, 20, 4)
    output = model(x)
    preds = output["preds"]

    assert preds.shape == (4, 5, 3), "Should predict 5 steps ahead with 3 output dims"
    assert torch.isfinite(preds).all()


def test_chunk_length_exceeds_sequence_length() -> None:
    """Verify model handles chunk_length > sequence_length gracefully."""
    model = _build_small_model(chunk_length=100)
    x = torch.randn(2, 10, 4)  # Sequence length 10 < chunk_length 100
    preds = model(x)["preds"]

    assert preds.shape == (2, 1, 2)
    assert torch.isfinite(preds).all()


def test_parameter_validation() -> None:
    """Test that invalid parameters raise appropriate errors."""
    import pytest

    # Invalid pred_len
    with pytest.raises(ValueError, match="pred_len must be positive"):
        _build_small_model(pred_len=0)

    # Invalid dropout
    with pytest.raises(ValueError, match="dropout must be in"):
        _build_small_model(dropout=1.5)

    # Invalid adapter_rank
    with pytest.raises(ValueError, match="adapter_rank must be non-negative"):
        _build_small_model(adapter_rank=-1)

    # Invalid ff_expansion
    with pytest.raises(ValueError, match="ff_expansion must be positive"):
        _build_small_model(ff_expansion=0)


def test_selective_states_returned() -> None:
    """Verify selective states are returned in extras for each layer."""
    model = _build_small_model(num_layers=3)
    x = torch.randn(2, 12, 4)
    output = model(x)

    selective_states = output["extras"]["selective_states"]
    assert len(selective_states) == 3, "Should have one state per layer"

    for state in selective_states:
        assert state.shape == (2, 8), "State should be [batch, state_dim]"
        assert torch.isfinite(state).all()
