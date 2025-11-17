"""Tests for the TimeXer implementation."""

import pytest
import torch

from airtrace.models import TimeXerModel


def test_timexer_forward_without_exogenous():
    """Test TimeXer without exogenous variables (endogenous only)."""
    model = TimeXerModel(
        input_dim=6,
        output_dim=3,
        exog_dim=0,
        patch_len=8,
        stride=4,
        d_model=32,
        nhead=4,
        num_layers=2,
        dim_feedforward=64,
        pred_len=1,
    )

    x = torch.randn(4, 32, 6)  # [B, T, D]
    output = model(x)

    assert "preds" in output
    assert "extras" in output
    assert output["preds"].shape == (4, 1, 3)  # [B, pred_len, output_dim]

    extras = output["extras"]
    assert "endogenous_output" in extras
    assert "global_tokens" in extras
    assert "patch_output" in extras
    assert "num_patches" in extras


def test_timexer_forward_with_exogenous():
    """Test TimeXer with exogenous variables."""
    model = TimeXerModel(
        input_dim=6,
        output_dim=3,
        exog_dim=4,
        patch_len=8,
        stride=4,
        d_model=32,
        nhead=4,
        num_layers=2,
        dim_feedforward=64,
        pred_len=1,
    )

    x = torch.randn(4, 32, 6)  # [B, T, D_endogenous]
    context = torch.randn(4, 32, 4)  # [B, T, D_exogenous]
    output = model(x, context=context)

    assert "preds" in output
    assert "extras" in output
    assert output["preds"].shape == (4, 1, 3)  # [B, pred_len, output_dim]

    extras = output["extras"]
    assert "endogenous_output" in extras
    assert "global_tokens" in extras
    assert "patch_output" in extras
    assert "num_patches" in extras
    assert "exog_embedded" in extras
    assert "cross_attn_weights" in extras


def test_timexer_static_exogenous():
    """Test TimeXer with static exogenous variables (no time dimension)."""
    model = TimeXerModel(
        input_dim=5,
        output_dim=2,
        exog_dim=3,
        patch_len=16,
        stride=8,
        d_model=64,
        nhead=8,
        num_layers=2,
        dim_feedforward=128,
        pred_len=2,
    )

    x = torch.randn(2, 64, 5)  # [B, T, D_endogenous]
    context = torch.randn(2, 3)  # [B, D_exogenous] - static
    output = model(x, context=context)

    assert output["preds"].shape == (2, 2, 2)  # [B, pred_len, output_dim]
    assert "cross_attn_weights" in output["extras"]


def test_timexer_multi_step_prediction():
    """Test TimeXer with multi-step prediction horizon."""
    pred_len = 5
    model = TimeXerModel(
        input_dim=4,
        output_dim=4,
        exog_dim=2,
        patch_len=8,
        stride=4,
        d_model=32,
        nhead=4,
        num_layers=2,
        dim_feedforward=64,
        pred_len=pred_len,
    )

    x = torch.randn(3, 32, 4)
    context = torch.randn(3, 32, 2)
    output = model(x, context=context)

    assert output["preds"].shape == (3, pred_len, 4)


def test_timexer_different_input_output_dims():
    """Test TimeXer when input_dim != output_dim."""
    model = TimeXerModel(
        input_dim=8,
        output_dim=3,
        exog_dim=5,
        patch_len=8,
        stride=4,
        d_model=32,
        nhead=4,
        num_layers=2,
        dim_feedforward=64,
        pred_len=1,
    )

    x = torch.randn(2, 32, 8)
    context = torch.randn(2, 32, 5)
    output = model(x, context=context)

    assert output["preds"].shape == (2, 1, 3)


def test_timexer_num_params():
    """Test that model parameter counting works."""
    model = TimeXerModel(
        input_dim=6,
        output_dim=3,
        exog_dim=4,
        patch_len=8,
        stride=4,
        d_model=32,
        nhead=4,
        num_layers=2,
        dim_feedforward=64,
        pred_len=1,
    )

    num_params = model.get_num_params()
    assert num_params > 0
    assert isinstance(num_params, int)


def test_timexer_repr():
    """Test model string representation."""
    model = TimeXerModel(
        input_dim=6,
        output_dim=3,
        exog_dim=4,
        patch_len=8,
        stride=4,
        d_model=32,
        nhead=4,
        num_layers=2,
        pred_len=2,
    )

    repr_str = repr(model)
    assert "TimeXerModel" in repr_str
    assert "input_dim=6" in repr_str
    assert "output_dim=3" in repr_str
    assert "exog_dim=4" in repr_str
    assert "pred_len=2" in repr_str


def test_timexer_variable_sequence_lengths():
    """Test TimeXer with different sequence lengths."""
    model = TimeXerModel(
        input_dim=4,
        output_dim=2,
        exog_dim=3,
        patch_len=8,
        stride=4,
        d_model=32,
        nhead=4,
        num_layers=2,
        dim_feedforward=64,
        pred_len=1,
    )

    # Test with sequence length 32
    x1 = torch.randn(2, 32, 4)
    context1 = torch.randn(2, 32, 3)
    output1 = model(x1, context=context1)
    assert output1["preds"].shape == (2, 1, 2)

    # Test with sequence length 64
    x2 = torch.randn(2, 64, 4)
    context2 = torch.randn(2, 64, 3)
    output2 = model(x2, context=context2)
    assert output2["preds"].shape == (2, 1, 2)


def test_timexer_batch_size_one():
    """Test TimeXer with batch size 1."""
    model = TimeXerModel(
        input_dim=5,
        output_dim=5,
        exog_dim=2,
        patch_len=16,
        stride=8,
        d_model=64,
        nhead=8,
        num_layers=2,
        dim_feedforward=128,
        pred_len=1,
    )

    x = torch.randn(1, 64, 5)
    context = torch.randn(1, 64, 2)
    output = model(x, context=context)

    assert output["preds"].shape == (1, 1, 5)


@pytest.mark.parametrize("activation", ["relu", "gelu"])
def test_timexer_activations(activation):
    """Test TimeXer with different activation functions."""
    model = TimeXerModel(
        input_dim=4,
        output_dim=2,
        exog_dim=0,
        patch_len=8,
        stride=4,
        d_model=32,
        nhead=4,
        num_layers=2,
        dim_feedforward=64,
        activation=activation,
        pred_len=1,
    )

    x = torch.randn(2, 32, 4)
    output = model(x)

    assert output["preds"].shape == (2, 1, 2)


def test_timexer_global_tokens():
    """Test TimeXer with multiple global tokens."""
    model = TimeXerModel(
        input_dim=6,
        output_dim=3,
        exog_dim=4,
        patch_len=8,
        stride=4,
        d_model=32,
        nhead=4,
        num_layers=2,
        dim_feedforward=64,
        num_global_tokens=2,
        pred_len=1,
    )

    x = torch.randn(2, 32, 6)
    context = torch.randn(2, 32, 4)
    output = model(x, context=context)

    assert output["preds"].shape == (2, 1, 3)
    # Global tokens should have shape [B, D_in, num_global_tokens, d_model]
    assert output["extras"]["global_tokens"].shape[2] == 2  # num_global_tokens
