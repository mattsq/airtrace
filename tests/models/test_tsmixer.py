"""Tests for the TSMixer implementation."""

import pytest
import torch

from airtrace.models import TSMixerModel
from airtrace.models.tsmixer import TSMixerBlock


def test_tsmixer_block_forward():
    """Test TSMixer block forward pass."""
    batch_size, seq_len, feature_dim = 4, 96, 10
    hidden_dim = 256

    block = TSMixerBlock(
        seq_len=seq_len,
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        dropout=0.0,
    )

    x = torch.randn(batch_size, seq_len, feature_dim)
    out = block(x)

    # Output shape should match input shape
    assert out.shape == (batch_size, seq_len, feature_dim)

    # Test with dropout
    block_dropout = TSMixerBlock(
        seq_len=seq_len,
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        dropout=0.1,
    )
    out_dropout = block_dropout(x)
    assert out_dropout.shape == (batch_size, seq_len, feature_dim)


def test_tsmixer_forward():
    """Test TSMixer model forward pass."""
    batch_size, seq_len, num_channels = 4, 96, 10
    pred_len = 24

    model = TSMixerModel(
        input_dim=num_channels,
        output_dim=num_channels,
        seq_len=seq_len,
        pred_len=pred_len,
        num_blocks=4,
        hidden_dim=256,
        dropout=0.1,
    )

    x = torch.randn(batch_size, seq_len, num_channels)
    output = model(x)

    assert "preds" in output
    assert "extras" in output
    assert output["preds"].shape == (batch_size, pred_len, num_channels)
    assert output["extras"]["block_outputs"] is not None


def test_tsmixer_residual_connections():
    """Test that residual connections work correctly."""
    batch_size, seq_len, feature_dim = 2, 96, 5
    hidden_dim = 128

    block = TSMixerBlock(
        seq_len=seq_len,
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        dropout=0.0,
    )

    x = torch.randn(batch_size, seq_len, feature_dim)
    out = block(x)

    # Output should differ from input (due to transformations)
    assert not torch.allclose(out, x)

    # But residuals should keep outputs stable
    assert torch.isfinite(out).all()


def test_tsmixer_different_input_output_dims():
    """Test TSMixer when input_dim != output_dim."""
    model = TSMixerModel(
        input_dim=8,
        output_dim=3,
        seq_len=96,
        pred_len=24,
        num_blocks=2,
        hidden_dim=128,
        dropout=0.1,
    )

    x = torch.randn(2, 96, 8)
    output = model(x)

    assert output["preds"].shape == (2, 24, 3)


def test_tsmixer_same_input_output_dims():
    """Test TSMixer when input_dim == output_dim (no output projection)."""
    model = TSMixerModel(
        input_dim=5,
        output_dim=5,
        seq_len=96,
        pred_len=24,
        num_blocks=3,
        hidden_dim=256,
        dropout=0.1,
    )

    x = torch.randn(3, 96, 5)
    output = model(x)

    assert output["preds"].shape == (3, 24, 5)
    # Check that output projection is None when dims match
    assert model.output_projection is None


def test_tsmixer_multi_step_prediction():
    """Test TSMixer with multi-step prediction horizon."""
    pred_len = 96
    model = TSMixerModel(
        input_dim=5,
        output_dim=5,
        seq_len=96,
        pred_len=pred_len,
        num_blocks=4,
        hidden_dim=256,
        dropout=0.1,
    )

    x = torch.randn(3, 96, 5)
    output = model(x)

    assert output["preds"].shape == (3, pred_len, 5)


def test_tsmixer_one_step_prediction():
    """Test TSMixer with single-step prediction."""
    model = TSMixerModel(
        input_dim=10,
        output_dim=5,
        seq_len=60,
        pred_len=1,
        num_blocks=2,
        hidden_dim=128,
        dropout=0.0,
    )

    x = torch.randn(4, 60, 10)
    output = model(x)

    assert output["preds"].shape == (4, 1, 5)


def test_tsmixer_variable_sequence_lengths():
    """Test TSMixer with different sequence lengths."""
    # Create model for seq_len=96
    model_96 = TSMixerModel(
        input_dim=4,
        output_dim=2,
        seq_len=96,
        pred_len=24,
        num_blocks=3,
        hidden_dim=128,
        dropout=0.1,
    )

    x1 = torch.randn(2, 96, 4)
    output1 = model_96(x1)
    assert output1["preds"].shape == (2, 24, 2)

    # Create model for seq_len=192
    model_192 = TSMixerModel(
        input_dim=4,
        output_dim=2,
        seq_len=192,
        pred_len=48,
        num_blocks=3,
        hidden_dim=128,
        dropout=0.1,
    )

    x2 = torch.randn(2, 192, 4)
    output2 = model_192(x2)
    assert output2["preds"].shape == (2, 48, 2)


def test_tsmixer_sequence_length_mismatch():
    """Test that TSMixer raises error on sequence length mismatch."""
    model = TSMixerModel(
        input_dim=5,
        output_dim=5,
        seq_len=96,
        pred_len=24,
        num_blocks=2,
        hidden_dim=128,
    )

    # Wrong sequence length
    x = torch.randn(2, 60, 5)

    with pytest.raises(ValueError, match="Expected input sequence length"):
        model(x)


def test_tsmixer_num_params():
    """Test that model parameter counting works."""
    model = TSMixerModel(
        input_dim=6,
        output_dim=3,
        seq_len=96,
        pred_len=24,
        num_blocks=4,
        hidden_dim=256,
        dropout=0.1,
    )

    num_params = model.get_num_params()
    assert num_params > 0
    assert isinstance(num_params, int)


def test_tsmixer_repr():
    """Test model string representation."""
    model = TSMixerModel(
        input_dim=6,
        output_dim=3,
        seq_len=96,
        pred_len=24,
        num_blocks=4,
        hidden_dim=256,
        dropout=0.1,
    )

    repr_str = repr(model)
    assert "TSMixerModel" in repr_str
    assert "input_dim=6" in repr_str
    assert "output_dim=3" in repr_str
    assert "seq_len=96" in repr_str
    assert "pred_len=24" in repr_str
    assert "num_blocks=4" in repr_str
    assert "hidden_dim=256" in repr_str


@pytest.mark.parametrize("num_blocks", [1, 2, 3, 4, 6])
def test_tsmixer_num_blocks(num_blocks):
    """Test TSMixer with different numbers of blocks."""
    model = TSMixerModel(
        input_dim=5,
        output_dim=5,
        seq_len=96,
        pred_len=24,
        num_blocks=num_blocks,
        hidden_dim=128,
        dropout=0.0,
    )

    x = torch.randn(2, 96, 5)
    output = model(x)

    assert output["preds"].shape == (2, 24, 5)
    # Check that all blocks are used
    assert len(model.mixer_blocks) == num_blocks
    # Check extras contain correct number of block outputs
    assert output["extras"]["block_outputs"].shape[0] == num_blocks


@pytest.mark.parametrize("hidden_dim", [64, 128, 256, 512])
def test_tsmixer_hidden_dim(hidden_dim):
    """Test TSMixer with different hidden dimensions."""
    model = TSMixerModel(
        input_dim=5,
        output_dim=5,
        seq_len=96,
        pred_len=24,
        num_blocks=2,
        hidden_dim=hidden_dim,
        dropout=0.0,
    )

    x = torch.randn(2, 96, 5)
    output = model(x)

    assert output["preds"].shape == (2, 24, 5)


@pytest.mark.parametrize("dropout", [0.0, 0.1, 0.2, 0.3])
def test_tsmixer_dropout(dropout):
    """Test TSMixer with different dropout rates."""
    model = TSMixerModel(
        input_dim=5,
        output_dim=5,
        seq_len=96,
        pred_len=24,
        num_blocks=3,
        hidden_dim=128,
        dropout=dropout,
    )

    x = torch.randn(2, 96, 5)

    # Training mode - dropout active
    model.train()
    out_train = model(x)
    assert out_train["preds"].shape == (2, 24, 5)

    # Eval mode - dropout inactive
    model.eval()
    out_eval = model(x)
    assert out_eval["preds"].shape == (2, 24, 5)


def test_tsmixer_batch_size_one():
    """Test TSMixer with batch size 1."""
    model = TSMixerModel(
        input_dim=5,
        output_dim=5,
        seq_len=96,
        pred_len=24,
        num_blocks=2,
        hidden_dim=128,
        dropout=0.1,
    )

    x = torch.randn(1, 96, 5)
    output = model(x)

    assert output["preds"].shape == (1, 24, 5)


def test_tsmixer_large_batch():
    """Test TSMixer with a larger batch size."""
    model = TSMixerModel(
        input_dim=5,
        output_dim=5,
        seq_len=96,
        pred_len=24,
        num_blocks=3,
        hidden_dim=128,
        dropout=0.1,
    )

    x = torch.randn(32, 96, 5)  # Batch size 32
    output = model(x)

    assert output["preds"].shape == (32, 24, 5)


def test_tsmixer_gradient_flow():
    """Test that gradients flow through the model."""
    model = TSMixerModel(
        input_dim=5,
        output_dim=5,
        seq_len=96,
        pred_len=24,
        num_blocks=3,
        hidden_dim=128,
        dropout=0.0,
    )

    x = torch.randn(2, 96, 5, requires_grad=True)
    output = model(x)

    # Compute a simple loss
    loss = output["preds"].mean()
    loss.backward()

    # Check that gradients exist
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    assert torch.isfinite(x.grad).all()


def test_tsmixer_deterministic_eval():
    """Test that eval mode is deterministic."""
    model = TSMixerModel(
        input_dim=5,
        output_dim=5,
        seq_len=96,
        pred_len=24,
        num_blocks=3,
        hidden_dim=128,
        dropout=0.1,
    )

    x = torch.randn(2, 96, 5)

    # Eval mode should be deterministic
    model.eval()
    torch.manual_seed(42)
    out_eval_1 = model(x)
    torch.manual_seed(43)
    out_eval_2 = model(x)

    # Outputs should be identical (deterministic in eval mode)
    assert torch.allclose(out_eval_1["preds"], out_eval_2["preds"])


def test_tsmixer_time_mixing():
    """Test that time-mixing operates on temporal dimension."""
    batch_size, seq_len, feature_dim = 2, 96, 5
    hidden_dim = 128

    block = TSMixerBlock(
        seq_len=seq_len,
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        dropout=0.0,
    )

    x = torch.randn(batch_size, seq_len, feature_dim)
    out = block(x)

    # Time-mixing should maintain feature dimension
    assert out.shape[2] == feature_dim

    # Check that time-mixing MLP exists and has correct dimensions
    assert block.time_mlp[0].in_features == seq_len
    assert block.time_mlp[3].out_features == seq_len


def test_tsmixer_feature_mixing():
    """Test that feature-mixing operates on feature dimension."""
    batch_size, seq_len, feature_dim = 2, 96, 5
    hidden_dim = 128

    block = TSMixerBlock(
        seq_len=seq_len,
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        dropout=0.0,
    )

    x = torch.randn(batch_size, seq_len, feature_dim)
    out = block(x)

    # Feature-mixing should maintain sequence length
    assert out.shape[1] == seq_len

    # Check that feature-mixing MLP exists and has correct dimensions
    assert block.feature_mlp[0].in_features == feature_dim
    assert block.feature_mlp[3].out_features == feature_dim


def test_tsmixer_temporal_projection():
    """Test temporal projection layer."""
    model = TSMixerModel(
        input_dim=5,
        output_dim=5,
        seq_len=96,
        pred_len=48,
        num_blocks=2,
        hidden_dim=128,
        dropout=0.0,
    )

    # Check temporal projection dimensions
    assert model.temporal_projection.in_features == 96
    assert model.temporal_projection.out_features == 48

    x = torch.randn(2, 96, 5)
    output = model(x)

    assert output["preds"].shape == (2, 48, 5)


def test_tsmixer_output_projection():
    """Test output projection when input_dim != output_dim."""
    input_dim, output_dim = 10, 3

    model = TSMixerModel(
        input_dim=input_dim,
        output_dim=output_dim,
        seq_len=96,
        pred_len=24,
        num_blocks=2,
        hidden_dim=128,
        dropout=0.0,
    )

    # Check output projection exists
    assert model.output_projection is not None
    assert model.output_projection.in_features == input_dim
    assert model.output_projection.out_features == output_dim

    x = torch.randn(2, 96, input_dim)
    output = model(x)

    assert output["preds"].shape == (2, 24, output_dim)


def test_tsmixer_no_context():
    """Test that TSMixer works without context (as expected)."""
    model = TSMixerModel(
        input_dim=5,
        output_dim=5,
        seq_len=96,
        pred_len=24,
        num_blocks=2,
        hidden_dim=128,
        dropout=0.0,
    )

    x = torch.randn(2, 96, 5)
    # TSMixer doesn't use context, but should accept it
    context = torch.randn(2, 96, 3)
    output = model(x, context=context)

    assert output["preds"].shape == (2, 24, 5)


def test_tsmixer_block_outputs_tracking():
    """Test that block outputs are tracked correctly in extras."""
    num_blocks = 3
    model = TSMixerModel(
        input_dim=5,
        output_dim=5,
        seq_len=96,
        pred_len=24,
        num_blocks=num_blocks,
        hidden_dim=128,
        dropout=0.0,
    )

    x = torch.randn(2, 96, 5)
    output = model(x)

    block_outputs = output["extras"]["block_outputs"]
    assert block_outputs is not None
    assert block_outputs.shape[0] == num_blocks
    assert block_outputs.shape[1:] == (2, 96, 5)  # [num_blocks, batch, seq, features]


def test_tsmixer_very_short_sequence():
    """Test TSMixer with very short sequence length."""
    model = TSMixerModel(
        input_dim=5,
        output_dim=5,
        seq_len=10,
        pred_len=5,
        num_blocks=2,
        hidden_dim=64,
        dropout=0.0,
    )

    x = torch.randn(2, 10, 5)
    output = model(x)

    assert output["preds"].shape == (2, 5, 5)


def test_tsmixer_long_horizon():
    """Test TSMixer with long prediction horizon."""
    model = TSMixerModel(
        input_dim=5,
        output_dim=5,
        seq_len=192,
        pred_len=192,
        num_blocks=4,
        hidden_dim=256,
        dropout=0.1,
    )

    x = torch.randn(2, 192, 5)
    output = model(x)

    assert output["preds"].shape == (2, 192, 5)


def test_tsmixer_numerical_stability():
    """Test that TSMixer produces numerically stable outputs."""
    model = TSMixerModel(
        input_dim=5,
        output_dim=5,
        seq_len=96,
        pred_len=24,
        num_blocks=3,
        hidden_dim=128,
        dropout=0.0,
    )

    # Test with large values
    x_large = torch.randn(2, 96, 5) * 1000
    output_large = model(x_large)
    assert torch.isfinite(output_large["preds"]).all()

    # Test with small values
    x_small = torch.randn(2, 96, 5) * 0.001
    output_small = model(x_small)
    assert torch.isfinite(output_small["preds"]).all()

    # Test with zeros
    x_zeros = torch.zeros(2, 96, 5)
    output_zeros = model(x_zeros)
    assert torch.isfinite(output_zeros["preds"]).all()


def test_tsmixer_single_feature():
    """Test TSMixer with univariate time series (single feature)."""
    model = TSMixerModel(
        input_dim=1,
        output_dim=1,
        seq_len=96,
        pred_len=24,
        num_blocks=2,
        hidden_dim=128,
        dropout=0.0,
    )

    x = torch.randn(4, 96, 1)
    output = model(x)

    assert output["preds"].shape == (4, 24, 1)


def test_tsmixer_many_features():
    """Test TSMixer with many features."""
    model = TSMixerModel(
        input_dim=50,
        output_dim=20,
        seq_len=96,
        pred_len=24,
        num_blocks=3,
        hidden_dim=256,
        dropout=0.1,
    )

    x = torch.randn(2, 96, 50)
    output = model(x)

    assert output["preds"].shape == (2, 24, 20)
