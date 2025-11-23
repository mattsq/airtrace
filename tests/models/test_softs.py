"""Tests for the SOFTS implementation."""

import pytest
import torch

from airtrace.models import SOFTS
from airtrace.models.softs import STARModule, EncoderLayerSOFTS


def test_star_module_forward():
    """Test STAR module forward pass."""
    batch_size, channels, d_series = 4, 10, 128
    d_core = 64

    star = STARModule(d_series, d_core)
    x = torch.randn(batch_size, channels, d_series)

    # Test training mode
    star.train()
    out_train = star(x)
    assert out_train.shape == (batch_size, channels, d_series)

    # Test eval mode
    star.eval()
    out_eval = star(x)
    assert out_eval.shape == (batch_size, channels, d_series)


def test_encoder_layer_forward():
    """Test encoder layer forward pass."""
    batch_size, channels, d_model = 4, 10, 128
    d_core, d_ff = 64, 256

    layer = EncoderLayerSOFTS(d_model, d_core, d_ff)
    x = torch.randn(batch_size, channels, d_model)

    out = layer(x)
    assert out.shape == (batch_size, channels, d_model)


def test_softs_forward():
    """Test SOFTS model forward pass."""
    batch_size, seq_len, num_channels = 4, 96, 10
    pred_len = 24
    hidden_dim = 128

    model = SOFTS(
        input_dim=num_channels,
        output_dim=num_channels,
        seq_len=seq_len,
        pred_len=pred_len,
        hidden_dim=hidden_dim,
        d_core=64,
        d_ff=256,
        e_layers=2,
    )

    x = torch.randn(batch_size, seq_len, num_channels)
    output = model(x)

    assert "preds" in output
    assert "extras" in output
    assert output["preds"].shape == (batch_size, pred_len, num_channels)


def test_softs_normalization():
    """Test that normalization is applied correctly."""
    batch_size, seq_len, num_channels = 4, 96, 10
    pred_len = 24

    # With normalization
    model_norm = SOFTS(
        input_dim=num_channels,
        output_dim=num_channels,
        seq_len=seq_len,
        pred_len=pred_len,
        hidden_dim=128,
        use_norm=True,
    )

    # Without normalization
    model_no_norm = SOFTS(
        input_dim=num_channels,
        output_dim=num_channels,
        seq_len=seq_len,
        pred_len=pred_len,
        hidden_dim=128,
        use_norm=False,
    )

    x = torch.randn(batch_size, seq_len, num_channels) * 100  # Large values

    out_norm = model_norm(x)
    out_no_norm = model_no_norm(x)

    # Both should produce valid outputs
    assert torch.isfinite(out_norm["preds"]).all()
    assert torch.isfinite(out_no_norm["preds"]).all()

    # Check extras
    means = out_norm["extras"]["means"]
    stdev = out_norm["extras"]["stdev"]
    assert means.shape == (num_channels,)
    assert stdev.shape == (num_channels,)
    assert torch.isfinite(means).all()
    assert torch.isfinite(stdev).all()
    assert out_no_norm["extras"]["means"] is None
    assert out_no_norm["extras"]["stdev"] is None


def test_softs_stochastic_pooling():
    """Test that stochastic pooling differs between train/eval."""
    batch_size, seq_len, num_channels = 4, 96, 10
    pred_len = 24

    model = SOFTS(
        input_dim=num_channels,
        output_dim=num_channels,
        seq_len=seq_len,
        pred_len=pred_len,
        hidden_dim=128,
    )

    x = torch.randn(batch_size, seq_len, num_channels)

    # Training mode (stochastic)
    model.train()
    torch.manual_seed(42)
    out_train_1 = model(x)
    torch.manual_seed(43)
    out_train_2 = model(x)

    # Outputs should differ (stochastic)
    assert not torch.allclose(out_train_1["preds"], out_train_2["preds"])

    # Eval mode (deterministic)
    model.eval()
    out_eval_1 = model(x)
    out_eval_2 = model(x)

    # Outputs should be identical (deterministic)
    assert torch.allclose(out_eval_1["preds"], out_eval_2["preds"])


def test_softs_different_input_output_dims():
    """Test SOFTS when input_dim != output_dim."""
    model = SOFTS(
        input_dim=8,
        output_dim=3,
        seq_len=96,
        pred_len=24,
        hidden_dim=128,
        d_core=64,
        d_ff=256,
        e_layers=2,
    )

    x = torch.randn(2, 96, 8)
    output = model(x)

    assert output["preds"].shape == (2, 24, 3)


def test_softs_multi_step_prediction():
    """Test SOFTS with multi-step prediction horizon."""
    pred_len = 96
    model = SOFTS(
        input_dim=5,
        output_dim=5,
        seq_len=96,
        pred_len=pred_len,
        hidden_dim=128,
        d_core=64,
        d_ff=256,
        e_layers=2,
    )

    x = torch.randn(3, 96, 5)
    output = model(x)

    assert output["preds"].shape == (3, pred_len, 5)


def test_softs_variable_sequence_lengths():
    """Test SOFTS with different sequence lengths."""
    # Create model for seq_len=96
    model_96 = SOFTS(
        input_dim=4,
        output_dim=2,
        seq_len=96,
        pred_len=24,
        hidden_dim=128,
        d_core=64,
        d_ff=256,
        e_layers=2,
    )

    x1 = torch.randn(2, 96, 4)
    output1 = model_96(x1)
    assert output1["preds"].shape == (2, 24, 2)

    # Create model for seq_len=192
    model_192 = SOFTS(
        input_dim=4,
        output_dim=2,
        seq_len=192,
        pred_len=48,
        hidden_dim=128,
        d_core=64,
        d_ff=256,
        e_layers=2,
    )

    x2 = torch.randn(2, 192, 4)
    output2 = model_192(x2)
    assert output2["preds"].shape == (2, 48, 2)


def test_softs_num_params():
    """Test that model parameter counting works."""
    model = SOFTS(
        input_dim=6,
        output_dim=3,
        seq_len=96,
        pred_len=24,
        hidden_dim=128,
        d_core=64,
        d_ff=256,
        e_layers=3,
    )

    num_params = model.get_num_params()
    assert num_params > 0
    assert isinstance(num_params, int)


def test_softs_repr():
    """Test model string representation."""
    model = SOFTS(
        input_dim=6,
        output_dim=3,
        seq_len=96,
        pred_len=24,
        hidden_dim=128,
    )

    repr_str = repr(model)
    assert "SOFTS" in repr_str
    assert "input_dim=6" in repr_str
    assert "output_dim=3" in repr_str


@pytest.mark.parametrize("activation", ["relu", "gelu"])
def test_softs_activations(activation):
    """Test SOFTS with different activation functions."""
    model = SOFTS(
        input_dim=4,
        output_dim=2,
        seq_len=96,
        pred_len=24,
        hidden_dim=128,
        d_core=64,
        d_ff=256,
        e_layers=2,
        activation=activation,
    )

    x = torch.randn(2, 96, 4)
    output = model(x)

    assert output["preds"].shape == (2, 24, 2)


@pytest.mark.parametrize("e_layers", [1, 2, 3, 4])
def test_softs_num_layers(e_layers):
    """Test SOFTS with different numbers of encoder layers."""
    model = SOFTS(
        input_dim=5,
        output_dim=5,
        seq_len=96,
        pred_len=24,
        hidden_dim=128,
        d_core=64,
        d_ff=256,
        e_layers=e_layers,
    )

    x = torch.randn(2, 96, 5)
    output = model(x)

    assert output["preds"].shape == (2, 24, 5)


def test_softs_batch_size_one():
    """Test SOFTS with batch size 1."""
    model = SOFTS(
        input_dim=5,
        output_dim=5,
        seq_len=96,
        pred_len=24,
        hidden_dim=128,
        d_core=64,
        d_ff=256,
        e_layers=2,
    )

    x = torch.randn(1, 96, 5)
    output = model(x)

    assert output["preds"].shape == (1, 24, 5)


def test_softs_d_core_variations():
    """Test SOFTS with different d_core values."""
    # Small compression
    model_small = SOFTS(
        input_dim=5,
        output_dim=5,
        seq_len=96,
        pred_len=24,
        hidden_dim=128,
        d_core=32,  # 1/4 of hidden_dim
        d_ff=256,
        e_layers=2,
    )

    # No compression
    model_no_compress = SOFTS(
        input_dim=5,
        output_dim=5,
        seq_len=96,
        pred_len=24,
        hidden_dim=128,
        d_core=128,  # Same as hidden_dim
        d_ff=256,
        e_layers=2,
    )

    x = torch.randn(2, 96, 5)

    out_small = model_small(x)
    out_no_compress = model_no_compress(x)

    assert out_small["preds"].shape == (2, 24, 5)
    assert out_no_compress["preds"].shape == (2, 24, 5)


def test_softs_dropout():
    """Test SOFTS with dropout enabled."""
    model = SOFTS(
        input_dim=5,
        output_dim=5,
        seq_len=96,
        pred_len=24,
        hidden_dim=128,
        d_core=64,
        d_ff=256,
        e_layers=2,
        dropout=0.1,
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


def test_softs_gradient_flow():
    """Test that gradients flow through the model."""
    model = SOFTS(
        input_dim=5,
        output_dim=5,
        seq_len=96,
        pred_len=24,
        hidden_dim=128,
        d_core=64,
        d_ff=256,
        e_layers=2,
    )

    x = torch.randn(2, 96, 5, requires_grad=True)
    output = model(x)

    # Compute a simple loss
    loss = output["preds"].mean()
    loss.backward()

    # Check that gradients exist and carry signal
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert torch.any(x.grad != 0)


def test_softs_large_batch():
    """Test SOFTS with a larger batch size."""
    model = SOFTS(
        input_dim=5,
        output_dim=5,
        seq_len=96,
        pred_len=24,
        hidden_dim=128,
        d_core=64,
        d_ff=256,
        e_layers=2,
    )

    x = torch.randn(32, 96, 5)  # Batch size 32
    output = model(x)

    assert output["preds"].shape == (32, 24, 5)
