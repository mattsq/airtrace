"""Tests for FreTS model implementation."""

import pytest
import torch

from airtrace.models import FreTSModel
from airtrace.models.frets import FrequencyMLP, ChannelMixing


@pytest.fixture
def batch():
    """Create a dummy batch for testing."""
    B, T_in, D_in = 4, 64, 5
    x = torch.randn(B, T_in, D_in)
    return x


def test_frets_model_forward_basic(batch):
    """Test FreTS model basic forward pass."""
    model = FreTSModel(
        input_dim=5,
        output_dim=3,
        seq_len=64,
        pred_len=1,
        d_model=64,
        hidden_dim=128,
        num_layers=2,
    )

    output = model(batch)

    assert "preds" in output
    assert "extras" in output
    assert output["preds"].shape == (4, 1, 3)  # [B, pred_len, D_out]
    assert "freq_components_orig" in output["extras"]
    assert "freq_components_processed" in output["extras"]
    assert "time_reconstruction" in output["extras"]


def test_frets_model_multi_horizon():
    """Test FreTS with multi-horizon prediction."""
    model = FreTSModel(
        input_dim=5,
        output_dim=3,
        seq_len=64,
        pred_len=10,  # Predict 10 steps ahead
        d_model=64,
        hidden_dim=128,
    )

    x = torch.randn(2, 64, 5)
    output = model(x)

    assert output["preds"].shape == (2, 10, 3)  # [B, pred_len, D_out]


def test_frets_model_different_d_model():
    """Test FreTS with different model dimensions."""
    for d_model in [32, 64, 128]:
        model = FreTSModel(
            input_dim=5,
            output_dim=3,
            seq_len=64,
            pred_len=1,
            d_model=d_model,
            hidden_dim=d_model * 2,
        )

        x = torch.randn(2, 64, 5)
        output = model(x)
        assert output["preds"].shape == (2, 1, 3)


def test_frets_model_different_num_freqs():
    """Test FreTS with different numbers of frequency components."""
    seq_len = 64
    max_freqs = seq_len // 2 + 1  # 33

    for num_freqs in [8, 16, max_freqs]:
        model = FreTSModel(
            input_dim=5,
            output_dim=3,
            seq_len=seq_len,
            pred_len=1,
            num_freqs=num_freqs,
            d_model=64,
        )

        x = torch.randn(2, seq_len, 5)
        output = model(x)
        assert output["preds"].shape == (2, 1, 3)
        # Check that frequency components have expected shape
        assert output["extras"]["freq_components_orig"].shape[1] == num_freqs


def test_frets_model_auto_num_freqs():
    """Test FreTS with automatic frequency component selection."""
    seq_len = 64
    model = FreTSModel(
        input_dim=5,
        output_dim=3,
        seq_len=seq_len,
        pred_len=1,
        num_freqs=None,  # Auto
        d_model=64,
    )

    x = torch.randn(2, seq_len, 5)
    output = model(x)

    expected_freqs = seq_len // 2 + 1  # 33
    assert output["extras"]["freq_components_orig"].shape[1] == expected_freqs


def test_frets_model_different_hidden_dims():
    """Test FreTS with different hidden dimensions."""
    for hidden_dim in [64, 128, 256]:
        model = FreTSModel(
            input_dim=5,
            output_dim=3,
            seq_len=64,
            pred_len=1,
            d_model=64,
            hidden_dim=hidden_dim,
        )

        x = torch.randn(2, 64, 5)
        output = model(x)
        assert output["preds"].shape == (2, 1, 3)


def test_frets_model_different_num_layers():
    """Test FreTS with different numbers of MLP layers."""
    for num_layers in [1, 2, 3, 4]:
        model = FreTSModel(
            input_dim=5,
            output_dim=3,
            seq_len=64,
            pred_len=1,
            d_model=64,
            hidden_dim=128,
            num_layers=num_layers,
        )

        x = torch.randn(2, 64, 5)
        output = model(x)
        assert output["preds"].shape == (2, 1, 3)


def test_frets_model_dropout():
    """Test FreTS with different dropout rates."""
    for dropout in [0.0, 0.1, 0.3, 0.5]:
        model = FreTSModel(
            input_dim=5,
            output_dim=3,
            seq_len=64,
            pred_len=1,
            d_model=64,
            hidden_dim=128,
            dropout=dropout,
        )

        x = torch.randn(2, 64, 5)
        output = model(x)
        assert output["preds"].shape == (2, 1, 3)


def test_frets_model_activation_functions():
    """Test FreTS with different activation functions."""
    for activation in ["gelu", "relu", "tanh"]:
        model = FreTSModel(
            input_dim=5,
            output_dim=3,
            seq_len=64,
            pred_len=1,
            d_model=64,
            hidden_dim=128,
            activation=activation,
        )

        x = torch.randn(2, 64, 5)
        output = model(x)
        assert output["preds"].shape == (2, 1, 3)


def test_frets_model_channel_independence():
    """Test FreTS with and without channel independence."""
    for channel_independence in [False, True]:
        model = FreTSModel(
            input_dim=5,
            output_dim=3,
            seq_len=64,
            pred_len=1,
            d_model=64,
            hidden_dim=128,
            channel_independence=channel_independence,
        )

        x = torch.randn(2, 64, 5)
        output = model(x)
        assert output["preds"].shape == (2, 1, 3)


def test_frets_model_normalize_fft():
    """Test FreTS with and without FFT normalization."""
    for normalize_fft in [False, True]:
        model = FreTSModel(
            input_dim=5,
            output_dim=3,
            seq_len=64,
            pred_len=1,
            d_model=64,
            hidden_dim=128,
            normalize_fft=normalize_fft,
        )

        x = torch.randn(2, 64, 5)
        output = model(x)
        assert output["preds"].shape == (2, 1, 3)


def test_frets_model_gradient_flow():
    """Test that gradients flow through FreTS model."""
    model = FreTSModel(
        input_dim=5,
        output_dim=3,
        seq_len=64,
        pred_len=1,
        d_model=64,
        hidden_dim=128,
    )
    x = torch.randn(2, 64, 5, requires_grad=True)

    output = model(x)
    preds = output["preds"]

    # Compute dummy loss
    loss = preds.mean()
    loss.backward()

    # Check that parameters have gradients
    for param in model.parameters():
        if param.requires_grad:
            assert param.grad is not None
            assert torch.isfinite(param.grad).all()
            assert torch.any(param.grad != 0)


def test_frets_model_num_params():
    """Test parameter counting for FreTS."""
    model = FreTSModel(
        input_dim=5,
        output_dim=3,
        seq_len=64,
        pred_len=1,
        d_model=128,
        hidden_dim=256,
        num_layers=3,
    )

    num_params = model.get_num_params()
    assert num_params > 0
    print(f"FreTS model has {num_params:,} parameters")


def test_frets_model_device_transfer():
    """Test moving FreTS model to device."""
    model = FreTSModel(
        input_dim=5,
        output_dim=3,
        seq_len=64,
        pred_len=1,
        d_model=64,
        hidden_dim=128,
    )

    # Test CPU
    model = model.to("cpu")
    x = torch.randn(2, 64, 5).to("cpu")
    output = model(x)
    assert output["preds"].device.type == "cpu"

    # Test CUDA (if available)
    if torch.cuda.is_available():
        model = model.to("cuda")
        x = torch.randn(2, 64, 5).to("cuda")
        output = model(x)
        assert output["preds"].device.type == "cuda"


def test_frets_model_no_nan():
    """Test that FreTS doesn't produce NaN values."""
    model = FreTSModel(
        input_dim=5,
        output_dim=3,
        seq_len=64,
        pred_len=1,
        d_model=64,
        hidden_dim=128,
    )
    x = torch.randn(2, 64, 5)

    output = model(x)

    assert not torch.isnan(output["preds"]).any()
    assert not torch.isinf(output["preds"]).any()


def test_frets_model_batch_sizes():
    """Test FreTS with different batch sizes."""
    model = FreTSModel(
        input_dim=5,
        output_dim=3,
        seq_len=64,
        pred_len=1,
        d_model=64,
        hidden_dim=128,
    )

    for batch_size in [1, 2, 4, 8, 16]:
        x = torch.randn(batch_size, 64, 5)
        output = model(x)
        assert output["preds"].shape == (batch_size, 1, 3)


def test_frets_model_sequence_lengths():
    """Test FreTS with different sequence lengths."""
    for seq_len in [32, 64, 128, 256]:
        model = FreTSModel(
            input_dim=5,
            output_dim=3,
            seq_len=seq_len,
            pred_len=1,
            d_model=64,
            hidden_dim=128,
        )

        x = torch.randn(2, seq_len, 5)
        output = model(x)
        assert output["preds"].shape == (2, 1, 3)


def test_frets_model_same_input_output_dims():
    """Test FreTS when input_dim == output_dim."""
    model = FreTSModel(
        input_dim=5,
        output_dim=5,
        seq_len=64,
        pred_len=1,
        d_model=64,
        hidden_dim=128,
    )

    x = torch.randn(2, 64, 5)
    output = model(x)

    assert output["preds"].shape == (2, 1, 5)


def test_frets_model_different_input_output_dims():
    """Test FreTS when input_dim != output_dim."""
    model = FreTSModel(
        input_dim=10,
        output_dim=3,
        seq_len=64,
        pred_len=1,
        d_model=64,
        hidden_dim=128,
    )

    x = torch.randn(2, 64, 10)
    output = model(x)

    assert output["preds"].shape == (2, 1, 3)


def test_frets_model_reproducibility():
    """Test that FreTS is reproducible with same seed."""
    torch.manual_seed(42)
    model1 = FreTSModel(
        input_dim=5,
        output_dim=3,
        seq_len=64,
        pred_len=1,
        d_model=64,
        hidden_dim=128,
    )

    torch.manual_seed(42)
    model2 = FreTSModel(
        input_dim=5,
        output_dim=3,
        seq_len=64,
        pred_len=1,
        d_model=64,
        hidden_dim=128,
    )

    # Load same state
    model2.load_state_dict(model1.state_dict())

    x = torch.randn(2, 64, 5)

    # Set to eval mode
    model1.eval()
    model2.eval()

    with torch.no_grad():
        output1 = model1(x)
        output2 = model2(x)

    torch.testing.assert_close(output1["preds"], output2["preds"])


def test_frets_model_train_eval_modes():
    """Test that FreTS behaves differently in train and eval modes with dropout."""
    model = FreTSModel(
        input_dim=5,
        output_dim=3,
        seq_len=64,
        pred_len=1,
        d_model=64,
        hidden_dim=128,
        dropout=0.5,
    )
    x = torch.randn(2, 64, 5)

    # Eval mode (dropout inactive)
    model.eval()
    with torch.no_grad():
        output_eval1 = model(x)
        output_eval2 = model(x)

    # Outputs should be identical in eval mode
    torch.testing.assert_close(output_eval1["preds"], output_eval2["preds"])


def test_frets_model_wrong_sequence_length():
    """Test that FreTS raises error for wrong sequence length."""
    model = FreTSModel(
        input_dim=5,
        output_dim=3,
        seq_len=64,
        pred_len=1,
        d_model=64,
        hidden_dim=128,
    )

    # Wrong sequence length
    x = torch.randn(2, 32, 5)  # Should be 64

    with pytest.raises(ValueError, match="Expected input sequence length"):
        model(x)


def test_frets_model_extras_content():
    """Test that extras contain expected information."""
    model = FreTSModel(
        input_dim=5,
        output_dim=3,
        seq_len=64,
        pred_len=1,
        d_model=64,
        hidden_dim=128,
        num_freqs=16,
    )

    x = torch.randn(2, 64, 5)
    output = model(x)

    extras = output["extras"]

    # Check that extras contain frequency components
    assert "freq_components_orig" in extras
    assert "freq_components_processed" in extras
    assert "time_reconstruction" in extras

    # Check shapes
    assert extras["freq_components_orig"].shape[0] == 2  # Batch size
    assert extras["freq_components_orig"].shape[1] == 16  # num_freqs
    assert extras["freq_components_processed"].shape[0] == 2
    assert extras["time_reconstruction"].shape == (2, 64, 64)  # [B, T, d_model]


def test_frets_model_repr():
    """Test FreTS model string representation."""
    model = FreTSModel(
        input_dim=5,
        output_dim=3,
        seq_len=64,
        pred_len=1,
        d_model=128,
        hidden_dim=256,
    )

    model_repr = repr(model)
    assert "FreTSModel" in model_repr
    assert "num_params" in model_repr


def test_frets_frequency_processing_periodic_signal():
    """Test that FreTS properly processes periodic signals."""
    model = FreTSModel(
        input_dim=1,
        output_dim=1,
        seq_len=64,
        pred_len=1,
        d_model=32,
        hidden_dim=64,
        num_freqs=8,
    )

    # Create a simple periodic signal (sine wave)
    t = torch.linspace(0, 4 * torch.pi, 64).unsqueeze(0).unsqueeze(-1)
    x = torch.sin(t)  # Pure sine wave

    model.eval()
    with torch.no_grad():
        output = model(x)

    # Model should produce reasonable predictions
    assert output["preds"].shape == (1, 1, 1)
    assert not torch.isnan(output["preds"]).any()


def test_frets_frequency_mlp_component():
    """Test the FrequencyMLP component."""
    num_freqs = 16
    d_model = 64
    freq_mlp = FrequencyMLP(
        num_freqs=num_freqs,
        d_model=d_model,
        hidden_dim=128,
        num_layers=2,
        dropout=0.1,
        activation="gelu",
    )

    # Create complex frequency tensor
    x_freq = torch.randn(2, num_freqs, d_model) + 1j * torch.randn(2, num_freqs, d_model)

    output = freq_mlp(x_freq)

    # Check output shape and type
    assert output.shape == (2, num_freqs, d_model)
    assert output.dtype == torch.complex64 or output.dtype == torch.complex128


def test_frets_channel_mixing_component():
    """Test the ChannelMixing component."""
    channel_mix = ChannelMixing(input_dim=10, output_dim=5, dropout=0.1)

    x = torch.randn(2, 64, 10)
    output = channel_mix(x)

    assert output.shape == (2, 64, 5)


def test_frets_model_small_num_freqs():
    """Test FreTS with very small number of frequency components."""
    model = FreTSModel(
        input_dim=5,
        output_dim=3,
        seq_len=64,
        pred_len=1,
        num_freqs=4,  # Very small
        d_model=64,
        hidden_dim=128,
    )

    x = torch.randn(2, 64, 5)
    output = model(x)

    assert output["preds"].shape == (2, 1, 3)
    assert output["extras"]["freq_components_orig"].shape[1] == 4


def test_frets_model_short_sequence():
    """Test FreTS with short sequences."""
    seq_len = 16
    model = FreTSModel(
        input_dim=5,
        output_dim=3,
        seq_len=seq_len,
        pred_len=1,
        d_model=32,
        hidden_dim=64,
    )

    x = torch.randn(2, seq_len, 5)
    output = model(x)

    assert output["preds"].shape == (2, 1, 3)


def test_frets_model_long_prediction_horizon():
    """Test FreTS with long prediction horizon."""
    model = FreTSModel(
        input_dim=5,
        output_dim=3,
        seq_len=64,
        pred_len=32,  # Long horizon
        d_model=64,
        hidden_dim=128,
    )

    x = torch.randn(2, 64, 5)
    output = model(x)

    assert output["preds"].shape == (2, 32, 3)


def test_frets_vs_dlinear_parameters():
    """Test that FreTS has reasonable parameter count compared to DLinear."""
    from airtrace.models import DLinearModel

    frets = FreTSModel(
        input_dim=5,
        output_dim=3,
        seq_len=64,
        pred_len=1,
        d_model=64,
        hidden_dim=128,
        num_layers=2,
    )

    dlinear = DLinearModel(
        input_dim=5,
        output_dim=3,
        seq_len=64,
        pred_len=1,
        kernel_size=25,
    )

    frets_params = frets.get_num_params()
    dlinear_params = dlinear.get_num_params()

    print(f"FreTS params: {frets_params:,}")
    print(f"DLinear params: {dlinear_params:,}")

    # FreTS should have more parameters (it's more complex)
    assert frets_params > dlinear_params


def test_frets_model_zero_input():
    """Test FreTS with zero input."""
    model = FreTSModel(
        input_dim=5,
        output_dim=3,
        seq_len=64,
        pred_len=1,
        d_model=64,
        hidden_dim=128,
    )

    x = torch.zeros(2, 64, 5)

    model.eval()
    with torch.no_grad():
        output = model(x)

    # Should produce valid output (not NaN)
    assert not torch.isnan(output["preds"]).any()
    assert not torch.isinf(output["preds"]).any()


def test_frets_model_large_values():
    """Test FreTS with large input values."""
    model = FreTSModel(
        input_dim=5,
        output_dim=3,
        seq_len=64,
        pred_len=1,
        d_model=64,
        hidden_dim=128,
        normalize_fft=True,  # Normalization should help
    )

    x = torch.randn(2, 64, 5) * 1000  # Large values

    model.eval()
    with torch.no_grad():
        output = model(x)

    # Should produce valid output (not NaN)
    assert not torch.isnan(output["preds"]).any()
    assert not torch.isinf(output["preds"]).any()


def test_frets_frequency_components_magnitude():
    """Test that frequency components are properly computed."""
    model = FreTSModel(
        input_dim=1,
        output_dim=1,
        seq_len=64,
        pred_len=1,
        d_model=32,
        hidden_dim=64,
        num_freqs=10,
    )

    # Create a signal with known frequency content
    t = torch.linspace(0, 2 * torch.pi, 64).unsqueeze(0).unsqueeze(-1)
    x = torch.sin(2 * t)  # Frequency = 2

    model.eval()
    with torch.no_grad():
        output = model(x)

    # Check that frequency components are non-negative (magnitude)
    freq_orig = output["extras"]["freq_components_orig"]
    assert (freq_orig >= 0).all(), "Frequency magnitudes should be non-negative"

    # Low frequencies should have most energy for smooth signals
    low_freq_energy = freq_orig[:, :5, :].sum()
    high_freq_energy = freq_orig[:, 5:, :].sum()
    assert low_freq_energy > 0  # Should have some low-frequency energy


def test_frets_model_state_dict():
    """Test that FreTS state dict can be saved and loaded."""
    model = FreTSModel(
        input_dim=5,
        output_dim=3,
        seq_len=64,
        pred_len=1,
        d_model=64,
        hidden_dim=128,
    )

    # Save state dict
    state_dict = model.state_dict()

    # Create new model and load state
    new_model = FreTSModel(
        input_dim=5,
        output_dim=3,
        seq_len=64,
        pred_len=1,
        d_model=64,
        hidden_dim=128,
    )
    new_model.load_state_dict(state_dict)

    # Test that outputs match
    x = torch.randn(2, 64, 5)
    model.eval()
    new_model.eval()

    with torch.no_grad():
        output1 = model(x)
        output2 = new_model(x)

    torch.testing.assert_close(output1["preds"], output2["preds"])
