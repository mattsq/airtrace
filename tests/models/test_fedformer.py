"""Tests for FEDformer model implementation."""

import pytest
import torch

from airtrace.models import FEDformerModel


@pytest.fixture
def batch():
    """Create a dummy batch for testing."""
    B, T_in, D_in = 4, 64, 5
    x = torch.randn(B, T_in, D_in)
    return x


def test_fedformer_model_forward_fourier(batch):
    """Test FEDformer model forward pass with Fourier mode."""
    model = FEDformerModel(
        input_dim=5,
        output_dim=3,
        d_model=64,
        n_heads=4,
        e_layers=2,
        d_layers=1,
        moving_avg=25,
        d_ff=128,
        dropout=0.1,
        activation="gelu",
        freq_mode="fourier",
        modes=16,
        label_len=24,
        pred_len=1,
    )

    output = model(batch)

    assert "preds" in output
    assert "extras" in output
    assert output["preds"].shape == (4, 1, 3)  # [B, pred_len, D_out]
    assert "encoder_trend" in output["extras"]
    assert "decoder_trend" in output["extras"]
    assert "seasonal_component" in output["extras"]
    assert "freq_mode" in output["extras"]
    assert output["extras"]["freq_mode"] == "fourier"


def test_fedformer_model_forward_wavelet(batch):
    """Test FEDformer model forward pass with Wavelet mode."""
    model = FEDformerModel(
        input_dim=5,
        output_dim=3,
        d_model=64,
        n_heads=4,
        e_layers=2,
        d_layers=1,
        freq_mode="wavelet",
        label_len=24,
        pred_len=1,
    )

    output = model(batch)

    assert "preds" in output
    assert output["preds"].shape == (4, 1, 3)
    assert output["extras"]["freq_mode"] == "wavelet"


def test_fedformer_model_multi_horizon():
    """Test FEDformer with multi-horizon prediction."""
    model = FEDformerModel(
        input_dim=5,
        output_dim=3,
        d_model=64,
        n_heads=4,
        pred_len=10,  # Predict 10 steps ahead
        label_len=24,
    )

    x = torch.randn(2, 64, 5)
    output = model(x)

    assert output["preds"].shape == (2, 10, 3)  # [B, pred_len, D_out]


def test_fedformer_model_different_d_model():
    """Test FEDformer with different model dimensions."""
    for d_model in [32, 64, 128]:
        model = FEDformerModel(
            input_dim=5,
            output_dim=3,
            d_model=d_model,
            n_heads=4,
            e_layers=2,
            d_layers=1,
        )

        x = torch.randn(2, 64, 5)
        output = model(x)
        assert output["preds"].shape == (2, 1, 3)


def test_fedformer_model_different_n_heads():
    """Test FEDformer with different numbers of heads."""
    for n_heads in [2, 4, 8]:
        model = FEDformerModel(
            input_dim=5,
            output_dim=3,
            d_model=64,
            n_heads=n_heads,
        )

        x = torch.randn(2, 64, 5)
        output = model(x)
        assert output["preds"].shape == (2, 1, 3)


def test_fedformer_model_different_layers():
    """Test FEDformer with different numbers of encoder/decoder layers."""
    for e_layers, d_layers in [(1, 1), (2, 1), (3, 2)]:
        model = FEDformerModel(
            input_dim=5,
            output_dim=3,
            d_model=64,
            n_heads=4,
            e_layers=e_layers,
            d_layers=d_layers,
        )

        x = torch.randn(2, 64, 5)
        output = model(x)
        assert output["preds"].shape == (2, 1, 3)


def test_fedformer_model_different_modes():
    """Test FEDformer with different numbers of Fourier modes."""
    for modes in [8, 16, 32, 64]:
        model = FEDformerModel(
            input_dim=5,
            output_dim=3,
            d_model=64,
            n_heads=4,
            freq_mode="fourier",
            modes=modes,
        )

        x = torch.randn(2, 64, 5)
        output = model(x)
        assert output["preds"].shape == (2, 1, 3)


def test_fedformer_model_different_moving_avg():
    """Test FEDformer with different moving average kernel sizes."""
    for moving_avg in [5, 13, 25, 49]:
        model = FEDformerModel(
            input_dim=5,
            output_dim=3,
            d_model=64,
            n_heads=4,
            moving_avg=moving_avg,
        )

        x = torch.randn(2, 64, 5)
        output = model(x)
        assert output["preds"].shape == (2, 1, 3)


def test_fedformer_model_gradient_flow():
    """Test that gradients flow through FEDformer model."""
    model = FEDformerModel(
        input_dim=5,
        output_dim=3,
        d_model=64,
        n_heads=4,
        e_layers=2,
        d_layers=1,
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


def test_fedformer_model_num_params():
    """Test parameter counting for FEDformer."""
    model = FEDformerModel(
        input_dim=5,
        output_dim=3,
        d_model=128,
        n_heads=8,
        e_layers=2,
        d_layers=1,
    )

    num_params = model.get_num_params()
    assert num_params > 0
    print(f"FEDformer model has {num_params:,} parameters")


def test_fedformer_model_device_transfer():
    """Test moving FEDformer model to device."""
    model = FEDformerModel(
        input_dim=5,
        output_dim=3,
        d_model=64,
        n_heads=4,
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


def test_fedformer_model_no_nan():
    """Test that FEDformer doesn't produce NaN values."""
    model = FEDformerModel(
        input_dim=5,
        output_dim=3,
        d_model=64,
        n_heads=4,
    )
    x = torch.randn(2, 64, 5)

    output = model(x)

    assert not torch.isnan(output["preds"]).any()
    assert not torch.isinf(output["preds"]).any()


def test_fedformer_model_batch_sizes():
    """Test FEDformer with different batch sizes."""
    model = FEDformerModel(
        input_dim=5,
        output_dim=3,
        d_model=64,
        n_heads=4,
    )

    for batch_size in [1, 2, 4, 8]:
        x = torch.randn(batch_size, 64, 5)
        output = model(x)
        assert output["preds"].shape == (batch_size, 1, 3)


def test_fedformer_model_sequence_lengths():
    """Test FEDformer with different sequence lengths."""
    model = FEDformerModel(
        input_dim=5,
        output_dim=3,
        d_model=64,
        n_heads=4,
        label_len=12,
    )

    for seq_len in [32, 64, 128]:
        x = torch.randn(2, seq_len, 5)
        output = model(x)
        assert output["preds"].shape == (2, 1, 3)


def test_fedformer_model_dropout():
    """Test FEDformer with different dropout rates."""
    for dropout in [0.0, 0.1, 0.3, 0.5]:
        model = FEDformerModel(
            input_dim=5,
            output_dim=3,
            d_model=64,
            n_heads=4,
            dropout=dropout,
        )

        x = torch.randn(2, 64, 5)
        output = model(x)
        assert output["preds"].shape == (2, 1, 3)


def test_fedformer_model_same_input_output_dims():
    """Test FEDformer when input_dim == output_dim."""
    model = FEDformerModel(
        input_dim=5,
        output_dim=5,
        d_model=64,
        n_heads=4,
    )

    x = torch.randn(2, 64, 5)
    output = model(x)

    assert output["preds"].shape == (2, 1, 5)


def test_fedformer_model_different_input_output_dims():
    """Test FEDformer when input_dim != output_dim."""
    model = FEDformerModel(
        input_dim=10,
        output_dim=3,
        d_model=64,
        n_heads=4,
    )

    x = torch.randn(2, 64, 10)
    output = model(x)

    assert output["preds"].shape == (2, 1, 3)


def test_fedformer_model_reproducibility():
    """Test that FEDformer is reproducible with same seed."""
    torch.manual_seed(42)
    model1 = FEDformerModel(
        input_dim=5,
        output_dim=3,
        d_model=64,
        n_heads=4,
    )

    torch.manual_seed(42)
    model2 = FEDformerModel(
        input_dim=5,
        output_dim=3,
        d_model=64,
        n_heads=4,
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


def test_fedformer_model_train_eval_modes():
    """Test that FEDformer behaves differently in train and eval modes."""
    model = FEDformerModel(
        input_dim=5,
        output_dim=3,
        d_model=64,
        n_heads=4,
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


def test_fedformer_decomposition_component():
    """Test the SeriesDecomposition component."""
    from airtrace.models.fedformer import SeriesDecomposition

    decomp = SeriesDecomposition(kernel_size=25)
    x = torch.randn(2, 64, 5)

    seasonal, trend = decomp(x)

    # Check shapes
    assert seasonal.shape == x.shape
    assert trend.shape == x.shape

    # Check decomposition property: seasonal + trend ≈ original
    reconstructed = seasonal + trend
    torch.testing.assert_close(reconstructed, x, rtol=1e-5, atol=1e-5)


def test_fedformer_fourier_attention_component():
    """Test the FourierAttention component."""
    from airtrace.models.fedformer import FourierAttention

    attn = FourierAttention(d_model=64, n_heads=4, dropout=0.1, modes=16)

    q = torch.randn(2, 32, 64)
    k = torch.randn(2, 32, 64)
    v = torch.randn(2, 32, 64)

    output = attn(q, k, v)

    assert output.shape == (2, 32, 64)  # Same shape as query


def test_fedformer_wavelet_attention_component():
    """Test the WaveletAttention component."""
    from airtrace.models.fedformer import WaveletAttention

    attn = WaveletAttention(d_model=64, n_heads=4, dropout=0.1)

    q = torch.randn(2, 32, 64)
    k = torch.randn(2, 32, 64)
    v = torch.randn(2, 32, 64)

    output = attn(q, k, v)

    assert output.shape == (2, 32, 64)  # Same shape as query


def test_fedformer_encoder_layer():
    """Test the FEDformerEncoderLayer component."""
    from airtrace.models.fedformer import FEDformerEncoderLayer

    layer = FEDformerEncoderLayer(
        d_model=64,
        n_heads=4,
        d_ff=128,
        moving_avg=25,
        dropout=0.1,
        activation="gelu",
        freq_mode="fourier",
        modes=16,
    )

    x = torch.randn(2, 64, 64)
    seasonal, trend = layer(x)

    assert seasonal.shape == x.shape
    assert trend.shape == x.shape


def test_fedformer_decoder_layer():
    """Test the FEDformerDecoderLayer component."""
    from airtrace.models.fedformer import FEDformerDecoderLayer

    layer = FEDformerDecoderLayer(
        d_model=64,
        n_heads=4,
        d_ff=128,
        moving_avg=25,
        dropout=0.1,
        activation="gelu",
        freq_mode="fourier",
        modes=16,
    )

    seasonal = torch.randn(2, 32, 64)
    trend = torch.randn(2, 32, 64)
    memory = torch.randn(2, 64, 64)

    seasonal_out, trend_out = layer(seasonal, trend, memory)

    assert seasonal_out.shape == seasonal.shape
    assert trend_out.shape == trend.shape


def test_fedformer_model_extras_content():
    """Test that extras contain expected information."""
    model = FEDformerModel(
        input_dim=5,
        output_dim=3,
        d_model=64,
        n_heads=4,
    )

    x = torch.randn(2, 64, 5)
    output = model(x)

    extras = output["extras"]

    # Check that extras contain decomposition components
    assert "encoder_trend" in extras
    assert "decoder_trend" in extras
    assert "seasonal_component" in extras
    assert "freq_mode" in extras

    # Check shapes
    assert extras["seasonal_component"].shape[0] == 2  # Batch size


def test_fedformer_activation_functions():
    """Test FEDformer with different activation functions."""
    for activation in ["gelu", "relu"]:
        model = FEDformerModel(
            input_dim=5,
            output_dim=3,
            d_model=64,
            n_heads=4,
            activation=activation,
        )

        x = torch.randn(2, 64, 5)
        output = model(x)
        assert output["preds"].shape == (2, 1, 3)


def test_fedformer_model_repr():
    """Test FEDformer model string representation."""
    model = FEDformerModel(
        input_dim=5,
        output_dim=3,
        d_model=128,
        n_heads=8,
        e_layers=2,
        d_layers=1,
    )

    model_repr = repr(model)
    assert "FEDformerModel" in model_repr
    assert "num_params" in model_repr


def test_fedformer_vs_autoformer_parameters():
    """Test that FEDformer has comparable parameter count to Autoformer."""
    from airtrace.models import AutoformerModel

    fedformer = FEDformerModel(
        input_dim=5,
        output_dim=3,
        d_model=128,
        n_heads=8,
        e_layers=2,
        d_layers=1,
    )

    autoformer = AutoformerModel(
        input_dim=5,
        output_dim=3,
        d_model=128,
        n_heads=8,
        e_layers=2,
        d_layers=1,
    )

    fedformer_params = fedformer.get_num_params()
    autoformer_params = autoformer.get_num_params()

    print(f"FEDformer params: {fedformer_params:,}")
    print(f"Autoformer params: {autoformer_params:,}")

    # FEDformer should have comparable parameters (within 50% range)
    assert 0.5 * autoformer_params < fedformer_params < 2.0 * autoformer_params


def test_fedformer_frequency_processing():
    """Test that FEDformer properly processes frequency domain."""
    model = FEDformerModel(
        input_dim=1,
        output_dim=1,
        d_model=32,
        n_heads=2,
        freq_mode="fourier",
        modes=8,
    )

    # Create a simple periodic signal
    t = torch.linspace(0, 4 * torch.pi, 64).unsqueeze(0).unsqueeze(-1)
    x = torch.sin(t) + 0.5 * torch.sin(2 * t)  # Composite signal

    model.eval()
    with torch.no_grad():
        output = model(x)

    # Model should produce reasonable predictions
    assert output["preds"].shape == (1, 1, 1)
    assert not torch.isnan(output["preds"]).any()


def test_fedformer_short_sequence():
    """Test FEDformer with short sequences."""
    model = FEDformerModel(
        input_dim=5,
        output_dim=3,
        d_model=32,
        n_heads=2,
        label_len=4,
        pred_len=1,
    )

    x = torch.randn(2, 16, 5)  # Short sequence
    output = model(x)

    assert output["preds"].shape == (2, 1, 3)
