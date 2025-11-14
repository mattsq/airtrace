"""Tests for model implementations."""

import pytest
import torch

from airtrace.models import (
    GRUARModel,
    LinearTrendModel,
    MeanModel,
    MovingAverageModel,
    PersistenceModel,
    TCNModel,
    TransformerModel,
    ZeroModel,
)


@pytest.fixture
def batch():
    """Create a dummy batch for testing."""
    B, T_in, D_in = 4, 32, 5
    x = torch.randn(B, T_in, D_in)
    return x


def test_gru_ar_model_forward(batch):
    """Test GRU AR model forward pass."""
    model = GRUARModel(
        input_dim=5,
        output_dim=3,
        hidden_size=64,
        num_layers=2
    )

    output = model(batch)

    assert "preds" in output
    assert "extras" in output
    assert output["preds"].shape == (4, 1, 3)  # [B, 1, D_out]


def test_tcn_model_forward(batch):
    """Test TCN model forward pass."""
    model = TCNModel(
        input_dim=5,
        output_dim=3,
        num_channels=[32, 64, 64],
        kernel_size=3
    )

    output = model(batch)

    assert "preds" in output
    assert output["preds"].shape == (4, 1, 3)


def test_transformer_model_forward(batch):
    """Test Transformer model forward pass."""
    model = TransformerModel(
        input_dim=5,
        output_dim=3,
        d_model=64,
        nhead=4,
        num_encoder_layers=2
    )

    output = model(batch)

    assert "preds" in output
    assert output["preds"].shape == (4, 1, 3)


def test_model_num_params():
    """Test parameter counting."""
    model = GRUARModel(
        input_dim=5,
        output_dim=3,
        hidden_size=64,
        num_layers=2
    )

    num_params = model.get_num_params()
    assert num_params > 0
    print(f"GRU model has {num_params:,} parameters")


def test_model_device_transfer():
    """Test moving model to device."""
    model = GRUARModel(input_dim=5, output_dim=3)

    # Test CPU
    model = model.to("cpu")
    x = torch.randn(2, 16, 5).to("cpu")
    output = model(x)
    assert output["preds"].device.type == "cpu"

    # Test CUDA (if available)
    if torch.cuda.is_available():
        model = model.to("cuda")
        x = torch.randn(2, 16, 5).to("cuda")
        output = model(x)
        assert output["preds"].device.type == "cuda"


def test_model_gradient_flow():
    """Test that gradients flow through model."""
    model = GRUARModel(input_dim=5, output_dim=3)
    x = torch.randn(2, 16, 5, requires_grad=True)

    output = model(x)
    preds = output["preds"]

    # Compute dummy loss
    loss = preds.mean()
    loss.backward()

    # Check that parameters have gradients
    for param in model.parameters():
        if param.requires_grad:
            assert param.grad is not None


@pytest.mark.parametrize("model_class,kwargs", [
    (GRUARModel, {"hidden_size": 32, "num_layers": 1}),
    (TCNModel, {"num_channels": [32, 32], "kernel_size": 3}),
    (TransformerModel, {"d_model": 32, "nhead": 4, "num_encoder_layers": 1}),
])
def test_model_output_shape(model_class, kwargs):
    """Test output shapes for different models."""
    model = model_class(input_dim=5, output_dim=3, **kwargs)

    x = torch.randn(2, 16, 5)
    output = model(x)

    assert output["preds"].shape[0] == 2  # Batch size
    assert output["preds"].shape[2] == 3  # Output dim


# ============================================================================
# Baseline Model Tests
# ============================================================================


def test_persistence_model_forward(batch):
    """Test persistence model forward pass."""
    model = PersistenceModel(input_dim=5, output_dim=3)

    output = model(batch)

    assert "preds" in output
    assert output["preds"].shape == (4, 1, 3)  # [B, 1, D_out]


def test_persistence_model_same_dims():
    """Test persistence model with same input/output dims."""
    model = PersistenceModel(input_dim=5, output_dim=5)
    x = torch.randn(2, 10, 5)

    output = model(x)

    # Should return last value
    expected = x[:, -1:, :]  # [2, 1, 5]
    assert output["preds"].shape == expected.shape
    torch.testing.assert_close(output["preds"], expected)


def test_zero_model_forward(batch):
    """Test zero model forward pass."""
    model = ZeroModel(input_dim=5, output_dim=3)

    output = model(batch)

    assert "preds" in output
    assert output["preds"].shape == (4, 1, 3)
    # Check all predictions are zero
    torch.testing.assert_close(output["preds"], torch.zeros_like(output["preds"]))


def test_mean_model_forward(batch):
    """Test mean model forward pass."""
    model = MeanModel(input_dim=5, output_dim=3)

    output = model(batch)

    assert "preds" in output
    assert output["preds"].shape == (4, 1, 3)


def test_mean_model_same_dims():
    """Test mean model with same input/output dims."""
    model = MeanModel(input_dim=5, output_dim=5)
    x = torch.randn(2, 10, 5)

    output = model(x)

    # Should return mean across time
    expected = x.mean(dim=1, keepdim=True)  # [2, 1, 5]
    assert output["preds"].shape == expected.shape
    torch.testing.assert_close(output["preds"], expected)


def test_moving_average_model_forward(batch):
    """Test moving average model forward pass."""
    model = MovingAverageModel(input_dim=5, output_dim=3, window_size=5)

    output = model(batch)

    assert "preds" in output
    assert output["preds"].shape == (4, 1, 3)
    assert "window_size" in output["extras"]


def test_moving_average_model_full_window():
    """Test moving average with full window (None)."""
    model = MovingAverageModel(input_dim=5, output_dim=5, window_size=None)
    x = torch.randn(2, 10, 5)

    output = model(x)

    # Should return mean of all timesteps
    expected = x.mean(dim=1, keepdim=True)  # [2, 1, 5]
    assert output["preds"].shape == expected.shape
    torch.testing.assert_close(output["preds"], expected)


def test_moving_average_model_partial_window():
    """Test moving average with partial window."""
    model = MovingAverageModel(input_dim=5, output_dim=5, window_size=3)
    x = torch.randn(2, 10, 5)

    output = model(x)

    # Should return mean of last 3 timesteps
    expected = x[:, -3:, :].mean(dim=1, keepdim=True)  # [2, 1, 5]
    assert output["preds"].shape == expected.shape
    torch.testing.assert_close(output["preds"], expected)


def test_linear_trend_model_forward(batch):
    """Test linear trend model forward pass."""
    model = LinearTrendModel(input_dim=5, output_dim=3)

    output = model(batch)

    assert "preds" in output
    assert output["preds"].shape == (4, 1, 3)
    assert "slope" in output["extras"]
    assert "intercept" in output["extras"]


def test_linear_trend_model_constant_sequence():
    """Test linear trend on constant sequence."""
    model = LinearTrendModel(input_dim=5, output_dim=5)

    # Create constant sequence
    x = torch.ones(2, 10, 5) * 5.0

    output = model(x)

    # For constant sequence, prediction should be the constant value
    assert output["preds"].shape == (2, 1, 5)
    # Slope should be near zero
    assert output["extras"]["slope"].abs().max() < 1e-5
    # Prediction should be close to constant value
    torch.testing.assert_close(
        output["preds"],
        torch.ones_like(output["preds"]) * 5.0,
        atol=1e-5,
        rtol=1e-5
    )


def test_linear_trend_model_linear_sequence():
    """Test linear trend on actual linear sequence."""
    model = LinearTrendModel(input_dim=1, output_dim=1)

    # Create linear sequence: y = 2 + 3*t
    t = torch.arange(10, dtype=torch.float32)
    x = (2 + 3 * t).reshape(1, 10, 1)  # [1, 10, 1]

    output = model(x)

    # Should predict next value: y(10) = 2 + 3*10 = 32
    expected = torch.tensor([[[32.0]]])
    torch.testing.assert_close(output["preds"], expected, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("model_class,kwargs", [
    (PersistenceModel, {}),
    (ZeroModel, {}),
    (MeanModel, {}),
    (MovingAverageModel, {"window_size": 5}),
    (LinearTrendModel, {}),
])
def test_baseline_models_no_nan(model_class, kwargs):
    """Test that baseline models don't produce NaN values."""
    model = model_class(input_dim=5, output_dim=3, **kwargs)
    x = torch.randn(2, 16, 5)

    output = model(x)

    assert not torch.isnan(output["preds"]).any()
    assert not torch.isinf(output["preds"]).any()


@pytest.mark.parametrize("model_class,kwargs", [
    (PersistenceModel, {}),
    (ZeroModel, {}),
    (MeanModel, {}),
    (MovingAverageModel, {"window_size": 5}),
    (LinearTrendModel, {}),
])
def test_baseline_models_num_params(model_class, kwargs):
    """Test that baseline models have minimal parameters."""
    model = model_class(input_dim=5, output_dim=3, **kwargs)

    num_params = model.get_num_params()
    # Baselines should have very few parameters (only projection if input_dim != output_dim)
    assert num_params <= 5 * 3  # At most a linear projection


def test_baseline_models_deterministic():
    """Test that baseline models are deterministic."""
    models = [
        PersistenceModel(input_dim=5, output_dim=3),
        ZeroModel(input_dim=5, output_dim=3),
        MeanModel(input_dim=5, output_dim=3),
        MovingAverageModel(input_dim=5, output_dim=3, window_size=5),
        LinearTrendModel(input_dim=5, output_dim=3),
    ]

    x = torch.randn(2, 16, 5)

    for model in models:
        # Run twice
        output1 = model(x)
        output2 = model(x)

        # Should produce identical results
        torch.testing.assert_close(output1["preds"], output2["preds"])
