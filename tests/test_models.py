"""Tests for model implementations."""

import pytest
import torch

from airtrace.models import (
    DriftModel,
    ExponentialSmoothingModel,
    GRUARModel,
    GRUSeq2SeqModel,
    HoltLinearTrendModel,
    LinearARModel,
    LinearTrendModel,
    LSTMARModel,
    LSTMSeq2SeqModel,
    MeanModel,
    MedianModel,
    MLPARModel,
    MovingAverageModel,
    PatchTSTModel,
    PersistenceModel,
    PolynomialTrendModel,
    SeasonalNaiveModel,
    TCNModel,
    ThetaModel,
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
    (PatchTSTModel, {"patch_len": 8, "stride": 4, "d_model": 32, "nhead": 4, "num_layers": 2}),
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


def test_median_model_forward(batch):
    """Test median model forward pass."""
    model = MedianModel(input_dim=5, output_dim=3)

    output = model(batch)

    assert "preds" in output
    assert output["preds"].shape == (4, 1, 3)


def test_median_model_same_dims():
    """Test median model with same input/output dims."""
    model = MedianModel(input_dim=5, output_dim=5)
    x = torch.randn(2, 10, 5)

    output = model(x)

    # Should return median across time
    expected = x.median(dim=1, keepdim=True).values  # [2, 1, 5]
    assert output["preds"].shape == expected.shape
    torch.testing.assert_close(output["preds"], expected)


def test_drift_model_forward(batch):
    """Test drift model forward pass."""
    model = DriftModel(input_dim=5, output_dim=3)

    output = model(batch)

    assert "preds" in output
    assert output["preds"].shape == (4, 1, 3)
    assert "drift" in output["extras"]


def test_drift_model_constant_sequence():
    """Test drift on constant sequence."""
    model = DriftModel(input_dim=5, output_dim=5)

    # Create constant sequence
    x = torch.ones(2, 10, 5) * 5.0

    output = model(x)

    # For constant sequence, drift should be zero
    assert output["extras"]["drift"].abs().max() < 1e-6
    # Prediction should be the constant value
    torch.testing.assert_close(
        output["preds"],
        torch.ones_like(output["preds"]) * 5.0,
        atol=1e-5,
        rtol=1e-5
    )


def test_drift_model_linear_sequence():
    """Test drift on linear sequence."""
    model = DriftModel(input_dim=1, output_dim=1)

    # Create linear sequence with slope 3: y = 2 + 3*t
    t = torch.arange(10, dtype=torch.float32)
    x = (2 + 3 * t).reshape(1, 10, 1)  # [1, 10, 1]

    output = model(x)

    # Drift should be 3.0
    expected_drift = torch.tensor([[3.0]])
    torch.testing.assert_close(
        output["extras"]["drift"], expected_drift, atol=1e-4, rtol=1e-4
    )

    # Should predict: last_value + drift = 29 + 3 = 32
    expected_pred = torch.tensor([[[32.0]]])
    torch.testing.assert_close(output["preds"], expected_pred, atol=1e-4, rtol=1e-4)


def test_lazy_baseline_model_repr_handles_uninitialized_parameters():
    """Ensure lazy modules don't break model representations or validation."""
    model = LinearARModel(input_dim=5, output_dim=3)

    # ``repr`` calls ``get_num_params`` which previously failed for lazy params.
    model_repr = repr(model)

    assert "LinearARModel" in model_repr
    assert "num_params" in model_repr


def test_exponential_smoothing_model_forward(batch):
    """Test exponential smoothing model forward pass."""
    model = ExponentialSmoothingModel(input_dim=5, output_dim=3, alpha=0.3)

    output = model(batch)

    assert "preds" in output
    assert output["preds"].shape == (4, 1, 3)
    assert "alpha" in output["extras"]
    assert output["extras"]["alpha"] == 0.3


def test_exponential_smoothing_alpha_validation():
    """Test that invalid alpha values raise errors."""
    with pytest.raises(ValueError):
        ExponentialSmoothingModel(input_dim=5, output_dim=3, alpha=0.0)

    with pytest.raises(ValueError):
        ExponentialSmoothingModel(input_dim=5, output_dim=3, alpha=1.5)

    with pytest.raises(ValueError):
        ExponentialSmoothingModel(input_dim=5, output_dim=3, alpha=-0.1)


def test_exponential_smoothing_high_alpha():
    """Test exponential smoothing with high alpha (more weight on recent)."""
    model = ExponentialSmoothingModel(input_dim=1, output_dim=1, alpha=0.9)

    # Create sequence where last value is very different
    x = torch.cat([
        torch.ones(1, 9, 1) * 1.0,
        torch.ones(1, 1, 1) * 10.0
    ], dim=1)

    output = model(x)

    # With high alpha, prediction should be closer to last value
    # Should be heavily weighted toward 10.0
    assert output["preds"][0, 0, 0] > 5.0


def test_seasonal_naive_model_forward(batch):
    """Test seasonal naive model forward pass."""
    model = SeasonalNaiveModel(input_dim=5, output_dim=3, season_length=10)

    output = model(batch)

    assert "preds" in output
    assert output["preds"].shape == (4, 1, 3)
    assert "season_length" in output["extras"]
    assert "used_seasonal" in output["extras"]


def test_seasonal_naive_with_sufficient_history():
    """Test seasonal naive with enough history."""
    model = SeasonalNaiveModel(input_dim=5, output_dim=5, season_length=5)

    # Create data with clear seasonal pattern
    x = torch.randn(2, 10, 5)

    output = model(x)

    # Should use seasonal value (5 timesteps back from end)
    expected = x[:, -5, :].unsqueeze(1)  # [2, 1, 5]
    assert output["preds"].shape == expected.shape
    torch.testing.assert_close(output["preds"], expected)
    assert output["extras"]["used_seasonal"] is True


def test_seasonal_naive_insufficient_history():
    """Test seasonal naive with insufficient history."""
    model = SeasonalNaiveModel(input_dim=5, output_dim=5, season_length=20)

    # Only 10 timesteps, but season_length is 20
    x = torch.randn(2, 10, 5)

    output = model(x)

    # Should fall back to persistence (last value)
    expected = x[:, -1, :].unsqueeze(1)  # [2, 1, 5]
    assert output["preds"].shape == expected.shape
    torch.testing.assert_close(output["preds"], expected)
    assert output["extras"]["used_seasonal"] is False


def test_seasonal_naive_season_validation():
    """Test that invalid season_length raises error."""
    with pytest.raises(ValueError):
        SeasonalNaiveModel(input_dim=5, output_dim=3, season_length=0)

    with pytest.raises(ValueError):
        SeasonalNaiveModel(input_dim=5, output_dim=3, season_length=-1)


@pytest.mark.parametrize("model_class,kwargs", [
    (PersistenceModel, {}),
    (ZeroModel, {}),
    (MeanModel, {}),
    (MedianModel, {}),
    (MovingAverageModel, {"window_size": 5}),
    (LinearTrendModel, {}),
    (DriftModel, {}),
    (ExponentialSmoothingModel, {"alpha": 0.3}),
    (SeasonalNaiveModel, {"season_length": 10}),
    (PolynomialTrendModel, {"degree": 2}),
    (HoltLinearTrendModel, {"alpha": 0.3, "beta": 0.1}),
    (ThetaModel, {"theta": 2.0, "alpha": 0.5}),
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
    (MedianModel, {}),
    (MovingAverageModel, {"window_size": 5}),
    (LinearTrendModel, {}),
    (DriftModel, {}),
    (ExponentialSmoothingModel, {"alpha": 0.3}),
    (SeasonalNaiveModel, {"season_length": 10}),
    (PolynomialTrendModel, {"degree": 2}),
    (HoltLinearTrendModel, {"alpha": 0.3, "beta": 0.1}),
    (ThetaModel, {"theta": 2.0, "alpha": 0.5}),
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
        MedianModel(input_dim=5, output_dim=3),
        MovingAverageModel(input_dim=5, output_dim=3, window_size=5),
        LinearTrendModel(input_dim=5, output_dim=3),
        DriftModel(input_dim=5, output_dim=3),
        ExponentialSmoothingModel(input_dim=5, output_dim=3, alpha=0.3),
        SeasonalNaiveModel(input_dim=5, output_dim=3, season_length=10),
        PolynomialTrendModel(input_dim=5, output_dim=3, degree=2),
        HoltLinearTrendModel(input_dim=5, output_dim=3, alpha=0.3, beta=0.1),
        ThetaModel(input_dim=5, output_dim=3, theta=2.0, alpha=0.5),
    ]

    x = torch.randn(2, 16, 5)

    for model in models:
        # Run twice
        output1 = model(x)
        output2 = model(x)

        # Should produce identical results
        torch.testing.assert_close(output1["preds"], output2["preds"])


# ============================================================================
# New Baseline Model Tests
# ============================================================================


def test_polynomial_trend_model_forward(batch):
    """Test polynomial trend model forward pass."""
    model = PolynomialTrendModel(input_dim=5, output_dim=3, degree=2)

    output = model(batch)

    assert "preds" in output
    assert output["preds"].shape == (4, 1, 3)
    assert "degree" in output["extras"]
    assert output["extras"]["degree"] == 2


def test_polynomial_trend_quadratic_sequence():
    """Test polynomial trend on quadratic sequence."""
    model = PolynomialTrendModel(input_dim=1, output_dim=1, degree=2)

    # Create quadratic sequence: y = 1 + 2*t + 3*t^2
    t = torch.arange(10, dtype=torch.float32)
    x = (1 + 2 * t + 3 * t ** 2).reshape(1, 10, 1)  # [1, 10, 1]

    output = model(x)

    # Should predict next value: y(10) = 1 + 2*10 + 3*100 = 321
    expected = torch.tensor([[[321.0]]])
    torch.testing.assert_close(output["preds"], expected, atol=1.0, rtol=1e-3)


def test_polynomial_trend_degree_validation():
    """Test that invalid degree raises error."""
    with pytest.raises(ValueError):
        PolynomialTrendModel(input_dim=5, output_dim=3, degree=0)

    with pytest.raises(ValueError):
        PolynomialTrendModel(input_dim=5, output_dim=3, degree=-1)


def test_holt_linear_trend_model_forward(batch):
    """Test Holt's linear trend model forward pass."""
    model = HoltLinearTrendModel(input_dim=5, output_dim=3, alpha=0.3, beta=0.1)

    output = model(batch)

    assert "preds" in output
    assert output["preds"].shape == (4, 1, 3)
    assert "alpha" in output["extras"]
    assert "beta" in output["extras"]
    assert "level" in output["extras"]
    assert "trend" in output["extras"]


def test_holt_linear_trend_alpha_beta_validation():
    """Test that invalid alpha/beta values raise errors."""
    with pytest.raises(ValueError):
        HoltLinearTrendModel(input_dim=5, output_dim=3, alpha=0.0, beta=0.1)

    with pytest.raises(ValueError):
        HoltLinearTrendModel(input_dim=5, output_dim=3, alpha=1.5, beta=0.1)

    with pytest.raises(ValueError):
        HoltLinearTrendModel(input_dim=5, output_dim=3, alpha=0.3, beta=0.0)

    with pytest.raises(ValueError):
        HoltLinearTrendModel(input_dim=5, output_dim=3, alpha=0.3, beta=1.5)


def test_holt_linear_trend_constant_sequence():
    """Test Holt's linear trend on constant sequence."""
    model = HoltLinearTrendModel(input_dim=5, output_dim=5, alpha=0.3, beta=0.1)

    # Create constant sequence
    x = torch.ones(2, 10, 5) * 5.0

    output = model(x)

    # For constant sequence, trend should approach zero
    # and prediction should be close to constant value
    assert output["preds"].shape == (2, 1, 5)
    # Trend should be small
    assert output["extras"]["trend"].abs().max() < 1.0


def test_theta_model_forward(batch):
    """Test Theta model forward pass."""
    model = ThetaModel(input_dim=5, output_dim=3, theta=2.0, alpha=0.5)

    output = model(batch)

    assert "preds" in output
    assert output["preds"].shape == (4, 1, 3)
    assert "theta" in output["extras"]
    assert "alpha" in output["extras"]
    assert "theta0_forecast" in output["extras"]
    assert "theta2_forecast" in output["extras"]


def test_theta_model_linear_sequence():
    """Test Theta model on linear sequence."""
    model = ThetaModel(input_dim=1, output_dim=1, theta=2.0, alpha=0.5)

    # Create linear sequence: y = 2 + 3*t
    t = torch.arange(10, dtype=torch.float32)
    x = (2 + 3 * t).reshape(1, 10, 1)  # [1, 10, 1]

    output = model(x)

    # For linear sequence, theta-0 should give good forecast
    # Expected: y(10) = 2 + 3*10 = 32
    # Prediction should be close to 32
    assert output["preds"].shape == (1, 1, 1)
    # Allow some tolerance due to theta-2 line
    assert 28.0 < output["preds"][0, 0, 0] < 36.0


def test_linear_ar_model_forward(batch):
    """Test Linear AR model forward pass."""
    model = LinearARModel(input_dim=5, output_dim=3)

    output = model(batch)

    assert "preds" in output
    assert output["preds"].shape == (4, 1, 3)


def test_linear_ar_model_trainable():
    """Test that Linear AR model has trainable parameters."""
    model = LinearARModel(input_dim=5, output_dim=3)
    x = torch.randn(2, 10, 5)

    # Parameters should be registered immediately (as UninitializedParameter)
    # This ensures optimizers can capture them before first forward pass
    params_before = list(model.parameters())
    assert len(params_before) > 0, "LazyLinear should register parameters immediately"

    # First forward pass materializes the lazy parameters
    output1 = model(x)

    # Check that model has parameters with correct shape
    num_params = model.get_num_params()
    assert num_params > 0
    # Should have input_dim * T_in * output_dim + output_dim (bias) params
    # For T_in=10, D_in=5, D_out=3: 10*5*3 + 3 = 153
    assert num_params == 10 * 5 * 3 + 3


def test_linear_ar_model_gradient_flow():
    """Test that gradients flow through Linear AR model."""
    model = LinearARModel(input_dim=5, output_dim=3)
    x = torch.randn(2, 10, 5, requires_grad=True)

    output = model(x)
    preds = output["preds"]

    # Compute dummy loss
    loss = preds.mean()
    loss.backward()

    # Check that model parameters have gradients
    for param in model.parameters():
        if param.requires_grad:
            assert param.grad is not None


# ============================================================================
# New Model Tests (MLP AR, LSTM AR, Seq2Seq)
# ============================================================================


def test_mlp_ar_model_forward(batch):
    """Test MLP AR model forward pass."""
    model = MLPARModel(
        input_dim=5,
        output_dim=3,
        hidden_dims=[128, 64],
        dropout=0.1
    )

    output = model(batch)

    assert "preds" in output
    assert output["preds"].shape == (4, 1, 3)


def test_mlp_ar_model_default_hidden_dims():
    """Test MLP AR with default hidden dimensions."""
    model = MLPARModel(input_dim=5, output_dim=3)
    x = torch.randn(2, 10, 5)

    output = model(x)

    assert output["preds"].shape == (2, 1, 3)


def test_mlp_ar_model_trainable():
    """Test that MLP AR model has trainable parameters."""
    model = MLPARModel(input_dim=5, output_dim=3, hidden_dims=[64, 32])
    x = torch.randn(2, 10, 5)

    # Parameters should be registered immediately (as UninitializedParameter for first layer)
    # This ensures optimizers can capture them before first forward pass
    params_before = list(model.parameters())
    assert len(params_before) > 0, "LazyLinear should register parameters immediately"

    # First forward pass materializes the lazy parameters
    output = model(x)

    # Check that model has parameters
    num_params = model.get_num_params()
    assert num_params > 0
    # Should have multiple layers with non-zero parameters
    assert num_params > 100


def test_mlp_ar_model_gradient_flow():
    """Test that gradients flow through MLP AR model."""
    model = MLPARModel(input_dim=5, output_dim=3, hidden_dims=[64, 32])
    x = torch.randn(2, 10, 5, requires_grad=True)

    output = model(x)
    preds = output["preds"]

    # Compute dummy loss
    loss = preds.mean()
    loss.backward()

    # Check that model parameters have gradients
    for param in model.parameters():
        if param.requires_grad:
            assert param.grad is not None


def test_mlp_ar_model_activation():
    """Test MLP AR with different activations."""
    for activation in ["relu", "gelu", "tanh"]:
        model = MLPARModel(
            input_dim=5,
            output_dim=3,
            hidden_dims=[32],
            activation=activation
        )
        x = torch.randn(2, 10, 5)
        output = model(x)
        assert output["preds"].shape == (2, 1, 3)


def test_mlp_ar_model_invalid_activation():
    """Test that invalid activation raises error."""
    with pytest.raises(ValueError):
        MLPARModel(input_dim=5, output_dim=3, activation="invalid")


def test_lstm_ar_model_forward(batch):
    """Test LSTM AR model forward pass."""
    model = LSTMARModel(
        input_dim=5,
        output_dim=3,
        hidden_size=64,
        num_layers=2
    )

    output = model(batch)

    assert "preds" in output
    assert "extras" in output
    assert output["preds"].shape == (4, 1, 3)
    assert "hidden" in output["extras"]
    assert "cell" in output["extras"]


def test_lstm_ar_model_with_attention(batch):
    """Test LSTM AR model with attention."""
    model = LSTMARModel(
        input_dim=5,
        output_dim=3,
        hidden_size=64,
        num_layers=2,
        use_attention=True
    )

    output = model(batch)

    assert output["preds"].shape == (4, 1, 3)
    assert "attention_weights" in output["extras"]


def test_lstm_ar_model_bidirectional(batch):
    """Test bidirectional LSTM AR model."""
    model = LSTMARModel(
        input_dim=5,
        output_dim=3,
        hidden_size=64,
        num_layers=2,
        bidirectional=True
    )

    output = model(batch)

    assert output["preds"].shape == (4, 1, 3)


def test_lstm_ar_model_gradient_flow():
    """Test that gradients flow through LSTM AR model."""
    model = LSTMARModel(input_dim=5, output_dim=3, hidden_size=32)
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


def test_gru_seq2seq_model_forward(batch):
    """Test GRU Seq2Seq model forward pass."""
    model = GRUSeq2SeqModel(
        input_dim=5,
        output_dim=3,
        hidden_size=64,
        num_layers=2,
        use_attention=False
    )

    output = model(batch, pred_len=5)

    assert "preds" in output
    assert "extras" in output
    assert output["preds"].shape == (4, 5, 3)  # [B, pred_len, D_out]


def test_gru_seq2seq_model_single_step(batch):
    """Test GRU Seq2Seq with single step prediction."""
    model = GRUSeq2SeqModel(
        input_dim=5,
        output_dim=3,
        hidden_size=64,
        num_layers=1
    )

    output = model(batch, pred_len=1)

    assert output["preds"].shape == (4, 1, 3)


def test_gru_seq2seq_model_with_attention(batch):
    """Test GRU Seq2Seq model with attention."""
    model = GRUSeq2SeqModel(
        input_dim=5,
        output_dim=3,
        hidden_size=64,
        num_layers=2,
        use_attention=True
    )

    output = model(batch, pred_len=3)

    assert output["preds"].shape == (4, 3, 3)
    assert "attention_weights" in output["extras"]


def test_gru_seq2seq_model_teacher_forcing():
    """Test GRU Seq2Seq with teacher forcing."""
    model = GRUSeq2SeqModel(
        input_dim=5,
        output_dim=3,
        hidden_size=32,
        num_layers=1,
        teacher_forcing_ratio=1.0  # Always use teacher forcing
    )
    model.train()

    x = torch.randn(2, 10, 5)
    target = torch.randn(2, 5, 3)

    output = model(x, target=target, pred_len=5)

    assert output["preds"].shape == (2, 5, 3)


def test_gru_seq2seq_model_gradient_flow():
    """Test that gradients flow through GRU Seq2Seq model."""
    model = GRUSeq2SeqModel(input_dim=5, output_dim=3, hidden_size=32)
    x = torch.randn(2, 10, 5, requires_grad=True)

    output = model(x, pred_len=3)
    preds = output["preds"]

    # Compute dummy loss
    loss = preds.mean()
    loss.backward()

    # Check that parameters have gradients
    for param in model.parameters():
        if param.requires_grad:
            assert param.grad is not None


def test_lstm_seq2seq_model_forward(batch):
    """Test LSTM Seq2Seq model forward pass."""
    model = LSTMSeq2SeqModel(
        input_dim=5,
        output_dim=3,
        hidden_size=64,
        num_layers=2,
        use_attention=False
    )

    output = model(batch, pred_len=5)

    assert "preds" in output
    assert "extras" in output
    assert output["preds"].shape == (4, 5, 3)  # [B, pred_len, D_out]
    assert "cell" in output["extras"]


def test_lstm_seq2seq_model_with_attention(batch):
    """Test LSTM Seq2Seq model with attention."""
    model = LSTMSeq2SeqModel(
        input_dim=5,
        output_dim=3,
        hidden_size=64,
        num_layers=2,
        use_attention=True
    )

    output = model(batch, pred_len=3)

    assert output["preds"].shape == (4, 3, 3)
    assert "attention_weights" in output["extras"]


def test_lstm_seq2seq_model_gradient_flow():
    """Test that gradients flow through LSTM Seq2Seq model."""
    model = LSTMSeq2SeqModel(input_dim=5, output_dim=3, hidden_size=32)
    x = torch.randn(2, 10, 5, requires_grad=True)

    output = model(x, pred_len=3)
    preds = output["preds"]

    # Compute dummy loss
    loss = preds.mean()
    loss.backward()

    # Check that parameters have gradients
    for param in model.parameters():
        if param.requires_grad:
            assert param.grad is not None


@pytest.mark.parametrize("model_class,kwargs", [
    (MLPARModel, {"hidden_dims": [64, 32]}),
    (LSTMARModel, {"hidden_size": 32, "num_layers": 1}),
    (GRUSeq2SeqModel, {"hidden_size": 32, "num_layers": 1}),
    (LSTMSeq2SeqModel, {"hidden_size": 32, "num_layers": 1}),
])
def test_new_models_no_nan(model_class, kwargs):
    """Test that new models don't produce NaN values."""
    model = model_class(input_dim=5, output_dim=3, **kwargs)
    x = torch.randn(2, 16, 5)

    if "Seq2Seq" in model_class.__name__:
        output = model(x, pred_len=1)
    else:
        output = model(x)

    assert not torch.isnan(output["preds"]).any()
    assert not torch.isinf(output["preds"]).any()


# ============================================================================
# PatchTST Model Tests
# ============================================================================


def test_patchtst_model_forward(batch):
    """Test PatchTST model forward pass."""
    model = PatchTSTModel(
        input_dim=5,
        output_dim=3,
        patch_len=8,
        stride=4,
        d_model=64,
        nhead=4,
        num_layers=2
    )

    output = model(batch)

    assert "preds" in output
    assert "extras" in output
    assert output["preds"].shape == (4, 1, 3)  # [B, 1, D_out]
    assert "num_patches" in output["extras"]
    assert "channel_representations" in output["extras"]


def test_patchtst_model_patching():
    """Test that patching works correctly."""
    model = PatchTSTModel(
        input_dim=3,
        output_dim=2,
        patch_len=8,
        stride=4,
        d_model=32,
        nhead=4
    )

    # Input of length 32 with patch_len=8, stride=4
    # Should produce (32 - 8) // 4 + 1 = 7 patches
    x = torch.randn(2, 32, 3)
    output = model(x)

    assert output["preds"].shape == (2, 1, 2)
    assert output["extras"]["num_patches"] == 7
    assert output["extras"]["patch_len"] == 8


def test_patchtst_model_different_patch_configs():
    """Test PatchTST with different patch configurations."""
    # Non-overlapping patches (stride = patch_len)
    model1 = PatchTSTModel(
        input_dim=5,
        output_dim=3,
        patch_len=16,
        stride=16,
        d_model=32,
        nhead=4
    )
    x = torch.randn(2, 64, 5)
    output1 = model1(x)
    # (64 - 16) // 16 + 1 = 4 patches
    assert output1["extras"]["num_patches"] == 4

    # Overlapping patches
    model2 = PatchTSTModel(
        input_dim=5,
        output_dim=3,
        patch_len=16,
        stride=8,
        d_model=32,
        nhead=4
    )
    output2 = model2(x)
    # (64 - 16) // 8 + 1 = 7 patches
    assert output2["extras"]["num_patches"] == 7


def test_patchtst_model_gradient_flow():
    """Test that gradients flow through PatchTST model."""
    model = PatchTSTModel(
        input_dim=5,
        output_dim=3,
        patch_len=8,
        stride=4,
        d_model=32,
        nhead=4,
        num_layers=2
    )
    x = torch.randn(2, 32, 5, requires_grad=True)

    output = model(x)
    preds = output["preds"]

    # Compute dummy loss
    loss = preds.mean()
    loss.backward()

    # Check that parameters have gradients
    for param in model.parameters():
        if param.requires_grad:
            assert param.grad is not None


def test_patchtst_model_num_params():
    """Test parameter counting for PatchTST."""
    model = PatchTSTModel(
        input_dim=5,
        output_dim=3,
        patch_len=16,
        stride=8,
        d_model=64,
        nhead=4,
        num_layers=2
    )

    num_params = model.get_num_params()
    assert num_params > 0
    print(f"PatchTST model has {num_params:,} parameters")


def test_patchtst_model_channel_independence():
    """Test that channels are processed independently."""
    model = PatchTSTModel(
        input_dim=3,
        output_dim=3,
        patch_len=8,
        stride=4,
        d_model=32,
        nhead=4,
        num_layers=1
    )

    x = torch.randn(2, 32, 3)
    output = model(x)

    # Check that channel representations are extracted
    channel_repr = output["extras"]["channel_representations"]
    assert channel_repr.shape == (2, 3, 32)  # [B, D_in, d_model]


def test_patchtst_model_different_activations():
    """Test PatchTST with different activation functions."""
    for activation in ["relu", "gelu"]:
        model = PatchTSTModel(
            input_dim=5,
            output_dim=3,
            patch_len=8,
            stride=4,
            d_model=32,
            nhead=4,
            activation=activation
        )
        x = torch.randn(2, 32, 5)
        output = model(x)
        assert output["preds"].shape == (2, 1, 3)


def test_patchtst_model_repr():
    """Test PatchTST model string representation."""
    model = PatchTSTModel(
        input_dim=5,
        output_dim=3,
        patch_len=16,
        stride=8,
        d_model=128
    )

    model_repr = repr(model)
    assert "PatchTSTModel" in model_repr
    assert "patch_len=16" in model_repr
    assert "stride=8" in model_repr
    assert "d_model=128" in model_repr
    assert "num_params" in model_repr
