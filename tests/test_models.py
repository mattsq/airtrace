"""Tests for model implementations."""

import pytest
import torch

from airtrace.models import GRUARModel, TCNModel, TransformerModel


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
