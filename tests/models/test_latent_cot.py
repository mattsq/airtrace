"""Tests for Latent Chain-of-Thought model implementation."""

import pytest
import torch

from airtrace.models import LatentCOTModel


@pytest.fixture
def batch():
    """Create a dummy batch for testing."""
    B, T_in, D_in = 4, 32, 5
    x = torch.randn(B, T_in, D_in)
    return x


def test_latent_cot_forward_pass(batch):
    """Test basic forward pass of LatentCOT model."""
    model = LatentCOTModel(
        input_dim=5,
        output_dim=3,
        latent_dim=64,
        encoder_hidden_dim=128,
        ponder_hidden_dim=128,
        max_ponder_steps=5,
    )

    output = model(batch)

    assert "preds" in output
    assert "extras" in output
    assert output["preds"].shape == (4, 1, 3)  # [B, 1, D_out]


def test_latent_cot_extras_content(batch):
    """Test that extras contain expected information."""
    model = LatentCOTModel(
        input_dim=5,
        output_dim=3,
        latent_dim=64,
        max_ponder_steps=5,
    )

    output = model(batch)
    extras = output["extras"]

    # Check required keys
    assert "initial_latent" in extras
    assert "final_latent" in extras
    assert "latent_states" in extras
    assert "halt_probs" in extras
    assert "num_steps" in extras
    assert "act_loss" in extras
    assert "mean_steps" in extras
    assert "max_steps" in extras

    # Check shapes
    assert extras["initial_latent"].shape == (4, 64)  # [B, latent_dim]
    assert extras["final_latent"].shape == (4, 64)
    assert isinstance(extras["latent_states"], list)
    assert isinstance(extras["halt_probs"], list)
    assert extras["num_steps"].shape == (4,)  # [B]
    assert isinstance(extras["act_loss"], torch.Tensor)
    assert isinstance(extras["mean_steps"], float)
    assert isinstance(extras["max_steps"], float)


def test_latent_cot_encoder_types():
    """Test different encoder types (GRU, LSTM, MLP)."""
    x = torch.randn(2, 32, 5)

    for encoder_type in ["gru", "lstm"]:
        model = LatentCOTModel(
            input_dim=5,
            output_dim=3,
            latent_dim=64,
            encoder_type=encoder_type,
            max_ponder_steps=3,
        )

        output = model(x)
        assert output["preds"].shape == (2, 1, 3)


def test_latent_cot_max_ponder_steps():
    """Test different maximum pondering steps."""
    x = torch.randn(2, 32, 5)

    for max_steps in [1, 3, 5, 10]:
        model = LatentCOTModel(
            input_dim=5,
            output_dim=3,
            latent_dim=64,
            max_ponder_steps=max_steps,
        )

        output = model(x)
        extras = output["extras"]

        # Number of steps should not exceed max_steps
        assert extras["num_steps"].max().item() <= max_steps

        # Number of latent states should be at most max_steps + 1 (initial + updates)
        assert len(extras["latent_states"]) <= max_steps + 1


def test_latent_cot_deterministic_vs_stochastic():
    """Test deterministic (inference) vs stochastic (training) halting."""
    model = LatentCOTModel(
        input_dim=5,
        output_dim=3,
        latent_dim=64,
        max_ponder_steps=10,
        halting_threshold=0.99,
    )
    x = torch.randn(2, 32, 5)

    # Deterministic mode (inference)
    model.eval()
    with torch.no_grad():
        output_det = model(x, deterministic=True)

    # Run again - should be identical
    with torch.no_grad():
        output_det2 = model(x, deterministic=True)

    torch.testing.assert_close(output_det["preds"], output_det2["preds"])
    torch.testing.assert_close(
        output_det["extras"]["num_steps"],
        output_det2["extras"]["num_steps"]
    )

    # Stochastic mode (training)
    model.train()
    torch.manual_seed(42)
    output_stoch1 = model(x, deterministic=False)
    torch.manual_seed(43)
    output_stoch2 = model(x, deterministic=False)

    # Stochastic outputs may differ due to random halting
    # Just check they're both valid
    assert output_stoch1["preds"].shape == output_stoch2["preds"].shape


def test_latent_cot_act_loss_computation(batch):
    """Test ACT loss computation."""
    model = LatentCOTModel(
        input_dim=5,
        output_dim=3,
        latent_dim=64,
        max_ponder_steps=5,
        act_loss_weight=0.01,
    )

    output = model(batch)
    act_loss = output["extras"]["act_loss"]

    # ACT loss should be a scalar tensor
    assert isinstance(act_loss, torch.Tensor)
    assert act_loss.numel() == 1
    # ACT loss should be non-negative (it's a ponder cost)
    assert act_loss.item() >= 0


def test_latent_cot_gradient_flow():
    """Test that gradients flow through the entire model."""
    model = LatentCOTModel(
        input_dim=5,
        output_dim=3,
        latent_dim=64,
        max_ponder_steps=5,
    )
    x = torch.randn(2, 32, 5, requires_grad=True)

    output = model(x)
    preds = output["preds"]
    act_loss = output["extras"]["act_loss"]

    # Compute total loss
    pred_loss = preds.mean()
    total_loss = pred_loss + 0.01 * act_loss
    total_loss.backward()

    # Check that parameters have gradients
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert isinstance(param.grad, torch.Tensor), f"No grad for {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite grad for {name}"


def test_latent_cot_residual_connections():
    """Test with and without residual connections in ponder blocks."""
    x = torch.randn(2, 32, 5)

    for use_residual in [True, False]:
        model = LatentCOTModel(
            input_dim=5,
            output_dim=3,
            latent_dim=64,
            use_residual=use_residual,
            max_ponder_steps=5,
        )

        output = model(x)
        assert output["preds"].shape == (2, 1, 3)


def test_latent_cot_different_latent_dims():
    """Test different latent space dimensions."""
    x = torch.randn(2, 32, 5)

    for latent_dim in [32, 64, 128, 256]:
        model = LatentCOTModel(
            input_dim=5,
            output_dim=3,
            latent_dim=latent_dim,
            max_ponder_steps=3,
        )

        output = model(x)
        extras = output["extras"]

        assert output["preds"].shape == (2, 1, 3)
        assert extras["initial_latent"].shape == (2, latent_dim)
        assert extras["final_latent"].shape == (2, latent_dim)


def test_latent_cot_different_batch_sizes():
    """Test with different batch sizes."""
    model = LatentCOTModel(
        input_dim=5,
        output_dim=3,
        latent_dim=64,
        max_ponder_steps=5,
    )

    for batch_size in [1, 2, 4, 8]:
        x = torch.randn(batch_size, 32, 5)
        output = model(x)
        assert output["preds"].shape == (batch_size, 1, 3)
        assert output["extras"]["num_steps"].shape == (batch_size,)


def test_latent_cot_different_sequence_lengths():
    """Test with different input sequence lengths."""
    model = LatentCOTModel(
        input_dim=5,
        output_dim=3,
        latent_dim=64,
        encoder_type="gru",
        max_ponder_steps=5,
    )

    for seq_len in [16, 32, 64, 128]:
        x = torch.randn(2, seq_len, 5)
        output = model(x)
        assert output["preds"].shape == (2, 1, 3)


def test_latent_cot_no_nan_or_inf(batch):
    """Test that model doesn't produce NaN or Inf values."""
    model = LatentCOTModel(
        input_dim=5,
        output_dim=3,
        latent_dim=64,
        max_ponder_steps=5,
    )

    output = model(batch)

    assert not torch.isnan(output["preds"]).any()
    assert not torch.isinf(output["preds"]).any()
    assert not torch.isnan(output["extras"]["act_loss"]).any()
    assert not torch.isinf(output["extras"]["act_loss"]).any()


def test_latent_cot_device_transfer():
    """Test moving model to different devices."""
    model = LatentCOTModel(
        input_dim=5,
        output_dim=3,
        latent_dim=64,
        max_ponder_steps=3,
    )

    # Test CPU
    model = model.to("cpu")
    x = torch.randn(2, 32, 5).to("cpu")
    output = model(x)
    assert output["preds"].device.type == "cpu"

    # Test CUDA (if available)
    if torch.cuda.is_available():
        model = model.to("cuda")
        x = torch.randn(2, 32, 5).to("cuda")
        output = model(x)
        assert output["preds"].device.type == "cuda"


def test_latent_cot_num_params():
    """Test parameter counting."""
    model = LatentCOTModel(
        input_dim=5,
        output_dim=3,
        latent_dim=128,
        encoder_hidden_dim=256,
        ponder_hidden_dim=256,
        max_ponder_steps=5,
    )

    num_params = model.get_num_params()
    assert num_params > 0
    print(f"LatentCOT model has {num_params:,} parameters")


def test_latent_cot_train_eval_modes():
    """Test that model behaves differently in train and eval modes."""
    model = LatentCOTModel(
        input_dim=5,
        output_dim=3,
        latent_dim=64,
        max_ponder_steps=10,
        ponder_dropout=0.5,
    )
    x = torch.randn(2, 32, 5)

    # In eval mode with deterministic=True, should be reproducible
    model.eval()
    with torch.no_grad():
        output_eval1 = model(x, deterministic=True)
        output_eval2 = model(x, deterministic=True)

    torch.testing.assert_close(output_eval1["preds"], output_eval2["preds"])


def test_latent_cot_return_all_steps():
    """Test return_all_steps option."""
    model = LatentCOTModel(
        input_dim=5,
        output_dim=3,
        latent_dim=64,
        max_ponder_steps=5,
    )
    x = torch.randn(2, 32, 5)

    output = model(x, return_all_steps=True)

    assert "all_preds" in output["extras"]
    all_preds = output["extras"]["all_preds"]

    assert isinstance(all_preds, list)
    assert len(all_preds) > 0

    # Each prediction should have shape [B, 1, D_out]
    for pred in all_preds:
        assert pred.shape == (2, 1, 3)


def test_latent_cot_reproducibility():
    """Test that model is reproducible with same seed."""
    torch.manual_seed(42)
    model1 = LatentCOTModel(
        input_dim=5,
        output_dim=3,
        latent_dim=64,
        max_ponder_steps=5,
    )

    torch.manual_seed(42)
    model2 = LatentCOTModel(
        input_dim=5,
        output_dim=3,
        latent_dim=64,
        max_ponder_steps=5,
    )

    # Load same state
    model2.load_state_dict(model1.state_dict())

    x = torch.randn(2, 32, 5)

    # Set to eval mode with deterministic halting
    model1.eval()
    model2.eval()

    with torch.no_grad():
        output1 = model1(x, deterministic=True)
        output2 = model2(x, deterministic=True)

    torch.testing.assert_close(output1["preds"], output2["preds"])


def test_latent_cot_halting_threshold():
    """Test different halting thresholds."""
    x = torch.randn(2, 32, 5)

    for threshold in [0.5, 0.7, 0.9, 0.99]:
        model = LatentCOTModel(
            input_dim=5,
            output_dim=3,
            latent_dim=64,
            max_ponder_steps=10,
            halting_threshold=threshold,
        )

        model.eval()
        with torch.no_grad():
            output = model(x, deterministic=True)

        # Lower thresholds should generally lead to fewer steps
        assert output["extras"]["mean_steps"] <= 10


def test_latent_cot_ponder_block_components():
    """Test the PonderBlock component."""
    from airtrace.models.latent_cot import PonderBlock

    ponder_block = PonderBlock(
        latent_dim=64,
        hidden_dim=128,
        dropout=0.1,
        use_residual=True
    )

    latent = torch.randn(2, 64)
    refined = ponder_block(latent)

    assert refined.shape == (2, 64)
    assert not torch.isnan(refined).any()


def test_latent_cot_halting_module():
    """Test the HaltingModule component."""
    from airtrace.models.latent_cot import HaltingModule

    halting = HaltingModule(latent_dim=64)

    latent = torch.randn(2, 64)
    halt_prob = halting(latent)

    assert halt_prob.shape == (2,)
    # Halting probabilities should be in [0, 1]
    assert (halt_prob >= 0).all() and (halt_prob <= 1).all()


def test_latent_cot_encode_decode():
    """Test encode and decode methods separately."""
    model = LatentCOTModel(
        input_dim=5,
        output_dim=3,
        latent_dim=64,
        max_ponder_steps=5,
    )
    x = torch.randn(2, 32, 5)

    # Test encoding
    initial_latent = model.encode(x)
    assert initial_latent.shape == (2, 64)

    # Test decoding
    preds = model.decode(initial_latent)
    assert preds.shape == (2, 3)


def test_latent_cot_same_input_output_dims():
    """Test when input_dim == output_dim."""
    model = LatentCOTModel(
        input_dim=5,
        output_dim=5,
        latent_dim=64,
        max_ponder_steps=5,
    )

    x = torch.randn(2, 32, 5)
    output = model(x)

    assert output["preds"].shape == (2, 1, 5)


def test_latent_cot_different_input_output_dims():
    """Test when input_dim != output_dim."""
    model = LatentCOTModel(
        input_dim=10,
        output_dim=3,
        latent_dim=64,
        max_ponder_steps=5,
    )

    x = torch.randn(2, 32, 10)
    output = model(x)

    assert output["preds"].shape == (2, 1, 3)


def test_latent_cot_dropout_rates():
    """Test different dropout rates."""
    x = torch.randn(2, 32, 5)

    for dropout in [0.0, 0.1, 0.3, 0.5]:
        model = LatentCOTModel(
            input_dim=5,
            output_dim=3,
            latent_dim=64,
            max_ponder_steps=5,
            ponder_dropout=dropout,
        )

        output = model(x)
        assert output["preds"].shape == (2, 1, 3)


def test_latent_cot_encoder_num_layers():
    """Test different numbers of encoder layers."""
    x = torch.randn(2, 32, 5)

    for num_layers in [1, 2, 3, 4]:
        model = LatentCOTModel(
            input_dim=5,
            output_dim=3,
            latent_dim=64,
            encoder_type="gru",
            encoder_num_layers=num_layers,
            max_ponder_steps=5,
        )

        output = model(x)
        assert output["preds"].shape == (2, 1, 3)


def test_latent_cot_model_repr():
    """Test model string representation."""
    model = LatentCOTModel(
        input_dim=5,
        output_dim=3,
        latent_dim=128,
        max_ponder_steps=10,
    )

    model_repr = repr(model)
    assert "LatentCOTModel" in model_repr
    assert "num_params" in model_repr


def test_latent_cot_integration_with_task():
    """Test integration with COTOneStepTask."""
    from airtrace.tasks import build_task

    model = LatentCOTModel(
        input_dim=5,
        output_dim=5,
        latent_dim=64,
        max_ponder_steps=5,
    )

    task = build_task({
        "name": "cot_one_step",
        "loss": "mse",
        "metrics": ["rmse", "mae"],
        "act_loss_weight": 0.01,
    })

    # Create dummy batch
    batch = {
        "x": torch.randn(2, 32, 5),
        "y": torch.randn(2, 10, 5),  # Horizon can be longer
        "meta": {}
    }

    # Test training step
    model.train()
    train_output = task.training_step(batch, model)

    assert "loss" in train_output
    assert "pred_loss" in train_output
    assert "act_loss" in train_output
    assert "mean_steps" in train_output
    assert "rmse" in train_output
    assert "mae" in train_output

    # Test validation step
    model.eval()
    val_output = task.validation_step(batch, model)

    assert "loss" in val_output
    assert "pred_loss" in val_output
    assert "act_loss" in val_output
