"""Tests for the Mamba2ARModel."""

import torch

from airtrace.models.mamba2 import Mamba2ARModel


def test_mamba2_forward_shapes() -> None:
    """The model should return predictions with the requested horizon."""

    model = Mamba2ARModel(
        input_dim=6,
        output_dim=4,
        pred_len=2,
        embed_dim=32,
        state_dim=32,
        num_layers=2,
        conv_kernel_size=3,
        chunk_size=8,
        adapter_kernel_size=3,
    )
    x = torch.randn(3, 40, 6)

    output = model(x)

    assert output["preds"].shape == (3, 2, 4)
    assert "extras" in output
    assert len(output["extras"]["block_states"]) == 2


def test_mamba2_backward_pass() -> None:
    """Gradients should propagate through the entire backbone."""

    model = Mamba2ARModel(
        input_dim=5,
        output_dim=3,
        pred_len=1,
        embed_dim=16,
        state_dim=16,
        num_layers=1,
        conv_kernel_size=3,
        chunk_size=4,
        adapter_kernel_size=3,
    )
    x = torch.randn(2, 17, 5)

    preds = model(x)["preds"]
    loss = preds.mean()
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)
