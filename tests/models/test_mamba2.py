"""Tests for the Temporal Mamba-2 model."""

from __future__ import annotations

import torch

from airtrace.models.mamba2 import Mamba2Model


def _build_small_model(**overrides):
    params = dict(
        input_dim=4,
        output_dim=2,
        pred_len=1,
        embed_dim=16,
        state_dim=8,
        num_layers=2,
        conv_kernel_size=3,
        chunk_length=4,
        bidirectional_scan=True,
        dropout=0.0,
        ff_expansion=2,
    )
    params.update(overrides)
    return Mamba2Model(**params)


def test_mamba2_forward_shapes() -> None:
    model = _build_small_model()
    x = torch.randn(3, 12, 4)
    output = model(x)
    preds = output["preds"]
    extras = output["extras"]
    assert preds.shape == (3, 1, 2)
    assert len(extras["selective_states"]) == 2
    for state in extras["selective_states"]:
        assert state.shape == (3, 8)


def test_chunked_scan_handles_long_contexts() -> None:
    model = _build_small_model(chunk_length=2)
    x = torch.randn(2, 33, 4)
    preds = model(x)["preds"]
    assert torch.isfinite(preds).all()


def test_lora_only_fine_tuning_path() -> None:
    model = _build_small_model(
        adapter_rank=2,
        adapter_alpha=4.0,
        freeze_backbone=True,
        train_head=False,
    )
    trainable = {name for name, param in model.named_parameters() if param.requires_grad}
    assert any("lora" in name for name in trainable)
    assert not any(
        name.startswith("input_proj") for name in trainable
    ), "Input projection should be frozen when freeze_backbone=True"
