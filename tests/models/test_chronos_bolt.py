"""Tests for the Chronos-Bolt model."""

from __future__ import annotations

from pathlib import Path

import torch

from airtrace.models.chronos_bolt import ChronosBoltModel


def _build_test_model(**overrides) -> ChronosBoltModel:
    params = dict(
        input_dim=6,
        output_dim=3,
        pred_len=5,
        embed_dim=64,
        patch_size=4,
        patch_stride=2,
        num_blocks=2,
        num_heads=4,
        dilation_growth=2,
        conv_kernel_size=3,
    )
    params.update(overrides)
    return ChronosBoltModel(**params)


def test_chronos_bolt_forward_shapes() -> None:
    model = _build_test_model()
    x = torch.randn(4, 32, 6)
    out = model(x)
    preds = out["preds"]
    assert preds.shape == (4, 5, 3)
    assert "attention_maps" in out["extras"]
    assert len(out["extras"]["attention_maps"]) == len(model.blocks)


def test_chronos_bolt_checkpoint_loading(tmp_path: Path) -> None:
    model = _build_test_model()
    checkpoint_path = tmp_path / "chronos_bolt.pt"
    torch.save(model.state_dict(), checkpoint_path)

    loaded = _build_test_model(pretrained_checkpoint=str(checkpoint_path), strict_checkpoint=True)
    for key, value in model.state_dict().items():
        assert torch.allclose(value, loaded.state_dict()[key])


def test_chronos_bolt_lora_freeze_flags() -> None:
    model = _build_test_model(lora_rank=2, freeze_backbone=True, train_head=False)
    trainable = [name for name, param in model.named_parameters() if param.requires_grad]
    assert trainable, "Expected LoRA parameters to remain trainable"
    assert all("lora" in name for name in trainable)
