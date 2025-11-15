"""Tests for the Moirai multiresolution SSM model."""

from __future__ import annotations

from pathlib import Path

import torch

from airtrace.models.moirai import MoiraiModel


def _build_model(**overrides) -> MoiraiModel:
    params = dict(
        input_dim=4,
        output_dim=2,
        pred_len=5,
        embed_dim=48,
        state_dim=32,
        num_layers=2,
        conv_kernel_size=5,
        dropout=0.05,
        ff_expansion=2,
        patch_scales=(4, 8),
        max_positions=256,
    )
    params.update(overrides)
    return MoiraiModel(**params)


def test_moirai_forward_shapes() -> None:
    model = _build_model()
    x = torch.randn(3, 32, 4)
    output = model(x)
    preds = output["preds"]
    assert preds.shape == (3, 5, 2)
    extras = output["extras"]
    assert len(extras["multiresolution_tokens"]) == len(model.patch_scales)
    assert len(extras["ssm_states"]) == len(model.layers)


def test_moirai_checkpoint_loading(tmp_path: Path) -> None:
    model = _build_model()
    checkpoint_path = tmp_path / "moirai.pt"
    torch.save(model.state_dict(), checkpoint_path)

    loaded = _build_model(pretrained_checkpoint=str(checkpoint_path), strict_checkpoint=True)
    for key, value in model.state_dict().items():
        assert torch.allclose(value, loaded.state_dict()[key])


def test_moirai_lora_freeze_behavior() -> None:
    model = _build_model(adapter_rank=2, freeze_backbone=True, train_head=False)
    trainable = [name for name, param in model.named_parameters() if param.requires_grad]
    assert trainable, "Expected some LoRA parameters to remain trainable"
    assert all("lora" in name for name in trainable)
