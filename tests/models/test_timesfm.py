from __future__ import annotations

import torch

from airtrace.models.timesfm import TimesFMModel, TimesFMPatcher


def test_timesfm_patcher_shapes():
    patcher = TimesFMPatcher(input_dim=4, patch_len=5, patch_stride=5, embed_dim=16)
    x = torch.randn(2, 11, 4)  # padding required to align stride

    patches, pad_len = patcher(x)

    assert patches.shape == (2, 3, 16)
    assert pad_len == 4  # 11 -> pad to 15 so three patches of length 5


def test_timesfm_forward_outputs():
    model = TimesFMModel(
        input_dim=6,
        output_dim=2,
        pred_len=12,
        patch_len=4,
        patch_stride=4,
        embed_dim=32,
        num_layers=2,
        num_heads=4,
        ff_expansion=2,
        dropout=0.0,
        max_positions=128,
    )
    model.eval()

    x = torch.randn(3, 20, 6)
    output = model(x)

    assert "preds" in output
    assert "extras" in output
    preds = output["preds"]
    extras = output["extras"]

    assert preds.shape == (3, 12, 2)
    assert extras["pred_patch_count"] == torch.tensor(3)
    assert extras["num_context_patches"] >= 1


def test_timesfm_encode_decode_matches_forward():
    model = TimesFMModel(
        input_dim=5,
        output_dim=3,
        pred_len=8,
        patch_len=4,
        patch_stride=4,
        embed_dim=48,
        num_layers=2,
        num_heads=4,
        ff_expansion=2,
        dropout=0.0,
        max_positions=128,
    )
    model.eval()

    x = torch.randn(2, 16, 5)
    latent, extras = model.encode(x)
    decoded = model.decode(latent, pred_len=8)
    forward_output = model(x)["preds"]

    assert decoded.shape == (2, 8, 3)
    assert torch.allclose(decoded, forward_output, atol=1e-5)
    assert extras["pad_len"] >= 0


def test_timesfm_forward_allows_pred_len_override():
    model = TimesFMModel(
        input_dim=4,
        output_dim=2,
        pred_len=6,
        patch_len=3,
        patch_stride=3,
        embed_dim=24,
        num_layers=1,
        num_heads=3,
        ff_expansion=2,
        dropout=0.0,
        max_positions=64,
    )
    model.eval()

    x = torch.randn(1, 9, 4)
    output = model(x, pred_len=2)

    assert output["preds"].shape == (1, 2, 2)
    assert output["extras"]["pred_patch_count"] == torch.tensor(1)
    assert output["extras"]["pred_len"] == torch.tensor(2)
