from __future__ import annotations

import torch

from airtrace.models.s_mamba import SMambaModel


def _build_small_model(**overrides: int | float | bool) -> SMambaModel:
    params = dict(
        input_dim=4,
        output_dim=2,
        seq_len=12,
        pred_len=2,
        embed_dim=16,
        state_dim=8,
        num_layers=2,
        ff_expansion=2,
        dropout=0.0,
        bidirectional_scan=True,
        conv_kernel_size=3,
    )
    params.update(overrides)
    return SMambaModel(**params)


def test_forward_shapes_and_extras() -> None:
    model = _build_small_model()
    x = torch.randn(3, 12, 4)

    output = model(x)
    preds = output["preds"]
    extras = output["extras"]

    assert preds.shape == (3, 2, 2)
    assert len(extras["selective_states"]) == 2
    for state in extras["selective_states"]:
        assert state.shape == (3, 8)
    assert extras["token_embeddings"].shape == (3, 12, 16)


def test_bidirectional_changes_outputs() -> None:
    x = torch.randn(2, 12, 4)
    model_bidir = _build_small_model(bidirectional_scan=True)
    model_unidir = _build_small_model(bidirectional_scan=False)

    model_unidir.load_state_dict(model_bidir.state_dict(), strict=False)

    with torch.no_grad():
        preds_bidir = model_bidir(x)["preds"]
        preds_unidir = model_unidir(x)["preds"]

    assert not torch.allclose(preds_bidir, preds_unidir)


def test_parameter_validation() -> None:
    import pytest

    with pytest.raises(ValueError, match="seq_len must be positive"):
        _build_small_model(seq_len=0)

    with pytest.raises(ValueError, match="pred_len must be positive"):
        _build_small_model(pred_len=0)

    with pytest.raises(ValueError, match="conv_kernel_size must be odd"):
        _build_small_model(conv_kernel_size=2)

    with pytest.raises(ValueError, match="dropout must be in"):
        _build_small_model(dropout=1.5)


def test_decode_pred_len_guard() -> None:
    import pytest

    model = _build_small_model()
    latent = torch.randn(2, 12, 16)

    with pytest.raises(ValueError, match="pred_len must match"):
        model.decode(latent, pred_len=3)
