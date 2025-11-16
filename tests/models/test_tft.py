"""Tests for the Temporal Fusion Transformer implementation."""

import torch

from airtrace.models import TemporalFusionTransformer


def test_tft_forward_shapes():
    """Model should produce correctly shaped point forecasts."""

    model = TemporalFusionTransformer(
        input_dim=6,
        output_dim=3,
        hidden_size=32,
        lstm_layers=1,
        num_heads=2,
        pred_len=2,
        quantiles=None,
    )

    x = torch.randn(4, 12, 6)
    output = model(x)

    assert "preds" in output
    assert "extras" in output
    assert output["preds"].shape == (4, 2, 3)

    extras = output["extras"]
    assert extras["encoder_variable_importance"].shape[:3] == (4, 12, 6)
    assert extras["attention_weights"].shape[0] == 4  # batch-first attention


def test_tft_quantile_head_and_known_future_support():
    """Quantile head should return median preds and full quantile cube."""

    quantiles = [0.1, 0.5, 0.9]
    model = TemporalFusionTransformer(
        input_dim=5,
        output_dim=2,
        hidden_size=16,
        lstm_layers=1,
        num_heads=2,
        pred_len=3,
        static_input_dim=2,
        known_future_dim=4,
        quantiles=quantiles,
    )

    x = torch.randn(2, 8, 5)
    known_future = torch.randn(2, 3, 4)
    static_covariates = torch.randn(2, 2)

    output = model(x, static_covariates=static_covariates, known_future=known_future)

    preds = output["preds"]
    assert preds.shape == (2, 3, 2)

    quantile_cube = output["extras"]["quantile_forecast"]
    assert quantile_cube.shape == (2, 3, 2, len(quantiles))

    # Median slice should correspond to the configured 0.5 quantile
    median_from_cube = quantile_cube[..., quantiles.index(0.5)]
    torch.testing.assert_close(preds, median_from_cube)

    # Variable selection weights should form a proper probability simplex
    enc_weights = output["extras"]["encoder_variable_importance"]
    weight_sums = enc_weights.sum(dim=-1)
    torch.testing.assert_close(weight_sums, torch.ones_like(weight_sums), atol=1e-5, rtol=1e-5)
