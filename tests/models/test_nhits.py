import torch

from airtrace.models.nhits import NHiTSModel


def test_nhits_forward_shapes():
    """N-HiTS should return forecasts and stack outputs with expected shapes."""

    batch_size, seq_len, input_dim = 3, 48, 4
    pred_len, output_dim = 6, 2

    model = NHiTSModel(
        input_dim=input_dim,
        output_dim=output_dim,
        pred_len=pred_len,
        pool_sizes=[1, 2],
        blocks_per_stack=2,
        hidden_size=64,
        num_layers=3,
    )

    x = torch.randn(batch_size, seq_len, input_dim)
    output = model(x)

    preds = output["preds"]
    assert preds.shape == (batch_size, pred_len, output_dim)

    stack_forecasts = output["extras"]["stack_forecasts"]
    assert stack_forecasts.shape == (
        batch_size,
        len(model.pool_sizes),
        pred_len,
        output_dim,
    )


def test_nhits_handles_misaligned_pools():
    """Pooling with non-divisible lengths should still produce valid outputs."""

    batch_size, seq_len, input_dim = 2, 37, 3
    pred_len, output_dim = 5, 3

    model = NHiTSModel(
        input_dim=input_dim,
        output_dim=output_dim,
        pred_len=pred_len,
        pool_sizes=[3, 4],
        blocks_per_stack=1,
        hidden_size=32,
        num_layers=2,
        interpolation_mode="nearest",
    )

    x = torch.randn(batch_size, seq_len, input_dim)
    output = model(x)

    preds = output["preds"]
    assert preds.shape == (batch_size, pred_len, output_dim)
    assert torch.isfinite(preds).all()
