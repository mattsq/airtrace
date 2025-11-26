import torch
import pytest

from airtrace.models.lag_llama import (
    LagLlamaModel,
    LagLlamaTokenizer,
    SinusoidalTimeEmbedding,
)


def test_tokenizer_padding_and_sensor_embeddings() -> None:
    tokenizer = LagLlamaTokenizer(
        input_dim=3,
        patch_size=3,
        stride=2,
        embed_dim=4,
        add_sensor_embeddings=True,
    )
    tokenizer.patch_proj.weight.data.zero_()
    tokenizer.patch_proj.bias.data.zero_()
    tokenizer.sensor_embed.weight.data.fill_(1.0)

    x = torch.zeros(1, 1, 3)
    tokens = tokenizer(x)

    assert tokens.shape == (1, 1, 4)
    assert torch.allclose(tokens, torch.ones_like(tokens))


def test_tokenizer_validation_errors() -> None:
    with pytest.raises(ValueError):
        LagLlamaTokenizer(input_dim=2, patch_size=0, stride=1, embed_dim=4, add_sensor_embeddings=False)
    with pytest.raises(ValueError):
        LagLlamaTokenizer(input_dim=2, patch_size=1, stride=0, embed_dim=4, add_sensor_embeddings=False)


def test_sinusoidal_time_embedding() -> None:
    emb = SinusoidalTimeEmbedding(dim=4)
    timesteps = torch.tensor([0.0, 1.0])
    result = emb(timesteps)

    assert result.shape == (2, 4)
    assert torch.allclose(result[0], torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=result.dtype))
    with pytest.raises(ValueError):
        SinusoidalTimeEmbedding(dim=3)


def test_retrieval_uses_stored_bank_and_limits_neighbors() -> None:
    model = LagLlamaModel(
        input_dim=2,
        output_dim=1,
        pred_len=1,
        embed_dim=4,
        patch_size=2,
        patch_stride=1,
        retrieval_mode="cosine",
        max_neighbors=1,
        diffusion_steps=0,
    )
    bank = torch.randn(2, 3, 2)
    model.update_retrieval_bank(bank)

    summaries = torch.randn(1, 4)
    neighbors = model._retrieve(summaries, x_device=torch.device("cpu"), retrieval_bank=None)

    assert neighbors is not None
    assert neighbors.shape == (1, 1, 4)


def test_retrieval_skips_when_disabled() -> None:
    model = LagLlamaModel(
        input_dim=2,
        output_dim=1,
        pred_len=1,
        embed_dim=4,
        patch_size=2,
        patch_stride=1,
        retrieval_mode="none",
        max_neighbors=0,
        diffusion_steps=0,
    )
    summaries = torch.randn(2, 4)
    result = model._retrieve(summaries, x_device=torch.device("cpu"), retrieval_bank=torch.randn(1, 3, 2))
    assert result is None


def test_forward_runs_diffusion_and_returns_context_summary() -> None:
    torch.manual_seed(0)
    model = LagLlamaModel(
        input_dim=2,
        output_dim=2,
        pred_len=2,
        embed_dim=8,
        patch_size=2,
        patch_stride=1,
        retrieval_mode="none",
        diffusion_steps=2,
        init_noise_scale=0.05,
    )

    x = torch.randn(2, 4, 2)
    context = torch.randn(2, 4, 2)
    output = model(x, context=context, num_samples=3)

    preds = output["preds"]
    extras = output["extras"]

    assert preds.shape == (2, 2, 2)
    assert extras["samples"].shape == (2, 3, 2, 2)
    assert extras["context_summary"] is not None
    assert extras["context_summary"].shape == (2, 1, 2)
