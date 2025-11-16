"""Retrieval-augmented Lag-Llama-style diffusion forecaster."""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from .base import ARBaseModel
from .registry import register


class LagLlamaTokenizer(nn.Module):
    """Patchify the time series into overlapping Lag-Llama tokens."""

    def __init__(
        self,
        input_dim: int,
        patch_size: int,
        stride: int,
        embed_dim: int,
        add_sensor_embeddings: bool,
    ) -> None:
        super().__init__()
        if patch_size <= 0:
            raise ValueError("patch_size must be positive")
        if stride <= 0:
            raise ValueError("stride must be positive")
        self.input_dim = input_dim
        self.patch_size = patch_size
        self.stride = stride
        self.add_sensor_embeddings = add_sensor_embeddings
        self.patch_proj = nn.Linear(patch_size * input_dim, embed_dim)
        if add_sensor_embeddings:
            self.sensor_embed = nn.Embedding(input_dim, embed_dim)

    def _pad_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """Pad sequence so that unfolding produces at least one patch."""

        B, T, _ = x.shape
        target = max(T, self.patch_size)
        remainder = (target - self.patch_size) % self.stride
        pad_extra = (self.stride - remainder) % self.stride
        total_pad = target + pad_extra - T
        if total_pad > 0:
            padding = torch.zeros(B, total_pad, self.input_dim, device=x.device, dtype=x.dtype)
            x = torch.cat([x, padding], dim=1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return patch tokens [B, num_tokens, embed_dim]."""

        x = self._pad_sequence(x)
        patches = x.unfold(1, self.patch_size, self.stride)
        patches = patches.contiguous().view(x.shape[0], -1, self.patch_size * self.input_dim)
        tokens = self.patch_proj(patches)
        if self.add_sensor_embeddings:
            sensor_ids = torch.arange(self.input_dim, device=x.device)
            sensor_tokens = self.sensor_embed(sensor_ids).mean(dim=0, keepdim=True)
            tokens = tokens + sensor_tokens
        return tokens


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal embedding used to condition diffusion timesteps."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("Time embedding dimension must be even")
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        device = timesteps.device
        exponent = torch.arange(half, device=device, dtype=timesteps.dtype)
        exponent = -math.log(10000.0) * exponent / (half - 1)
        freqs = torch.exp(exponent)
        angles = timesteps.unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return emb


class LagLlamaDiffusionBlock(nn.Module):
    """Self + cross attention block used inside the diffusion model."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float,
        ff_expansion: int,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * ff_expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * ff_expansion, embed_dim),
        )
        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(embed_dim, embed_dim))

    def forward(
        self,
        latents: torch.Tensor,
        context_tokens: Optional[torch.Tensor],
        retrieved_tokens: Optional[torch.Tensor],
        time_emb: torch.Tensor,
    ) -> torch.Tensor:
        x = latents + self.time_proj(time_emb).unsqueeze(1)
        attn_out, _ = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        memory: Optional[torch.Tensor] = context_tokens
        if retrieved_tokens is not None:
            memory = (
                retrieved_tokens if memory is None else torch.cat([memory, retrieved_tokens], dim=1)
            )
        if memory is not None:
            cross_out, _ = self.cross_attn(self.norm2(x), memory, memory)
            x = x + cross_out
        x = x + self.ff(self.norm3(x))
        return x


class LagLlamaNoisePredictor(nn.Module):
    """Predict noise inside the diffusion loop."""

    def __init__(
        self,
        embed_dim: int,
        output_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        ff_expansion: int,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        self.input_proj = nn.Linear(output_dim, embed_dim)
        self.blocks = nn.ModuleList(
            [
                LagLlamaDiffusionBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    ff_expansion=ff_expansion,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, output_dim)

    def forward(
        self,
        latents: torch.Tensor,
        context_tokens: Optional[torch.Tensor],
        retrieved_tokens: Optional[torch.Tensor],
        time_emb: torch.Tensor,
    ) -> torch.Tensor:
        h = self.input_proj(latents)
        for block in self.blocks:
            h = block(h, context_tokens, retrieved_tokens, time_emb)
        h = self.norm(h)
        return self.output_proj(h)


@register("lag_llama")
class LagLlamaModel(ARBaseModel):
    """Retrieval-augmented Lag-Llama-style probabilistic forecaster."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        pred_len: int = 1,
        embed_dim: int = 256,
        patch_size: int = 16,
        patch_stride: int = 8,
        add_sensor_embeddings: bool = True,
        max_positions: int = 4096,
        retrieval_mode: str = "none",
        max_neighbors: int = 4,
        diffusion_layers: int = 2,
        diffusion_heads: int = 4,
        diffusion_steps: int = 4,
        diffusion_dropout: float = 0.1,
        diffusion_ff_expansion: int = 4,
        init_noise_scale: float = 0.1,
        guidance_scale: float = 1.0,
    ) -> None:
        super().__init__(input_dim, output_dim)
        if pred_len <= 0:
            raise ValueError("pred_len must be positive")
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive")
        if max_positions <= 0:
            raise ValueError("max_positions must be positive")
        if diffusion_steps < 0:
            raise ValueError("diffusion_steps must be non-negative")
        if max_neighbors < 0:
            raise ValueError("max_neighbors must be non-negative")

        self.pred_len = pred_len
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.max_positions = max_positions
        self.retrieval_mode = retrieval_mode.lower()
        self.max_neighbors = max_neighbors
        self.diffusion_steps = diffusion_steps
        self.init_noise_scale = init_noise_scale
        self.guidance_scale = guidance_scale

        self.tokenizer = LagLlamaTokenizer(
            input_dim=input_dim,
            patch_size=patch_size,
            stride=patch_stride,
            embed_dim=embed_dim,
            add_sensor_embeddings=add_sensor_embeddings,
        )
        self.positional = nn.Parameter(torch.zeros(1, max_positions, embed_dim))
        nn.init.trunc_normal_(self.positional, std=0.02)
        self.context_norm = nn.LayerNorm(embed_dim)
        self.decoder = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, pred_len * output_dim),
        )
        self.time_embed = SinusoidalTimeEmbedding(embed_dim)
        self.noise_predictor = LagLlamaNoisePredictor(
            embed_dim=embed_dim,
            output_dim=output_dim,
            num_layers=diffusion_layers,
            num_heads=diffusion_heads,
            dropout=diffusion_dropout,
            ff_expansion=diffusion_ff_expansion,
        )
        self._retrieval_bank: Optional[torch.Tensor] = None

    def update_retrieval_bank(self, windows: torch.Tensor) -> None:
        """Register reference windows for retrieval (stored on CPU)."""

        self._retrieval_bank = windows.detach().cpu()

    def _apply_positional(self, tokens: torch.Tensor) -> torch.Tensor:
        length = tokens.shape[1]
        if length <= self.max_positions:
            pos = self.positional[:, :length, :]
        else:
            pos = F.interpolate(
                self.positional.transpose(1, 2),
                size=length,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        return tokens + pos

    def _encode_sequences(self, sequences: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(sequences)
        tokens = self._apply_positional(tokens)
        return self.context_norm(tokens)

    def _retrieve(
        self,
        summaries: torch.Tensor,
        x_device: torch.device,
        retrieval_bank: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if self.retrieval_mode == "none" or self.max_neighbors == 0:
            return None
        bank = retrieval_bank
        if bank is None:
            bank = self._retrieval_bank
        if bank is None:
            return None
        if isinstance(bank, (list, tuple)):
            bank = torch.stack(bank)
        if bank.dim() == 2:
            bank = bank.unsqueeze(0)
        bank = bank.to(x_device)
        encoded = self._encode_sequences(bank)
        bank_summaries = encoded.mean(dim=1)
        sims = F.cosine_similarity(summaries.unsqueeze(1), bank_summaries.unsqueeze(0), dim=-1)
        k = min(self.max_neighbors, sims.shape[1])
        if k == 0:
            return None
        top_indices = torch.topk(sims, k=k, dim=1).indices
        neighbors = torch.gather(
            bank_summaries.unsqueeze(0).expand(summaries.shape[0], -1, -1),
            dim=1,
            index=top_indices.unsqueeze(-1).expand(-1, -1, bank_summaries.shape[-1]),
        )
        return neighbors

    def _diffusion_sample(
        self,
        base: torch.Tensor,
        context_tokens: torch.Tensor,
        retrieved_tokens: Optional[torch.Tensor],
        num_samples: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = base.device
        B = base.shape[0]
        preds = []
        steps = max(1, self.diffusion_steps)
        step_size = self.guidance_scale / steps
        for _ in range(num_samples):
            latent = (
                torch.randn(B, self.pred_len, self.output_dim, device=device)
                * self.init_noise_scale
            )
            for step in range(self.diffusion_steps):
                t = torch.full(
                    (B,),
                    float(step) / max(1, self.diffusion_steps - 1),
                    device=device,
                )
                t_emb = self.time_embed(t)
                noise = self.noise_predictor(
                    latent,
                    context_tokens,
                    retrieved_tokens,
                    t_emb,
                )
                latent = latent - step_size * noise
            preds.append(base + latent)
        samples = torch.stack(preds, dim=1)
        mean_pred = samples.mean(dim=1)
        return mean_pred, samples

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        retrieval_bank: Optional[torch.Tensor] = None,
        num_samples: int = 1,
        **_: Dict,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass returning mean prediction and optional samples."""

        context_tokens = self._encode_sequences(x)
        summaries = context_tokens.mean(dim=1)
        retrieved = self._retrieve(summaries, x.device, retrieval_bank)
        base = self.decoder(summaries).view(x.shape[0], self.pred_len, self.output_dim)
        if num_samples <= 1 or self.diffusion_steps == 0:
            preds = base
            sample_bank = base.unsqueeze(1)
        else:
            preds, sample_bank = self._diffusion_sample(
                base, context_tokens, retrieved, num_samples
            )
        context_summary = (
            context.mean(dim=1, keepdim=True).detach()
            if context is not None and context.dim() >= 2
            else None
        )
        extras: Dict[str, Optional[torch.Tensor]] = {
            "context_tokens": context_tokens.detach(),
            "retrieved_neighbors": retrieved.detach() if retrieved is not None else None,
            "samples": sample_bank.detach(),
            "context_summary": context_summary,
        }
        return {"preds": preds, "extras": extras}
