from __future__ import annotations

"""TimesFM: Decoder-only patch-based foundation model for time series forecasting."""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ResidualWrapperCompatible
from .registry import register


class TimesFMPatcher(nn.Module):
    """Convert a time series into patch embeddings.

    This module mirrors the patch tokenizer used by TimesFM-style models: it
    slices the input window into overlapping patches, flattens each patch, and
    projects it into the model embedding dimension.
    """

    def __init__(
        self, input_dim: int, patch_len: int, patch_stride: int, embed_dim: int
    ) -> None:
        super().__init__()
        if patch_len <= 0:
            raise ValueError("patch_len must be positive")
        if patch_stride <= 0:
            raise ValueError("patch_stride must be positive")

        self.input_dim = input_dim
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.proj = nn.Linear(input_dim * patch_len, embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """Return patch embeddings and the applied padding length.

        Args:
            x: Input tensor of shape [B, T, D].

        Returns:
            Tuple of (patch_embeddings, pad_len) where ``patch_embeddings`` has
            shape [B, num_patches, embed_dim]. ``pad_len`` is the number of
            padded timesteps appended to make the window divisible by
            ``patch_stride``.
        """

        if x.dim() != 3:
            raise ValueError("TimesFMPatcher expects input of shape [B, T, D]")

        batch_size, seq_len, feature_dim = x.shape
        if feature_dim != self.input_dim:
            raise ValueError(
                f"Expected input_dim={self.input_dim}, received {feature_dim}"
            )

        # Ensure we can extract at least one patch and that unfold divides the
        # (possibly padded) sequence evenly according to ``patch_stride``.
        if seq_len < self.patch_len:
            pad_len = self.patch_len - seq_len
        else:
            remainder = (seq_len - self.patch_len) % self.patch_stride
            pad_len = (self.patch_stride - remainder) % self.patch_stride

        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))

        patches = x.unfold(dimension=1, size=self.patch_len, step=self.patch_stride)
        patches = patches.contiguous().view(batch_size, patches.size(1), -1)
        return self.proj(patches), int(pad_len)


def _build_causal_mask(length: int, device: torch.device) -> torch.Tensor:
    """Create an upper-triangular causal mask for autoregressive attention."""

    if length <= 0:
        raise ValueError("Mask length must be positive")
    mask = torch.full((length, length), float("-inf"), device=device)
    return torch.triu(mask, diagonal=1)


@register("timesfm")
class TimesFMModel(ResidualWrapperCompatible):
    """Decoder-only patch Transformer inspired by Google's TimesFM.

    The implementation follows the key design choices of TimesFM: patch-based
    tokenization, decoder-only causal attention, and a lightweight projection
    head for forecasting. While the original model leverages large-scale
    pre-training, this implementation focuses on compatibility and efficiency
    for AirTrace experiments.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        pred_len: int = 24,
        patch_len: int = 16,
        patch_stride: int = 16,
        embed_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        ff_expansion: int = 4,
        dropout: float = 0.1,
        max_positions: int = 2048,
        normalize_inputs: bool = True,
        **kwargs: Optional[Dict],
    ) -> None:
        super().__init__(input_dim, output_dim, **kwargs)
        if pred_len <= 0:
            raise ValueError("pred_len must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.pred_len = pred_len
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.embed_dim = embed_dim
        self.max_positions = max_positions
        self.normalize_inputs = normalize_inputs

        self.input_norm = nn.LayerNorm(input_dim) if normalize_inputs else None
        self.patcher = TimesFMPatcher(
            input_dim=input_dim,
            patch_len=patch_len,
            patch_stride=patch_stride,
            embed_dim=embed_dim,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * ff_expansion,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.pos_embedding = nn.Parameter(torch.zeros(1, max_positions, embed_dim))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        self.forecast_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.forecast_token, std=0.02)

        self.output_head = nn.Linear(embed_dim, output_dim * patch_len)

    def _embed_patches(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        if self.input_norm is not None:
            x = self.input_norm(x)
        return self.patcher(x)

    def _apply_transformer(self, tokens: torch.Tensor) -> torch.Tensor:
        total_len = tokens.size(1)
        if total_len > self.max_positions:
            raise ValueError(
                f"Sequence length {total_len} exceeds max_positions={self.max_positions}"
            )
        pos = self.pos_embedding[:, :total_len, :]
        mask = _build_causal_mask(total_len, tokens.device)
        return self.transformer(tokens + pos, mask=mask)

    def encode(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        del context
        tokens, pad_len = self._embed_patches(x)
        hidden = self._apply_transformer(tokens)
        extras: Dict[str, torch.Tensor] = {
            "pad_len": torch.tensor(pad_len, device=x.device),
            "num_context_patches": torch.tensor(tokens.shape[1], device=x.device),
        }
        return hidden, extras

    def decode(self, latent: torch.Tensor, pred_len: int) -> torch.Tensor:
        if pred_len <= 0:
            raise ValueError("pred_len must be positive")

        num_pred_patches = math.ceil(pred_len / self.patch_len)
        forecast_tokens = self.forecast_token.expand(
            latent.size(0), num_pred_patches, self.embed_dim
        )

        tokens = torch.cat([latent, forecast_tokens], dim=1)
        hidden = self._apply_transformer(tokens)
        pred_tokens = hidden[:, -num_pred_patches:, :]

        patch_preds = self.output_head(pred_tokens)
        patch_preds = patch_preds.view(
            latent.size(0), num_pred_patches, self.patch_len, self.output_dim
        )
        seq_preds = patch_preds.reshape(
            latent.size(0), num_pred_patches * self.patch_len, self.output_dim
        )
        return seq_preds[:, :pred_len, :]

    def forward(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None, **kwargs
    ) -> Dict[str, torch.Tensor]:
        del kwargs
        latent, extras = self.encode(x, context)
        preds = self.decode(latent, self.pred_len)
        extras.update(
            {
                "hidden_state": latent,
                "pred_patch_count": torch.tensor(
                    math.ceil(self.pred_len / self.patch_len), device=x.device
                ),
            }
        )
        return {"preds": preds, "extras": extras}
