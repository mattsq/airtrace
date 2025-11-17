"""Implementation of the Non-stationary Transformer (NeurIPS 2022)."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ARBaseModel
from .registry import register


class SeriesStationarizer(nn.Module):
    """Learnable per-series stationarization module.

    Applies mean-variance normalization with optional affine parameters and
    provides utilities to restore predictions to the original scale.
    """

    def __init__(self, num_features: int, affine: bool = True, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.ones(1, 1, num_features))
            self.beta = nn.Parameter(torch.zeros(1, 1, num_features))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normalize a batch of sequences.

        Args:
            x: Input tensor of shape [B, T, D].

        Returns:
            normalized: Stationarized tensor.
            mean: Per-feature mean with shape [B, 1, D].
            std: Per-feature std with shape [B, 1, D].
        """
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True) + self.eps

        normalized = (x - mean) / std
        if self.affine:
            normalized = normalized * self.gamma + self.beta

        return normalized, mean, std

    def denormalize(
        self,
        normalized: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> torch.Tensor:
        """Invert the stationarization transform.

        Args:
            normalized: Tensor in normalized space.
            mean: Mean used for normalization.
            std: Standard deviation used for normalization.

        Returns:
            Tensor restored to the original scale.
        """
        if self.affine:
            normalized = (normalized - self.beta[..., : normalized.shape[-1]]) / (
                self.gamma[..., : normalized.shape[-1]] + self.eps
            )
        return normalized * std[..., : normalized.shape[-1]] + mean[..., : normalized.shape[-1]]


class DeStationaryAttention(nn.Module):
    """Multi-head attention with learnable de-stationary projections."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float = 0.1,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.eps = eps

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.delta_proj = nn.Linear(d_model, d_model)
        self.tau_proj = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        series_mean: torch.Tensor,
        series_std: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply de-stationary self-attention.

        Args:
            x: Input embeddings [B, T, d_model].
            series_mean: Mean across time for each series [B, d_model].
            series_std: Std across time for each series [B, d_model].
            attn_mask: Optional attention mask broadcastable to [B, nhead, T, T].

        Returns:
            output: Attended representations [B, T, d_model].
            attn_map: Mean attention map across heads [B, T, T].
        """
        B, T, _ = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        delta = self.delta_proj(series_mean).unsqueeze(1)
        tau = torch.exp(self.tau_proj(torch.log(series_std + self.eps))).unsqueeze(1)

        q = (q + delta) / (tau + self.eps)
        k = (k + delta) / (tau + self.eps)

        q = q.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.nhead, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        attended = torch.matmul(attn, v)
        attended = attended.transpose(1, 2).reshape(B, T, self.d_model)
        output = self.out_proj(attended)
        output = self.output_dropout(output)

        attn_map = attn.mean(dim=1)
        return output, attn_map


class NonStationaryTransformerEncoderLayer(nn.Module):
    """Encoder layer with de-stationary attention and feedforward block."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        activation: str = "gelu",
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.self_attn = DeStationaryAttention(d_model, nhead, dropout=dropout, eps=eps)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(
        self,
        x: torch.Tensor,
        series_mean: torch.Tensor,
        series_std: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_output, attn_map = self.self_attn(
            x, series_mean=series_mean, series_std=series_std, attn_mask=attn_mask
        )
        x = self.norm1(x + attn_output)

        ff = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = self.norm2(x + self.dropout2(ff))
        return x, attn_map


class NonStationaryTransformerEncoder(nn.Module):
    """Stack of non-stationary transformer encoder layers."""

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        activation: str,
        eps: float,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                NonStationaryTransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation=activation,
                    eps=eps,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        series_mean: torch.Tensor,
        series_std: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        attn_maps: List[torch.Tensor] = []
        for layer in self.layers:
            x, attn_map = layer(x, series_mean=series_mean, series_std=series_std, attn_mask=attn_mask)
            attn_maps.append(attn_map)
        return x, attn_maps


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


@register("nonstationary_transformer")
class NonStationaryTransformerModel(ARBaseModel):
    """Non-stationary Transformer with de-stationary attention blocks."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        activation: str = "gelu",
        pred_len: int = 1,
        stationarization_eps: float = 1e-5,
        affine_stationary: bool = True,
        **kwargs: Dict[str, object],
    ) -> None:
        super().__init__(input_dim, output_dim, **kwargs)
        self.pred_len = pred_len
        self.stationarizer = SeriesStationarizer(
            num_features=input_dim, affine=affine_stationary, eps=stationarization_eps
        )
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model=d_model, dropout=dropout)
        self.encoder = NonStationaryTransformerEncoder(
            num_layers=num_layers,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            eps=stationarization_eps,
        )
        self.forecast_head = nn.Linear(d_model, output_dim * pred_len)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        **kwargs: Dict[str, object],
    ) -> Dict[str, torch.Tensor]:
        del context, kwargs
        stationary_x, mean, std = self.stationarizer(x)
        embedded = self.input_projection(stationary_x)
        embedded = self.positional_encoding(embedded)

        series_mean = embedded.mean(dim=1)
        series_std = embedded.std(dim=1) + self.stationarizer.eps

        encoded, attn_maps = self.encoder(embedded, series_mean=series_mean, series_std=series_std)
        final_hidden = encoded[:, -1, :]

        preds_normalized = self.forecast_head(final_hidden).view(
            x.size(0), self.pred_len, self.output_dim
        )
        preds = self._restore_scale(preds_normalized, mean, std)

        return {
            "preds": preds,
            "extras": {
                "encoder_output": encoded,
                "attention_maps": attn_maps,
                "series_mean": mean,
                "series_std": std,
            },
        }

    def _restore_scale(self, preds: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        if self.output_dim == self.input_dim:
            return self.stationarizer.denormalize(preds, mean, std)

        reduced_mean = mean.mean(dim=2, keepdim=True)
        reduced_std = std.mean(dim=2, keepdim=True)
        return preds * reduced_std + reduced_mean
