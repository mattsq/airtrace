"""Autoformer implementation with auto-correlation and series decomposition."""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ARBaseModel
from .registry import register


class PositionalEmbedding(nn.Module):
    """Sinusoidal positional embedding."""

    def __init__(self, d_model: int, max_len: int = 1000) -> None:
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.size(1)
        return self.pe[:, :length, :]


class DataEmbedding(nn.Module):
    """Project raw values and add positional encoding."""

    def __init__(self, input_dim: int, d_model: int, dropout: float, max_len: int) -> None:
        super().__init__()
        self.value_projection = nn.Linear(input_dim, d_model)
        self.position = PositionalEmbedding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.value_projection(x) + self.position(x))


class MovingAverage(nn.Module):
    """Moving average block used for trend extraction."""

    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = max(1, kernel_size)
        self.padding = (self.kernel_size - 1) // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kernel_size == 1:
            return x

        front_pad = x[:, :1, :].repeat(1, self.padding, 1)
        end_pad = x[:, -1:, :].repeat(1, self.kernel_size - 1 - self.padding, 1)
        x_padded = torch.cat([front_pad, x, end_pad], dim=1)

        x_padded = x_padded.transpose(1, 2)
        averaged = F.avg_pool1d(x_padded, kernel_size=self.kernel_size, stride=1)
        return averaged.transpose(1, 2)


class SeriesDecomposition(nn.Module):
    """Decompose a sequence into seasonal and trend components."""

    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.moving_avg = MovingAverage(kernel_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


class AutoCorrelation(nn.Module):
    """Auto-correlation based attention layer."""

    def __init__(
        self, d_model: int, n_heads: int, dropout: float = 0.1, top_k: int = 5
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.top_k = top_k

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _pad_to_length(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        current_len = x.shape[2]
        if current_len >= target_len:
            return x

        pad_shape = (x.shape[0], x.shape[1], target_len - current_len, x.shape[3])
        pad = torch.zeros(pad_shape, device=x.device, dtype=x.dtype)
        return torch.cat([x, pad], dim=2)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        B, L_q, _ = query.shape
        L_k = key.shape[1]
        target_len = max(L_q, L_k)

        q = self._pad_to_length(
            self.q_proj(query).view(B, L_q, self.n_heads, self.d_head).permute(0, 2, 1, 3),
            target_len,
        )
        k = self._pad_to_length(
            self.k_proj(key).view(B, L_k, self.n_heads, self.d_head).permute(0, 2, 1, 3),
            target_len,
        )
        v = self._pad_to_length(
            self.v_proj(value).view(B, L_k, self.n_heads, self.d_head).permute(0, 2, 1, 3),
            target_len,
        )

        q_fft = torch.fft.rfft(q, n=target_len, dim=2)
        k_fft = torch.fft.rfft(k, n=target_len, dim=2)
        corr = torch.fft.irfft(q_fft * torch.conj(k_fft), n=target_len, dim=2)

        scores = corr.mean(dim=-1)
        top_k = min(self.top_k, target_len)
        vals, indices = torch.topk(scores, k=top_k, dim=-1)
        weights = F.softmax(vals, dim=-1)

        aggregated = torch.zeros_like(q)
        for i in range(top_k):
            shift = indices[:, :, i]
            weight = weights[:, :, i].unsqueeze(-1).unsqueeze(-1)

            shifted = torch.zeros_like(q)
            for b in range(B):
                for h in range(self.n_heads):
                    shift_val = int(shift[b, h].item())
                    shifted[b, h] = torch.roll(v[b, h], shifts=-shift_val, dims=0)

            aggregated = aggregated + shifted * weight

        aggregated = aggregated[:, :, :L_q, :]
        aggregated = aggregated.permute(0, 2, 1, 3).contiguous().view(B, L_q, self.d_model)
        return self.out_proj(self.dropout(aggregated))


class FeedForward(nn.Module):
    """Position-wise feedforward network."""

    def __init__(self, d_model: int, d_ff: int, dropout: float, activation: str) -> None:
        super().__init__()
        act = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            act,
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AutoformerEncoderLayer(nn.Module):
    """Encoder layer with auto-correlation attention and decomposition."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        moving_avg: int,
        dropout: float,
        activation: str,
        top_k: int,
    ) -> None:
        super().__init__()
        self.attention = AutoCorrelation(d_model, n_heads, dropout=dropout, top_k=top_k)
        self.decomp1 = SeriesDecomposition(moving_avg)
        self.decomp2 = SeriesDecomposition(moving_avg)
        self.ffn = FeedForward(d_model, d_ff, dropout, activation)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_out = self.attention(x, x, x)
        x = x + self.dropout(attn_out)
        x, trend1 = self.decomp1(self.norm1(x))

        ffn_out = self.ffn(x)
        x = x + self.dropout(ffn_out)
        x, trend2 = self.decomp2(self.norm2(x))

        return x, trend1 + trend2


class AutoformerDecoderLayer(nn.Module):
    """Decoder layer with self/cross auto-correlation and decomposition."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        moving_avg: int,
        dropout: float,
        activation: str,
        top_k: int,
    ) -> None:
        super().__init__()
        self.self_attn = AutoCorrelation(d_model, n_heads, dropout=dropout, top_k=top_k)
        self.cross_attn = AutoCorrelation(d_model, n_heads, dropout=dropout, top_k=top_k)
        self.decomp1 = SeriesDecomposition(moving_avg)
        self.decomp2 = SeriesDecomposition(moving_avg)
        self.decomp3 = SeriesDecomposition(moving_avg)
        self.ffn = FeedForward(d_model, d_ff, dropout, activation)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        seasonal: torch.Tensor,
        trend: torch.Tensor,
        memory: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self_out = self.self_attn(seasonal, seasonal, seasonal)
        seasonal = seasonal + self.dropout(self_out)
        seasonal, trend1 = self.decomp1(self.norm1(seasonal))

        cross_out = self.cross_attn(seasonal, memory, memory)
        seasonal = seasonal + self.dropout(cross_out)
        seasonal, trend2 = self.decomp2(self.norm2(seasonal))

        ffn_out = self.ffn(seasonal)
        seasonal = seasonal + self.dropout(ffn_out)
        seasonal, trend3 = self.decomp3(self.norm3(seasonal))

        return seasonal, trend + trend1 + trend2 + trend3


class AutoformerEncoder(nn.Module):
    """Stacked Autoformer encoder."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        moving_avg: int,
        dropout: float,
        activation: str,
        top_k: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                AutoformerEncoderLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation,
                    top_k=top_k,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        trend_total = torch.zeros_like(x)
        for layer in self.layers:
            x, trend = layer(x)
            trend_total = trend_total + trend
        return x, trend_total


class AutoformerDecoder(nn.Module):
    """Stacked Autoformer decoder."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        moving_avg: int,
        dropout: float,
        activation: str,
        top_k: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                AutoformerDecoderLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation,
                    top_k=top_k,
                )
                for _ in range(num_layers)
            ]
        )
        self.decomp = SeriesDecomposition(moving_avg)

    def forward(
        self, seasonal: torch.Tensor, trend: torch.Tensor, memory: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        for layer in self.layers:
            seasonal, trend = layer(seasonal, trend, memory)
        seasonal, trend_extra = self.decomp(seasonal)
        return seasonal, trend + trend_extra


@register("autoformer")
class AutoformerModel(ARBaseModel):
    """Autoformer with series decomposition and auto-correlation."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        d_model: int = 128,
        n_heads: int = 8,
        e_layers: int = 2,
        d_layers: int = 1,
        moving_avg: int = 25,
        d_ff: int = 256,
        dropout: float = 0.1,
        activation: str = "gelu",
        label_len: int = 24,
        pred_len: int = 1,
        top_k: int = 5,
        max_len: int = 512,
        **kwargs: Dict[str, object],
    ) -> None:
        super().__init__(input_dim, output_dim, **kwargs)
        self.label_len = label_len
        self.pred_len = pred_len
        self.input_dim = input_dim

        self.enc_embedding = DataEmbedding(input_dim, d_model, dropout, max_len)
        self.dec_embedding = DataEmbedding(input_dim, d_model, dropout, max_len)
        self.encoder = AutoformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            moving_avg=moving_avg,
            dropout=dropout,
            activation=activation,
            top_k=top_k,
            num_layers=e_layers,
        )
        self.decoder = AutoformerDecoder(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            moving_avg=moving_avg,
            dropout=dropout,
            activation=activation,
            top_k=top_k,
            num_layers=d_layers,
        )
        self.decomp = SeriesDecomposition(moving_avg)
        self.projection = nn.Linear(d_model, output_dim)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        **kwargs: Dict[str, object],
    ) -> Dict[str, torch.Tensor]:
        del context, kwargs
        B, T_in, _ = x.shape

        enc_out, enc_trend = self.encoder(self.enc_embedding(x))

        label_len = min(self.label_len, T_in)
        dec_zeros = torch.zeros(
            B, self.pred_len, self.input_dim, device=x.device, dtype=x.dtype
        )
        dec_input = torch.cat([x[:, T_in - label_len :, :], dec_zeros], dim=1)
        seasonal_init, trend_init = self.decomp(self.dec_embedding(dec_input))

        enc_trend_aligned = enc_trend
        if enc_trend.size(1) != seasonal_init.size(1):
            enc_trend_aligned = F.interpolate(
                enc_trend.permute(0, 2, 1),
                size=seasonal_init.size(1),
                mode="linear",
                align_corners=False,
            ).permute(0, 2, 1)

        dec_out, trend_out = self.decoder(
            seasonal_init, trend_init + enc_trend_aligned, enc_out
        )

        output = dec_out + trend_out
        preds = self.projection(output[:, -self.pred_len :, :])

        return {
            "preds": preds,
            "extras": {
                "encoder_trend": enc_trend,
                "decoder_trend": trend_out,
                "seasonal_component": dec_out,
            },
        }
