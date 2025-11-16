"""FEDformer implementation with Frequency Enhanced Decomposition.

FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting
Reference: https://arxiv.org/abs/2201.12740 (ICML 2022)
"""

from __future__ import annotations

import math
from typing import Dict, Literal, Optional, Tuple

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
        """Get positional encoding.

        Args:
            x: Input tensor [B, T, D]

        Returns:
            Positional encoding [B, T, D_model]
        """
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
        """Embed input with value projection and positional encoding.

        Args:
            x: Input tensor [B, T, D]

        Returns:
            Embedded tensor [B, T, D_model]
        """
        return self.dropout(self.value_projection(x) + self.position(x))


class MovingAverage(nn.Module):
    """Moving average block used for trend extraction."""

    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = max(1, kernel_size)
        self.padding = (self.kernel_size - 1) // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply moving average.

        Args:
            x: Input tensor [B, T, D]

        Returns:
            Smoothed tensor [B, T, D]
        """
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
        """Decompose series into seasonal and trend.

        Args:
            x: Input tensor [B, T, D]

        Returns:
            Tuple of (seasonal, trend) tensors, each [B, T, D]
        """
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


class FourierAttention(nn.Module):
    """Fourier-based attention mechanism for frequency domain processing."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        modes: int = 32,
    ) -> None:
        """Initialize Fourier attention.

        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dropout: Dropout rate
            modes: Number of Fourier modes to keep (low-pass filtering)
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.modes = modes

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Apply Fourier attention.

        Args:
            query: Query tensor [B, L_q, D]
            key: Key tensor [B, L_k, D]
            value: Value tensor [B, L_v, D]

        Returns:
            Attention output [B, L_q, D]
        """
        B, L_q, _ = query.shape
        L_k = key.shape[1]

        # Project to multi-head format
        q = self.q_proj(query).view(B, L_q, self.n_heads, self.d_head)
        k = self.k_proj(key).view(B, L_k, self.n_heads, self.d_head)
        v = self.v_proj(value).view(B, L_k, self.n_heads, self.d_head)

        # Permute to [B, H, L, D_head]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # Apply FFT to query and key
        q_fft = torch.fft.rfft(q, dim=2, norm='ortho')
        k_fft = torch.fft.rfft(k, dim=2, norm='ortho')
        v_fft = torch.fft.rfft(v, dim=2, norm='ortho')

        # Limit to low frequency modes
        modes = min(self.modes, q_fft.size(2), k_fft.size(2), v_fft.size(2))

        # Frequency domain attention: element-wise multiplication
        # This is simplified - in practice you can do more complex operations
        out_fft = torch.zeros_like(v_fft)

        # For low frequencies, apply attention-like operation
        # Simplified: weight by query-key similarity in frequency domain
        for i in range(modes):
            # Compute attention weights in frequency domain
            attn_weight = (q_fft[:, :, i, :] * torch.conj(k_fft[:, :, i, :])).real
            attn_weight = F.softmax(attn_weight, dim=-1)

            # Apply attention to values
            out_fft[:, :, i, :] = v_fft[:, :, i, :] * attn_weight.unsqueeze(-1).expand_as(v_fft[:, :, i, :])

        # Inverse FFT
        out = torch.fft.irfft(out_fft, n=L_q, dim=2, norm='ortho')

        # Reshape and project
        out = out.permute(0, 2, 1, 3).contiguous().view(B, L_q, self.d_model)
        return self.out_proj(self.dropout(out))


class WaveletAttention(nn.Module):
    """Wavelet-based attention mechanism for multi-resolution processing."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        wavelet: str = 'haar',
    ) -> None:
        """Initialize Wavelet attention.

        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dropout: Dropout rate
            wavelet: Wavelet type ('haar' or 'db')
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.wavelet = wavelet

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # Simplified Haar wavelet filters
        if wavelet == 'haar':
            # Low-pass and high-pass filters
            self.register_buffer('low_filter', torch.tensor([1/math.sqrt(2), 1/math.sqrt(2)]))
            self.register_buffer('high_filter', torch.tensor([1/math.sqrt(2), -1/math.sqrt(2)]))

    def _wavelet_transform(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply simplified wavelet transform.

        Args:
            x: Input tensor [B, H, L, D]

        Returns:
            Tuple of (approximation, detail) coefficients
        """
        # Simplified: use average pooling and difference for Haar wavelet approximation
        # Real implementation would use proper wavelet transforms
        B, H, L, D = x.shape

        if L % 2 != 0:
            # Pad to even length
            x = F.pad(x, (0, 0, 0, 1))
            L = L + 1

        x_reshaped = x.view(B, H, L // 2, 2, D)
        approx = x_reshaped.mean(dim=3)  # Average (low-pass)
        detail = x_reshaped[:, :, :, 0, :] - x_reshaped[:, :, :, 1, :]  # Difference (high-pass)

        return approx, detail

    def _inverse_wavelet_transform(
        self,
        approx: torch.Tensor,
        detail: torch.Tensor,
        target_len: int,
    ) -> torch.Tensor:
        """Apply inverse wavelet transform.

        Args:
            approx: Approximation coefficients [B, H, L//2, D]
            detail: Detail coefficients [B, H, L//2, D]
            target_len: Target sequence length

        Returns:
            Reconstructed signal [B, H, L, D]
        """
        B, H, L_half, D = approx.shape

        # Reconstruct from approximation and detail
        x0 = approx + detail / 2
        x1 = approx - detail / 2

        # Interleave
        x = torch.stack([x0, x1], dim=3).view(B, H, L_half * 2, D)

        # Trim to target length
        if x.size(2) > target_len:
            x = x[:, :, :target_len, :]

        return x

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Apply Wavelet attention.

        Args:
            query: Query tensor [B, L_q, D]
            key: Key tensor [B, L_k, D]
            value: Value tensor [B, L_v, D]

        Returns:
            Attention output [B, L_q, D]
        """
        B, L_q, _ = query.shape
        L_k = key.shape[1]

        # Project to multi-head format
        q = self.q_proj(query).view(B, L_q, self.n_heads, self.d_head)
        k = self.k_proj(key).view(B, L_k, self.n_heads, self.d_head)
        v = self.v_proj(value).view(B, L_k, self.n_heads, self.d_head)

        # Permute to [B, H, L, D_head]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # Apply wavelet transform
        q_approx, q_detail = self._wavelet_transform(q)
        k_approx, k_detail = self._wavelet_transform(k)
        v_approx, v_detail = self._wavelet_transform(v)

        # Compute attention on approximation coefficients (low-frequency)
        attn_scores = torch.matmul(q_approx, k_approx.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to value approximations
        out_approx = torch.matmul(attn_weights, v_approx)

        # Keep detail coefficients (could also apply attention to details)
        # For simplicity, we use a simple average
        out_detail = v_detail.mean(dim=2, keepdim=True).expand_as(out_approx)

        # Inverse wavelet transform
        out = self._inverse_wavelet_transform(out_approx, out_detail, L_q)

        # Reshape and project
        out = out.permute(0, 2, 1, 3).contiguous().view(B, L_q, self.d_model)
        return self.out_proj(out)


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
        """Apply feedforward network.

        Args:
            x: Input tensor [B, T, D]

        Returns:
            Output tensor [B, T, D]
        """
        return self.net(x)


class FEDformerEncoderLayer(nn.Module):
    """FEDformer encoder layer with frequency attention and decomposition."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        moving_avg: int,
        dropout: float,
        activation: str,
        freq_mode: Literal['fourier', 'wavelet'],
        modes: int,
    ) -> None:
        super().__init__()

        if freq_mode == 'fourier':
            self.attention = FourierAttention(d_model, n_heads, dropout=dropout, modes=modes)
        else:  # wavelet
            self.attention = WaveletAttention(d_model, n_heads, dropout=dropout)

        self.decomp1 = SeriesDecomposition(moving_avg)
        self.decomp2 = SeriesDecomposition(moving_avg)
        self.ffn = FeedForward(d_model, d_ff, dropout, activation)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply encoder layer.

        Args:
            x: Input tensor [B, T, D]

        Returns:
            Tuple of (seasonal, trend) tensors
        """
        attn_out = self.attention(x, x, x)
        x = x + self.dropout(attn_out)
        x, trend1 = self.decomp1(self.norm1(x))

        ffn_out = self.ffn(x)
        x = x + self.dropout(ffn_out)
        x, trend2 = self.decomp2(self.norm2(x))

        return x, trend1 + trend2


class FEDformerDecoderLayer(nn.Module):
    """FEDformer decoder layer with self/cross frequency attention and decomposition."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        moving_avg: int,
        dropout: float,
        activation: str,
        freq_mode: Literal['fourier', 'wavelet'],
        modes: int,
    ) -> None:
        super().__init__()

        if freq_mode == 'fourier':
            self.self_attn = FourierAttention(d_model, n_heads, dropout=dropout, modes=modes)
            self.cross_attn = FourierAttention(d_model, n_heads, dropout=dropout, modes=modes)
        else:  # wavelet
            self.self_attn = WaveletAttention(d_model, n_heads, dropout=dropout)
            self.cross_attn = WaveletAttention(d_model, n_heads, dropout=dropout)

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
        """Apply decoder layer.

        Args:
            seasonal: Seasonal component [B, T, D]
            trend: Trend component [B, T, D]
            memory: Encoder memory [B, T_enc, D]

        Returns:
            Tuple of (seasonal, trend) tensors
        """
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


class FEDformerEncoder(nn.Module):
    """Stacked FEDformer encoder."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        moving_avg: int,
        dropout: float,
        activation: str,
        freq_mode: Literal['fourier', 'wavelet'],
        modes: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                FEDformerEncoderLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation,
                    freq_mode=freq_mode,
                    modes=modes,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply encoder stack.

        Args:
            x: Input tensor [B, T, D]

        Returns:
            Tuple of (seasonal, trend) tensors
        """
        trend_total = torch.zeros_like(x)
        for layer in self.layers:
            x, trend = layer(x)
            trend_total = trend_total + trend
        return x, trend_total


class FEDformerDecoder(nn.Module):
    """Stacked FEDformer decoder."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        moving_avg: int,
        dropout: float,
        activation: str,
        freq_mode: Literal['fourier', 'wavelet'],
        modes: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                FEDformerDecoderLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation,
                    freq_mode=freq_mode,
                    modes=modes,
                )
                for _ in range(num_layers)
            ]
        )
        self.decomp = SeriesDecomposition(moving_avg)

    def forward(
        self, seasonal: torch.Tensor, trend: torch.Tensor, memory: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply decoder stack.

        Args:
            seasonal: Seasonal component [B, T, D]
            trend: Trend component [B, T, D]
            memory: Encoder memory [B, T_enc, D]

        Returns:
            Tuple of (seasonal, trend) tensors
        """
        for layer in self.layers:
            seasonal, trend = layer(seasonal, trend, memory)
        seasonal, trend_extra = self.decomp(seasonal)
        return seasonal, trend + trend_extra


@register("fedformer")
class FEDformerModel(ARBaseModel):
    """FEDformer with Frequency Enhanced Decomposition."""

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
        freq_mode: Literal['fourier', 'wavelet'] = 'fourier',
        modes: int = 32,
        label_len: int = 24,
        pred_len: int = 1,
        max_len: int = 512,
        **kwargs: Dict[str, object],
    ) -> None:
        """Initialize FEDformer model.

        Args:
            input_dim: Input feature dimension
            output_dim: Output feature dimension
            d_model: Model dimension
            n_heads: Number of attention heads
            e_layers: Number of encoder layers
            d_layers: Number of decoder layers
            moving_avg: Moving average kernel size for decomposition
            d_ff: Feedforward dimension
            dropout: Dropout rate
            activation: Activation function ('gelu' or 'relu')
            freq_mode: Frequency processing mode ('fourier' or 'wavelet')
            modes: Number of Fourier modes to keep (for Fourier mode)
            label_len: Length of start token (decoder input from encoder)
            pred_len: Prediction horizon length
            max_len: Maximum sequence length
            **kwargs: Additional arguments
        """
        super().__init__(input_dim, output_dim, **kwargs)
        self.label_len = label_len
        self.pred_len = pred_len
        self.input_dim = input_dim
        self.freq_mode = freq_mode

        self.enc_embedding = DataEmbedding(input_dim, d_model, dropout, max_len)
        self.dec_embedding = DataEmbedding(input_dim, d_model, dropout, max_len)
        self.encoder = FEDformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            moving_avg=moving_avg,
            dropout=dropout,
            activation=activation,
            freq_mode=freq_mode,
            modes=modes,
            num_layers=e_layers,
        )
        self.decoder = FEDformerDecoder(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            moving_avg=moving_avg,
            dropout=dropout,
            activation=activation,
            freq_mode=freq_mode,
            modes=modes,
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
        """Forward pass of FEDformer.

        Args:
            x: Input tensor [B, T_in, D_in]
            context: Optional context (not used)
            **kwargs: Additional arguments

        Returns:
            Dictionary with:
                - preds: Predictions [B, pred_len, D_out]
                - extras: Dictionary with encoder/decoder components
        """
        del context, kwargs
        B, T_in, _ = x.shape

        # Encoder
        enc_out, enc_trend = self.encoder(self.enc_embedding(x))

        # Decoder input: last label_len from encoder + zeros for prediction
        label_len = min(self.label_len, T_in)
        dec_zeros = torch.zeros(
            B, self.pred_len, self.input_dim, device=x.device, dtype=x.dtype
        )
        dec_input = torch.cat([x[:, T_in - label_len :, :], dec_zeros], dim=1)
        seasonal_init, trend_init = self.decomp(self.dec_embedding(dec_input))

        # Align encoder trend with decoder input length
        enc_trend_aligned = enc_trend
        if enc_trend.size(1) != seasonal_init.size(1):
            enc_trend_aligned = F.interpolate(
                enc_trend.permute(0, 2, 1),
                size=seasonal_init.size(1),
                mode="linear",
                align_corners=False,
            ).permute(0, 2, 1)

        # Decoder
        dec_out, trend_out = self.decoder(
            seasonal_init, trend_init + enc_trend_aligned, enc_out
        )

        # Combine seasonal and trend, project to output dimension
        output = dec_out + trend_out
        preds = self.projection(output[:, -self.pred_len :, :])

        return {
            "preds": preds,
            "extras": {
                "encoder_trend": enc_trend,
                "decoder_trend": trend_out,
                "seasonal_component": dec_out,
                "freq_mode": self.freq_mode,
            },
        }
