"""TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis.

TimesNet transforms 1D time series into 2D tensors through period-based reshaping
and applies vision backbones (2D convolutions) to capture both intraperiod and
interperiod variations.

Reference:
    Wu et al. (2023): "TimesNet: Temporal 2D-Variation Modeling for General Time
    Series Analysis" (ICLR 2023)
    Paper: https://arxiv.org/abs/2210.02186
    Code: https://github.com/thuml/TimesNet
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ResidualWrapperCompatible
from .registry import register


class TimesBlock(nn.Module):
    """TimesBlock: Core module that applies 2D convolutions to period-reshaped tensors.

    This module:
    1. Detects top-k periods using FFT
    2. Reshapes 1D time series into 2D tensors based on each period
    3. Applies 2D Inception-like convolutions
    4. Aggregates multi-period representations
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_kernels: int = 6,
        top_k: int = 5,
        dropout: float = 0.1,
    ) -> None:
        """Initialize TimesBlock.

        Args:
            d_model: Model dimension
            d_ff: Feedforward dimension
            num_kernels: Number of kernels in Inception block
            top_k: Number of top periods to use
            dropout: Dropout rate
        """
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_kernels = num_kernels
        self.top_k = top_k

        # Inception-like 2D convolution block
        self.conv = nn.ModuleList()
        for i in range(num_kernels):
            # Different kernel sizes to capture multi-scale patterns
            kernel_size = 2 * i + 1
            padding = kernel_size // 2
            self.conv.append(
                nn.Conv2d(
                    in_channels=d_model,
                    out_channels=d_ff,
                    kernel_size=kernel_size,
                    padding=padding,
                )
            )

        # Projection to aggregate multi-kernel outputs
        self.projection = nn.Linear(num_kernels * d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def _detect_periods(self, x: torch.Tensor, top_k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Detect top-k periods using FFT.

        Args:
            x: Input tensor [B, T, D]
            top_k: Number of top periods to detect

        Returns:
            Tuple of (period_list, period_weights):
                - period_list: [B, top_k] tensor of period lengths
                - period_weights: [B, top_k] tensor of period importance weights
        """
        B, T, D = x.shape

        # Apply FFT along time dimension
        # Average across channels for period detection
        x_mean = x.mean(dim=-1)  # [B, T]

        # FFT to frequency domain
        x_fft = torch.fft.rfft(x_mean, dim=-1)  # [B, T//2 + 1]

        # Compute amplitude spectrum (power)
        amplitude = torch.abs(x_fft)  # [B, T//2 + 1]

        # Ignore DC component and very high frequencies
        # Focus on frequencies corresponding to periods in range [2, T//2]
        min_period = 2
        max_freq_idx = T // min_period
        amplitude[:, 0] = 0  # Remove DC component
        if max_freq_idx < amplitude.shape[1]:
            amplitude[:, max_freq_idx:] = 0  # Remove high frequencies

        # Find top-k frequencies
        top_k_actual = min(top_k, amplitude.shape[1] - 1)
        top_values, top_indices = torch.topk(amplitude, k=top_k_actual, dim=-1)

        # Convert frequency indices to periods
        # Period = T / frequency_index (avoiding division by zero)
        periods = T / (top_indices.float() + 1e-8)  # [B, top_k]
        periods = periods.clamp(min=min_period, max=T)
        periods = periods.long()

        # Normalize weights
        weights = top_values / (top_values.sum(dim=-1, keepdim=True) + 1e-8)  # [B, top_k]

        return periods, weights

    def _reshape_to_2d(self, x: torch.Tensor, period: int) -> Tuple[torch.Tensor, int]:
        """Reshape 1D time series to 2D tensor based on period.

        Args:
            x: Input tensor [B, T, D]
            period: Period length for reshaping

        Returns:
            Tuple of (reshaped, padding):
                - reshaped: [B, D, period, num_periods]
                - padding: Amount of padding added
        """
        B, T, D = x.shape

        # Calculate number of complete periods
        num_periods = T // period

        # Pad if necessary to have complete periods
        padding = period * num_periods - T
        if padding < 0:
            padding = period - (T % period)
            num_periods = (T + padding) // period

        if padding > 0:
            # Pad with replication of boundary values
            x = F.pad(x, (0, 0, 0, padding), mode='replicate')
            T = T + padding

        # Reshape to [B, num_periods, period, D]
        x_reshaped = x[:, :num_periods * period, :].reshape(B, num_periods, period, D)

        # Permute to [B, D, period, num_periods] for Conv2D
        x_reshaped = x_reshaped.permute(0, 3, 2, 1).contiguous()

        return x_reshaped, padding

    def _reshape_from_2d(self, x: torch.Tensor, original_len: int, padding: int) -> torch.Tensor:
        """Reshape 2D tensor back to 1D time series.

        Args:
            x: Input tensor [B, D, period, num_periods]
            original_len: Original sequence length before padding
            padding: Amount of padding to remove

        Returns:
            Reshaped tensor [B, T, D]
        """
        B, D, period, num_periods = x.shape

        # Permute back to [B, num_periods, period, D]
        x = x.permute(0, 3, 2, 1).contiguous()

        # Reshape to [B, T, D]
        x = x.reshape(B, num_periods * period, D)

        # Remove padding
        if padding > 0:
            x = x[:, :-padding, :]

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [B, T, D]

        Returns:
            Output tensor [B, T, D]
        """
        B, T, D = x.shape

        # Detect top-k periods
        periods, weights = self._detect_periods(x, self.top_k)  # [B, k], [B, k]

        # Process each period and aggregate
        outputs = []
        for k in range(self.top_k):
            # Get period for each sample in batch
            # Use the median period across batch for simplicity
            period_k = int(periods[:, k].float().median().item())
            period_k = max(2, min(period_k, T))  # Clamp to valid range

            # Reshape to 2D
            x_2d, padding = self._reshape_to_2d(x, period_k)  # [B, D, P, N]

            # Apply multi-kernel 2D convolutions
            conv_outputs = []
            for conv_layer in self.conv:
                conv_out = conv_layer(x_2d)  # [B, d_ff, P, N]
                conv_outputs.append(conv_out)

            # Concatenate multi-kernel outputs
            x_conv = torch.cat(conv_outputs, dim=1)  # [B, num_kernels * d_ff, P, N]

            # Permute for linear projection: [B, P, N, num_kernels * d_ff]
            x_conv = x_conv.permute(0, 2, 3, 1).contiguous()

            # Project back to d_model
            x_proj = self.projection(x_conv)  # [B, P, N, d_model]

            # Permute back to [B, d_model, P, N]
            x_proj = x_proj.permute(0, 3, 1, 2).contiguous()

            # Reshape back to 1D
            x_1d = self._reshape_from_2d(x_proj, T, padding)  # [B, T, D]

            # Weight by period importance (expand weights for broadcasting)
            weight_k = weights[:, k:k+1, None]  # [B, 1, 1]
            outputs.append(x_1d * weight_k)

        # Aggregate all period representations
        output = sum(outputs)  # [B, T, D]

        # Apply dropout
        output = self.dropout(output)

        return output


class PositionalEmbedding(nn.Module):
    """Fixed sinusoidal positional embedding."""

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()

        # Create positional encoding table
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding.

        Args:
            x: Input tensor [B, T, D]

        Returns:
            Tensor with positional encoding added
        """
        return x + self.pe[:, :x.size(1), :]


@register("timesnet")
class TimesNetModel(ResidualWrapperCompatible):
    """TimesNet model for time series forecasting.

    TimesNet transforms 1D time series into 2D space through period-based reshaping
    and applies 2D convolutions to capture complex temporal variations. It discovers
    multi-periodicity using FFT and processes each period with shared 2D kernels.

    Key Features:
        - Period detection via FFT
        - 2D reshaping based on detected periods
        - Inception-style multi-kernel convolutions
        - Multi-period aggregation

    Args:
        input_dim: Input feature dimension
        output_dim: Output feature dimension
        seq_len: Expected input sequence length
        pred_len: Prediction horizon
        d_model: Model hidden dimension
        d_ff: Feedforward dimension
        num_layers: Number of TimesBlock layers
        num_kernels: Number of kernels in Inception block
        top_k: Number of top periods to use
        dropout: Dropout rate
        embed_type: Type of embedding ('positional' or 'none')
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        seq_len: int = 96,
        pred_len: int = 1,
        d_model: int = 64,
        d_ff: int = 128,
        num_layers: int = 2,
        num_kernels: int = 6,
        top_k: int = 5,
        dropout: float = 0.1,
        embed_type: str = 'positional',
        **kwargs,
    ) -> None:
        super().__init__(input_dim, output_dim, **kwargs)

        # Validate parameters
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}")
        if pred_len <= 0:
            raise ValueError(f"pred_len must be positive, got {pred_len}")
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if d_ff <= 0:
            raise ValueError(f"d_ff must be positive, got {d_ff}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if num_kernels <= 0:
            raise ValueError(f"num_kernels must be positive, got {num_kernels}")
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model
        self.embed_type = embed_type

        # Input embedding
        self.input_projection = nn.Linear(input_dim, d_model)

        # Positional embedding
        if embed_type == 'positional':
            self.pos_embedding = PositionalEmbedding(d_model, max_len=seq_len + pred_len)
        else:
            self.pos_embedding = None

        # Stacked TimesBlocks
        self.layers = nn.ModuleList([
            TimesBlock(
                d_model=d_model,
                d_ff=d_ff,
                num_kernels=num_kernels,
                top_k=top_k,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # Layer normalization
        self.norm = nn.LayerNorm(d_model)

        # Prediction head
        # For multi-step prediction, we use a linear projection
        self.prediction_head = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, pred_len * output_dim),
        )

        self.dropout = nn.Dropout(dropout)

    def encode(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        del context
        x_proj = self.input_projection(x)
        if self.pos_embedding is not None:
            x_proj = self.pos_embedding(x_proj)

        x_proj = self.dropout(x_proj)

        layer_outputs = []
        hidden = x_proj
        for layer in self.layers:
            residual = hidden
            hidden = layer(hidden)
            hidden = self.norm(hidden + residual)
            layer_outputs.append(hidden)

        last_hidden = hidden[:, -1, :]
        extras: Dict[str, torch.Tensor] = {
            "embeddings": hidden,
            "layer_outputs": layer_outputs,
            "last_hidden": last_hidden,
        }
        return last_hidden, extras

    def decode(self, latent: torch.Tensor, pred_len: int) -> torch.Tensor:
        if pred_len != self.pred_len:
            raise ValueError(
                f"pred_len {pred_len} does not match configured pred_len {self.pred_len}"
            )

        pred_flat = self.prediction_head(latent)
        return pred_flat.reshape(latent.size(0), self.pred_len, self.output_dim)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor [B, T, D_in]
            context: Optional context tensor (unused)
            **kwargs: Additional arguments

        Returns:
            Dictionary containing:
                - preds: Predictions [B, pred_len, D_out]
                - extras: Additional outputs (embeddings, etc.)
        """
        pred_len = int(kwargs.get("pred_len", self.pred_len))
        latent, extras = self.encode(x, context=context)
        preds = self.decode(latent, pred_len)
        extras["representation"] = latent
        return {"preds": preds, "extras": extras}
