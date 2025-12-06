"""TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting.

Implementation of TimeMixer from:
"TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting" (ICLR 2024)

Key innovations:
1. Multiscale decomposition: Analyzes time series at multiple temporal resolutions
2. Decomposable mixing: Separates seasonal and trend components, mixes them separately
3. Bottom-up seasonal mixing: Aggregates fine-grained seasonal patterns to coarse scales
4. Top-down trend mixing: Propagates macroscopic trend information to fine scales
5. Future multipredictor mixing: Ensembles predictions from multiple scales

This is particularly effective for multivariate sensor data where patterns exist at
multiple time scales (e.g., aircraft sensors with sub-second noise, multi-minute
maneuvers, and hour-long fuel consumption trends).

Reference: https://arxiv.org/abs/2405.14616
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ResidualWrapperCompatible
from .registry import register


class SeriesDecomposition(nn.Module):
    """Decompose time series into seasonal and trend components.

    Uses moving average to extract trend, with seasonal as residual.
    This is a classical and interpretable decomposition method.
    """

    def __init__(self, kernel_size: int = 25):
        """Initialize series decomposition.

        Args:
            kernel_size: Size of moving average kernel for trend extraction
        """
        super().__init__()
        self.kernel_size = kernel_size
        # Use average pooling for moving average
        self.avg_pool = nn.AvgPool1d(
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            count_include_pad=False
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decompose series into seasonal and trend.

        Args:
            x: Input tensor [B, T, D] or [B, D, T]

        Returns:
            seasonal: Seasonal component (high frequency)
            trend: Trend component (low frequency)
        """
        # Expect [B, T, D], convert to [B, D, T] for pooling
        if x.dim() == 3:
            permuted = x.permute(0, 2, 1)  # [B, D, T]
        else:
            permuted = x

        # Extract trend via moving average
        trend = self.avg_pool(permuted)  # [B, D, T]

        # Seasonal is residual
        seasonal = permuted - trend  # [B, D, T]

        # Convert back to [B, T, D]
        if x.dim() == 3:
            seasonal = seasonal.permute(0, 2, 1)
            trend = trend.permute(0, 2, 1)

        return seasonal, trend


class MultiScaleSeasonMixing(nn.Module):
    """Bottom-up mixing of seasonal components across scales.

    Progressively aggregates detailed seasonal information from fine to coarse scales.
    Each finer scale contributes residual information to the coarser scale.
    """

    def __init__(
        self,
        seq_len: int,
        down_sampling_layers: int,
        d_model: int,
        dropout: float = 0.1
    ):
        """Initialize multi-scale season mixing.

        Args:
            seq_len: Input sequence length
            down_sampling_layers: Number of downsampling scales
            d_model: Model dimension
            dropout: Dropout probability
        """
        super().__init__()
        self.down_sampling_layers = down_sampling_layers

        # Downsampling convolutions to create multiple scales
        self.down_sampling_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    in_channels=d_model,
                    out_channels=d_model,
                    kernel_size=3,
                    stride=2,
                    padding=1
                ),
                nn.GELU(),
                nn.Dropout(dropout)
            ) for _ in range(down_sampling_layers)
        ])

        # Mixing layers for each scale (bottom-up)
        # Need down_sampling_layers + 1 layers for all scales including coarsest
        self.mixing_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout)
            ) for _ in range(down_sampling_layers + 1)
        ])

    def forward(self, seasonal_init: torch.Tensor) -> torch.Tensor:
        """Apply bottom-up seasonal mixing.

        Args:
            seasonal_init: Initial seasonal component [B, T, D]

        Returns:
            Mixed seasonal representation [B, T, D]
        """
        B, T, D = seasonal_init.shape

        # Create multi-scale representations by progressive downsampling
        seasonal_scales = [seasonal_init.permute(0, 2, 1)]  # [B, D, T]

        for down_conv in self.down_sampling_convs:
            seasonal_scales.append(down_conv(seasonal_scales[-1]))

        # Bottom-up mixing: fine to coarse
        # Mix all scales, with finer scales contributing to coarser ones
        for i in range(self.down_sampling_layers + 1):
            # Mix current scale information
            # Permute to [B, T_i, D] for linear layer
            curr_scale = seasonal_scales[i].permute(0, 2, 1)
            mixed = self.mixing_layers[i](curr_scale)  # [B, T_i, D]

            # Add to next (coarser) scale via downsampling (if not at coarsest)
            if i < self.down_sampling_layers:
                next_scale = seasonal_scales[i + 1].permute(0, 2, 1)  # [B, T_next, D]
                # Downsample mixed to match next scale
                mixed_down = F.interpolate(
                    mixed.permute(0, 2, 1),
                    size=next_scale.shape[1],
                    mode='linear',
                    align_corners=False
                ).permute(0, 2, 1)
                # Update next scale with residual
                seasonal_scales[i + 1] = (next_scale + mixed_down).permute(0, 2, 1)
            else:
                # Coarsest scale - update with mixed output
                seasonal_scales[i] = mixed.permute(0, 2, 1)

        # Return coarsest scale with all aggregated information
        # Upsample final coarsest scale back to original resolution
        out = F.interpolate(
            seasonal_scales[-1],
            size=T,
            mode='linear',
            align_corners=False
        )

        return out.permute(0, 2, 1)  # [B, T, D]


class MultiScaleTrendMixing(nn.Module):
    """Top-down mixing of trend components across scales.

    Propagates macroscopic trend information from coarse to fine scales.
    Coarser scales provide prior knowledge to guide finer scale trends.
    """

    def __init__(
        self,
        seq_len: int,
        down_sampling_layers: int,
        d_model: int,
        dropout: float = 0.1
    ):
        """Initialize multi-scale trend mixing.

        Args:
            seq_len: Input sequence length
            down_sampling_layers: Number of downsampling scales
            d_model: Model dimension
            dropout: Dropout probability
        """
        super().__init__()
        self.down_sampling_layers = down_sampling_layers

        # Downsampling convolutions to create multiple scales
        self.down_sampling_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    in_channels=d_model,
                    out_channels=d_model,
                    kernel_size=3,
                    stride=2,
                    padding=1
                ),
                nn.GELU(),
                nn.Dropout(dropout)
            ) for _ in range(down_sampling_layers)
        ])

        # Mixing layers for each scale (top-down)
        # Need down_sampling_layers + 1 layers for all scales including finest
        self.mixing_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout)
            ) for _ in range(down_sampling_layers + 1)
        ])

    def forward(self, trend_init: torch.Tensor) -> torch.Tensor:
        """Apply top-down trend mixing.

        Args:
            trend_init: Initial trend component [B, T, D]

        Returns:
            Mixed trend representation [B, T, D]
        """
        B, T, D = trend_init.shape

        # Create multi-scale representations by progressive downsampling
        trend_scales = [trend_init.permute(0, 2, 1)]  # [B, D, T]

        for down_conv in self.down_sampling_convs:
            trend_scales.append(down_conv(trend_scales[-1]))

        # Top-down mixing: coarse to fine
        # Mix all scales, with coarser scales contributing to finer ones
        for i in range(self.down_sampling_layers, -1, -1):
            # Mix current scale
            curr_scale = trend_scales[i].permute(0, 2, 1)  # [B, T_i, D]
            mixed = self.mixing_layers[i](curr_scale)  # [B, T_i, D]

            if i > 0:
                # Upsample to next (finer) scale and add as residual
                prev_scale = trend_scales[i - 1].permute(0, 2, 1)  # [B, T_prev, D]
                mixed_up = F.interpolate(
                    mixed.permute(0, 2, 1),
                    size=prev_scale.shape[1],
                    mode='linear',
                    align_corners=False
                ).permute(0, 2, 1)
                # Update previous (finer) scale with residual
                trend_scales[i - 1] = (prev_scale + mixed_up).permute(0, 2, 1)
            else:
                # Finest scale - update with mixed output
                trend_scales[i] = mixed.permute(0, 2, 1)

        # Return finest scale with all trend information
        return trend_scales[0].permute(0, 2, 1)  # [B, T, D]


class PastDecomposableMixing(nn.Module):
    """Past-Decomposable-Mixing (PDM) Block.

    Core building block of TimeMixer that:
    1. Decomposes input into seasonal and trend components
    2. Applies bottom-up mixing to seasonal patterns
    3. Applies top-down mixing to trend patterns
    4. Recombines with residual connection
    """

    def __init__(
        self,
        seq_len: int,
        d_model: int,
        down_sampling_layers: int = 3,
        decomp_kernel: int = 25,
        dropout: float = 0.1
    ):
        """Initialize PDM block.

        Args:
            seq_len: Input sequence length
            d_model: Model dimension
            down_sampling_layers: Number of scales for multi-scale mixing
            decomp_kernel: Kernel size for series decomposition
            dropout: Dropout probability
        """
        super().__init__()

        # Series decomposition
        self.decomposition = SeriesDecomposition(kernel_size=decomp_kernel)

        # Multi-scale seasonal mixing (bottom-up)
        self.season_mixing = MultiScaleSeasonMixing(
            seq_len=seq_len,
            down_sampling_layers=down_sampling_layers,
            d_model=d_model,
            dropout=dropout
        )

        # Multi-scale trend mixing (top-down)
        self.trend_mixing = MultiScaleTrendMixing(
            seq_len=seq_len,
            down_sampling_layers=down_sampling_layers,
            d_model=d_model,
            dropout=dropout
        )

        # Layer normalization
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through PDM block.

        Args:
            x: Input tensor [B, T, D]

        Returns:
            Output tensor [B, T, D]
        """
        # Store input for residual
        residual = x

        # Decompose into seasonal and trend
        seasonal, trend = self.decomposition(x)

        # Mix seasonal (bottom-up) and trend (top-down)
        seasonal_mixed = self.season_mixing(seasonal)
        trend_mixed = self.trend_mixing(trend)

        # Recombine
        out = seasonal_mixed + trend_mixed

        # Add residual and normalize
        out = self.norm(out + residual)

        return out


@register("timemixer")
class TimeMixerModel(ResidualWrapperCompatible):
    """TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting.

    Fully MLP-based architecture that achieves SOTA performance through:
    - Multi-scale temporal analysis (microscopic to macroscopic)
    - Decomposable mixing of seasonal and trend components
    - Bottom-up aggregation of seasonal patterns
    - Top-down propagation of trend information
    - Ensemble predictions from multiple scales

    Particularly effective for aircraft sensor data with patterns at multiple
    time scales: sensor noise (fine), maneuvers (medium), fuel trends (coarse).

    Architecture:
    1. Input projection to d_model dimension
    2. Stack of PDM blocks for feature extraction
    3. Multi-scale prediction ensemble
    4. Output projection to target dimension
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        pred_len: int = 1,
        seq_len: int = 96,
        d_model: int = 64,
        num_layers: int = 2,
        down_sampling_layers: int = 3,
        decomp_kernel: int = 25,
        dropout: float = 0.1,
        **kwargs
    ):
        """Initialize TimeMixer model.

        Args:
            input_dim: Dimension of input features (number of sensors)
            output_dim: Dimension of output predictions
            pred_len: Prediction horizon length
            seq_len: Input sequence length (for multi-scale calculation)
            d_model: Model dimension
            num_layers: Number of PDM blocks
            down_sampling_layers: Number of scales for multi-scale mixing
            decomp_kernel: Kernel size for series decomposition
            dropout: Dropout probability
            **kwargs: Additional arguments
        """
        super().__init__(input_dim, output_dim, **kwargs)

        self.pred_len = pred_len
        self.seq_len = seq_len
        self.d_model = d_model
        self.num_layers = num_layers
        self.down_sampling_layers = down_sampling_layers

        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)

        # Stack of PDM blocks
        self.pdm_blocks = nn.ModuleList([
            PastDecomposableMixing(
                seq_len=seq_len,
                d_model=d_model,
                down_sampling_layers=down_sampling_layers,
                decomp_kernel=decomp_kernel,
                dropout=dropout
            ) for _ in range(num_layers)
        ])

        # Layer normalization
        self.norm = nn.LayerNorm(d_model)

        # Multi-scale prediction heads
        # Create predictions at different temporal resolutions
        self.prediction_heads = nn.ModuleList()
        curr_len = seq_len
        for i in range(down_sampling_layers + 1):
            self.prediction_heads.append(
                nn.Linear(d_model, pred_len * output_dim)
            )
            curr_len = curr_len // 2 if curr_len > 1 else 1

        # Final projection for ensemble
        self.output_projection = nn.Linear(
            (down_sampling_layers + 1) * pred_len * output_dim,
            pred_len * output_dim
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Initialize parameters
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _create_multi_scale(
        self,
        x: torch.Tensor,
        num_scales: int
    ) -> List[torch.Tensor]:
        """Create multi-scale representations via downsampling.

        Args:
            x: Input tensor [B, T, D]
            num_scales: Number of scales to create

        Returns:
            List of tensors at different scales
        """
        scales = [x]

        # Progressively downsample
        for i in range(num_scales):
            # Permute to [B, D, T] for conv
            prev = scales[-1].permute(0, 2, 1)
            # Downsample by 2
            if prev.shape[2] > 1:
                downsampled = F.avg_pool1d(prev, kernel_size=2, stride=2)
                scales.append(downsampled.permute(0, 2, 1))
            else:
                # Can't downsample further
                break

        return scales

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        del context, kwargs
        latent, extras = self.encode(x)
        preds = self.decode(latent, pred_len=self.pred_len)

        return {"preds": preds, "extras": extras}

    def encode(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Encode sequence into multi-scale pooled predictions."""

        del context
        _batch_size, _, _ = x.shape

        x = self.input_projection(x)
        x = self.dropout(x)

        for pdm_block in self.pdm_blocks:
            x = pdm_block(x)

        x = self.norm(x)
        multi_scale_features = self._create_multi_scale(x, self.down_sampling_layers)

        scale_predictions = []
        for features, pred_head in zip(multi_scale_features, self.prediction_heads):
            pooled = features.mean(dim=1)
            scale_predictions.append(pred_head(pooled))

        latent = torch.cat(scale_predictions, dim=1)
        extras = {
            "multi_scale_features": multi_scale_features,
            "scale_predictions": scale_predictions,
        }

        return latent, extras

    def decode(self, latent: torch.Tensor, pred_len: int) -> torch.Tensor:
        """Decode ensembled scale predictions into final forecast."""

        if pred_len != self.pred_len:
            raise ValueError(
                f"TimeMixerModel only supports pred_len={self.pred_len}, received {pred_len}."
            )

        final_pred = self.output_projection(latent)
        preds = final_pred.reshape(latent.shape[0], self.pred_len, self.output_dim)
        return preds

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  input_dim={self.input_dim},\n"
            f"  output_dim={self.output_dim},\n"
            f"  pred_len={self.pred_len},\n"
            f"  seq_len={self.seq_len},\n"
            f"  d_model={self.d_model},\n"
            f"  num_layers={self.num_layers},\n"
            f"  down_sampling_layers={self.down_sampling_layers},\n"
            f"  num_params={self.get_num_params():,}\n"
            f")"
        )
