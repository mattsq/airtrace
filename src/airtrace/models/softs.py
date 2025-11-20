"""SOFTS: Efficient Multivariate Time Series Forecasting with Series-Core Fusion.

Reference:
    Han et al. (2024): "SOFTS: Efficient Multivariate Time Series Forecasting with
    Series-Core Fusion", NeurIPS 2024.
    arXiv: https://arxiv.org/abs/2404.14197
    Code: https://github.com/Secilia-Cxy/SOFTS
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ARBaseModel
from .registry import register


class STARModule(nn.Module):
    """STar Aggregate-Redistribute Module for efficient channel mixing.

    The STAR module performs aggregate-redistribute operations across channels:
    1. Aggregate: Compress channel information to a core representation
    2. Stochastic Pooling: Sample or average across channels
    3. Redistribute: Fuse core with original features

    This design achieves O(C) complexity instead of O(C²) for attention.
    """

    def __init__(self, d_series: int, d_core: int):
        """Initialize STAR module.

        Args:
            d_series: Input dimension per channel (typically d_model)
            d_core: Core compression dimension (bottleneck size)
        """
        super().__init__()
        self.gen1 = nn.Linear(d_series, d_series)  # FFN preprocessing
        self.gen2 = nn.Linear(d_series, d_core)  # Aggregate to core
        self.gen3 = nn.Linear(d_series + d_core, d_series)  # Fuse
        self.gen4 = nn.Linear(d_series, d_series)  # Final projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with stochastic pooling.

        Args:
            x: Input tensor [B, C, d_series] where
               B = batch size, C = channels, d_series = features

        Returns:
            Output tensor [B, C, d_series]
        """
        batch_size, channels, _ = x.shape

        # Aggregate: FFN + compression to core
        h = F.gelu(self.gen1(x))  # [B, C, d_series]
        z = self.gen2(h)  # [B, C, d_core]

        # Stochastic pooling (training) or weighted average (eval)
        if self.training:
            # Multinomial sampling across channels
            ratio = F.softmax(z, dim=1)  # [B, C, d_core]
            ratio = ratio.permute(0, 2, 1).reshape(-1, channels)  # [B*d_core, C]
            indices = torch.multinomial(ratio, 1)  # [B*d_core, 1]
            indices = indices.view(batch_size, -1, 1).permute(0, 2, 1)  # [B, 1, d_core]
            z_agg = torch.gather(z, 1, indices)  # [B, 1, d_core]
            z_agg = z_agg.repeat(1, channels, 1)  # [B, C, d_core]
        else:
            # Weighted average
            weight = F.softmax(z, dim=1)  # [B, C, d_core]
            z_agg = torch.sum(z * weight, dim=1, keepdim=True)  # [B, 1, d_core]
            z_agg = z_agg.repeat(1, channels, 1)  # [B, C, d_core]

        # Redistribute: concatenate + fusion MLP
        fused = torch.cat([x, z_agg], dim=-1)  # [B, C, d_series + d_core]
        fused = F.gelu(self.gen3(fused))  # [B, C, d_series]
        output = self.gen4(fused)  # [B, C, d_series]

        return output


class EncoderLayerSOFTS(nn.Module):
    """Encoder layer with STAR module and feedforward network."""

    def __init__(
        self,
        d_model: int,
        d_core: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        """Initialize encoder layer.

        Args:
            d_model: Model dimension
            d_core: STAR core compression dimension
            d_ff: Feedforward network dimension
            dropout: Dropout probability
            activation: Activation function ('gelu' or 'relu')
        """
        super().__init__()
        self.star = STARModule(d_model, d_core)

        # Pointwise feedforward (1x1 convolutions)
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)

        # Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Regularization
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder layer.

        Args:
            x: Input tensor [B, C, d_model]

        Returns:
            Output tensor [B, C, d_model]
        """
        # STAR module + residual
        residual = x
        x = self.star(x)
        x = residual + self.dropout(x)
        x = self.norm1(x)

        # Feedforward network + residual
        residual = x
        y = x.transpose(-1, 1)  # [B, d_model, C]
        y = self.activation(self.conv1(y))  # [B, d_ff, C]
        y = self.dropout(y)
        y = self.conv2(y)  # [B, d_model, C]
        y = self.dropout(y).transpose(-1, 1)  # [B, C, d_model]
        x = residual + y
        x = self.norm2(x)

        return x


@register("softs")
class SOFTS(ARBaseModel):
    """SOFTS: Series-cOre Fused Time Series forecaster.

    A pure MLP-based model for multivariate time series forecasting using
    the STAR (STar Aggregate-Redistribute) module for efficient channel mixing.

    Key features:
    - Channel-as-token paradigm: treats each variable as a token
    - STAR module: efficient O(C) channel mixing with stochastic pooling
    - Pure MLP architecture: no attention mechanisms
    - Instance normalization: handles non-stationary data

    Reference:
        Han et al. (2024): "SOFTS: Efficient Multivariate Time Series Forecasting
        with Series-Core Fusion", NeurIPS 2024.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        seq_len: int,
        pred_len: int,
        horizon: Optional[int] = None,
        hidden_dim: int = 512,
        d_core: int = 128,
        d_ff: int = 512,
        e_layers: int = 3,
        dropout: float = 0.0,
        activation: str = "gelu",
        use_norm: bool = True,
        **kwargs,
    ):
        """Initialize SOFTS model.

        Args:
            input_dim: Number of input channels (D)
            output_dim: Number of output channels (typically same as input_dim)
            seq_len: Input sequence length (T)
            pred_len: Prediction window length (for data windowing)
            horizon: Actual prediction horizon (defaults to pred_len if not provided)
            hidden_dim: Model dimension (d_model)
            d_core: STAR core compression dimension
            d_ff: Feedforward network dimension
            e_layers: Number of encoder layers
            dropout: Dropout probability
            activation: Activation function ('gelu' or 'relu')
            use_norm: Whether to use instance normalization
        """
        super().__init__(input_dim, output_dim, **kwargs)

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.horizon = horizon if horizon is not None else pred_len
        self.use_norm = use_norm

        # Channel-as-token embedding: [B, T, D] -> [B, D, d_model]
        self.embedding = nn.Linear(seq_len, hidden_dim)
        self.embed_dropout = nn.Dropout(dropout)

        # Encoder: stack of STAR-based layers
        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayerSOFTS(
                    d_model=hidden_dim,
                    d_core=d_core,
                    d_ff=d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(e_layers)
            ]
        )

        # Projection head: [B, D, d_model] -> [B, D, horizon] -> [B, horizon, D]
        self.projection = nn.Linear(hidden_dim, self.horizon)

    def forward(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None, **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through SOFTS model.

        Args:
            x: Input tensor [B, T, D]
            context: Optional context (not used in SOFTS)
            **kwargs: Additional arguments

        Returns:
            Dictionary containing:
                - preds: Predictions [B, horizon, D_out]
                - extras: Dictionary with normalization statistics
        """
        batch_size, seq_len, num_channels = x.shape

        # Instance normalization (per channel)
        if self.use_norm:
            means = x.mean(dim=1, keepdim=True)  # [B, 1, D]
            x = x - means
            stdev = torch.sqrt(
                torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            x = x / stdev
        else:
            means, stdev = None, None

        # Embedding: [B, T, D] -> [B, D, d_model]
        x = x.permute(0, 2, 1)  # [B, D, T]
        x = self.embedding(x)  # [B, D, d_model]
        x = self.embed_dropout(x)

        # Encoder: [B, D, d_model] -> [B, D, d_model]
        for layer in self.encoder_layers:
            x = layer(x)

        # Projection: [B, D, d_model] -> [B, D, horizon]
        x = self.projection(x)  # [B, D, horizon]

        # Permute to output format: [B, horizon, D]
        x = x.permute(0, 2, 1)  # [B, horizon, D]

        # Ensure output matches expected channels
        x = x[:, :, : self.output_dim]

        # De-normalization
        if self.use_norm and means is not None:
            if self.output_dim <= num_channels:
                stdev_out = stdev[:, :, : self.output_dim]
                means_out = means[:, :, : self.output_dim]
            else:
                pad_size = self.output_dim - num_channels
                stdev_out = F.pad(stdev, (0, pad_size), mode="replicate")
                means_out = F.pad(means, (0, pad_size), mode="replicate")

            x = x * stdev_out
            x = x + means_out

        return {
            "preds": x,
            "extras": {
                "means": means,
                "stdev": stdev,
            },
        }
