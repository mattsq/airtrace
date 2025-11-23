"""Implementation of TSMixer architecture.

This module implements TSMixer introduced in:
  Chen et al., "TSMixer: An All-MLP Architecture for Time Series Forecasting" (KDD 2023).

Key design principles:
- All-MLP architecture (no attention, convolutions, or recurrence)
- Alternating time-mixing and feature-mixing MLPs
- Residual connections and layer normalization for stable training
- Supports multivariate time series forecasting

Architecture:
  Input [B, T_in, D_in]
    -> Time-Mixing MLP (operates on time dimension)
    -> Feature-Mixing MLP (operates on feature dimension)
    -> ... (repeated N times)
    -> Temporal Projection (T_in -> T_out)
    -> Output Projection (D_in -> D_out if needed)
  Output [B, T_out, D_out]
"""

from typing import Dict, Optional

import torch
import torch.nn as nn

from .base import ARBaseModel
from .registry import register


class TSMixerBlock(nn.Module):
    """Single TSMixer block with time-mixing and feature-mixing MLPs.

    This block applies two sequential operations:
    1. Time-mixing: MLP operates along the time dimension
    2. Feature-mixing: MLP operates along the feature dimension

    Both operations use residual connections and layer normalization.
    """

    def __init__(
        self,
        seq_len: int,
        feature_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ) -> None:
        """Initialize TSMixer block.

        Args:
            seq_len: Length of input sequence
            feature_dim: Number of features/channels
            hidden_dim: Hidden dimension for MLP expansion
            dropout: Dropout probability
        """
        super().__init__()

        # Time-mixing components
        self.time_norm = nn.LayerNorm(feature_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(seq_len, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, seq_len),
            nn.Dropout(dropout),
        )

        # Feature-mixing components
        self.feature_norm = nn.LayerNorm(feature_dim)
        self.feature_mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through TSMixer block.

        Args:
            x: Input tensor [B, T, D]

        Returns:
            Output tensor [B, T, D]
        """
        # Time-mixing: [B, T, D] -> [B, D, T] -> MLP -> [B, D, T] -> [B, T, D]
        residual = x
        x = self.time_norm(x)  # [B, T, D]
        x = x.transpose(1, 2)  # [B, D, T]
        x = self.time_mlp(x)   # [B, D, T]
        x = x.transpose(1, 2)  # [B, T, D]
        x = x + residual       # Residual connection

        # Feature-mixing: [B, T, D] -> MLP on D -> [B, T, D]
        residual = x
        x = self.feature_norm(x)   # [B, T, D]
        x = self.feature_mlp(x)    # [B, T, D]
        x = x + residual           # Residual connection

        return x


class BlockOutputs(list[torch.Tensor]):
    """Container for TSMixer block outputs that exposes tensor-like shape."""

    def __init__(self, outputs: list[torch.Tensor]) -> None:
        super().__init__(outputs)
        self._stacked = torch.stack(outputs, dim=0) if outputs else torch.empty(0)

    @property
    def shape(self) -> torch.Size:
        """Return stacked shape, mirroring torch.Tensor.shape."""

        return self._stacked.shape


@register("tsmixer")
class TSMixerModel(ARBaseModel):
    """TSMixer model for time series forecasting.

    TSMixer is an all-MLP architecture that alternates between time-mixing
    and feature-mixing operations. It's simple, efficient, and competitive
    with transformer-based models for time series forecasting.

    Key features:
    - All-MLP architecture (no attention or convolution)
    - Time and feature mixing in alternating blocks
    - Residual connections for deep networks
    - Temporal projection for flexible horizon forecasting
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        seq_len: int = 60,
        pred_len: int = 1,
        num_blocks: int = 4,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        **kwargs,
    ) -> None:
        """Initialize TSMixer model.

        Args:
            input_dim: Dimension of input features
            output_dim: Dimension of output predictions
            seq_len: Length of input sequence
            pred_len: Length of prediction horizon
            num_blocks: Number of TSMixer blocks to stack
            hidden_dim: Hidden dimension for MLP expansion (default: 256)
            dropout: Dropout probability (default: 0.1)
            **kwargs: Additional arguments (ignored for compatibility)
        """
        super().__init__(input_dim, output_dim, **kwargs)

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_blocks = num_blocks
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # Stack of TSMixer blocks
        self.mixer_blocks = nn.ModuleList([
            TSMixerBlock(
                seq_len=seq_len,
                feature_dim=input_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
            for _ in range(num_blocks)
        ])

        # Temporal projection: map from seq_len to pred_len
        self.temporal_projection = nn.Linear(seq_len, pred_len)

        # Feature/output projection if input_dim != output_dim
        if input_dim != output_dim:
            self.output_projection = nn.Linear(input_dim, output_dim)
        else:
            self.output_projection = None

        # Final layer norm
        self.final_norm = nn.LayerNorm(input_dim)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through TSMixer.

        Args:
            x: Input tensor [B, T_in, D_in]
            context: Optional context tensor (not used in TSMixer)
            **kwargs: Additional arguments (ignored)

        Returns:
            Dictionary containing:
                - preds: Predictions [B, T_out, D_out]
                - extras: Additional outputs (block outputs for analysis)
        """
        if x.size(1) != self.seq_len:
            raise ValueError(
                f"Expected input sequence length {self.seq_len}, got {x.size(1)}"
            )

        # Store intermediate outputs for potential analysis
        block_outputs: list[torch.Tensor] = []

        # Pass through mixer blocks
        out = x
        for block in self.mixer_blocks:
            out = block(out)
            block_outputs.append(out.detach().clone())

        # Final normalization
        out = self.final_norm(out)  # [B, T_in, D_in]

        # Temporal projection: [B, T_in, D_in] -> [B, D_in, T_in] -> [B, D_in, T_out] -> [B, T_out, D_in]
        out = out.transpose(1, 2)                    # [B, D_in, T_in]
        out = self.temporal_projection(out)          # [B, D_in, T_out]
        out = out.transpose(1, 2)                    # [B, T_out, D_in]

        # Output projection if needed: [B, T_out, D_in] -> [B, T_out, D_out]
        if self.output_projection is not None:
            out = self.output_projection(out)

        return {
            "preds": out,
            "extras": {"block_outputs": BlockOutputs(block_outputs)},
        }

    def __repr__(self) -> str:
        """String representation of the model."""
        return (
            f"{self.__class__.__name__}(\n"
            f"  input_dim={self.input_dim},\n"
            f"  output_dim={self.output_dim},\n"
            f"  seq_len={self.seq_len},\n"
            f"  pred_len={self.pred_len},\n"
            f"  num_blocks={self.num_blocks},\n"
            f"  hidden_dim={self.hidden_dim},\n"
            f"  dropout={self.dropout},\n"
            f"  num_params={self.get_num_params():,}\n"
            f")"
        )
