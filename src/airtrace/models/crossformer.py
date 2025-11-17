"""Crossformer: Transformer utilizing cross-dimension dependencies."""

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ARBaseModel
from .registry import register


def _sinusoidal_encoding(length: int, dim: int, device: torch.device) -> torch.Tensor:
    """Create sinusoidal positional encoding.

    Args:
        length: Sequence length to encode.
        dim: Embedding dimension.
        device: Target device.

    Returns:
        Positional encoding tensor of shape [length, dim].
    """
    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / dim))
    pe = torch.zeros(length, dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class DimensionSegmentEmbedding(nn.Module):
    """Dimension-Segment-Wise (DSW) embedding used in Crossformer.

    The module segments variates into groups, flattens temporal patches within each
    segment, and projects them into the model dimension while adding temporal and
    dimensional positional encodings.
    """

    def __init__(
        self,
        input_dim: int,
        seg_len: int,
        dim_seg_size: int,
        d_model: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.seg_len = seg_len
        self.dim_seg_size = dim_seg_size
        self.d_model = d_model
        self.num_groups = math.ceil(input_dim / dim_seg_size)

        self.project = nn.Linear(dim_seg_size * seg_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """Embed segmented patches.

        Args:
            patches: Tensor of shape [B, N_patches, D, seg_len].

        Returns:
            Embedded tensor of shape [B, N_patches, num_groups, d_model].
        """
        device = patches.device
        B, n_patches, D, _ = patches.shape

        # Pad dimension axis so it divides evenly into groups
        pad_dims = (self.dim_seg_size - (D % self.dim_seg_size)) % self.dim_seg_size
        if pad_dims > 0:
            patches = F.pad(patches, (0, 0, 0, pad_dims))
            D += pad_dims

        num_groups = D // self.dim_seg_size
        patches = patches.view(B, n_patches, num_groups, self.dim_seg_size, self.seg_len)
        segments = patches.reshape(B, n_patches, num_groups, self.dim_seg_size * self.seg_len)

        embedded = self.project(segments)

        temporal_pe = _sinusoidal_encoding(n_patches, self.d_model, device).view(1, n_patches, 1, self.d_model)
        dim_pe = _sinusoidal_encoding(num_groups, self.d_model, device).view(1, 1, num_groups, self.d_model)

        embedded = embedded + temporal_pe + dim_pe
        return self.dropout(embedded)


class CrossformerEncoder(nn.Module):
    """Two-stage encoder capturing temporal and cross-dimension dependencies."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        temporal_depth: int,
        spatial_depth: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        cross_dim_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )

        self.temporal_encoder = nn.TransformerEncoder(temporal_layer, num_layers=temporal_depth)
        self.cross_dim_encoder = nn.TransformerEncoder(cross_dim_layer, num_layers=spatial_depth)

    def forward(self, tokens: torch.Tensor, num_groups: int) -> torch.Tensor:
        """Apply temporal then cross-dimension attention.

        Args:
            tokens: Input embeddings [B, N_patches, num_groups, d_model].
            num_groups: Number of variable groups.

        Returns:
            Encoded tokens with the same shape as input.
        """
        B, n_patches, _, d_model = tokens.shape

        # Stage 1: temporal attention per dimension group
        temporal_input = tokens.reshape(B * num_groups, n_patches, d_model)
        temporal_output = self.temporal_encoder(temporal_input)
        temporal_output = temporal_output.view(B, num_groups, n_patches, d_model).transpose(1, 2)

        # Stage 2: cross-dimension attention per temporal slice
        cross_input = temporal_output.reshape(B * n_patches, num_groups, d_model)
        cross_output = self.cross_dim_encoder(cross_input)
        cross_output = cross_output.view(B, n_patches, num_groups, d_model)

        return cross_output


@register("crossformer")
class CrossformerModel(ARBaseModel):
    """Crossformer model for multivariate forecasting.

    Implements the two-stage attention pipeline: temporal attention within each
    dimension segment followed by cross-dimension attention across variable groups.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        seg_len: int = 16,
        seg_stride: int = 8,
        dim_seg_size: int = 4,
        d_model: int = 128,
        nhead: int = 8,
        temporal_depth: int = 2,
        spatial_depth: int = 1,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        pred_len: int = 1,
        pooling: str = "last",
        **kwargs,
    ) -> None:
        super().__init__(input_dim, output_dim, **kwargs)

        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead for multi-head attention.")
        if seg_len <= 0 or seg_stride <= 0:
            raise ValueError("seg_len and seg_stride must be positive integers.")

        self.seg_len = seg_len
        self.seg_stride = seg_stride
        self.dim_seg_size = dim_seg_size
        self.d_model = d_model
        self.pred_len = pred_len
        self.pooling = pooling
        self.num_groups = math.ceil(input_dim / dim_seg_size)

        self.embedding = DimensionSegmentEmbedding(
            input_dim=input_dim,
            seg_len=seg_len,
            dim_seg_size=dim_seg_size,
            d_model=d_model,
            dropout=dropout,
        )
        self.encoder = CrossformerEncoder(
            d_model=d_model,
            nhead=nhead,
            temporal_depth=temporal_depth,
            spatial_depth=spatial_depth,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.projection = nn.Linear(self.num_groups * d_model, output_dim * pred_len)
        self.dropout = nn.Dropout(dropout)

    def _create_patches(self, x: torch.Tensor) -> torch.Tensor:
        """Create overlapping temporal patches along the sequence dimension."""
        _, T, _ = x.shape
        if T < self.seg_len:
            pad_len = self.seg_len - T
            pad = torch.zeros(
                x.size(0), pad_len, x.size(2), device=x.device, dtype=x.dtype
            )
            x = torch.cat([x, pad], dim=1)
            T = x.shape[1]

        n_patches = (T - self.seg_len) // self.seg_stride + 1
        patches = x.unfold(dimension=1, size=self.seg_len, step=self.seg_stride)
        return patches  # [B, N_patches, D, seg_len]

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for Crossformer.

        Args:
            x: Input tensor of shape [B, T_in, D_in].
            context: Optional context tensor (unused).
            **kwargs: Additional arguments.

        Returns:
            Dictionary containing predictions and intermediate features.
        """
        del context  # Crossformer currently ignores context inputs

        patches = self._create_patches(x)
        B, n_patches, _, _ = patches.shape

        embedded = self.embedding(patches)
        encoded = self.encoder(embedded, num_groups=self.num_groups)

        if self.pooling == "last":
            pooled = encoded[:, -1]
        elif self.pooling == "mean":
            pooled = encoded.mean(dim=1)
        else:
            raise ValueError("pooling must be either 'last' or 'mean'.")

        pooled = self.dropout(pooled)
        flattened = pooled.reshape(B, self.num_groups * self.d_model)
        preds = self.projection(flattened).view(B, self.pred_len, self.output_dim)

        return {
            "preds": preds,
            "extras": {
                "num_patches": torch.tensor(n_patches, device=x.device),
                "num_groups": torch.tensor(self.num_groups, device=x.device),
                "pooled_tokens": pooled,
            },
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  input_dim={self.input_dim},\n"
            f"  output_dim={self.output_dim},\n"
            f"  seg_len={self.seg_len},\n"
            f"  seg_stride={self.seg_stride},\n"
            f"  dim_seg_size={self.dim_seg_size},\n"
            f"  pred_len={self.pred_len},\n"
            f"  num_params={self.get_num_params():,}\n"
            f")"
        )
