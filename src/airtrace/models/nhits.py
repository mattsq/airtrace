"""N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting.

This implementation follows Challu et al., "N-HiTS: Neural Hierarchical
Interpolation for Time Series Forecasting" (AAAI 2023). It extends the N-BEATS
idea with multi-resolution pooling and interpolation blocks that reconstruct the
backcast and forecast at different temporal scales.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ResidualWrapperCompatible
from .registry import register


class NHiTSBlock(nn.Module):
    """Single N-HiTS block with pooling and interpolation heads."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        pred_len: int,
        pool_size: int = 2,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.0,
        interpolation_mode: str = "linear",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.pred_len = pred_len
        self.pool_size = pool_size
        self.dropout = dropout
        self.interpolation_mode = interpolation_mode
        self._align_corners = interpolation_mode in {
            "linear",
            "bilinear",
            "bicubic",
            "trilinear",
        }

        layers: List[nn.Linear] = [nn.Linear(input_dim, hidden_size)]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
        self.layers = nn.ModuleList(layers)
        self.backcast_head = nn.Linear(hidden_size, input_dim)
        self.forecast_head = nn.Linear(hidden_size, output_dim)

    def _interpolate(self, tensor: torch.Tensor, size: int) -> torch.Tensor:
        align_corners: Optional[bool] = False if self._align_corners else None
        return F.interpolate(
            tensor,
            size=size,
            mode=self.interpolation_mode,
            align_corners=align_corners,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute backcast and forecast components for a single block."""

        backcast_length = x.size(1)
        pooled = F.max_pool1d(
            x.transpose(1, 2),
            kernel_size=self.pool_size,
            stride=self.pool_size,
            ceil_mode=True,
        ).transpose(1, 2)

        hidden = pooled
        for layer in self.layers:
            hidden = F.relu(layer(hidden))
            if self.dropout > 0:
                hidden = F.dropout(hidden, p=self.dropout, training=self.training)

        backcast_low = self.backcast_head(hidden)
        forecast_low = self.forecast_head(hidden)

        backcast = self._interpolate(
            backcast_low.transpose(1, 2), size=backcast_length
        ).transpose(1, 2)
        forecast = self._interpolate(
            forecast_low.transpose(1, 2), size=self.pred_len
        ).transpose(1, 2)

        return backcast, forecast


@register("nhits")
class NHiTSModel(ResidualWrapperCompatible):
    """N-HiTS model composed of hierarchical interpolation stacks."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        pred_len: int = 1,
        pool_sizes: Optional[List[int]] = None,
        blocks_per_stack: int = 1,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.0,
        interpolation_mode: str = "linear",
    ) -> None:
        super().__init__(input_dim=input_dim, output_dim=output_dim)
        self.pred_len = pred_len
        self.pool_sizes = pool_sizes or [1, 2, 3]

        self.stacks = nn.ModuleList()
        for pool_size in self.pool_sizes:
            blocks = nn.ModuleList(
                [
                    NHiTSBlock(
                        input_dim=input_dim,
                        output_dim=output_dim,
                        pred_len=pred_len,
                        pool_size=pool_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        dropout=dropout,
                        interpolation_mode=interpolation_mode,
                    )
                    for _ in range(blocks_per_stack)
                ]
            )
            self.stacks.append(blocks)

    def _compute_block_forecasts(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        residual = x
        batch_size = x.size(0)
        forecast = torch.zeros(
            batch_size, self.pred_len, self.output_dim, device=x.device, dtype=x.dtype
        )
        stack_outputs: List[torch.Tensor] = []

        for blocks in self.stacks:
            stack_forecast = torch.zeros_like(forecast)
            for block in blocks:
                backcast, block_forecast = block(residual)
                residual = residual - backcast
                forecast = forecast + block_forecast
                stack_forecast = stack_forecast + block_forecast
            stack_outputs.append(stack_forecast)

        stacked = (
            torch.stack(stack_outputs, dim=1)
            if stack_outputs
            else torch.empty(0, device=x.device, dtype=x.dtype)
        )

        return forecast, stacked

    def encode(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        del context
        forecast, stacked = self._compute_block_forecasts(x)
        extras: Dict[str, torch.Tensor] = {"stack_forecasts": stacked}
        return stacked if stacked.numel() > 0 else forecast.unsqueeze(1), extras

    def decode(self, latent: torch.Tensor, pred_len: int) -> torch.Tensor:
        if pred_len != self.pred_len:
            raise ValueError(
                f"NHiTSModel only supports pred_len={self.pred_len}, got {pred_len}"
            )

        if latent.dim() == 4:
            return latent.sum(dim=1)
        if latent.dim() == 3 and latent.size(1) == pred_len:
            return latent
        raise ValueError(
            "Latent for NHiTSModel decode must be stacked forecasts [B, S, T, D] "
            "or a single forecast [B, T, D]."
        )

    def forward(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None, **kwargs
    ) -> Dict[str, torch.Tensor]:
        pred_len = int(kwargs.get("pred_len", self.pred_len))
        latent, extras = self.encode(x, context=context)
        preds = self.decode(latent, pred_len)
        extras["representation"] = latent
        return {"preds": preds, "extras": extras}
