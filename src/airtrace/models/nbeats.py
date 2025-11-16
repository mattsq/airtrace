"""Implementation of the N-BEATS architecture.

This module implements the N-BEATS model introduced in:
  Oreshkin et al., "N-BEATS: Neural basis expansion analysis for interpretable
  time series forecasting" (ICLR 2020).

Key design choices for AirTrace:
- Supports multivariate inputs/outputs by flattening the window and projecting
  back to `[B, T, D]` tensors.
- Provides generic, trend, and seasonality blocks consistent with the original
  paper's interpretable stacks.
- Uses residual stacking with backcast/forecast decomposition to enable deep
  ensembles of blocks while maintaining stable gradients.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ARBaseModel
from .registry import register


class NBeatsBlock(nn.Module):
    """Single N-BEATS block producing a backcast and forecast component."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        pred_len: int,
        hidden_size: int = 256,
        num_layers: int = 4,
        block_type: str = "generic",
        degree: int = 2,
        harmonics: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.pred_len = pred_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.block_type = block_type
        self.degree = degree
        self.harmonics = harmonics
        self.dropout = dropout

        # First layer is lazy to adapt to variable window lengths.
        layers = [nn.LazyLinear(hidden_size)]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
        self.fc_stack = nn.ModuleList(layers)
        self.theta_layer: Optional[nn.Linear] = None

    def _create_theta_layer(
        self, backcast_size: int, forecast_size: int, device: torch.device
    ) -> None:
        theta_size = self._theta_size(backcast_size, forecast_size)
        theta = nn.Linear(self.hidden_size, theta_size)
        self.theta_layer = theta.to(device)

    def _theta_size(self, backcast_size: int, forecast_size: int) -> int:
        if self.block_type == "generic":
            return backcast_size + forecast_size
        if self.block_type in {"trend", "seasonality"}:
            theta_dim = self._basis_dim(backcast_size, forecast_size)[0]
            return 2 * theta_dim
        raise ValueError(f"Unsupported block type: {self.block_type}")

    def _basis_dim(self, backcast_size: int, forecast_size: int) -> Tuple[int, int]:
        if self.block_type == "trend":
            return self.degree + 1, self.degree + 1
        if self.block_type == "seasonality":
            backcast_len = backcast_size // self.input_dim
            num_harmonics = self.harmonics or max(1, backcast_len // 2)
            dim = 2 * num_harmonics + 1
            return dim, dim
        raise ValueError(f"Unsupported block type: {self.block_type}")

    def _trend_basis(self, length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        t = torch.linspace(-1.0, 1.0, steps=length, device=device, dtype=dtype)
        return torch.stack([t ** i for i in range(self.degree + 1)], dim=0)

    def _seasonality_basis(
        self, length: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        harmonics = self.harmonics or max(1, length // 2)
        t = torch.linspace(0, 2 * torch.pi, steps=length, device=device, dtype=dtype)
        basis = [torch.ones_like(t)]
        for k in range(1, harmonics + 1):
            basis.append(torch.cos(k * t))
            basis.append(torch.sin(k * t))
        return torch.stack(basis, dim=0)

    def _compute_components(
        self,
        theta: torch.Tensor,
        backcast_size: int,
        forecast_size: int,
        backcast_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.block_type == "generic":
            theta_b = theta[:, :backcast_size]
            theta_f = theta[:, backcast_size:]
            backcast = theta_b
            forecast = theta_f
        elif self.block_type == "trend":
            theta_dim, _ = self._basis_dim(backcast_size, forecast_size)
            theta_b = theta[:, :theta_dim]
            theta_f = theta[:, theta_dim:]
            backcast_basis = self._trend_basis(backcast_length, device, dtype)
            forecast_basis = self._trend_basis(self.pred_len, device, dtype)
            backcast = theta_b @ backcast_basis.repeat_interleave(
                self.input_dim, dim=1
            )
            forecast = theta_f @ forecast_basis.repeat_interleave(
                self.output_dim, dim=1
            )
        elif self.block_type == "seasonality":
            theta_dim, _ = self._basis_dim(backcast_size, forecast_size)
            theta_b = theta[:, :theta_dim]
            theta_f = theta[:, theta_dim:]
            backcast_basis = self._seasonality_basis(backcast_length, device, dtype)
            forecast_basis = self._seasonality_basis(self.pred_len, device, dtype)
            backcast = theta_b @ backcast_basis.repeat_interleave(
                self.input_dim, dim=1
            )
            forecast = theta_f @ forecast_basis.repeat_interleave(
                self.output_dim, dim=1
            )
        else:
            raise ValueError(f"Unsupported block type: {self.block_type}")

        return backcast, forecast

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute block backcast and forecast components."""

        B, backcast_length, _ = x.shape
        backcast_size = backcast_length * self.input_dim
        forecast_size = self.pred_len * self.output_dim

        flat = x.reshape(B, backcast_size)
        for layer in self.fc_stack:
            flat = F.relu(layer(flat))
            if self.dropout > 0:
                flat = F.dropout(flat, p=self.dropout, training=self.training)

        if self.theta_layer is None:
            self._create_theta_layer(backcast_size, forecast_size, flat.device)
        theta = self.theta_layer(flat)  # type: ignore[operator]

        backcast, forecast = self._compute_components(
            theta, backcast_size, forecast_size, backcast_length, flat.device, flat.dtype
        )

        backcast = backcast.view(B, backcast_length, self.input_dim)
        forecast = forecast.view(B, self.pred_len, self.output_dim)
        return backcast, forecast


@register("nbeats")
class NBeatsModel(ARBaseModel):
    """N-BEATS model with configurable stacks of interpretable blocks."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        stack_types: Optional[List[str]] = None,
        num_blocks_per_stack: int = 2,
        hidden_size: int = 256,
        num_layers: int = 4,
        pred_len: int = 1,
        degree: int = 2,
        harmonics: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(input_dim=input_dim, output_dim=output_dim)
        self.stack_types = stack_types or ["trend", "seasonality"]
        self.num_blocks_per_stack = num_blocks_per_stack
        self.pred_len = pred_len

        self.stacks = nn.ModuleList()
        for stack_type in self.stack_types:
            blocks = nn.ModuleList(
                [
                    NBeatsBlock(
                        input_dim=input_dim,
                        output_dim=output_dim,
                        pred_len=pred_len,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        block_type=stack_type,
                        degree=degree,
                        harmonics=harmonics,
                        dropout=dropout,
                    )
                    for _ in range(num_blocks_per_stack)
                ]
            )
            self.stacks.append(blocks)

    def forward(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None, **kwargs
    ) -> Dict[str, torch.Tensor]:
        residual = x
        B = x.size(0)
        forecast = torch.zeros(
            B, self.pred_len, self.output_dim, device=x.device, dtype=x.dtype
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

        extras: Dict[str, torch.Tensor] = {
            "stack_forecasts": torch.stack(stack_outputs, dim=1)
            if stack_outputs
            else torch.empty(0, device=x.device, dtype=x.dtype)
        }

        return {"preds": forecast, "extras": extras}
