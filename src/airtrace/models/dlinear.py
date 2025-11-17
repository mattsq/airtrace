"""DLinear and NLinear baseline models."""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ARBaseModel
from .registry import register


def _ensure_seq_len(x: torch.Tensor, expected_len: int) -> None:
    if x.size(1) != expected_len:
        raise ValueError(
            f"Expected input sequence length {expected_len}, got {x.size(1)}"
        )


@register("dlinear")
class DLinearModel(ARBaseModel):
    """Decomposition-Linear model for time series forecasting.

    This model applies a simple moving average decomposition to split the input
    into seasonal and trend components and fits independent linear projections
    over the temporal dimension for each component. Predictions from both
    components are summed to form the final forecast.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        seq_len: int = 60,
        pred_len: int = 1,
        kernel_size: int = 25,
        **kwargs,
    ) -> None:
        super().__init__(input_dim, output_dim, **kwargs)

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.kernel_size = min(kernel_size, seq_len)

        self.seasonal_linear = nn.Linear(seq_len, pred_len)
        self.trend_linear = nn.Linear(seq_len, pred_len)

        if input_dim != output_dim:
            self.output_projection = nn.Linear(input_dim, output_dim)
        else:
            self.output_projection = None

    def _moving_average(self, x: torch.Tensor) -> torch.Tensor:
        """Compute moving average along the temporal dimension."""

        if self.kernel_size == 1:
            return x

        pad_front = x[:, :1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        pad_back = x[:, -1:, :].repeat(1, self.kernel_size // 2, 1)
        x_padded = torch.cat([pad_front, x, pad_back], dim=1)

        moving_mean = F.avg_pool1d(
            x_padded.permute(0, 2, 1),
            kernel_size=self.kernel_size,
            stride=1,
        ).permute(0, 2, 1)

        return moving_mean

    def forward(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None, **kwargs
    ) -> Dict[str, torch.Tensor]:
        _ensure_seq_len(x, self.seq_len)

        moving_mean = self._moving_average(x)
        seasonal_init = x - moving_mean
        trend_init = moving_mean

        seasonal = self.seasonal_linear(seasonal_init.transpose(1, 2)).transpose(1, 2)
        trend = self.trend_linear(trend_init.transpose(1, 2)).transpose(1, 2)

        preds = seasonal + trend

        if self.output_projection is not None:
            preds = self.output_projection(preds)

        return {
            "preds": preds,
            "extras": {
                "seasonal_component": seasonal,
                "trend_component": trend,
            },
        }


@register("nlinear")
class NLinearModel(ARBaseModel):
    """Non-stationary Linear model.

    The NLinear variant subtracts the per-feature mean before applying a linear
    projection over time, then adds the mean back to stabilize forecasts on
    non-stationary series.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        seq_len: int = 60,
        pred_len: int = 1,
        center_data: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(input_dim, output_dim, **kwargs)

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.center_data = center_data

        self.linear = nn.Linear(seq_len, pred_len)
        if input_dim != output_dim:
            self.output_projection = nn.Linear(input_dim, output_dim)
        else:
            self.output_projection = None

    def forward(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None, **kwargs
    ) -> Dict[str, torch.Tensor]:
        _ensure_seq_len(x, self.seq_len)

        if self.center_data:
            mean = x.mean(dim=1, keepdim=True)
            x_centered = x - mean
        else:
            mean = None
            x_centered = x

        preds = self.linear(x_centered.transpose(1, 2)).transpose(1, 2)

        if mean is not None:
            preds = preds + mean

        if self.output_projection is not None:
            preds = self.output_projection(preds)

        return {"preds": preds, "extras": {"mean_offset": mean}}
