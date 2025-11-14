"""Simple baseline models for time series prediction.

These models provide simple baselines to compare more sophisticated models against.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn

from .base import ARBaseModel
from .registry import register


@register("persistence")
class PersistenceModel(ARBaseModel):
    """Persistence (naive) baseline model.

    Predicts the last observed value as the next value.
    Also known as the "naive forecast" or "random walk model".

    This is one of the most common baselines for time series prediction.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        **kwargs
    ):
        """Initialize persistence model.

        Args:
            input_dim: Dimension of input features
            output_dim: Dimension of output predictions
            **kwargs: Additional arguments (ignored)
        """
        super().__init__(input_dim, output_dim, **kwargs)

        # No trainable parameters needed
        # But we need to handle input_dim != output_dim case
        if input_dim != output_dim:
            # Simple linear projection (could also just take first output_dim features)
            self.projection = nn.Linear(input_dim, output_dim, bias=False)
        else:
            self.projection = None

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass - return last value.

        Args:
            x: Input tensor [B, T_in, D_in]
            context: Optional context tensor (ignored)
            **kwargs: Additional arguments (ignored)

        Returns:
            Dictionary with 'preds' [B, 1, D_out]
        """
        # Get last timestep
        last_value = x[:, -1, :]  # [B, D_in]

        # Handle dimension mismatch
        if self.projection is not None:
            last_value = self.projection(last_value)  # [B, D_out]

        # Reshape to [B, 1, D_out]
        preds = last_value.unsqueeze(1)

        return {
            "preds": preds,
            "extras": {}
        }


@register("moving_average")
class MovingAverageModel(ARBaseModel):
    """Moving average baseline model.

    Predicts the mean of the last k values.
    Uses all available values in the input window if k is not specified.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        window_size: Optional[int] = None,
        **kwargs
    ):
        """Initialize moving average model.

        Args:
            input_dim: Dimension of input features
            output_dim: Dimension of output predictions
            window_size: Number of recent values to average (None = use all)
            **kwargs: Additional arguments (ignored)
        """
        super().__init__(input_dim, output_dim, **kwargs)

        self.window_size = window_size

        # Handle dimension mismatch
        if input_dim != output_dim:
            self.projection = nn.Linear(input_dim, output_dim, bias=False)
        else:
            self.projection = None

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass - return moving average.

        Args:
            x: Input tensor [B, T_in, D_in]
            context: Optional context tensor (ignored)
            **kwargs: Additional arguments (ignored)

        Returns:
            Dictionary with 'preds' [B, 1, D_out]
        """
        # Select window
        if self.window_size is not None:
            window = x[:, -self.window_size:, :]  # [B, k, D_in]
        else:
            window = x  # [B, T_in, D_in]

        # Compute mean
        avg_value = window.mean(dim=1)  # [B, D_in]

        # Handle dimension mismatch
        if self.projection is not None:
            avg_value = self.projection(avg_value)  # [B, D_out]

        # Reshape to [B, 1, D_out]
        preds = avg_value.unsqueeze(1)

        return {
            "preds": preds,
            "extras": {"window_size": window.shape[1]}
        }


@register("zero")
class ZeroModel(ARBaseModel):
    """Zero baseline model.

    Always predicts zero. Useful baseline for:
    - Anomaly detection (assume normal = zero deviation)
    - Differenced/normalized data where mean is zero
    - Measuring if a model learns anything at all
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        **kwargs
    ):
        """Initialize zero model.

        Args:
            input_dim: Dimension of input features
            output_dim: Dimension of output predictions
            **kwargs: Additional arguments (ignored)
        """
        super().__init__(input_dim, output_dim, **kwargs)
        # No parameters needed

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass - return zeros.

        Args:
            x: Input tensor [B, T_in, D_in]
            context: Optional context tensor (ignored)
            **kwargs: Additional arguments (ignored)

        Returns:
            Dictionary with 'preds' [B, 1, D_out] of zeros
        """
        B = x.shape[0]

        # Create zero tensor
        preds = torch.zeros(B, 1, self.output_dim, device=x.device, dtype=x.dtype)

        return {
            "preds": preds,
            "extras": {}
        }


@register("linear_trend")
class LinearTrendModel(ARBaseModel):
    """Linear trend baseline model.

    Fits a simple linear trend to the input sequence and extrapolates
    one step ahead. Uses least squares fitting.

    For each feature independently:
        y_pred = a + b * (T_in + 1)
    where a and b are fitted to the input window.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        **kwargs
    ):
        """Initialize linear trend model.

        Args:
            input_dim: Dimension of input features
            output_dim: Dimension of output predictions
            **kwargs: Additional arguments (ignored)
        """
        super().__init__(input_dim, output_dim, **kwargs)

        # Handle dimension mismatch
        if input_dim != output_dim:
            self.projection = nn.Linear(input_dim, output_dim, bias=False)
        else:
            self.projection = None

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass - fit linear trend and extrapolate.

        Args:
            x: Input tensor [B, T_in, D_in]
            context: Optional context tensor (ignored)
            **kwargs: Additional arguments (ignored)

        Returns:
            Dictionary with 'preds' [B, 1, D_out]
        """
        B, T_in, D_in = x.shape

        # Create time indices [0, 1, 2, ..., T_in-1]
        t = torch.arange(T_in, device=x.device, dtype=x.dtype)  # [T_in]

        # Fit linear trend using least squares for each batch and feature
        # y = a + b*t
        # Using closed-form solution:
        # b = (n*sum(t*y) - sum(t)*sum(y)) / (n*sum(t^2) - sum(t)^2)
        # a = mean(y) - b*mean(t)

        t_mean = t.mean()
        t_sum = t.sum()
        t_sq_sum = (t ** 2).sum()

        # Reshape for broadcasting: [T_in] -> [1, T_in, 1]
        t_bc = t.unsqueeze(0).unsqueeze(2)

        # Compute sums over time dimension
        y_mean = x.mean(dim=1)  # [B, D_in]
        ty_sum = (t_bc * x).sum(dim=1)  # [B, D_in]
        y_sum = x.sum(dim=1)  # [B, D_in]

        # Compute slope (b) and intercept (a)
        numerator = T_in * ty_sum - t_sum * y_sum
        denominator = T_in * t_sq_sum - t_sum ** 2

        # Avoid division by zero (happens with constant time series)
        b = torch.where(
            denominator.abs() > 1e-8,
            numerator / denominator,
            torch.zeros_like(numerator)
        )  # [B, D_in]

        a = y_mean - b * t_mean  # [B, D_in]

        # Predict next timestep (t = T_in)
        next_value = a + b * T_in  # [B, D_in]

        # Handle dimension mismatch
        if self.projection is not None:
            next_value = self.projection(next_value)  # [B, D_out]

        # Reshape to [B, 1, D_out]
        preds = next_value.unsqueeze(1)

        return {
            "preds": preds,
            "extras": {
                "slope": b,
                "intercept": a
            }
        }


@register("mean")
class MeanModel(ARBaseModel):
    """Historical mean baseline model.

    Always predicts the historical mean of the input sequence.
    Equivalent to a moving average with window_size = all available data.

    Useful baseline for stationary time series.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        **kwargs
    ):
        """Initialize mean model.

        Args:
            input_dim: Dimension of input features
            output_dim: Dimension of output predictions
            **kwargs: Additional arguments (ignored)
        """
        super().__init__(input_dim, output_dim, **kwargs)

        # Handle dimension mismatch
        if input_dim != output_dim:
            self.projection = nn.Linear(input_dim, output_dim, bias=False)
        else:
            self.projection = None

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass - return historical mean.

        Args:
            x: Input tensor [B, T_in, D_in]
            context: Optional context tensor (ignored)
            **kwargs: Additional arguments (ignored)

        Returns:
            Dictionary with 'preds' [B, 1, D_out]
        """
        # Compute mean across time dimension
        mean_value = x.mean(dim=1)  # [B, D_in]

        # Handle dimension mismatch
        if self.projection is not None:
            mean_value = self.projection(mean_value)  # [B, D_out]

        # Reshape to [B, 1, D_out]
        preds = mean_value.unsqueeze(1)

        return {
            "preds": preds,
            "extras": {}
        }
