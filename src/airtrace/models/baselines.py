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


@register("median")
class MedianModel(ARBaseModel):
    """Historical median baseline model.

    Always predicts the historical median of the input sequence.
    More robust to outliers than the mean model.

    Useful baseline for time series with outliers or heavy-tailed distributions.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        **kwargs
    ):
        """Initialize median model.

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
        """Forward pass - return historical median.

        Args:
            x: Input tensor [B, T_in, D_in]
            context: Optional context tensor (ignored)
            **kwargs: Additional arguments (ignored)

        Returns:
            Dictionary with 'preds' [B, 1, D_out]
        """
        # Compute median across time dimension
        median_value = x.median(dim=1).values  # [B, D_in]

        # Handle dimension mismatch
        if self.projection is not None:
            median_value = self.projection(median_value)  # [B, D_out]

        # Reshape to [B, 1, D_out]
        preds = median_value.unsqueeze(1)

        return {
            "preds": preds,
            "extras": {}
        }


@register("drift")
class DriftModel(ARBaseModel):
    """Drift (random walk with drift) baseline model.

    Predicts the last value plus the average change over the input window.
    Also known as "naive with drift" or "random walk with drift".

    Equivalent to: y_pred = y_T + (y_T - y_1) / (T - 1)

    Useful for time series with a trend but no clear pattern.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        **kwargs
    ):
        """Initialize drift model.

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
        """Forward pass - return last value plus average drift.

        Args:
            x: Input tensor [B, T_in, D_in]
            context: Optional context tensor (ignored)
            **kwargs: Additional arguments (ignored)

        Returns:
            Dictionary with 'preds' [B, 1, D_out]
        """
        B, T_in, D_in = x.shape

        # Get first and last values
        first_value = x[:, 0, :]  # [B, D_in]
        last_value = x[:, -1, :]  # [B, D_in]

        # Compute average drift
        if T_in > 1:
            drift = (last_value - first_value) / (T_in - 1)  # [B, D_in]
        else:
            drift = torch.zeros_like(last_value)

        # Predict: last value + one step of drift
        next_value = last_value + drift  # [B, D_in]

        # Handle dimension mismatch
        if self.projection is not None:
            next_value = self.projection(next_value)  # [B, D_out]

        # Reshape to [B, 1, D_out]
        preds = next_value.unsqueeze(1)

        return {
            "preds": preds,
            "extras": {"drift": drift}
        }


@register("exponential_smoothing")
class ExponentialSmoothingModel(ARBaseModel):
    """Exponential smoothing baseline model.

    Uses exponentially weighted moving average (EWMA) for prediction.
    More recent values get higher weight with exponential decay.

    y_pred = alpha * y_T + (1-alpha) * EWMA_{T-1}

    Common baseline in time series forecasting.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        alpha: float = 0.3,
        **kwargs
    ):
        """Initialize exponential smoothing model.

        Args:
            input_dim: Dimension of input features
            output_dim: Dimension of output predictions
            alpha: Smoothing parameter (0 < alpha <= 1)
                  Higher alpha = more weight on recent values
                  Lower alpha = smoother, more weight on history
            **kwargs: Additional arguments (ignored)
        """
        super().__init__(input_dim, output_dim, **kwargs)

        if not 0 < alpha <= 1:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")

        self.alpha = alpha

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
        """Forward pass - compute EWMA and predict.

        Args:
            x: Input tensor [B, T_in, D_in]
            context: Optional context tensor (ignored)
            **kwargs: Additional arguments (ignored)

        Returns:
            Dictionary with 'preds' [B, 1, D_out]
        """
        B, T_in, D_in = x.shape

        # Compute EWMA iteratively
        # Start with first value
        ewma = x[:, 0, :]  # [B, D_in]

        # Update for each subsequent timestep
        for t in range(1, T_in):
            ewma = self.alpha * x[:, t, :] + (1 - self.alpha) * ewma

        # The EWMA at the last timestep is our prediction
        next_value = ewma  # [B, D_in]

        # Handle dimension mismatch
        if self.projection is not None:
            next_value = self.projection(next_value)  # [B, D_out]

        # Reshape to [B, 1, D_out]
        preds = next_value.unsqueeze(1)

        return {
            "preds": preds,
            "extras": {"alpha": self.alpha}
        }


@register("seasonal_naive")
class SeasonalNaiveModel(ARBaseModel):
    """Seasonal naive baseline model.

    Predicts the value from the same position in the previous seasonal cycle.
    For example, with season_length=24 (hourly data, daily seasonality),
    predicts tomorrow's 3pm value using today's 3pm value.

    If season_length > input length, falls back to persistence model.

    Classic baseline for seasonal time series.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        season_length: int = 24,
        **kwargs
    ):
        """Initialize seasonal naive model.

        Args:
            input_dim: Dimension of input features
            output_dim: Dimension of output predictions
            season_length: Length of seasonal cycle (e.g., 24 for daily pattern in hourly data)
            **kwargs: Additional arguments (ignored)
        """
        super().__init__(input_dim, output_dim, **kwargs)

        if season_length < 1:
            raise ValueError(f"season_length must be >= 1, got {season_length}")

        self.season_length = season_length

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
        """Forward pass - return value from previous season.

        Args:
            x: Input tensor [B, T_in, D_in]
            context: Optional context tensor (ignored)
            **kwargs: Additional arguments (ignored)

        Returns:
            Dictionary with 'preds' [B, 1, D_out]
        """
        B, T_in, D_in = x.shape

        # If we have enough history, use seasonal value
        if T_in >= self.season_length:
            # Get value from one season ago (relative to last timestep)
            seasonal_value = x[:, -self.season_length, :]  # [B, D_in]
        else:
            # Fall back to persistence if not enough history
            seasonal_value = x[:, -1, :]  # [B, D_in]

        # Handle dimension mismatch
        if self.projection is not None:
            seasonal_value = self.projection(seasonal_value)  # [B, D_out]

        # Reshape to [B, 1, D_out]
        preds = seasonal_value.unsqueeze(1)

        return {
            "preds": preds,
            "extras": {
                "season_length": self.season_length,
                "used_seasonal": T_in >= self.season_length
            }
        }
