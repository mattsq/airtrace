"""Simple baseline models for time series prediction.

These models provide simple baselines to compare more sophisticated models against.
"""

from typing import Dict, List, Optional, Tuple, Union

import warnings

import numpy as np
import torch
import torch.nn as nn
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .base import ARBaseModel, ResidualWrapperCompatible
from .registry import register


def _batched_eye(size: int, batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Create a batched identity matrix."""

    eye = torch.eye(size, device=device, dtype=dtype)
    return eye.unsqueeze(0).expand(batch, -1, -1)


def _match_output_dim(values: torch.Tensor, output_dim: int) -> torch.Tensor:
    """Align a prediction vector to ``output_dim`` via truncation or padding."""

    if values.shape[1] == output_dim:
        return values

    if values.shape[1] > output_dim:
        return values[:, :output_dim]

    padding = torch.zeros(
        values.shape[0],
        output_dim - values.shape[1],
        device=values.device,
        dtype=values.dtype,
    )
    return torch.cat([values, padding], dim=1)


def _repeat_predictions(values: torch.Tensor, pred_len: int) -> torch.Tensor:
    """Expand a single-step prediction across the requested horizon."""

    preds = values.unsqueeze(1)
    if pred_len != 1:
        preds = preds.expand(-1, pred_len, -1)
    return preds


@register("persistence")
class PersistenceModel(ResidualWrapperCompatible):
    """Persistence (naive) baseline model.

    Predicts the last observed value of target sensor(s) as the next value.
    Also known as the "naive forecast" or "random walk model".

    This is a truly non-trainable baseline with 0 learnable parameters.
    It uses metadata to identify which input dimensions correspond to target sensors.
    """

    def __init__(self, input_dim: int, output_dim: int, **kwargs):
        """Initialize persistence model.

        Args:
            input_dim: Dimension of input features
            output_dim: Dimension of output predictions
            **kwargs: Additional arguments (ignored)
        """
        super().__init__(input_dim, output_dim, **kwargs)
        # No trainable parameters at all

    def encode(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        meta: Optional[Union[List[Dict[str, int]], Dict[str, int]]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        del context
        last_value = x[:, -1, :]
        extras: Dict[str, torch.Tensor] = {}

        if meta:
            first_meta = meta[0] if isinstance(meta, list) else meta
            if "target_sensors" in first_meta and "input_sensor_indices" in first_meta:
                target_sensors = first_meta["target_sensors"]
                input_indices = first_meta["input_sensor_indices"]
                original_dim = first_meta.get("original_sensor_dim", self.input_dim)

                target_indices = []
                for target in target_sensors:
                    if target in input_indices:
                        idx = input_indices[target]
                        if idx < original_dim:
                            target_indices.append(idx)

                if target_indices:
                    index_tensor = torch.tensor(target_indices, device=x.device)
                    last_value = last_value.index_select(dim=1, index=index_tensor)
                    extras["target_indices"] = index_tensor

        aligned_value = _match_output_dim(last_value, self.output_dim)
        extras["latent"] = aligned_value
        return aligned_value, extras

    def decode(self, latent: torch.Tensor, pred_len: int) -> torch.Tensor:
        return _repeat_predictions(latent, pred_len)

    def forward(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None, **kwargs
    ) -> Dict[str, torch.Tensor]:
        pred_len = int(kwargs.get("pred_len", 1))
        latent, extras = self.encode(x, context=context, meta=kwargs.get("meta"))
        preds = self.decode(latent, pred_len)
        return {"preds": preds, "extras": extras}


@register("moving_average")
class MovingAverageModel(ResidualWrapperCompatible):
    """Moving average baseline model.

    Predicts the mean of the last k values.
    Uses all available values in the input window if k is not specified.
    This is a truly non-trainable baseline with 0 learnable parameters.
    """

    def __init__(
        self, input_dim: int, output_dim: int, window_size: Optional[int] = None, **kwargs
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
        # No trainable parameters at all

    def encode(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Union[torch.Tensor, bool]]]:
        del context
        if self.window_size is not None:
            window = x[:, -self.window_size :, :]
            window_size = self.window_size
        else:
            window = x
            window_size = x.shape[1]

        avg_value = window.mean(dim=1)
        aligned_avg = _match_output_dim(avg_value, self.output_dim)
        extras = {"window_size": torch.tensor(window_size, device=x.device)}
        return aligned_avg, extras

    def decode(self, latent: torch.Tensor, pred_len: int) -> torch.Tensor:
        return _repeat_predictions(latent, pred_len)

    def forward(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None, **kwargs
    ) -> Dict[str, torch.Tensor]:
        pred_len = int(kwargs.get("pred_len", 1))
        latent, extras = self.encode(x, context=context)
        preds = self.decode(latent, pred_len)
        return {"preds": preds, "extras": extras}


@register("zero")
class ZeroModel(ResidualWrapperCompatible):
    """Zero baseline model.

    Always predicts zero. Useful baseline for:
    - Anomaly detection (assume normal = zero deviation)
    - Differenced/normalized data where mean is zero
    - Measuring if a model learns anything at all
    """

    def __init__(self, input_dim: int, output_dim: int, **kwargs):
        """Initialize zero model.

        Args:
            input_dim: Dimension of input features
            output_dim: Dimension of output predictions
            **kwargs: Additional arguments (ignored)
        """
        super().__init__(input_dim, output_dim, **kwargs)
        # No parameters needed

    def encode(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        del context
        latent = torch.zeros(x.shape[0], self.output_dim, device=x.device, dtype=x.dtype)
        return latent, {"latent": latent}

    def decode(self, latent: torch.Tensor, pred_len: int) -> torch.Tensor:
        return _repeat_predictions(latent, pred_len)

    def forward(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None, **kwargs
    ) -> Dict[str, torch.Tensor]:
        pred_len = int(kwargs.get("pred_len", 1))
        latent, extras = self.encode(x, context=context)
        preds = self.decode(latent, pred_len)
        return {"preds": preds, "extras": extras}


@register("linear_trend")
class LinearTrendModel(ResidualWrapperCompatible):
    """Linear trend baseline model.

    Fits a simple linear trend to the input sequence and extrapolates
    one step ahead. Uses least squares fitting.

    For each feature independently:
        y_pred = a + b * (T_in + 1)
    where a and b are fitted to the input window.
    This is a truly non-trainable baseline with 0 learnable parameters.
    """

    def __init__(self, input_dim: int, output_dim: int, **kwargs):
        """Initialize linear trend model.

        Args:
            input_dim: Dimension of input features
            output_dim: Dimension of output predictions
            **kwargs: Additional arguments (ignored)
        """
        super().__init__(input_dim, output_dim, **kwargs)
        # No trainable parameters at all

    def encode(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        del context
        B, T_in, _ = x.shape

        t = torch.arange(T_in, device=x.device, dtype=x.dtype)
        t_mean = t.mean()
        t_sum = t.sum()
        t_sq_sum = (t**2).sum()
        t_b = t.view(1, T_in, 1)

        y_sum = x.sum(dim=1)
        ty_sum = (t_b * x).sum(dim=1)

        n = torch.tensor(float(T_in), device=x.device, dtype=x.dtype)
        denominator = torch.clamp(n * t_sq_sum - t_sum**2, min=1e-8)

        b = (n * ty_sum - t_sum * y_sum) / denominator
        a = (y_sum / n) - b * t_mean

        next_t = torch.tensor(float(T_in), device=x.device, dtype=x.dtype)
        next_value = a + b * next_t

        aligned = _match_output_dim(next_value, self.output_dim)
        extras = {"slope": b, "intercept": a}
        return aligned, extras

    def decode(self, latent: torch.Tensor, pred_len: int) -> torch.Tensor:
        return _repeat_predictions(latent, pred_len)

    def forward(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None, **kwargs
    ) -> Dict[str, torch.Tensor]:
        pred_len = int(kwargs.get("pred_len", 1))
        latent, extras = self.encode(x, context=context)
        preds = self.decode(latent, pred_len)
        return {"preds": preds, "extras": extras}


@register("mean")
class MeanModel(ResidualWrapperCompatible):
    """Historical mean baseline model.

    Always predicts the historical mean of the input sequence.
    Equivalent to a moving average with window_size = all available data.

    Useful baseline for stationary time series.
    This is a truly non-trainable baseline with 0 learnable parameters.
    """

    def __init__(self, input_dim: int, output_dim: int, **kwargs):
        """Initialize mean model.

        Args:
            input_dim: Dimension of input features
            output_dim: Dimension of output predictions
            **kwargs: Additional arguments (ignored)
        """
        super().__init__(input_dim, output_dim, **kwargs)
        # No trainable parameters at all

    def encode(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        del context
        mean_value = x.mean(dim=1)
        aligned_mean = _match_output_dim(mean_value, self.output_dim)
        return aligned_mean, {"latent": aligned_mean}

    def decode(self, latent: torch.Tensor, pred_len: int) -> torch.Tensor:
        return _repeat_predictions(latent, pred_len)

    def forward(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None, **kwargs
    ) -> Dict[str, torch.Tensor]:
        pred_len = int(kwargs.get("pred_len", 1))
        latent, extras = self.encode(x, context=context)
        preds = self.decode(latent, pred_len)
        return {"preds": preds, "extras": extras}


@register("median")
class MedianModel(ResidualWrapperCompatible):
    """Historical median baseline model.

    Always predicts the historical median of the input sequence.
    Useful baseline for robust forecasting when data has outliers.
    This is a truly non-trainable baseline with 0 learnable parameters.
    """

    def __init__(self, input_dim: int, output_dim: int, **kwargs):
        """Initialize median model.

        Args:
            input_dim: Dimension of input features
            output_dim: Dimension of output predictions
            **kwargs: Additional arguments (ignored)
        """
        super().__init__(input_dim, output_dim, **kwargs)
        # No trainable parameters at all

    def encode(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        del context
        median_value = x.median(dim=1).values
        aligned = _match_output_dim(median_value, self.output_dim)
        return aligned, {"latent": aligned}

    def decode(self, latent: torch.Tensor, pred_len: int) -> torch.Tensor:
        return _repeat_predictions(latent, pred_len)

    def forward(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None, **kwargs
    ) -> Dict[str, torch.Tensor]:
        pred_len = int(kwargs.get("pred_len", 1))
        latent, extras = self.encode(x, context=context)
        preds = self.decode(latent, pred_len)
        return {"preds": preds, "extras": extras}


@register("drift")
class DriftModel(ResidualWrapperCompatible):
    """Drift (random walk with drift) baseline model.

    Predicts the last value plus the average change over the input window.
    Also known as "naive with drift" or "random walk with drift".

    Equivalent to: y_pred = y_T + (y_T - y_1) / (T - 1)

    Useful for time series with a trend but no clear pattern.
    This is a truly non-trainable baseline with 0 learnable parameters.
    """

    def __init__(self, input_dim: int, output_dim: int, **kwargs):
        """Initialize drift model.

        Args:
            input_dim: Dimension of input features
            output_dim: Dimension of output predictions
            **kwargs: Additional arguments (ignored)
        """
        super().__init__(input_dim, output_dim, **kwargs)
        # No trainable parameters at all

    def encode(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        del context
        B, T_in, _ = x.shape
        first_value = x[:, 0, :]
        last_value = x[:, -1, :]

        if T_in > 1:
            drift = (last_value - first_value) / (T_in - 1)
        else:
            drift = torch.zeros_like(last_value)

        next_value = last_value + drift
        aligned = _match_output_dim(next_value, self.output_dim)
        extras = {"drift": drift}
        return aligned, extras

    def decode(self, latent: torch.Tensor, pred_len: int) -> torch.Tensor:
        return _repeat_predictions(latent, pred_len)

    def forward(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None, **kwargs
    ) -> Dict[str, torch.Tensor]:
        pred_len = int(kwargs.get("pred_len", 1))
        latent, extras = self.encode(x, context=context)
        preds = self.decode(latent, pred_len)
        return {"preds": preds, "extras": extras}


@register("exponential_smoothing")
class ExponentialSmoothingModel(ResidualWrapperCompatible):
    """Exponential smoothing baseline model.

    Uses exponentially weighted moving average (EWMA) for prediction.
    More recent values get higher weight with exponential decay.

    y_pred = alpha * y_T + (1-alpha) * EWMA_{T-1}

    Common baseline in time series forecasting.
    This is a truly non-trainable baseline with 0 learnable parameters.
    """

    def __init__(self, input_dim: int, output_dim: int, alpha: float = 0.3, **kwargs):
        """Initialize exponential smoothing model.

        Args:
            input_dim: Dimension of input features
            output_dim: Dimension of output predictions
            alpha: Smoothing factor (0 < alpha <= 1)
            **kwargs: Additional arguments (ignored)
        """
        if alpha <= 0 or alpha > 1:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")

        super().__init__(input_dim, output_dim, **kwargs)
        self.alpha = alpha
        # No trainable parameters at all

    def encode(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        del context
        B, T_in, _ = x.shape

        ewma = x[:, 0, :].clone()
        for t in range(1, T_in):
            ewma = self.alpha * x[:, t, :] + (1 - self.alpha) * ewma

        next_value = ewma
        aligned = _match_output_dim(next_value, self.output_dim)
        extras = {"alpha": torch.tensor(self.alpha, device=x.device)}
        return aligned, extras

    def decode(self, latent: torch.Tensor, pred_len: int) -> torch.Tensor:
        return _repeat_predictions(latent, pred_len)

    def forward(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None, **kwargs
    ) -> Dict[str, torch.Tensor]:
        pred_len = int(kwargs.get("pred_len", 1))
        latent, extras = self.encode(x, context=context)
        preds = self.decode(latent, pred_len)
        return {"preds": preds, "extras": extras}


@register("seasonal_naive")
class SeasonalNaiveModel(ResidualWrapperCompatible):
    """Seasonal naive baseline model.

    Predicts the value from the same position in the previous seasonal cycle.
    For example, with season_length=24 (hourly data, daily seasonality),
    predicts tomorrow's 3pm value using today's 3pm value.

    If season_length > input length, falls back to persistence model.

    Classic baseline for seasonal time series.
    This is a truly non-trainable baseline with 0 learnable parameters.
    """

    def __init__(self, input_dim: int, output_dim: int, season_length: int = 24, **kwargs):
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
        # No trainable parameters at all

    def encode(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Union[torch.Tensor, bool]]]:
        del context
        B, T_in, _ = x.shape

        if T_in >= self.season_length:
            seasonal_value = x[:, -self.season_length, :]
            used_seasonal = True
        else:
            seasonal_value = x[:, -1, :]
            used_seasonal = False

        aligned = _match_output_dim(seasonal_value, self.output_dim)
        extras = {
            "season_length": torch.tensor(self.season_length, device=x.device),
            "used_seasonal": used_seasonal,
        }
        return aligned, extras

    def decode(self, latent: torch.Tensor, pred_len: int) -> torch.Tensor:
        return _repeat_predictions(latent, pred_len)

    def forward(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None, **kwargs
    ) -> Dict[str, torch.Tensor]:
        pred_len = int(kwargs.get("pred_len", 1))
        latent, extras = self.encode(x, context=context)
        preds = self.decode(latent, pred_len)
        return {"preds": preds, "extras": extras}


@register("polynomial_trend")
class PolynomialTrendModel(ResidualWrapperCompatible):
    """Polynomial trend baseline model.

    Fits a polynomial trend (quadratic, cubic, etc.) to the input sequence
    and extrapolates one step ahead using least squares fitting.

    For each feature independently:
        y_pred = a_0 + a_1*t + a_2*t^2 + ... + a_n*t^n
    where coefficients are fitted to the input window.

    Useful for time series with curved trends.
    This is a truly non-trainable baseline with 0 learnable parameters.
    """

    def __init__(self, input_dim: int, output_dim: int, degree: int = 2, **kwargs):
        """Initialize polynomial trend model.

        Args:
            input_dim: Dimension of input features
            output_dim: Dimension of output predictions
            degree: Degree of polynomial (2=quadratic, 3=cubic, etc.)
            **kwargs: Additional arguments (ignored)
        """
        super().__init__(input_dim, output_dim, **kwargs)

        if degree < 1:
            raise ValueError(f"degree must be >= 1, got {degree}")

        self.degree = degree
        # No trainable parameters at all

    def encode(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        del context
        B, T_in, D_in = x.shape

        # Create time indices [0, 1, 2, ..., T_in-1]
        t = torch.arange(T_in, device=x.device, dtype=x.dtype)  # [T_in]

        # Create design matrix for polynomial regression
        # X = [1, t, t^2, ..., t^degree]
        # Shape: [T_in, degree+1]
        X = torch.stack([t**i for i in range(self.degree + 1)], dim=1)  # [T_in, degree+1]

        # Solve least squares for each batch and feature
        # y = X @ coeffs
        # coeffs = (X^T X)^{-1} X^T y
        # Using PyTorch's lstsq for numerical stability

        # Reshape x for batch processing: [B, T_in, D_in] -> [B*D_in, T_in]
        y = x.transpose(1, 2).reshape(B * D_in, T_in)  # [B*D_in, T_in]

        # Solve least squares
        # coeffs shape: [B*D_in, degree+1]
        try:
            solution = torch.linalg.lstsq(X, y.T)  # X: [T_in, degree+1], y.T: [T_in, B*D_in]
            coeffs = solution.solution.T  # [B*D_in, degree+1]
        except RuntimeError:
            # If lstsq fails (singular matrix), fall back to zeros
            coeffs = torch.zeros(B * D_in, self.degree + 1, device=x.device, dtype=x.dtype)

        # Predict at next timestep (t = T_in)
        t_next = torch.tensor(T_in, device=x.device, dtype=x.dtype)
        x_next = torch.stack([t_next**i for i in range(self.degree + 1)], dim=0)  # [degree+1]

        # Compute predictions: coeffs @ x_next
        next_value = (coeffs * x_next.unsqueeze(0)).sum(dim=1)  # [B*D_in]

        # Reshape back: [B*D_in] -> [B, D_in]
        next_value = next_value.reshape(B, D_in)
        aligned = _match_output_dim(next_value, self.output_dim)

        extras: Dict[str, torch.Tensor] = {
            "degree": torch.tensor(self.degree, device=x.device),
            "coeffs": coeffs.reshape(B, D_in, -1),
        }
        extras["latent"] = aligned
        return aligned, extras

    def decode(self, latent: torch.Tensor, pred_len: int) -> torch.Tensor:
        return _repeat_predictions(latent, pred_len)

    def forward(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None, **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass - fit polynomial trend and extrapolate."""

        pred_len = int(kwargs.get("pred_len", 1))
        latent, extras = self.encode(x, context=context)
        preds = self.decode(latent, pred_len)

        return {"preds": preds, "extras": extras}


@register("holt_linear_trend")
class HoltLinearTrendModel(ResidualWrapperCompatible):
    """Holt's linear trend model (double exponential smoothing).

    Uses two smoothing equations:
    - Level equation: l_t = alpha * y_t + (1 - alpha) * (l_{t-1} + b_{t-1})
    - Trend equation: b_t = beta * (l_t - l_{t-1}) + (1 - beta) * b_{t-1}

    Prediction: y_{t+1} = l_t + b_t

    This is a classic baseline for trending time series that is more
    sophisticated than simple exponential smoothing.
    This is a truly non-trainable baseline with 0 learnable parameters.
    """

    def __init__(
        self, input_dim: int, output_dim: int, alpha: float = 0.3, beta: float = 0.1, **kwargs
    ):
        """Initialize Holt's linear trend model.

        Args:
            input_dim: Dimension of input features
            output_dim: Dimension of output predictions
            alpha: Level smoothing parameter (0 < alpha <= 1)
            beta: Trend smoothing parameter (0 < beta <= 1)
            **kwargs: Additional arguments (ignored)
        """
        super().__init__(input_dim, output_dim, **kwargs)

        if not 0 < alpha <= 1:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        if not 0 < beta <= 1:
            raise ValueError(f"beta must be in (0, 1], got {beta}")

        self.alpha = alpha
        self.beta = beta
        # No trainable parameters at all

    def encode(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        del context
        B, T_in, D_in = x.shape

        # Initialize level and trend
        # l_0 = y_0, b_0 = y_1 - y_0 (or 0 if T_in == 1)
        level = x[:, 0, :]  # [B, D_in]

        if T_in > 1:
            trend = x[:, 1, :] - x[:, 0, :]  # [B, D_in]
        else:
            trend = torch.zeros_like(level)

        # Apply Holt's equations iteratively
        for t in range(1, T_in):
            y_t = x[:, t, :]  # [B, D_in]

            # Update level
            level_prev = level
            level = self.alpha * y_t + (1 - self.alpha) * (level_prev + trend)

            # Update trend
            trend = self.beta * (level - level_prev) + (1 - self.beta) * trend

        # Forecast: l_T + b_T
        next_value = level + trend  # [B, D_in]
        aligned = _match_output_dim(next_value, self.output_dim)

        extras: Dict[str, torch.Tensor] = {
            "alpha": torch.tensor(self.alpha, device=x.device),
            "beta": torch.tensor(self.beta, device=x.device),
            "level": level,
            "trend": trend,
        }
        extras["latent"] = aligned
        return aligned, extras

    def decode(self, latent: torch.Tensor, pred_len: int) -> torch.Tensor:
        return _repeat_predictions(latent, pred_len)

    def forward(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None, **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass - apply Holt's linear trend method."""

        pred_len = int(kwargs.get("pred_len", 1))
        latent, extras = self.encode(x, context=context)
        preds = self.decode(latent, pred_len)

        return {"preds": preds, "extras": extras}


@register("holt_winters")
class HoltWintersModel(ResidualWrapperCompatible):
    """Holt-Winters (triple exponential smoothing) baseline model.

    Extends Holt's method with a seasonal component controlled by ``gamma`` and ``season_length``.
    Supports additive and multiplicative seasonality following the classical formulation:

    - Level: ``l_t = alpha * (y_t / s_{t-m}) + (1 - alpha) * (l_{t-1} + b_{t-1})`` (multiplicative)
      or ``alpha * (y_t - s_{t-m}) + (1 - alpha) * (l_{t-1} + b_{t-1})`` (additive)
    - Trend: ``b_t = beta * (l_t - l_{t-1}) + (1 - beta) * b_{t-1}``
    - Seasonality updates differ for additive vs multiplicative

    Forecast is ``(l_T + b_T) * s_{T+1-m}`` or ``l_T + b_T + s_{T+1-m}`` respectively.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        season_length: int = 24,
        alpha: float = 0.3,
        beta: float = 0.1,
        gamma: float = 0.1,
        seasonal: str = "additive",
        **kwargs,
    ) -> None:
        """Initialize Holt-Winters model.

        Args:
            input_dim: Dimension of input features.
            output_dim: Dimension of output predictions.
            season_length: Number of steps in one seasonal cycle (> 1).
            alpha: Level smoothing parameter (0 < alpha <= 1).
            beta: Trend smoothing parameter (0 < beta <= 1).
            gamma: Seasonal smoothing parameter (0 < gamma <= 1).
            seasonal: Type of seasonality ("additive" or "multiplicative").
            **kwargs: Additional arguments (ignored).
        """

        super().__init__(input_dim, output_dim, **kwargs)

        if season_length < 2:
            raise ValueError(f"season_length must be >= 2, got {season_length}")
        if not 0 < alpha <= 1:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        if not 0 < beta <= 1:
            raise ValueError(f"beta must be in (0, 1], got {beta}")
        if not 0 < gamma <= 1:
            raise ValueError(f"gamma must be in (0, 1], got {gamma}")

        seasonal_lower = seasonal.lower()
        if seasonal_lower not in {"additive", "multiplicative"}:
            raise ValueError("seasonal must be 'additive' or 'multiplicative'")

        self.season_length = season_length
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.seasonal = seasonal_lower
        # No trainable parameters at all

    def _initialize_states(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Initialize level, trend, and seasonality states."""

        B, T_in, D_in = x.shape
        seasonals = torch.zeros(
            B,
            self.season_length,
            D_in,
            device=x.device,
            dtype=x.dtype,
        )

        effective_length = min(T_in, self.season_length)
        if self.seasonal == "additive":
            baseline = x[:, :effective_length, :].mean(dim=1, keepdim=True)
            seasonals[:, :effective_length, :] = x[:, :effective_length, :] - baseline
        else:
            baseline = x[:, :effective_length, :].mean(dim=1, keepdim=True).clamp(min=1e-6)
            seasonals[:, :effective_length, :] = (
                x[:, :effective_length, :].clamp(min=1e-6) / baseline
            )
            seasonals[:, effective_length:, :] = 1.0

        if effective_length < self.season_length:
            if self.seasonal == "additive":
                seasonals[:, effective_length:, :] = 0.0
            else:
                seasonals[:, effective_length:, :] = 1.0

        if self.seasonal == "additive":
            level = x[:, 0, :] - seasonals[:, 0, :]
        else:
            level = x[:, 0, :].clamp(min=1e-6) / seasonals[:, 0, :].clamp(min=1e-6)

        if T_in > 1:
            if self.seasonal == "additive":
                next_level = x[:, 1, :] - seasonals[:, 1 % self.season_length, :]
                trend = next_level - level
            else:
                next_level = (
                    x[:, 1, :].clamp(min=1e-6)
                    / seasonals[:, 1 % self.season_length, :].clamp(min=1e-6)
                )
                trend = next_level - level
        else:
            trend = torch.zeros_like(level)

        return level, trend, seasonals

    def encode(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        del context

        B, T_in, D_in = x.shape
        level, trend, seasonals = self._initialize_states(x)

        for t in range(1, T_in):
            season_idx = t % self.season_length
            season_component = seasonals[:, season_idx, :]
            y_t = x[:, t, :]

            if self.seasonal == "additive":
                deseasonalized = y_t - season_component
                new_level = self.alpha * deseasonalized + (1 - self.alpha) * (level + trend)
                new_trend = self.beta * (new_level - level) + (1 - self.beta) * trend
                seasonals[:, season_idx, :] = (
                    self.gamma * (y_t - new_level) + (1 - self.gamma) * season_component
                )
            else:
                deseasonalized = y_t / season_component.clamp(min=1e-6)
                new_level = self.alpha * deseasonalized + (1 - self.alpha) * (level + trend)
                new_trend = self.beta * (new_level - level) + (1 - self.beta) * trend
                seasonals[:, season_idx, :] = (
                    self.gamma * (y_t / new_level.clamp(min=1e-6))
                    + (1 - self.gamma) * season_component
                )

            level = new_level
            trend = new_trend

        next_idx = T_in % self.season_length
        next_season = seasonals[:, next_idx, :]

        if self.seasonal == "additive":
            next_value = level + trend + next_season
        else:
            next_value = (level + trend) * next_season

        aligned = _match_output_dim(next_value, self.output_dim)

        extras = {
            "alpha": torch.tensor(self.alpha, device=x.device),
            "beta": torch.tensor(self.beta, device=x.device),
            "gamma": torch.tensor(self.gamma, device=x.device),
            "season_length": torch.tensor(self.season_length, device=x.device),
            # Preserve human-readable seasonal type for downstream consumers/tests.
            "seasonal_type": self.seasonal,
            "level": level,
            "trend": trend,
            "seasonals": seasonals,
        }
        extras["latent"] = aligned
        return aligned, extras

    def decode(self, latent: torch.Tensor, pred_len: int) -> torch.Tensor:
        return _repeat_predictions(latent, pred_len)

    def forward(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None, **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass applying Holt-Winters smoothing."""

        pred_len = int(kwargs.get("pred_len", 1))
        latent, extras = self.encode(x, context=context)
        preds = self.decode(latent, pred_len)

        return {"preds": preds, "extras": extras}


@register("theta")
class ThetaModel(ResidualWrapperCompatible):
    """Theta method baseline model.

    The Theta method decomposes the time series into two "theta lines":
    - Theta-0 line: Long-term linear trend
    - Theta-2 line: Short-term curvature (double the local curve)

    The final forecast combines these with weights (default: 0.5 each).

    This simple method won the M3 forecasting competition and is a
    strong baseline for many time series.

    Reference: Assimakopoulos & Nikolopoulos (2000)
    """

    def __init__(
        self, input_dim: int, output_dim: int, theta: float = 2.0, alpha: float = 0.5, **kwargs
    ):
        """Initialize Theta model.

        Args:
            input_dim: Dimension of input features
            output_dim: Dimension of output predictions
            theta: Theta coefficient for curvature line (typically 2.0)
            alpha: Weight for combining lines (0.5 = equal weight)
            **kwargs: Additional arguments (ignored)
        """
        super().__init__(input_dim, output_dim, **kwargs)

        self.theta = theta
        self.alpha = alpha
        # No trainable parameters at all

    def encode(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        del context
        B, T_in, D_in = x.shape

        # Create time indices
        t = torch.arange(T_in, device=x.device, dtype=x.dtype)  # [T_in]

        # Theta-0 line: Linear regression (long-term trend)
        t_mean = t.mean()
        t_sum = t.sum()
        t_sq_sum = (t**2).sum()

        t_bc = t.unsqueeze(0).unsqueeze(2)  # [1, T_in, 1]

        y_mean = x.mean(dim=1)  # [B, D_in]
        ty_sum = (t_bc * x).sum(dim=1)  # [B, D_in]
        y_sum = x.sum(dim=1)  # [B, D_in]

        numerator = T_in * ty_sum - t_sum * y_sum
        denominator = T_in * t_sq_sum - t_sum**2

        b = torch.where(
            denominator.abs() > 1e-8, numerator / denominator, torch.zeros_like(numerator)
        )  # [B, D_in]

        a = y_mean - b * t_mean  # [B, D_in]

        # Theta-0 forecast
        theta0_forecast = a + b * T_in  # [B, D_in]

        # Theta-2 line: Apply second differences and extrapolate
        if T_in >= 2:
            detrended = x - (a.unsqueeze(1) + b.unsqueeze(1) * t_bc)  # [B, T_in, D_in]
            ses_alpha = 0.3  # Can be tuned
            smoothed = detrended[:, 0, :]  # [B, D_in]

            for i in range(1, T_in):
                smoothed = ses_alpha * detrended[:, i, :] + (1 - ses_alpha) * smoothed

            theta2_forecast = theta0_forecast + smoothed  # [B, D_in]
        else:
            theta2_forecast = theta0_forecast

        next_value = self.alpha * theta0_forecast + (1 - self.alpha) * theta2_forecast  # [B, D_in]
        aligned = _match_output_dim(next_value, self.output_dim)

        extras = {
            "theta": torch.tensor(self.theta, device=x.device),
            "alpha": torch.tensor(self.alpha, device=x.device),
            "theta0_forecast": theta0_forecast,
            "theta2_forecast": theta2_forecast,
        }
        extras["latent"] = aligned
        return aligned, extras

    def decode(self, latent: torch.Tensor, pred_len: int) -> torch.Tensor:
        return _repeat_predictions(latent, pred_len)

    def forward(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None, **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass - apply Theta method."""

        pred_len = int(kwargs.get("pred_len", 1))
        latent, extras = self.encode(x, context=context)
        preds = self.decode(latent, pred_len)

        return {"preds": preds, "extras": extras}


@register("sarima")
class SARIMAModel(ResidualWrapperCompatible):
    """Seasonal ARIMA baseline model using statsmodels."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        order: Tuple[int, int, int] = (1, 1, 0),
        seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0),
        trend: Optional[str] = None,
        enforce_stationarity: bool = True,
        enforce_invertibility: bool = True,
        maxiter: int = 50,
        measurement_error: bool = False,
        **kwargs,
    ) -> None:
        """Initialise SARIMA model."""

        if len(order) != 3:
            raise ValueError("order must be a tuple of length 3")
        if len(seasonal_order) != 4:
            raise ValueError("seasonal_order must be a tuple of length 4")

        super().__init__(input_dim, output_dim, **kwargs)

        self.order = tuple(int(v) for v in order)
        self.seasonal_order = tuple(int(v) for v in seasonal_order)
        self.trend = trend
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility
        self.maxiter = maxiter
        self.measurement_error = measurement_error

        if input_dim != output_dim:
            self.projection = nn.Linear(input_dim, output_dim, bias=False)
        else:
            self.projection = None

    def _fit_and_forecast(self, series: np.ndarray) -> float:
        """Fit SARIMAX to a 1D series and return one-step forecast."""

        numeric_series = np.asarray(series, dtype=float)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            model = SARIMAX(
                numeric_series,
                order=self.order,
                seasonal_order=self.seasonal_order,
                trend=self.trend,
                enforce_stationarity=self.enforce_stationarity,
                enforce_invertibility=self.enforce_invertibility,
                measurement_error=self.measurement_error,
            )
            result = model.fit(disp=False, maxiter=self.maxiter)
        forecast = result.forecast(steps=1)
        return float(forecast[0])

    def encode(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Union[torch.Tensor, str]]]:
        del context
        B, _, D_in = x.shape
        preds = torch.zeros(B, D_in, dtype=torch.double)
        success_mask = torch.zeros(B, D_in, dtype=torch.bool)
        failure_mask = torch.zeros(B, D_in, dtype=torch.bool)
        last_error: Optional[str] = None

        series_np = x.detach().cpu().numpy()

        for b in range(B):
            for d in range(D_in):
                target_series = series_np[b, :, d]
                fallback = float(target_series[-1])

                if np.isnan(target_series).all():
                    preds[b, d] = fallback
                    failure_mask[b, d] = True
                    last_error = "all_nan_series"
                    continue

                try:
                    preds[b, d] = self._fit_and_forecast(target_series)
                    success_mask[b, d] = True
                except Exception as exc:  # noqa: BLE001 - fallback to persistence
                    preds[b, d] = fallback
                    failure_mask[b, d] = True
                    last_error = str(exc)

        preds_tensor = preds.to(device=x.device, dtype=x.dtype)

        if self.projection is not None:
            preds_tensor = self.projection(preds_tensor)

        extras: Dict[str, Union[torch.Tensor, str]] = {
            "success_mask": success_mask.to(device=x.device),
            "failure_mask": failure_mask.to(device=x.device),
        }

        extras["num_failures"] = torch.tensor(
            failure_mask.sum().item(),
            device=x.device,
            dtype=torch.int64,
        )

        if last_error is not None:
            extras["last_error"] = last_error

        extras["latent"] = preds_tensor
        return preds_tensor, extras

    def decode(self, latent: torch.Tensor, pred_len: int) -> torch.Tensor:
        return _repeat_predictions(latent, pred_len)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass - fit SARIMA per feature and forecast next timestep."""

        pred_len = int(kwargs.get("pred_len", 1))
        latent, extras = self.encode(x, context=context)
        preds = self.decode(latent, pred_len)

        return {"preds": preds, "extras": extras}


@register("var")
class VARModel(ResidualWrapperCompatible):
    """Vector autoregression (VAR) baseline model.

    Fits a multivariate autoregression on-the-fly using the incoming window and
    produces a one-step forecast. Unlike per-feature baselines, this captures
    cross-channel dependencies by solving a small regularised least squares
    problem for each sample in the batch.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        order: int = 1,
        fit_intercept: bool = True,
        regularization: float = 1e-4,
        **kwargs,
    ) -> None:
        """Initialise VAR model.

        Args:
            input_dim: Dimension of input features.
            output_dim: Dimension of output predictions.
            order: Number of autoregressive lags (p in VAR(p)). Must be >= 1.
            fit_intercept: Whether to include an intercept term.
            regularization: Ridge penalty applied to coefficient matrices. Set to
                zero for an ordinary least squares fit.
            **kwargs: Additional arguments (ignored).
        """

        if order < 1:
            raise ValueError("VAR order must be at least 1")

        super().__init__(input_dim, output_dim, **kwargs)

        self.order = order
        self.fit_intercept = fit_intercept
        self.regularization = regularization
        # No trainable parameters at all

    def _prepare_design(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Construct design and target matrices for batched least squares."""

        B, T_in, D_in = x.shape
        if T_in <= self.order:
            raise ValueError("Input sequence must be longer than VAR order")

        targets = x[:, self.order :, :]  # [B, T - order, D]

        lagged = []
        for lag in range(1, self.order + 1):
            lagged.append(x[:, self.order - lag : -lag, :])  # [B, T - order, D]
        design = torch.cat(lagged, dim=2)  # [B, T - order, D * order]

        if self.fit_intercept:
            ones = torch.ones(
                B,
                design.shape[1],
                1,
                device=x.device,
                dtype=x.dtype,
            )
            design = torch.cat([design, ones], dim=2)

        return design, targets

    def _solve_coefficients(
        self, design: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Solve regularised least squares for batched VAR coefficients."""

        B, _, num_features = design.shape
        device = design.device
        dtype = design.dtype

        design_t = design.transpose(1, 2)  # [B, F, R]
        xtx = design_t @ design  # [B, F, F]

        if self.regularization > 0:
            reg_eye = _batched_eye(num_features, B, device=device, dtype=dtype)
            xtx = xtx + self.regularization * reg_eye
            if self.fit_intercept:
                xtx[:, -1, -1] -= self.regularization  # do not regularise intercept

        xty = design_t @ targets  # [B, F, D]

        try:
            coefs = torch.linalg.solve(xtx, xty)
        except RuntimeError:
            pinv = torch.linalg.pinv(xtx)
            coefs = pinv @ xty

        if self.fit_intercept:
            intercept = coefs[:, -1, :]
            weights = coefs[:, :-1, :]
        else:
            intercept = torch.zeros(B, targets.shape[2], device=device, dtype=dtype)
            weights = coefs

        return weights, intercept

    def encode(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        del context

        B, T_in, D_in = x.shape
        extras: Dict[str, torch.Tensor] = {}

        if T_in <= self.order:
            last_value = x[:, -1, :]
            aligned = _match_output_dim(last_value, self.output_dim)
            extras["fitted"] = torch.tensor(False, device=x.device)
            extras["latent"] = aligned
            return aligned, extras

        design, targets = self._prepare_design(x)
        weights, intercept = self._solve_coefficients(design, targets)

        lag_vectors = []
        for lag in range(1, self.order + 1):
            lag_vectors.append(x[:, -lag, :].unsqueeze(1))
        lag_tensor = torch.cat(lag_vectors, dim=1)  # [B, order, D]

        lag_matrices = weights.reshape(B, self.order, D_in, D_in)
        forecast_core = torch.einsum("bod,bodk->bk", lag_tensor, lag_matrices)
        next_value = forecast_core + intercept

        aligned = _match_output_dim(next_value, self.output_dim)

        extras.update(
            {
                "fitted": torch.tensor(True, device=x.device),
                "lag_matrices": lag_matrices,
                "intercept": intercept,
                "order": torch.tensor(self.order, device=x.device),
            }
        )
        extras["latent"] = aligned

        return aligned, extras

    def decode(self, latent: torch.Tensor, pred_len: int) -> torch.Tensor:
        return _repeat_predictions(latent, pred_len)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass - fit VAR coefficients and forecast next timestep."""

        pred_len = int(kwargs.get("pred_len", 1))
        latent, extras = self.encode(x, context=context)
        preds = self.decode(latent, pred_len)

        return {"preds": preds, "extras": extras}


@register("linear_ar")
class LinearARModel(ResidualWrapperCompatible):
    """Simple linear autoregressive (AR) baseline model.

    Learns linear weights to combine past values:
        y_{t+1} = w_1 * y_t + w_2 * y_{t-1} + ... + w_p * y_{t-p+1} + b

    This is a trainable baseline (unlike the other non-trainable baselines).
    It provides a simple learned baseline to compare neural models against.

    The model uses all available timesteps in the input window as predictors.
    """

    def __init__(self, input_dim: int, output_dim: int, **kwargs):
        """Initialize linear AR model.

        Args:
            input_dim: Dimension of input features
            output_dim: Dimension of output predictions
            **kwargs: Additional arguments (ignored)
        """
        super().__init__(input_dim, output_dim, **kwargs)

        # Use LazyLinear to defer input size determination until first forward pass
        # This ensures parameters are registered immediately for optimizer creation
        self.linear = nn.LazyLinear(output_dim)

    def encode(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        del context
        latent = x.reshape(x.shape[0], -1)
        extras: Dict[str, torch.Tensor] = {
            "input_shape": torch.tensor(x.shape[1:], device=x.device)
        }
        return latent, extras

    def decode(self, latent: torch.Tensor, pred_len: int) -> torch.Tensor:
        preds = self.linear(latent).unsqueeze(1)
        if pred_len != 1:
            preds = preds.expand(-1, pred_len, -1)
        return preds

    def forward(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None, **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass - apply linear AR model.

        Args:
            x: Input tensor [B, T_in, D_in]
            context: Optional context tensor (ignored)
            **kwargs: Additional arguments (ignored)

        Returns:
            Dictionary with 'preds' [B, 1, D_out]
        """
        pred_len = int(kwargs.get("pred_len", 1))
        latent, extras = self.encode(x, context=context)
        preds = self.decode(latent, pred_len)
        extras["latent"] = latent

        return {"preds": preds, "extras": extras}


@register("mlp_ar")
class MLPARModel(ResidualWrapperCompatible):
    """Windowed MLP autoregressive baseline model.

    Flattens the input window [T_in, D_in] into a vector and passes it through
    a multi-layer perceptron (MLP) to predict the next timestep.

    This is a non-sequential baseline that treats the window as a static feature
    vector. It's useful for checking whether sequence modeling (RNNs, Transformers)
    actually provides value over a simple feedforward network.

    Unlike linear_ar, this model has hidden layers with nonlinearities, giving it
    more capacity while still ignoring temporal structure.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        activation: str = "relu",
        **kwargs,
    ):
        """Initialize MLP AR model.

        Args:
            input_dim: Dimension of input features
            output_dim: Dimension of output predictions
            hidden_dims: List of hidden layer dimensions (default: [256, 128])
            dropout: Dropout probability
            activation: Activation function ('relu', 'gelu', 'tanh')
            **kwargs: Additional arguments (ignored)
        """
        super().__init__(input_dim, output_dim, **kwargs)

        if hidden_dims is None:
            hidden_dims = [256, 128]

        self.hidden_dims = hidden_dims
        self.dropout_p = dropout

        # Select activation function
        if activation == "relu":
            activation_fn = nn.ReLU
        elif activation == "gelu":
            activation_fn = nn.GELU
        elif activation == "tanh":
            activation_fn = nn.Tanh
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build MLP using LazyLinear for first layer
        # This allows parameters to be registered immediately while deferring
        # input size determination until first forward pass
        feature_layers = []

        # First hidden layer - use LazyLinear since we don't know T_in
        feature_layers.append(nn.LazyLinear(hidden_dims[0]))
        feature_layers.append(activation_fn())
        feature_layers.append(nn.Dropout(dropout))

        # Remaining hidden layers
        for i in range(1, len(hidden_dims)):
            feature_layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
            feature_layers.append(activation_fn())
            feature_layers.append(nn.Dropout(dropout))

        self.feature_extractor = nn.Sequential(*feature_layers)
        self.head = nn.Linear(hidden_dims[-1], output_dim)

    def encode(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        del context
        latent = self.feature_extractor(x.reshape(x.shape[0], -1))
        extras: Dict[str, torch.Tensor] = {
            "input_shape": torch.tensor(x.shape[1:], device=x.device),
            "hidden_dims": torch.tensor(self.hidden_dims, device=x.device),
        }
        return latent, extras

    def decode(self, latent: torch.Tensor, pred_len: int) -> torch.Tensor:
        preds = self.head(latent).unsqueeze(1)
        if pred_len != 1:
            preds = preds.expand(-1, pred_len, -1)
        return preds

    def forward(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None, **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass - flatten and pass through MLP.

        Args:
            x: Input tensor [B, T_in, D_in]
            context: Optional context tensor (ignored)
            **kwargs: Additional arguments (ignored)

        Returns:
            Dictionary with 'preds' [B, 1, D_out]
        """
        pred_len = int(kwargs.get("pred_len", 1))
        latent, extras = self.encode(x, context=context)
        preds = self.decode(latent, pred_len)
        extras["latent"] = latent

        return {"preds": preds, "extras": extras}
