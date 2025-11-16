"""Detrending transforms for sensor data."""

from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.signal import detrend as scipy_detrend

from .base import Transform
from .registry import register


@register("detrend")
class DetrendTransform(Transform):
    """Polynomial detrending transform.

    Removes deterministic trends (e.g., linear weight decrease from fuel burn)
    to make data stationary without losing information to differencing.
    """

    def __init__(
        self,
        method: str = "linear",
        degree: int = 1,
        per_sensor: bool = True
    ):
        """Initialize detrending transform.

        Args:
            method: Detrending method - 'linear', 'constant', or 'polynomial'
            degree: Polynomial degree for 'polynomial' method (1=linear, 2=quadratic, etc.)
            per_sensor: If True, detrend each sensor independently
        """
        super().__init__()
        self.method = method
        self.degree = degree
        self.per_sensor = per_sensor

        # Store trend coefficients for inverse transform
        self.x_coeffs = None
        self.y_coeffs = None

    def fit(self, dataset) -> "DetrendTransform":
        """Fit transform (no-op - trends computed per sample).

        Args:
            dataset: Dataset to fit on

        Returns:
            self for method chaining
        """
        # Detrending is computed per sample, no global fit needed
        self.is_fitted = True
        return self

    def _fit_trend(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fit polynomial trend to data.

        Args:
            data: Input array [T, D]

        Returns:
            Tuple of (detrended_data, coefficients)
        """
        T, D = data.shape
        time_idx = np.arange(T)

        if self.method == "constant":
            # Remove mean
            coeffs = np.mean(data, axis=0, keepdims=True)  # [1, D]
            detrended = data - coeffs

        elif self.method == "linear":
            # Use scipy's linear detrend
            detrended = np.zeros_like(data)
            coeffs = np.zeros((2, D))  # [2, D] for slope and intercept

            for col in range(D):
                detrended[:, col] = scipy_detrend(data[:, col], type='linear')
                # Compute original trend for inverse
                coeffs[:, col] = np.polyfit(time_idx, data[:, col], deg=1)

        elif self.method == "polynomial":
            # Polynomial detrending
            detrended = np.zeros_like(data)
            coeffs = np.zeros((self.degree + 1, D))

            for col in range(D):
                poly_coeffs = np.polyfit(time_idx, data[:, col], deg=self.degree)
                trend = np.polyval(poly_coeffs, time_idx)
                detrended[:, col] = data[:, col] - trend
                coeffs[:, col] = poly_coeffs

        else:
            raise ValueError(f"Unknown detrending method: {self.method}")

        return detrended, coeffs

    def _reconstruct_trend(self, data: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
        """Reconstruct original data from detrended data and coefficients.

        Args:
            data: Detrended array [T, D]
            coeffs: Trend coefficients

        Returns:
            Original data with trend restored
        """
        T, D = data.shape
        time_idx = np.arange(T)

        if self.method == "constant":
            # Add back mean
            return data + coeffs

        elif self.method == "linear" or self.method == "polynomial":
            reconstructed = np.zeros_like(data)
            for col in range(D):
                trend = np.polyval(coeffs[:, col], time_idx)
                reconstructed[:, col] = data[:, col] + trend
            return reconstructed

        else:
            raise ValueError(f"Unknown detrending method: {self.method}")

    def __call__(
        self, x: np.ndarray, y: np.ndarray, meta: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Apply detrending.

        Args:
            x: Input sequence [T_in, D_in]
            y: Target sequence [T_out, D_out]
            meta: Metadata dict

        Returns:
            Detrended (x, y, meta)
        """
        if not self.is_fitted:
            raise RuntimeError("Transform not fitted. Call fit() first.")

        # Detrend x
        x_detrended, x_coeffs = self._fit_trend(x)

        # Detrend y
        if len(y) > 0:
            y_detrended, y_coeffs = self._fit_trend(y)
        else:
            y_detrended = y
            y_coeffs = None

        # Store coefficients in meta for inverse transform
        meta["detrend_x_coeffs"] = x_coeffs
        meta["detrend_y_coeffs"] = y_coeffs
        meta["detrend_method"] = self.method
        meta["detrend_degree"] = self.degree

        return x_detrended, y_detrended, meta

    def inverse(self, x: np.ndarray, y: Optional[np.ndarray] = None, meta: Optional[Dict[str, Any]] = None):
        """Inverse detrending (restore trend).

        Args:
            x: Detrended input
            y: Detrended target
            meta: Metadata with stored coefficients

        Returns:
            Original (x, y) with trends restored
        """
        # Note: This requires meta with coefficients from forward pass
        # For now, return as-is if meta not provided
        if meta is None or "detrend_x_coeffs" not in meta:
            return x, y

        x_reconstructed = self._reconstruct_trend(x, meta["detrend_x_coeffs"])

        if y is not None and meta.get("detrend_y_coeffs") is not None:
            y_reconstructed = self._reconstruct_trend(y, meta["detrend_y_coeffs"])
        else:
            y_reconstructed = y

        return x_reconstructed, y_reconstructed
