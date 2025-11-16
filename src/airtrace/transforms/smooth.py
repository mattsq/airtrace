"""Smoothing transforms for sensor data."""

from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import uniform_filter1d

from .base import Transform
from .registry import register


@register("smooth")
class SmoothTransform(Transform):
    """Moving average smoothing transform.

    Denoises high-frequency sensor jitter (e.g., altitude, speed during turbulence).
    Useful as pre-smoothing before autoregressive modeling.
    """

    def __init__(
        self,
        window_size: int = 3,
        method: str = "uniform",
        mode: str = "nearest"
    ):
        """Initialize smoothing transform.

        Args:
            window_size: Size of the smoothing window (must be odd)
            method: Smoothing method - 'uniform' (moving average) or 'gaussian'
            mode: How to handle boundaries - 'nearest', 'reflect', or 'wrap'
        """
        super().__init__()
        self.window_size = window_size
        self.method = method
        self.mode = mode

        # Ensure window size is odd
        if self.window_size % 2 == 0:
            self.window_size += 1

    def fit(self, dataset) -> "SmoothTransform":
        """Fit transform (no-op for smoothing).

        Args:
            dataset: Dataset to fit on

        Returns:
            self for method chaining
        """
        self.is_fitted = True
        return self

    def _smooth_array(self, data: np.ndarray) -> np.ndarray:
        """Apply smoothing to a 2D array [T, D].

        Args:
            data: Input array [T, D]

        Returns:
            Smoothed array [T, D]
        """
        if self.method == "uniform":
            # Apply uniform filter along time axis (axis=0) for each sensor
            smoothed = uniform_filter1d(
                data,
                size=self.window_size,
                axis=0,
                mode=self.mode
            )
        elif self.method == "gaussian":
            from scipy.ndimage import gaussian_filter1d
            sigma = self.window_size / 6.0  # Approximate: 99.7% within window
            smoothed = gaussian_filter1d(
                data,
                sigma=sigma,
                axis=0,
                mode=self.mode
            )
        else:
            raise ValueError(f"Unknown smoothing method: {self.method}")

        return smoothed

    def __call__(
        self, x: np.ndarray, y: np.ndarray, meta: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Apply smoothing.

        Args:
            x: Input sequence [T_in, D_in]
            y: Target sequence [T_out, D_out]
            meta: Metadata dict

        Returns:
            Smoothed (x, y, meta)
        """
        if not self.is_fitted:
            raise RuntimeError("Transform not fitted. Call fit() first.")

        # Smooth x and y independently
        x_smooth = self._smooth_array(x)
        y_smooth = self._smooth_array(y) if len(y) > 0 else y

        meta["smoothed"] = True
        meta["smooth_window"] = self.window_size
        meta["smooth_method"] = self.method

        return x_smooth, y_smooth, meta

    def inverse(self, x: np.ndarray, y: Optional[np.ndarray] = None):
        """Inverse smoothing (not possible - information is lost).

        Args:
            x: Smoothed input
            y: Smoothed target

        Returns:
            (x, y) unchanged (smoothing is irreversible)
        """
        # Smoothing is irreversible - return as-is
        return x, y
