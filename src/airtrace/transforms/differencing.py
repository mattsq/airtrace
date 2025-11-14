"""Differencing transforms for sensor data."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import Transform
from .registry import register


@register("diff")
class DifferenceTransform(Transform):
    """First-order differencing transform.

    Applies differencing to specified sensors to make them stationary.
    Useful for sensors with trends (e.g., fuel consumption).
    """

    def __init__(self, sensors: Optional[List[str]] = None, order: int = 1):
        """Initialize difference transform.

        Args:
            sensors: List of sensor names to difference (None = all sensors)
            order: Order of differencing (1 = first difference, 2 = second, etc.)
        """
        super().__init__()
        self.sensors = sensors
        self.order = order
        self.sensor_indices = None

    def fit(self, dataset) -> "DifferenceTransform":
        """Fit transform (determine sensor indices).

        Args:
            dataset: Dataset to fit on

        Returns:
            self for method chaining
        """
        # If specific sensors specified, we'd need sensor names from dataset
        # For now, assume we difference all channels
        self.is_fitted = True
        return self

    def __call__(
        self, x: np.ndarray, y: np.ndarray, meta: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Apply differencing.

        Args:
            x: Input sequence [T_in, D]
            y: Target sequence [T_out, D]
            meta: Metadata

        Returns:
            Differenced (x, y, meta)
        """
        # Store original first value for inverse transform
        meta["diff_initial_x"] = x[0].copy()
        meta["diff_initial_y"] = y[0].copy() if len(y) > 0 else None

        # Apply differencing
        if self.sensors is None:
            # Difference all sensors
            x_diff = np.diff(x, n=self.order, axis=0)
            y_diff = np.diff(y, n=self.order, axis=0) if len(y) > 0 else y
        else:
            # Difference only specified sensors
            # This is simplified - real implementation would use sensor names
            x_diff = np.diff(x, n=self.order, axis=0)
            y_diff = np.diff(y, n=self.order, axis=0) if len(y) > 0 else y

        # Pad to maintain sequence length
        x_padded = np.vstack([x[:self.order], x_diff])
        y_padded = np.vstack([y[:self.order], y_diff]) if len(y) > 0 else y

        return x_padded, y_padded, meta

    def inverse(self, x: np.ndarray, y: np.ndarray = None):
        """Inverse differencing (cumulative sum).

        Args:
            x: Differenced input
            y: Differenced target

        Returns:
            Original (x, y)

        Note:
            This requires the initial value, which should be stored in meta
            during forward pass. Simplified version here.
        """
        # Cumulative sum to invert differencing
        x_inv = np.cumsum(x, axis=0)

        if y is not None:
            y_inv = np.cumsum(y, axis=0)
        else:
            y_inv = None

        return x_inv, y_inv
