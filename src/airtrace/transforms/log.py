"""Logarithmic transforms for sensor data."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import Transform
from .registry import register


@register("log")
class LogTransform(Transform):
    """Logarithmic transform for sensors with exponential/multiplicative behavior.

    Useful for fuel consumption, flow rates, and other sensors that exhibit
    exponential or multiplicative dynamics. Stabilizes variance in these processes.
    """

    def __init__(
        self,
        base: str = "natural",
        offset: float = 1e-8,
        sensors: Optional[List[int]] = None,
        per_sensor: bool = True
    ):
        """Initialize log transform.

        Args:
            base: Logarithm base - 'natural' (ln), '10', or '2'
            offset: Small constant to avoid log(0), added before log
            sensors: Indices of sensors to transform (None = all sensors)
            per_sensor: If True, track min values per sensor for offset
        """
        super().__init__()
        self.base = base
        self.offset = offset
        self.sensors = sensors
        self.per_sensor = per_sensor

        # Will store minimum values per sensor (for offset adjustment)
        self.min_values = None

    def fit(self, dataset) -> "LogTransform":
        """Fit transform (compute min values for offset).

        Args:
            dataset: Dataset to fit on

        Returns:
            self for method chaining
        """
        # Collect samples to compute minimum values
        all_x = []
        for i in range(min(len(dataset), 1000)):
            sample = dataset[i]
            x = sample["x"] if isinstance(sample, dict) else sample[0]
            all_x.append(x.numpy() if hasattr(x, "numpy") else x)

        all_x = np.concatenate(all_x, axis=0)

        # Compute min per sensor (to ensure positive values)
        if self.per_sensor:
            self.min_values = np.min(all_x, axis=0)
        else:
            min_val = np.min(all_x)
            self.min_values = np.full(all_x.shape[1], min_val)

        self.is_fitted = True
        return self

    def _get_min_values(self, data: np.ndarray, sensor_idx: Optional[int] = None) -> np.ndarray:
        """Return minimum values aligned with the provided data slice."""

        if self.min_values is None:
            raise RuntimeError("Transform not fitted. Call fit() first.")

        if sensor_idx is None:
            return self.min_values

        end_idx = sensor_idx + data.shape[1]
        return self.min_values[sensor_idx:end_idx]

    def _apply_log(self, data: np.ndarray, sensor_idx: Optional[int] = None) -> np.ndarray:
        """Apply logarithm with appropriate base.

        Args:
            data: Input data

        Returns:
            Log-transformed data
        """
        # Ensure positive values by subtracting min and adding offset
        # If data is already positive, min_values will be <= 0
        min_values = np.minimum(self._get_min_values(data, sensor_idx), 0)
        adjusted = data - min_values + self.offset

        if self.base == "natural":
            return np.log(adjusted)
        elif self.base == "10":
            return np.log10(adjusted)
        elif self.base == "2":
            return np.log2(adjusted)
        else:
            raise ValueError(f"Unknown base: {self.base}")

    def _apply_exp(self, data: np.ndarray, sensor_idx: Optional[int] = None) -> np.ndarray:
        """Apply exponential (inverse of log).

        Args:
            data: Log-transformed data

        Returns:
            Original scale data
        """
        if self.base == "natural":
            result = np.exp(data)
        elif self.base == "10":
            result = np.power(10, data)
        elif self.base == "2":
            result = np.power(2, data)
        else:
            raise ValueError(f"Unknown base: {self.base}")

        # Reverse the offset adjustment
        min_values = np.minimum(self._get_min_values(data, sensor_idx), 0)
        result = result - self.offset + min_values
        return result

    def __call__(
        self, x: np.ndarray, y: np.ndarray, meta: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Apply log transform.

        Args:
            x: Input sequence [T_in, D_in]
            y: Target sequence [T_out, D_out]
            meta: Metadata dict

        Returns:
            Log-transformed (x, y, meta)
        """
        if not self.is_fitted:
            raise RuntimeError("Transform not fitted. Call fit() first.")

        # Apply log to specified sensors or all
        if self.sensors is None:
            x = self._apply_log(x)
            y = self._apply_log(y)
        else:
            x_copy = x.copy()
            y_copy = y.copy()
            for sensor_idx in self.sensors:
                x_copy[:, sensor_idx] = self._apply_log(
                    x[:, sensor_idx : sensor_idx + 1], sensor_idx
                ).squeeze()
                if y.shape[1] > sensor_idx:
                    y_copy[:, sensor_idx] = self._apply_log(
                        y[:, sensor_idx : sensor_idx + 1], sensor_idx
                    ).squeeze()
            x = x_copy
            y = y_copy

        meta["log_transformed"] = True
        meta["log_base"] = self.base

        return x, y, meta

    def inverse(self, x: np.ndarray, y: Optional[np.ndarray] = None):
        """Inverse log transform (exponential).

        Args:
            x: Log-transformed input
            y: Log-transformed target

        Returns:
            Original scale (x, y)
        """
        if self.sensors is None:
            x = self._apply_exp(x)
            if y is not None:
                y = self._apply_exp(y)
        else:
            x_copy = x.copy()
            for sensor_idx in self.sensors:
                x_copy[:, sensor_idx] = self._apply_exp(
                    x[:, sensor_idx : sensor_idx + 1], sensor_idx
                ).squeeze()
            x = x_copy

            if y is not None:
                y_copy = y.copy()
                for sensor_idx in self.sensors:
                    if y.shape[1] > sensor_idx:
                        y_copy[:, sensor_idx] = self._apply_exp(
                            y[:, sensor_idx : sensor_idx + 1], sensor_idx
                        ).squeeze()
                y = y_copy

        return x, y
