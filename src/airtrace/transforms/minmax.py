"""MinMax scaling transforms for sensor data."""

from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from .base import Transform
from .registry import register


@register("minmax")
class MinMaxTransform(Transform):
    """MinMax scaling transform.

    Scales data to a specified range (default [0, 1]).
    Useful for bounded sensors like throttle %, flaps position.
    """

    def __init__(
        self,
        feature_range: Tuple[float, float] = (0.0, 1.0),
        per_sensor: bool = True,
        clip: bool = False
    ):
        """Initialize MinMax transform.

        Args:
            feature_range: Target range for scaling (min, max)
            per_sensor: If True, scale each sensor independently
            clip: If True, clip values outside feature_range during transform
        """
        super().__init__()
        self.feature_range = feature_range
        self.per_sensor = per_sensor
        self.clip = clip
        self.scaler = MinMaxScaler(feature_range=feature_range, clip=clip)

    def fit(self, dataset) -> "MinMaxTransform":
        """Fit scaler on dataset.

        Args:
            dataset: Dataset to fit on

        Returns:
            self for method chaining
        """
        # Collect samples for fitting
        all_x = []
        for i in range(min(len(dataset), 1000)):  # Sample first 1000 for efficiency
            sample = dataset[i]
            x = sample["x"] if isinstance(sample, dict) else sample[0]
            all_x.append(x.numpy() if hasattr(x, "numpy") else x)

        # Reshape to [N, D] for fitting
        all_x = np.concatenate(all_x, axis=0)

        # Fit scaler
        self.scaler.fit(all_x)
        self.is_fitted = True

        return self

    def __call__(
        self, x: np.ndarray, y: np.ndarray, meta: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Apply MinMax scaling.

        Args:
            x: Input sequence [T_in, D_in]
            y: Target sequence [T_out, D_out]
            meta: Metadata dict

        Returns:
            Transformed (x, y, meta)
        """
        if not self.is_fitted:
            raise RuntimeError("Transform not fitted. Call fit() first.")

        # Transform x
        x_shape = x.shape
        x_flat = x.reshape(-1, x_shape[-1])
        x_transformed = self.scaler.transform(x_flat)
        x = x_transformed.reshape(x_shape)

        # Transform y (same scaler)
        y_shape = y.shape
        y_flat = y.reshape(-1, y_shape[-1])
        y_transformed = self.scaler.transform(y_flat)
        y = y_transformed.reshape(y_shape)

        return x, y, meta

    def inverse(self, x: np.ndarray, y: Optional[np.ndarray] = None):
        """Inverse MinMax scaling.

        Args:
            x: Scaled input
            y: Scaled target

        Returns:
            Original scale (x, y)
        """
        x_shape = x.shape
        x_flat = x.reshape(-1, x_shape[-1])
        x_inv = self.scaler.inverse_transform(x_flat)
        x = x_inv.reshape(x_shape)

        if y is not None:
            y_shape = y.shape
            y_flat = y.reshape(-1, y_shape[-1])
            y_inv = self.scaler.inverse_transform(y_flat)
            y = y_inv.reshape(y_shape)

        return x, y
