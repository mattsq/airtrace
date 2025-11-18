"""Scaling transforms for sensor data."""

from typing import Any, Dict, Tuple

import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler

from .base import Transform
from .registry import register


@register("zscore")
class ZScoreTransform(Transform):
    """Z-score normalization transform.

    Normalizes data to zero mean and unit variance per sensor.
    """

    def __init__(self, per_sensor: bool = True, center: bool = True, scale: bool = True):
        """Initialize z-score transform.

        Args:
            per_sensor: If True, normalize each sensor independently
            center: If True, center the data (subtract mean)
            scale: If True, scale the data (divide by std)
        """
        super().__init__()
        self.per_sensor = per_sensor
        self.center = center
        self.scale = scale
        self.scaler_x = StandardScaler(with_mean=center, with_std=scale)
        self.scaler_y = StandardScaler(with_mean=center, with_std=scale)

    def fit(self, dataset) -> "ZScoreTransform":
        """Fit scaler on dataset.

        Args:
            dataset: Dataset to fit on

        Returns:
            self for method chaining
        """
        # Collect samples for fitting with batch loading for speed
        num_samples = min(len(dataset), 200)  # Reduced from 1000 for faster fitting
        batch_size = 10  # Load 10 samples at a time

        all_x = []
        all_y = []

        # Batch load for better I/O performance
        for batch_start in range(0, num_samples, batch_size):
            batch_end = min(batch_start + batch_size, num_samples)
            for i in range(batch_start, batch_end):
                sample = dataset[i]
                x = sample["x"] if isinstance(sample, dict) else sample[0]
                y = sample["y"] if isinstance(sample, dict) else sample[1]
                all_x.append(x.numpy() if hasattr(x, "numpy") else x)
                all_y.append(y.numpy() if hasattr(y, "numpy") else y)

        # Reshape to [N, D] for fitting
        all_x = np.concatenate(all_x, axis=0)
        all_y = np.concatenate(all_y, axis=0)

        # Fit scalers
        self.scaler_x.fit(all_x)
        self.scaler_y.fit(all_y)
        self.is_fitted = True

        return self

    def get_stats(self) -> Dict[str, Any]:
        """Get scaler statistics for caching.

        Returns:
            Dictionary containing scaler parameters
        """
        if not self.is_fitted:
            raise RuntimeError("Transform not fitted. Call fit() first.")

        return {
            'scaler_x_mean': self.scaler_x.mean_,
            'scaler_x_scale': self.scaler_x.scale_,
            'scaler_x_var': self.scaler_x.var_,
            'scaler_x_n_features_in': self.scaler_x.n_features_in_,
            'scaler_x_n_samples_seen': self.scaler_x.n_samples_seen_,
            'scaler_y_mean': self.scaler_y.mean_,
            'scaler_y_scale': self.scaler_y.scale_,
            'scaler_y_var': self.scaler_y.var_,
            'scaler_y_n_features_in': self.scaler_y.n_features_in_,
            'scaler_y_n_samples_seen': self.scaler_y.n_samples_seen_,
            'per_sensor': self.per_sensor,
            'center': self.center,
            'scale': self.scale,
        }

    def set_stats(self, stats: Dict[str, Any]) -> None:
        """Set scaler statistics from cache.

        Args:
            stats: Dictionary containing scaler parameters
        """
        self.scaler_x.mean_ = stats['scaler_x_mean']
        self.scaler_x.scale_ = stats['scaler_x_scale']
        self.scaler_x.var_ = stats['scaler_x_var']
        self.scaler_x.n_features_in_ = stats['scaler_x_n_features_in']
        self.scaler_x.n_samples_seen_ = stats['scaler_x_n_samples_seen']
        self.scaler_y.mean_ = stats['scaler_y_mean']
        self.scaler_y.scale_ = stats['scaler_y_scale']
        self.scaler_y.var_ = stats['scaler_y_var']
        self.scaler_y.n_features_in_ = stats['scaler_y_n_features_in']
        self.scaler_y.n_samples_seen_ = stats['scaler_y_n_samples_seen']
        self.is_fitted = True

    def __call__(
        self, x: np.ndarray, y: np.ndarray, meta: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Apply z-score normalization.

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
        x_transformed = self.scaler_x.transform(x_flat)
        x = x_transformed.reshape(x_shape)

        # Transform y (separate scaler)
        y_shape = y.shape
        y_flat = y.reshape(-1, y_shape[-1])
        y_transformed = self.scaler_y.transform(y_flat)
        y = y_transformed.reshape(y_shape)

        return x, y, meta

    def inverse(self, x: np.ndarray, y: np.ndarray = None):
        """Inverse z-score normalization.

        Args:
            x: Normalized input
            y: Normalized target

        Returns:
            Original scale (x, y)
        """
        x_shape = x.shape
        x_flat = x.reshape(-1, x_shape[-1])
        x_inv = self.scaler_x.inverse_transform(x_flat)
        x = x_inv.reshape(x_shape)

        if y is not None:
            y_shape = y.shape
            y_flat = y.reshape(-1, y_shape[-1])
            y_inv = self.scaler_y.inverse_transform(y_flat)
            y = y_inv.reshape(y_shape)

        return x, y


@register("robust_scaler")
class RobustScalerTransform(Transform):
    """Robust scaler using median and IQR.

    More robust to outliers than z-score normalization.
    """

    def __init__(
        self,
        per_sensor: bool = True,
        quantile_range: Tuple[float, float] = (25.0, 75.0),
        with_centering: bool = True,
        with_scaling: bool = True
    ):
        """Initialize robust scaler.

        Args:
            per_sensor: If True, scale each sensor independently
            quantile_range: Quantile range for IQR calculation
            with_centering: If True, center using median
            with_scaling: If True, scale using IQR
        """
        super().__init__()
        self.per_sensor = per_sensor
        self.quantile_range = quantile_range
        self.scaler_x = RobustScaler(
            quantile_range=quantile_range,
            with_centering=with_centering,
            with_scaling=with_scaling
        )
        self.scaler_y = RobustScaler(
            quantile_range=quantile_range,
            with_centering=with_centering,
            with_scaling=with_scaling
        )

    def fit(self, dataset) -> "RobustScalerTransform":
        """Fit scaler on dataset.

        Args:
            dataset: Dataset to fit on

        Returns:
            self for method chaining
        """
        # Collect samples with batch loading for speed (optimized from one-by-one)
        num_samples = min(len(dataset), 200)  # Reduced from 1000 for faster fitting
        batch_size = 10  # Load 10 samples at a time

        all_x = []
        all_y = []

        # Batch load for better I/O performance
        for batch_start in range(0, num_samples, batch_size):
            batch_end = min(batch_start + batch_size, num_samples)
            for i in range(batch_start, batch_end):
                sample = dataset[i]
                x = sample["x"] if isinstance(sample, dict) else sample[0]
                y = sample["y"] if isinstance(sample, dict) else sample[1]
                all_x.append(x.numpy() if hasattr(x, "numpy") else x)
                all_y.append(y.numpy() if hasattr(y, "numpy") else y)

        all_x = np.concatenate(all_x, axis=0)
        all_y = np.concatenate(all_y, axis=0)

        # Fit scalers
        self.scaler_x.fit(all_x)
        self.scaler_y.fit(all_y)
        self.is_fitted = True

        return self

    def get_stats(self) -> Dict[str, Any]:
        """Get scaler statistics for caching.

        Returns:
            Dictionary containing scaler parameters
        """
        if not self.is_fitted:
            raise RuntimeError("Transform not fitted. Call fit() first.")

        return {
            'scaler_x_center': self.scaler_x.center_,
            'scaler_x_scale': self.scaler_x.scale_,
            'scaler_x_n_features_in': self.scaler_x.n_features_in_,
            'scaler_y_center': self.scaler_y.center_,
            'scaler_y_scale': self.scaler_y.scale_,
            'scaler_y_n_features_in': self.scaler_y.n_features_in_,
            'per_sensor': self.per_sensor,
            'quantile_range': self.quantile_range,
        }

    def set_stats(self, stats: Dict[str, Any]) -> None:
        """Set scaler statistics from cache.

        Args:
            stats: Dictionary containing scaler parameters
        """
        self.scaler_x.center_ = stats['scaler_x_center']
        self.scaler_x.scale_ = stats['scaler_x_scale']
        self.scaler_x.n_features_in_ = stats['scaler_x_n_features_in']
        self.scaler_y.center_ = stats['scaler_y_center']
        self.scaler_y.scale_ = stats['scaler_y_scale']
        self.scaler_y.n_features_in_ = stats['scaler_y_n_features_in']
        self.is_fitted = True

    def __call__(
        self, x: np.ndarray, y: np.ndarray, meta: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Apply robust scaling.

        Args:
            x: Input sequence
            y: Target sequence
            meta: Metadata

        Returns:
            Transformed (x, y, meta)
        """
        if not self.is_fitted:
            raise RuntimeError("Transform not fitted. Call fit() first.")

        # Transform x and y
        x_shape = x.shape
        x = self.scaler_x.transform(x.reshape(-1, x_shape[-1])).reshape(x_shape)

        y_shape = y.shape
        y = self.scaler_y.transform(y.reshape(-1, y_shape[-1])).reshape(y_shape)

        return x, y, meta

    def inverse(self, x: np.ndarray, y: np.ndarray = None):
        """Inverse robust scaling.

        Args:
            x: Scaled input
            y: Scaled target

        Returns:
            Original scale (x, y)
        """
        x_shape = x.shape
        x = self.scaler_x.inverse_transform(x.reshape(-1, x_shape[-1])).reshape(x_shape)

        if y is not None:
            y_shape = y.shape
            y = self.scaler_y.inverse_transform(y.reshape(-1, y_shape[-1])).reshape(y_shape)

        return x, y
