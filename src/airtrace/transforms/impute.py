"""Missing value imputation transforms for sensor data."""

from typing import Any, Dict, Optional, Tuple

import numpy as np

from .base import Transform
from .registry import register


@register("impute")
class ImputeTransform(Transform):
    """Missing value imputation transform.

    Handles sensor dropouts and telemetry gaps in real flight data.
    Supports forward-fill, backward-fill, interpolation, and mean imputation.
    """

    def __init__(
        self,
        method: str = "forward",
        fill_value: Optional[float] = None,
        limit: Optional[int] = None
    ):
        """Initialize imputation transform.

        Args:
            method: Imputation method - 'forward', 'backward', 'linear', 'mean', or 'constant'
            fill_value: Value to use for 'constant' method
            limit: Maximum number of consecutive NaNs to fill (None = no limit)
        """
        super().__init__()
        self.method = method
        self.fill_value = fill_value
        self.limit = limit

        # Will store mean values per sensor (for 'mean' method)
        self.mean_values = None

    def fit(self, dataset) -> "ImputeTransform":
        """Fit transform (compute mean values if needed).

        Args:
            dataset: Dataset to fit on

        Returns:
            self for method chaining
        """
        if self.method == "mean":
            # Collect samples to compute mean per sensor
            all_x = []
            for i in range(min(len(dataset), 1000)):
                sample = dataset[i]
                x = sample["x"] if isinstance(sample, dict) else sample[0]
                all_x.append(x.numpy() if hasattr(x, "numpy") else x)

            all_x = np.concatenate(all_x, axis=0)
            # Compute mean ignoring NaNs
            self.mean_values = np.nanmean(all_x, axis=0)

        self.is_fitted = True
        return self

    def _impute_forward(self, data: np.ndarray) -> np.ndarray:
        """Forward fill missing values.

        Args:
            data: Input array [T, D]

        Returns:
            Imputed array [T, D]
        """
        result = data.copy()
        for col in range(data.shape[1]):
            mask = np.isnan(result[:, col])
            if not mask.any():
                continue

            # Get indices of non-NaN values
            valid_idx = np.where(~mask)[0]
            if len(valid_idx) == 0:
                # All NaN - fill with 0
                result[:, col] = 0
                continue

            # Forward fill
            last_valid = None
            fill_count = 0
            for i in range(len(result)):
                if not mask[i]:
                    last_valid = result[i, col]
                    fill_count = 0
                elif last_valid is not None:
                    if self.limit is None or fill_count < self.limit:
                        result[i, col] = last_valid
                        fill_count += 1

        return result

    def _impute_backward(self, data: np.ndarray) -> np.ndarray:
        """Backward fill missing values.

        Args:
            data: Input array [T, D]

        Returns:
            Imputed array [T, D]
        """
        result = data.copy()
        for col in range(data.shape[1]):
            mask = np.isnan(result[:, col])
            if not mask.any():
                continue

            # Backward fill
            next_valid = None
            fill_count = 0
            for i in range(len(result) - 1, -1, -1):
                if not mask[i]:
                    next_valid = result[i, col]
                    fill_count = 0
                elif next_valid is not None:
                    if self.limit is None or fill_count < self.limit:
                        result[i, col] = next_valid
                        fill_count += 1

        return result

    def _impute_linear(self, data: np.ndarray) -> np.ndarray:
        """Linear interpolation for missing values.

        Args:
            data: Input array [T, D]

        Returns:
            Imputed array [T, D]
        """
        result = data.copy()
        for col in range(data.shape[1]):
            mask = ~np.isnan(result[:, col])
            if mask.sum() < 2:
                # Not enough points for interpolation
                result[:, col] = np.nan_to_num(result[:, col], nan=0.0)
                continue

            # Get valid indices and values
            valid_idx = np.where(mask)[0]
            valid_vals = result[mask, col]

            # Interpolate
            all_idx = np.arange(len(result))
            result[:, col] = np.interp(all_idx, valid_idx, valid_vals)

        return result

    def _impute_mean(self, data: np.ndarray) -> np.ndarray:
        """Fill missing values with mean.

        Args:
            data: Input array [T, D]

        Returns:
            Imputed array [T, D]
        """
        result = data.copy()
        for col in range(data.shape[1]):
            mask = np.isnan(result[:, col])
            if mask.any():
                result[mask, col] = self.mean_values[col]
        return result

    def _impute_constant(self, data: np.ndarray) -> np.ndarray:
        """Fill missing values with constant.

        Args:
            data: Input array [T, D]

        Returns:
            Imputed array [T, D]
        """
        fill_val = self.fill_value if self.fill_value is not None else 0.0
        return np.nan_to_num(data, nan=fill_val)

    def __call__(
        self, x: np.ndarray, y: np.ndarray, meta: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Apply imputation.

        Args:
            x: Input sequence [T_in, D_in]
            y: Target sequence [T_out, D_out]
            meta: Metadata dict

        Returns:
            Imputed (x, y, meta)
        """
        if not self.is_fitted:
            raise RuntimeError("Transform not fitted. Call fit() first.")

        # Apply imputation based on method
        if self.method == "forward":
            x = self._impute_forward(x)
            y = self._impute_forward(y) if len(y) > 0 else y
        elif self.method == "backward":
            x = self._impute_backward(x)
            y = self._impute_backward(y) if len(y) > 0 else y
        elif self.method == "linear":
            x = self._impute_linear(x)
            y = self._impute_linear(y) if len(y) > 0 else y
        elif self.method == "mean":
            x = self._impute_mean(x)
            y = self._impute_mean(y) if len(y) > 0 else y
        elif self.method == "constant":
            x = self._impute_constant(x)
            y = self._impute_constant(y) if len(y) > 0 else y
        else:
            raise ValueError(f"Unknown imputation method: {self.method}")

        meta["imputed"] = True
        meta["impute_method"] = self.method

        return x, y, meta

    def inverse(self, x: np.ndarray, y: Optional[np.ndarray] = None):
        """Inverse imputation (not possible - information is lost).

        Args:
            x: Imputed input
            y: Imputed target

        Returns:
            (x, y) unchanged (imputation is irreversible)
        """
        # Imputation is irreversible - return as-is
        return x, y
