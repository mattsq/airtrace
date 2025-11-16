"""Clipping and outlier removal transforms for sensor data."""

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from .base import Transform
from .registry import register


@register("clip")
class ClipTransform(Transform):
    """Clip outliers using percentiles or standard deviations.

    Removes extreme outliers before normalization to improve model robustness.
    Essential for handling sensor glitches and transient spikes.
    """

    def __init__(
        self,
        method: str = "percentile",
        lower: Optional[float] = 1.0,
        upper: Optional[float] = 99.0,
        n_std: float = 3.0,
        per_sensor: bool = True
    ):
        """Initialize clip transform.

        Args:
            method: Clipping method - 'percentile', 'std', or 'absolute'
            lower: Lower bound (percentile or absolute value)
            upper: Upper bound (percentile or absolute value)
            n_std: Number of standard deviations for 'std' method
            per_sensor: If True, compute bounds per sensor independently
        """
        super().__init__()
        self.method = method
        self.lower = lower
        self.upper = upper
        self.n_std = n_std
        self.per_sensor = per_sensor

        # Will be computed during fit
        self.lower_bounds = None
        self.upper_bounds = None

    def fit(self, dataset) -> "ClipTransform":
        """Fit clipping bounds on dataset.

        Args:
            dataset: Dataset to fit on

        Returns:
            self for method chaining
        """
        # Collect samples
        all_x = []
        for i in range(min(len(dataset), 1000)):
            sample = dataset[i]
            x = sample["x"] if isinstance(sample, dict) else sample[0]
            all_x.append(x.numpy() if hasattr(x, "numpy") else x)

        all_x = np.concatenate(all_x, axis=0)  # [N, D]

        if self.method == "percentile":
            # Compute percentile bounds per sensor
            if self.per_sensor:
                self.lower_bounds = np.percentile(all_x, self.lower, axis=0)
                self.upper_bounds = np.percentile(all_x, self.upper, axis=0)
            else:
                lower_val = np.percentile(all_x, self.lower)
                upper_val = np.percentile(all_x, self.upper)
                self.lower_bounds = np.full(all_x.shape[1], lower_val)
                self.upper_bounds = np.full(all_x.shape[1], upper_val)

        elif self.method == "std":
            # Compute mean ± n*std bounds per sensor
            if self.per_sensor:
                mean = np.mean(all_x, axis=0)
                std = np.std(all_x, axis=0)
                self.lower_bounds = mean - self.n_std * std
                self.upper_bounds = mean + self.n_std * std
            else:
                mean = np.mean(all_x)
                std = np.std(all_x)
                self.lower_bounds = np.full(all_x.shape[1], mean - self.n_std * std)
                self.upper_bounds = np.full(all_x.shape[1], mean + self.n_std * std)

        elif self.method == "absolute":
            # Use fixed absolute bounds
            self.lower_bounds = np.full(all_x.shape[1], self.lower)
            self.upper_bounds = np.full(all_x.shape[1], self.upper)

        else:
            raise ValueError(f"Unknown clipping method: {self.method}")

        self.is_fitted = True
        return self

    def __call__(
        self, x: np.ndarray, y: np.ndarray, meta: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Apply clipping.

        Args:
            x: Input sequence [T_in, D_in]
            y: Target sequence [T_out, D_out]
            meta: Metadata dict

        Returns:
            Clipped (x, y, meta)
        """
        if not self.is_fitted:
            raise RuntimeError("Transform not fitted. Call fit() first.")

        # Clip x
        x = np.clip(x, self.lower_bounds, self.upper_bounds)

        # Clip y
        y = np.clip(y, self.lower_bounds, self.upper_bounds)

        # Store clipping info in meta (useful for debugging)
        meta["clipped"] = True
        meta["clip_method"] = self.method

        return x, y, meta

    def inverse(self, x: np.ndarray, y: Optional[np.ndarray] = None):
        """Inverse clipping (no-op, information is lost).

        Args:
            x: Clipped input
            y: Clipped target

        Returns:
            (x, y) unchanged (clipping is irreversible)
        """
        # Clipping is irreversible - return as-is
        return x, y
