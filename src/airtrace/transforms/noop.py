"""No-op (identity) transform."""

from typing import Any, Dict, Optional, Tuple

import numpy as np

from .base import Transform
from .registry import register


@register("noop")
class NoOpTransform(Transform):
    """Identity transform (does nothing).

    Useful for baseline experiments on raw data without requiring
    special handling in configs.
    """

    def __init__(self):
        """Initialize no-op transform."""
        super().__init__()

    def fit(self, dataset) -> "NoOpTransform":
        """Fit transform (no-op).

        Args:
            dataset: Dataset to fit on

        Returns:
            self for method chaining
        """
        self.is_fitted = True
        return self

    def __call__(
        self, x: np.ndarray, y: np.ndarray, meta: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Apply transform (no-op).

        Args:
            x: Input sequence [T_in, D_in]
            y: Target sequence [T_out, D_out]
            meta: Metadata dict

        Returns:
            Unchanged (x, y, meta)
        """
        return x, y, meta

    def inverse(self, x: np.ndarray, y: Optional[np.ndarray] = None):
        """Inverse transform (no-op).

        Args:
            x: Input
            y: Target

        Returns:
            Unchanged (x, y)
        """
        return x, y
