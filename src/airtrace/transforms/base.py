"""Base classes for data transformations."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch


class Transform(ABC):
    """Base class for all data transforms.

    Transforms operate on (x, y, meta) tuples where:
    - x: input sequence [T_in, D_in]
    - y: target sequence [T_out, D_out]
    - meta: dictionary of metadata
    """

    def __init__(self):
        self.is_fitted = False

    @abstractmethod
    def fit(self, dataset) -> "Transform":
        """Fit transform parameters on a dataset.

        Args:
            dataset: Dataset object to fit on

        Returns:
            self for method chaining
        """
        pass

    @abstractmethod
    def __call__(
        self, x: np.ndarray, y: np.ndarray, meta: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Apply transform to a single sample.

        Args:
            x: Input sequence [T_in, D_in]
            y: Target sequence [T_out, D_out]
            meta: Metadata dictionary

        Returns:
            Transformed (x, y, meta) tuple
        """
        pass

    def inverse(
        self, x: np.ndarray, y: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Inverse transform (if applicable).

        Args:
            x: Transformed input sequence
            y: Transformed target sequence (optional)

        Returns:
            Original (x, y) tuple
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support inverse transform")


class Compose:
    """Compose multiple transforms together."""

    def __init__(self, transforms):
        """Initialize composed transform.

        Args:
            transforms: List of Transform objects
        """
        self.transforms = transforms

    def fit(self, dataset):
        """Fit all transforms in sequence.

        Args:
            dataset: Dataset to fit on

        Returns:
            self for method chaining
        """
        for transform in self.transforms:
            transform.fit(dataset)
        return self

    def __call__(self, x, y, meta):
        """Apply all transforms in sequence.

        Args:
            x: Input sequence
            y: Target sequence
            meta: Metadata

        Returns:
            Transformed (x, y, meta)
        """
        for transform in self.transforms:
            x, y, meta = transform(x, y, meta)
        return x, y, meta

    def inverse(self, x, y=None):
        """Apply inverse transforms in reverse order.

        Args:
            x: Transformed input
            y: Transformed target

        Returns:
            Original (x, y)
        """
        for transform in reversed(self.transforms):
            x, y = transform.inverse(x, y)
        return x, y

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics from all transforms for caching.

        Returns:
            Dictionary mapping transform names to their statistics
        """
        stats = {}
        for i, transform in enumerate(self.transforms):
            transform_name = f"{transform.__class__.__name__}_{i}"
            if hasattr(transform, 'get_stats'):
                stats[transform_name] = transform.get_stats()
        return stats

    def set_stats(self, stats: Dict[str, Any]) -> None:
        """Set statistics for all transforms from cache.

        Args:
            stats: Dictionary mapping transform names to their statistics
        """
        for i, transform in enumerate(self.transforms):
            transform_name = f"{transform.__class__.__name__}_{i}"
            if transform_name in stats and hasattr(transform, 'set_stats'):
                transform.set_stats(stats[transform_name])

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n    {0}'.format(t)
        format_string += '\n)'
        return format_string
