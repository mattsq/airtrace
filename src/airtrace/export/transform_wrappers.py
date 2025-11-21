"""PyTorch nn.Module wrappers for transforms to enable ONNX export."""

from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import numpy as np


class ZScoreWrapper(nn.Module):
    """PyTorch wrapper for ZScore transform to enable ONNX export.

    This module applies z-score normalization: (x - mean) / std
    """

    def __init__(self, mean: np.ndarray, std: np.ndarray, epsilon: float = 1e-8):
        """Initialize z-score wrapper.

        Args:
            mean: Mean values for each feature [D,]
            std: Standard deviation for each feature [D,]
            epsilon: Small constant for numerical stability
        """
        super().__init__()
        # Register as buffers so they're included in state_dict but not trained
        self.register_buffer('mean', torch.from_numpy(mean).float())
        self.register_buffer('std', torch.from_numpy(std).float())
        self.register_buffer('epsilon', torch.tensor(epsilon))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply z-score normalization.

        Args:
            x: Input tensor [..., D]

        Returns:
            Normalized tensor [..., D]
        """
        return (x - self.mean) / (self.std + self.epsilon)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """Apply inverse z-score normalization.

        Args:
            x: Normalized tensor [..., D]

        Returns:
            Original scale tensor [..., D]
        """
        return x * (self.std + self.epsilon) + self.mean


class RobustScalerWrapper(nn.Module):
    """PyTorch wrapper for RobustScaler transform to enable ONNX export.

    This module applies robust scaling: (x - center) / scale
    where center is the median and scale is the IQR.
    """

    def __init__(self, center: np.ndarray, scale: np.ndarray, epsilon: float = 1e-8):
        """Initialize robust scaler wrapper.

        Args:
            center: Center values (median) for each feature [D,]
            scale: Scale values (IQR) for each feature [D,]
            epsilon: Small constant for numerical stability
        """
        super().__init__()
        self.register_buffer('center', torch.from_numpy(center).float())
        self.register_buffer('scale', torch.from_numpy(scale).float())
        self.register_buffer('epsilon', torch.tensor(epsilon))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply robust scaling.

        Args:
            x: Input tensor [..., D]

        Returns:
            Scaled tensor [..., D]
        """
        return (x - self.center) / (self.scale + self.epsilon)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """Apply inverse robust scaling.

        Args:
            x: Scaled tensor [..., D]

        Returns:
            Original scale tensor [..., D]
        """
        return x * (self.scale + self.epsilon) + self.center


class TransformCompose(nn.Module):
    """Composes multiple transform wrappers."""

    def __init__(self, transforms: list):
        """Initialize composed transforms.

        Args:
            transforms: List of transform nn.Module objects
        """
        super().__init__()
        self.transforms = nn.ModuleList(transforms)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply all transforms in sequence.

        Args:
            x: Input tensor

        Returns:
            Transformed tensor
        """
        for transform in self.transforms:
            x = transform(x)
        return x

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """Apply inverse transforms in reverse order.

        Args:
            x: Transformed tensor

        Returns:
            Original tensor
        """
        for transform in reversed(self.transforms):
            x = transform.inverse(x)
        return x


def create_transform_wrapper(transform_stats: Dict[str, Any], transform_class_name: str) -> Optional[nn.Module]:
    """Create a PyTorch wrapper for a transform from its statistics.

    Args:
        transform_stats: Dictionary containing transform statistics
        transform_class_name: Name of the transform class (e.g., 'ZScoreTransform_0')

    Returns:
        PyTorch nn.Module wrapper or None if transform is not supported
    """
    # Extract the base class name without the index
    base_name = transform_class_name.rsplit('_', 1)[0]

    if base_name == 'ZScoreTransform':
        # For y-only transforms (typical for predictions), we only need the y scaler
        mean = transform_stats.get('scaler_y_mean')
        scale = transform_stats.get('scaler_y_scale')

        if mean is None or scale is None:
            return None

        return ZScoreWrapper(mean=mean, std=scale)

    elif base_name == 'RobustScalerTransform':
        # For robust scaler, sklearn stores center_ and scale_
        # We need to extract these from the transform stats
        # Note: This may need adjustment based on actual stats structure
        center = transform_stats.get('scaler_y_center')
        scale = transform_stats.get('scaler_y_scale')

        if center is None or scale is None:
            return None

        return RobustScalerWrapper(center=center, scale=scale)

    else:
        # Unsupported transform
        return None


def create_inverse_transform_pipeline(transform_stats: Dict[str, Any]) -> nn.Module:
    """Create a PyTorch module that applies inverse transforms.

    Args:
        transform_stats: Dictionary mapping transform names to their statistics

    Returns:
        PyTorch nn.Module that applies inverse transforms in reverse order
    """
    wrappers = []

    # Sort transform keys to maintain order
    sorted_keys = sorted(transform_stats.keys())

    for key in sorted_keys:
        wrapper = create_transform_wrapper(transform_stats[key], key)
        if wrapper is not None:
            wrappers.append(wrapper)

    if not wrappers:
        # Return identity module if no transforms
        return nn.Identity()

    return TransformCompose(wrappers)
