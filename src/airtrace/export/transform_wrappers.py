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


class DifferenceWrapper(nn.Module):
    """PyTorch wrapper for DifferenceTransform to enable ONNX export.

    Applies first-order differencing: x[t] - x[t-1]
    """

    def __init__(self, order: int = 1):
        """Initialize difference wrapper.

        Args:
            order: Order of differencing (1=first difference, 2=second, etc.)
        """
        super().__init__()
        self.order = order

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply differencing along time axis.

        Args:
            x: Input tensor [B, T, D] (batched) or [T, D] (unbatched)

        Returns:
            Differenced tensor with same shape (padded to maintain length)
        """
        # Determine if batched (3D) or unbatched (2D)
        is_batched = x.dim() == 3

        x_diff = x
        for _ in range(self.order):
            # Diff along time dimension
            if is_batched:
                # [B, T, D] -> [B, T-1, D]
                diff = x_diff[:, 1:, :] - x_diff[:, :-1, :]
                # Pad at beginning: [B, 1, D] + [B, T-1, D] -> [B, T, D]
                x_diff = torch.cat([x_diff[:, :1, :], diff], dim=1)
            else:
                # [T, D] -> [T-1, D]
                diff = x_diff[1:, :] - x_diff[:-1, :]
                # Pad at beginning: [1, D] + [T-1, D] -> [T, D]
                x_diff = torch.cat([x_diff[:1, :], diff], dim=0)
        return x_diff

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """Apply inverse differencing (cumulative sum) along time axis.

        Args:
            x: Differenced tensor [B, T, D] (batched) or [T, D] (unbatched)

        Returns:
            Original scale tensor with same shape
        """
        # Determine if batched (3D) or unbatched (2D)
        is_batched = x.dim() == 3
        time_dim = 1 if is_batched else 0

        x_inv = x
        for _ in range(self.order):
            x_inv = torch.cumsum(x_inv, dim=time_dim)
        return x_inv


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


def create_transform_wrapper(
    transform_stats: Dict[str, Any],
    transform_class_name: str,
    use_x_scaler: bool = False
) -> Optional[nn.Module]:
    """Create a PyTorch wrapper for a transform from its statistics.

    Args:
        transform_stats: Dictionary containing transform statistics
        transform_class_name: Name of the transform class (e.g., 'ZScoreTransform_0')
        use_x_scaler: If True, use scaler_x (for preprocessing inputs), else scaler_y (for postprocessing outputs)

    Returns:
        PyTorch nn.Module wrapper or None if transform is not supported
    """
    # Extract the base class name without the index
    base_name = transform_class_name.rsplit('_', 1)[0]

    if base_name == 'ZScoreTransform':
        # Choose appropriate scaler based on use case
        prefix = 'scaler_x' if use_x_scaler else 'scaler_y'
        mean = transform_stats.get(f'{prefix}_mean')
        scale = transform_stats.get(f'{prefix}_scale')

        if mean is None or scale is None:
            return None

        return ZScoreWrapper(mean=mean, std=scale)

    elif base_name == 'RobustScalerTransform':
        # For robust scaler, sklearn stores center_ and scale_
        prefix = 'scaler_x' if use_x_scaler else 'scaler_y'
        center = transform_stats.get(f'{prefix}_center')
        scale = transform_stats.get(f'{prefix}_scale')

        if center is None or scale is None:
            return None

        return RobustScalerWrapper(center=center, scale=scale)

    elif base_name == 'DifferenceTransform':
        order = transform_stats.get('order', 1)
        return DifferenceWrapper(order=order)

    else:
        # Unsupported transform
        return None


def _extract_transform_index(transform_name: str) -> int:
    """Extract the numeric index from a transform name.

    Args:
        transform_name: Transform name like 'ZScoreTransform_0'

    Returns:
        The numeric index
    """
    try:
        return int(transform_name.rsplit('_', 1)[1])
    except (ValueError, IndexError):
        return 0


def create_forward_transform_pipeline(transform_stats: Dict[str, Any]) -> nn.Module:
    """Create a PyTorch module that applies forward transforms for preprocessing.

    Args:
        transform_stats: Dictionary mapping transform names to their statistics
                        (keys should be in the format 'TransformName_N' where N is the order)

    Returns:
        PyTorch nn.Module that applies forward transforms in pipeline order
    """
    wrappers = []

    # Sort by the numeric index to preserve pipeline order (not alphabetically!)
    sorted_keys = sorted(transform_stats.keys(), key=_extract_transform_index)

    for key in sorted_keys:
        wrapper = create_transform_wrapper(transform_stats[key], key, use_x_scaler=True)
        if wrapper is not None:
            wrappers.append(wrapper)

    if not wrappers:
        # Return identity module if no transforms
        return nn.Identity()

    return TransformCompose(wrappers)


def create_inverse_transform_pipeline(transform_stats: Dict[str, Any]) -> nn.Module:
    """Create a PyTorch module that applies inverse transforms for postprocessing.

    Args:
        transform_stats: Dictionary mapping transform names to their statistics
                        (keys should be in the format 'TransformName_N' where N is the order)

    Returns:
        PyTorch nn.Module that applies inverse transforms in reverse pipeline order
    """
    wrappers = []

    # Sort by the numeric index to preserve pipeline order (not alphabetically!)
    sorted_keys = sorted(transform_stats.keys(), key=_extract_transform_index)

    for key in sorted_keys:
        wrapper = create_transform_wrapper(transform_stats[key], key, use_x_scaler=False)
        if wrapper is not None:
            wrappers.append(wrapper)

    if not wrappers:
        # Return identity module if no transforms
        return nn.Identity()

    return TransformCompose(wrappers)
