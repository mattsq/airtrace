"""Registry for transforms."""

from typing import Any, Dict, List

from .base import Compose, Transform

TRANSFORM_REGISTRY: Dict[str, type] = {}


def register(name: str):
    """Decorator to register a transform class.

    Args:
        name: Name to register the transform under

    Returns:
        Decorator function

    Example:
        >>> @register("zscore")
        >>> class ZScoreTransform(Transform):
        ...     pass
    """

    def decorator(cls):
        if name in TRANSFORM_REGISTRY:
            raise ValueError(f"Transform '{name}' is already registered")
        if not issubclass(cls, Transform):
            raise TypeError(f"Class {cls.__name__} must inherit from Transform")
        TRANSFORM_REGISTRY[name] = cls
        return cls

    return decorator


def build_transforms(config_list: List[Dict[str, Any]]) -> Compose:
    """Build a composed transform from a config list.

    Args:
        config_list: List of transform configs, each with 'name' and optional params
            Example: [{"name": "zscore", "per_sensor": true},
                     {"name": "diff", "order": 1}]

    Returns:
        Composed transform

    Raises:
        ValueError: If transform name not found in registry
    """
    transforms = []
    for cfg in config_list:
        name = cfg["name"]
        if name not in TRANSFORM_REGISTRY:
            available = ", ".join(TRANSFORM_REGISTRY.keys())
            raise ValueError(
                f"Transform '{name}' not found in registry. "
                f"Available transforms: {available}"
            )

        cls = TRANSFORM_REGISTRY[name]
        # Extract parameters (everything except 'name')
        params = {k: v for k, v in cfg.items() if k != "name"}
        transforms.append(cls(**params))

    return Compose(transforms)


def list_transforms() -> List[str]:
    """List all registered transforms.

    Returns:
        List of registered transform names
    """
    return sorted(TRANSFORM_REGISTRY.keys())
