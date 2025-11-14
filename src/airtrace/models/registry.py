"""Registry for models."""

from typing import Any, Dict

from .base import ARBaseModel

MODEL_REGISTRY: Dict[str, type] = {}


def register(name: str):
    """Decorator to register a model class.

    Args:
        name: Name to register the model under

    Returns:
        Decorator function

    Example:
        >>> @register("gru_ar")
        >>> class GRUARModel(ARBaseModel):
        ...     pass
    """

    def decorator(cls):
        if name in MODEL_REGISTRY:
            raise ValueError(f"Model '{name}' is already registered")
        if not issubclass(cls, ARBaseModel):
            raise TypeError(f"Class {cls.__name__} must inherit from ARBaseModel")
        MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def build_model(
    config: Dict[str, Any],
    input_dim: int,
    output_dim: int
) -> ARBaseModel:
    """Build a model from a config dictionary.

    Args:
        config: Model config with 'name' and 'params' keys
            Example: {"name": "gru_ar", "params": {"hidden_size": 128}}
        input_dim: Input feature dimension
        output_dim: Output prediction dimension

    Returns:
        Instantiated model

    Raises:
        ValueError: If model name not found in registry
    """
    name = config["name"]
    if name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Model '{name}' not found in registry. "
            f"Available models: {available}"
        )

    cls = MODEL_REGISTRY[name]
    params = config.get("params", {})
    return cls(input_dim=input_dim, output_dim=output_dim, **params)


def list_models():
    """List all registered models.

    Returns:
        List of registered model names
    """
    return sorted(MODEL_REGISTRY.keys())
