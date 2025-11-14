"""Registry for tasks."""

from typing import Any, Dict

from .base import Task

TASK_REGISTRY: Dict[str, type] = {}


def register(name: str):
    """Decorator to register a task class.

    Args:
        name: Name to register the task under

    Returns:
        Decorator function

    Example:
        >>> @register("one_step")
        >>> class OneStepTask(Task):
        ...     pass
    """

    def decorator(cls):
        if name in TASK_REGISTRY:
            raise ValueError(f"Task '{name}' is already registered")
        if not issubclass(cls, Task):
            raise TypeError(f"Class {cls.__name__} must inherit from Task")
        TASK_REGISTRY[name] = cls
        return cls

    return decorator


def build_task(config: Dict[str, Any]) -> Task:
    """Build a task from a config dictionary.

    Args:
        config: Task config with 'name' and other task-specific params
            Example: {"name": "one_step", "loss": "mse", "metrics": ["rmse"]}

    Returns:
        Instantiated task

    Raises:
        ValueError: If task name not found in registry
    """
    name = config["name"]
    if name not in TASK_REGISTRY:
        available = ", ".join(TASK_REGISTRY.keys())
        raise ValueError(
            f"Task '{name}' not found in registry. "
            f"Available tasks: {available}"
        )

    cls = TASK_REGISTRY[name]
    return cls(config)


def list_tasks():
    """List all registered tasks.

    Returns:
        List of registered task names
    """
    return sorted(TASK_REGISTRY.keys())
