"""AirTrace: Autoregressive modeling framework for aircraft sensor timeseries."""
from __future__ import annotations

__version__ = "0.1.0"

# Ensure optional dependency shims (like parquet fallbacks) are installed as soon as the
# package is imported. The module exposes no public symbols but performs the necessary
# side effects to keep local development frictionless.
from . import _compat as _compat

# Expose key submodules for convenient imports
from . import data, models, tasks, transforms, training

# Expose commonly used functions
from .models import build_model, list_models

__all__ = [
    "__version__",
    "data",
    "models",
    "tasks",
    "transforms",
    "training",
    "build_model",
    "list_models",
]
