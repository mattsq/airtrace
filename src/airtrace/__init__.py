"""AirTrace package initialization."""
from __future__ import annotations

# Ensure optional dependency shims (like parquet fallbacks) are installed as soon as the
# package is imported. The module exposes no public symbols but performs the necessary
# side effects to keep local development frictionless.
from . import _compat as _compat

__all__ = ["_compat"]
