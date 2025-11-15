"""Parquet compatibility helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd

_PATCH_APPLIED = False


def ensure_parquet_support() -> None:
    """Ensure ``pandas`` has working ``to_parquet``/``read_parquet`` helpers."""

    global _PATCH_APPLIED

    if _PATCH_APPLIED:
        return

    # If either optional engine is available we can rely on pandas' native
    # parquet implementation without any patching.
    if _parquet_engine_available():
        _PATCH_APPLIED = True
        return

    _apply_fallback_patch()
    _PATCH_APPLIED = True


def _parquet_engine_available() -> bool:
    """Return ``True`` if pandas can use an optional parquet engine."""

    try:  # pragma: no cover - simply checks import availability
        import pyarrow  # type: ignore  # noqa: F401

        return True
    except ImportError:
        try:
            import fastparquet  # type: ignore  # noqa: F401

            return True
        except ImportError:
            return False


def _apply_fallback_patch() -> None:
    """Patch pandas parquet helpers to fall back to pickle serialization."""

    dataframe_cls = pd.DataFrame
    original_to_pickle: Callable[..., None] = dataframe_cls.to_pickle
    original_read_pickle: Callable[..., pd.DataFrame] = pd.read_pickle

    def _to_parquet_fallback(
        self: pd.DataFrame,
        path: Any,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Serialize the frame using pickle when parquet engines are missing."""

        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Remove parquet-only arguments that pickle does not understand.
        kwargs.pop("engine", None)
        kwargs.pop("index", None)

        original_to_pickle(self, path_obj, **kwargs)

    def _read_parquet_fallback(
        path: Any,
        *args: Any,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Read pickled data when parquet engines are unavailable."""

        path_obj = Path(path)

        # ``columns`` is parquet-specific; emulate it manually for pickle data.
        columns: Optional[list[str]] = kwargs.pop("columns", None)
        kwargs.pop("engine", None)

        df = original_read_pickle(path_obj, **kwargs)
        if columns is not None:
            df = df[columns]
        return df

    dataframe_cls.to_parquet = _to_parquet_fallback  # type: ignore[assignment]
    pd.read_parquet = _read_parquet_fallback  # type: ignore[assignment]

