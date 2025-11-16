"""Compatibility helpers that make optional dependencies graceful."""
from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path
from typing import Any

import pandas as pd


def _parquet_engine_available() -> bool:
    """Return True if pandas can rely on a native parquet engine."""
    for engine in ("pyarrow", "fastparquet"):
        if importlib.util.find_spec(engine) is not None:
            return True
    return False


def _install_pickle_backed_parquet() -> None:
    """Install a pickle-backed fallback for pandas parquet helpers."""
    warnings.warn(
        "Neither `pyarrow` nor `fastparquet` is installed. Falling back to a pickle-based "
        "implementation for pandas parquet IO. Install `pyarrow` for optimal performance.",
        RuntimeWarning,
        stacklevel=2,
    )

    def _to_parquet_fallback(self: pd.DataFrame, path: str | Path, *args: Any, **kwargs: Any) -> None:
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        # ``to_pickle`` does not require external dependencies and preserves data fidelity
        # for the synthetic datasets used in tests.
        self.to_pickle(file_path, protocol=5)

    def _read_parquet_fallback(path: str | Path, *args: Any, **kwargs: Any) -> pd.DataFrame:
        file_path = Path(path)
        return pd.read_pickle(file_path)

    pd.DataFrame.to_parquet = _to_parquet_fallback  # type: ignore[assignment]
    pd.read_parquet = _read_parquet_fallback  # type: ignore[assignment]


if not _parquet_engine_available():
    _install_pickle_backed_parquet()
