import importlib
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import pytest

import airtrace._compat as compat


def test_parquet_engine_available(monkeypatch):
    def fake_find_spec(name):
        return object() if name == "pyarrow" else None

    monkeypatch.setattr(compat.importlib.util, "find_spec", fake_find_spec)
    assert compat._parquet_engine_available() is True


def test_pickle_backed_parquet_installed(monkeypatch):
    original_to_parquet = pd.DataFrame.to_parquet
    original_read_parquet = pd.read_parquet

    monkeypatch.setattr(compat.importlib.util, "find_spec", lambda name: None)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.reload(compat)

    assert any("Falling back" in str(w.message) for w in caught)

    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "data.parquet"
        df = pd.DataFrame({"value": [1, 2, 3]})
        df.to_parquet(path)
        loaded = pd.read_parquet(path)
        pd.testing.assert_frame_equal(df, loaded)

    # Restore original parquet helpers
    monkeypatch.setattr(pd.DataFrame, "to_parquet", original_to_parquet, raising=False)
    monkeypatch.setattr(pd, "read_parquet", original_read_parquet, raising=False)
    importlib.reload(compat)
