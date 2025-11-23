"""Pytest configuration ensuring the `airtrace` package is importable during local runs."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd
import pytest

# Add the repository's ``src`` directory to ``sys.path`` so that ``import airtrace``
# works without requiring an editable installation before running pytest.
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


@pytest.fixture()
def minimal_data_root(tmp_path: Path) -> Generator[Path, None, None]:
    """Create a tiny on-disk dataset usable by integration-style tests."""

    data_root = tmp_path / "data"
    processed = data_root / "processed"
    metadata = data_root / "metadata"
    processed.mkdir(parents=True)
    metadata.mkdir(parents=True)

    # Minimal flight with two sensors and five timesteps.
    flight_id = "flight_1"
    flight_df = pd.DataFrame(
        {
            "s1": np.linspace(0.0, 1.0, num=5, dtype=np.float32),
            "s2": np.linspace(1.0, 2.0, num=5, dtype=np.float32),
        }
    )
    flight_df.to_parquet(processed / f"{flight_id}.parquet")

    index_df = pd.DataFrame({"flight_id": [flight_id], "start_idx": [0], "end_idx": [5]})
    index_df.to_parquet(metadata / "train.parquet")
    index_df.to_parquet(metadata / "val.parquet")
    index_df.to_parquet(metadata / "test.parquet")

    static_df = pd.DataFrame({"flight_id": [flight_id], "aircraft_type": ["Q400"]})
    static_df.to_csv(metadata / "q400_flight_metadata.csv", index=False)

    yield data_root
