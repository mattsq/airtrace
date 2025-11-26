import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import pyarrow.parquet as pq
import pytest

from airtrace.data.ingest.indexer import WindowIndexer


def _make_flight(path: Path, length: int) -> None:
    df = pd.DataFrame({"sensor": range(length)})
    df.to_parquet(path, engine="pyarrow", index=False)


def test_get_length_falls_back_and_caches(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    flight_path = tmp_path / "flight.parquet"
    _make_flight(flight_path, length=5)

    calls: List[str] = []

    def fail_read_metadata(_: Path) -> None:
        calls.append("pyarrow")
        raise RuntimeError("metadata read failed")

    def stub_read_parquet(_: Path, columns=None):
        calls.append("pandas")
        return pd.DataFrame({"sensor": range(5)})

    monkeypatch.setattr("airtrace.data.ingest.indexer.pq.read_metadata", fail_read_metadata)
    monkeypatch.setattr("airtrace.data.ingest.indexer.pd.read_parquet", stub_read_parquet)

    indexer = WindowIndexer(input_len=2, pred_len=1, stride=1, processed_dir=tmp_path)

    assert indexer._get_flight_length(flight_path) == 5
    assert indexer._get_flight_length(flight_path) == 5
    # pq.read_metadata should only be invoked on the first call because the result is cached
    assert calls == ["pyarrow", "pandas"]


@pytest.mark.parametrize("partitioned", [True, False])
def test_create_index_streaming_output(tmp_path: Path, partitioned: bool) -> None:
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    _make_flight(processed_dir / "f1.parquet", length=6)

    indexer = WindowIndexer(input_len=2, pred_len=2, stride=2, processed_dir=processed_dir)

    output_path = tmp_path / "index_out"
    index_df, count = indexer.create_index(
        ["f1"],
        split_name="train",
        num_workers=1,
        output_path=output_path,
        partitioned=partitioned,
        materialize_dataframe=False,
        return_count=True,
    )

    assert count == 2
    assert index_df.empty

    if partitioned:
        partition_dirs = list(output_path.glob("flight_id=*"))
        assert partition_dirs, "Expected partitioned output with flight_id directory"
        table = pq.read_table(output_path)
    else:
        assert output_path.exists()
        table = pq.read_table(output_path)

    assert table.num_rows == 2
    assert table.column("start_idx").to_pylist() == [0, 2]
    assert table.column("end_idx").to_pylist() == [4, 6]


def test_create_all_indices_reuses_metadata(tmp_path: Path) -> None:
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    _make_flight(processed_dir / "train.parquet", length=8)
    _make_flight(processed_dir / "val.parquet", length=7)
    _make_flight(processed_dir / "test.parquet", length=6)

    processed_meta: Dict[str, Dict[str, object]] = {
        "train": {"length": 8, "source_signature": {"checksum": "abc"}},
        "val": {"length": 7, "source_signature": {"checksum": "def"}},
        "test": {"length": 6, "source_signature": {"checksum": "ghi"}},
    }

    indexer = WindowIndexer(input_len=3, pred_len=2, stride=1, processed_dir=processed_dir, metadata_dir=tmp_path)
    output_dir = tmp_path / "indices"
    first_train_path, _, _, counts = indexer.create_all_indices(
        train_ids=["train"],
        val_ids=["val"],
        test_ids=["test"],
        output_dir=output_dir,
        dataset_name="demo",
        processed_metadata=processed_meta,
    )

    first_mtime = first_train_path.stat().st_mtime
    assert counts == {"train": 4, "val": 3, "test": 2}

    # Remove processed data to ensure reuse path does not touch flight files
    for file_path in processed_dir.glob("*.parquet"):
        file_path.unlink()

    indexer_reuse = WindowIndexer(input_len=3, pred_len=2, stride=1, processed_dir=processed_dir, metadata_dir=tmp_path)
    reuse_train_path, _, _, reuse_counts = indexer_reuse.create_all_indices(
        train_ids=["train"],
        val_ids=["val"],
        test_ids=["test"],
        output_dir=output_dir,
        dataset_name="demo",
        processed_metadata=processed_meta,
    )

    assert reuse_counts == {"train": 4, "val": 3, "test": 2}
    assert reuse_train_path == first_train_path
    assert reuse_train_path.stat().st_mtime == first_mtime

    meta_path = tmp_path / "demo_index_meta.json"
    assert json.loads(meta_path.read_text()).get("window_counts") == reuse_counts
