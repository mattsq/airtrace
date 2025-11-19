import numpy as np
import pandas as pd
import pytest
import torch

from airtrace.data.dataset import DataStore, SensorWindowDataset


class _DummyStore:
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame
        self.calls = []

    def get_full_window(self, flight_id: str, start_idx: int, end_idx: int, column_names):
        self.calls.append((flight_id, start_idx, end_idx, tuple(column_names)))
        window = self.frame.iloc[start_idx:end_idx][column_names].to_numpy(dtype=np.float32)
        return window, {"flight_id": flight_id, "start_idx": start_idx, "end_idx": end_idx}


class _SimpleWindowSpec:
    def __init__(self, input_len: int):
        self.input_len = input_len


def test_sensor_window_dataset_splits_and_applies_transforms():
    frame = pd.DataFrame({"s1": range(6), "s2": np.arange(6, 12)})
    dataset = SensorWindowDataset(
        index_df=pd.DataFrame({"flight_id": ["F1"], "start_idx": [0], "end_idx": [6]}),
        data_store=_DummyStore(frame),
        transforms=lambda x, y, meta: (x + 1.0, y - 1.0, {**meta, "t": True}),
        sensor_names=["s1"],
        target_sensors=["s2"],
        window_spec=_SimpleWindowSpec(input_len=4),
    )

    sample = dataset[0]

    assert torch.equal(sample["x"], torch.tensor([[1.0], [2.0], [3.0], [4.0]]))
    assert torch.equal(sample["y"], torch.tensor([[9.0], [10.0]]))
    assert sample["meta"]["t"] is True
    assert sample["meta"]["sensor_names"] == ["s1"]
    assert dataset.data_store.calls == [("F1", 0, 6, ("s1", "s2"))]


def test_sensor_window_dataset_falls_back_to_ratio_split_when_window_spec_absent():
    frame = pd.DataFrame({"s": range(10)})
    dataset = SensorWindowDataset(
        index_df=pd.DataFrame({"flight_id": ["F2"], "start_idx": [0], "end_idx": [10]}),
        data_store=_DummyStore(frame),
        sensor_names=["s"],
        target_sensors=["s"],
        window_spec=None,
    )

    sample = dataset[0]

    # 90% of 10 points becomes 9 samples for x and 1 for y.
    assert sample["x"].shape[0] == 9
    assert sample["y"].shape[0] == 1


@pytest.fixture
def parquet_datastore(tmp_path) -> DataStore:
    processed = tmp_path / "processed"
    metadata_dir = tmp_path / "metadata"
    processed.mkdir()
    metadata_dir.mkdir()

    flight_df = pd.DataFrame({
        "sensor_a": np.linspace(0.0, 1.0, num=8),
        "sensor_b": np.linspace(1.0, 2.0, num=8),
        "target": np.linspace(2.0, 3.0, num=8),
    })
    flight_id = "flight_001"
    flight_df.to_parquet(processed / f"{flight_id}.parquet", index=False)

    metadata = pd.DataFrame({
        "flight_id": [flight_id],
        "tail_number": ["C-GQFW"],
        "aircraft_type": ["Q400"],
    })
    metadata.to_csv(metadata_dir / "q400_flight_metadata.csv", index=False)

    return DataStore(tmp_path)


def test_datastore_get_full_window_includes_metadata_and_caches_indices(parquet_datastore):
    store = parquet_datastore
    window, meta = store.get_full_window("flight_001", 1, 4, ["sensor_a", "target"])

    assert window.shape == (3, 2)
    assert meta["tail_number"] == "C-GQFW"
    cache_key = ("flight_001", ("sensor_a", "target"))
    assert cache_key in store._column_index_cache

    with pytest.raises(ValueError):
        store.get_full_window("flight_001", 0, 2, ["missing_sensor"])


def test_datastore_get_window_uses_heuristic_split(parquet_datastore):
    store = parquet_datastore
    total_len = 8  # flight fixture writes 8 rows
    x, y, meta = store.get_window("flight_001", 0, total_len, ["sensor_a", "sensor_b"], ["target"])

    assert x.shape[0] == int(total_len * 0.9)
    assert y.shape[1] == 1
    assert meta["flight_id"] == "flight_001"


def test_datastore_clear_cache_empties_all_structures(parquet_datastore):
    store = parquet_datastore
    store.get_full_window("flight_001", 0, 2, ["sensor_a"])

    assert store._flight_cache
    assert store._column_index_cache
    assert store._column_positions_cache

    store.clear_cache()

    assert store._flight_cache == {}
    assert store._column_index_cache == {}
    assert store._column_positions_cache == {}
