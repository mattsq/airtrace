import pandas as pd
import pytest

from airtrace.data.loaders import InterimDataProcessor, RawDataLoader


@pytest.fixture(autouse=True)
def _patch_parquet_io(monkeypatch):
    """Patch parquet IO to avoid optional dependencies in tests.

    The production code uses pandas parquet helpers, which require optional
    engines like ``pyarrow``. The project environment used for tests does not
    include those extras, so we patch parquet helpers to write/read CSV data
    under the same file names. The patch is reverted automatically after each
    test via the ``monkeypatch`` fixture.
    """

    def _to_parquet(self, path, *_, **__):
        # Write CSV content while keeping the expected extension.
        return self.to_csv(path, index=True)

    def _read_parquet(path, *_, **__):
        return pd.read_csv(path, index_col=0)

    monkeypatch.setattr(pd.DataFrame, "to_parquet", _to_parquet)
    monkeypatch.setattr(pd, "read_parquet", _read_parquet)


def test_load_raw_flight_prefers_csv(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    loader = RawDataLoader(tmp_path)

    csv_path = raw_dir / "flight123.csv"
    sample_df = pd.DataFrame({"timestamp": ["2024-01-01T00:00:00"], "value": [1.0]})
    sample_df.to_csv(csv_path, index=False)

    loaded = loader.load_raw_flight("flight123")

    assert list(loaded.columns) == ["timestamp", "value"]
    assert loaded.iloc[0]["value"] == pytest.approx(1.0)


def test_load_raw_flight_raises_for_missing_file(tmp_path):
    loader = RawDataLoader(tmp_path)
    with pytest.raises(FileNotFoundError):
        loader.load_raw_flight("does_not_exist")


def test_process_to_interim_resamples_and_filters(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    loader = RawDataLoader(tmp_path)

    raw_records = [
        {"timestamp": "2024-01-01T00:00:00", "sensor_name": "temp", "value": 10.0},
        {"timestamp": "2024-01-01T00:00:00", "sensor_name": "alt", "value": 1000.0},
        {"timestamp": "2024-01-01T00:00:02", "sensor_name": "temp", "value": 12.0},
    ]
    pd.DataFrame(raw_records).to_csv(raw_dir / "flight123.csv", index=False)

    interim_df = loader.process_to_interim(
        flight_id="flight123", resample_rate="1S", sensor_list=["temp", "unknown"]
    )

    assert list(interim_df.columns) == ["temp"]
    assert interim_df.index.inferred_type == "datetime64"
    assert list(interim_df.index) == list(
        pd.to_datetime(["2024-01-01T00:00:00", "2024-01-01T00:00:01", "2024-01-01T00:00:02"])
    )
    assert interim_df["temp"].tolist() == [10.0, 10.0, 12.0]


def test_create_windows_combines_flights_and_saves_index(tmp_path):
    processor = InterimDataProcessor(tmp_path)

    flights = {
        "A": pd.DataFrame({"a": [1, 2]}),
        "B": pd.DataFrame({"a": [3, 4]}),
    }
    processor.load_interim_flight = lambda flight_id: flights[flight_id]

    class DummyWindowSpec:
        def __init__(self):
            self.calls = []

        def create_windows(self, data, flight_id):
            self.calls.append((flight_id, data.shape))
            return pd.DataFrame({"flight_id": [flight_id], "start": [0], "end": [data.shape[0] - 1]})

    window_spec = DummyWindowSpec()

    index_df = processor.create_windows(["A", "B"], window_spec, output_name="combined")

    assert index_df.shape == (2, 3)
    assert set(index_df["flight_id"]) == {"A", "B"}
    assert window_spec.calls == [("A", (2, 1)), ("B", (2, 1))]

    saved_index = tmp_path / "metadata" / "combined_index.parquet"
    assert saved_index.exists()
    saved_df = pd.read_csv(saved_index)
    assert set(saved_df["flight_id"]) == {"A", "B"}
