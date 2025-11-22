from pathlib import Path

import pandas as pd
import pytest

from airtrace.data.ingest.config_gen import ConfigGenerator
from airtrace.data.ingest.indexer import WindowIndexer
from airtrace.data.ingest.processor import FlightProcessor
from airtrace.data.ingest.validator import FlightValidator


def test_config_generator_respects_force(tmp_path):
    output_path = tmp_path / "dataset.yaml"
    generator = ConfigGenerator(
        dataset_name="demo",
        sensors=["a", "b"],
        input_len=12,
        pred_len=3,
        stride=2,
        target_sensors=["b"],
    )

    generator.generate(output_path, n_flights=5, input_path="/data/raw")
    first_content = output_path.read_text()
    assert "dataset_name" in first_content

    output_path.write_text("original")
    generator.generate(output_path)
    assert output_path.read_text() == "original"

    generator.generate(output_path, force=True)
    assert "demo" in output_path.read_text()


def test_window_indexer_skips_short_and_missing_flights(tmp_path):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    flight_path = processed_dir / "flight_a.parquet"
    short_path = processed_dir / "short.parquet"

    df = pd.DataFrame({"s1": range(10)}, index=pd.date_range("2024-01-01", periods=10, freq="s"))
    df.to_parquet(flight_path, engine="pyarrow")

    short_df = pd.DataFrame({"s1": range(3)}, index=pd.date_range("2024-01-01", periods=3, freq="s"))
    short_df.to_parquet(short_path, engine="pyarrow")

    indexer = WindowIndexer(input_len=3, pred_len=2, stride=2, processed_dir=processed_dir)

    index_df = indexer.create_index(["flight_a", "missing", "short"], split_name="train")
    assert index_df["flight_id"].unique().tolist() == ["flight_a"]
    assert index_df[["start_idx", "end_idx"]].values.tolist() == [[0, 5], [2, 7], [4, 9]]


def test_flight_processor_process_all_respects_min_length(tmp_path):
    output_dir = tmp_path / "processed"
    processor = FlightProcessor(
        output_dir=output_dir,
        sensors=["a", "b"],
        timestamp_column="time",
        resample_rate="1S",
        dataset_name="demo",
    )

    long_df = pd.DataFrame(
        {"time": pd.date_range("2024-01-01", periods=5, freq="1s"), "a": range(5), "b": range(5, 10)}
    )
    short_df = pd.DataFrame(
        {"time": pd.date_range("2024-01-01", periods=1, freq="1s"), "a": [1], "b": [2]}
    )

    registry = {
        "long": tmp_path / "long.parquet",
        "short": tmp_path / "short.parquet",
    }
    long_df.to_parquet(registry["long"], engine="pyarrow")
    short_df.to_parquet(registry["short"], engine="pyarrow")

    processed = processor.process_all(registry, min_length=3)
    assert processed == ["long"]

    saved_path = output_dir / "long.parquet"
    assert saved_path.exists()
    saved_df = pd.read_parquet(saved_path)
    assert list(saved_df.columns) == ["a", "b"]
    assert saved_df.index.name == "timestamp"
    assert not (output_dir / "short.parquet").exists()


def test_flight_processor_standardize_timestamp_errors_on_missing_column():
    processor = FlightProcessor(output_dir=Path("/tmp/airtrace"), sensors=["a"], timestamp_column="time")
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValueError):
        processor._standardize_timestamp(df)


def test_flight_validator_detects_schema_and_quality(tmp_path):
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=6, freq="1s"),
            "sensor_a": [1, 2, 3, None, None, 6],
            "sensor_b": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
        }
    )
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    file_path = data_dir / "flight.parquet"
    df.to_parquet(file_path, engine="pyarrow")

    validator = FlightValidator(input_path=data_dir)
    report = validator.validate()
    assert not report.has_errors()
    assert report.warnings

    metadata = validator.detect_schema()
    assert metadata.timestamp_column == "timestamp"
    assert set(metadata.sensors) == {"sensor_a", "sensor_b"}
    assert metadata.nan_percentages["sensor_a"] > 0

