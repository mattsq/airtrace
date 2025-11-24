from pathlib import Path

import pandas as pd
import pytest

from airtrace.data.ingest.config_gen import ConfigGenerator
from airtrace.data.ingest.indexer import WindowIndexer
from airtrace.data.ingest.processor import FlightProcessResult, FlightProcessor
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


def test_window_indexer_create_all_indices(tmp_path):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()

    train_df = pd.DataFrame(
        {"s1": range(8)}, index=pd.date_range("2024-01-01", periods=8, freq="s")
    )
    val_df = pd.DataFrame(
        {"s1": range(6)}, index=pd.date_range("2024-01-02", periods=6, freq="s")
    )
    test_df = pd.DataFrame(
        {"s1": range(7)}, index=pd.date_range("2024-01-03", periods=7, freq="s")
    )

    train_df.to_parquet(processed_dir / "train.parquet", engine="pyarrow")
    val_df.to_parquet(processed_dir / "val.parquet", engine="pyarrow")
    test_df.to_parquet(processed_dir / "test.parquet", engine="pyarrow")

    indexer = WindowIndexer(input_len=3, pred_len=2, stride=2, processed_dir=processed_dir)
    (
        train_path,
        val_path,
        test_path,
        train_len,
        val_len,
        test_len,
    ) = indexer.create_all_indices(
        train_ids=["train"], val_ids=["val"], test_ids=["test"], output_dir=tmp_path, dataset_name="demo"
    )

    assert train_path.exists()
    assert val_path.exists()
    assert test_path.exists()

    assert train_len == 2
    assert val_len == 1
    assert test_len == 2

    train_index = pd.read_parquet(train_path)
    assert len(train_index) == train_len
    assert train_index.iloc[0].to_dict() == {"flight_id": "train", "start_idx": 0, "end_idx": 5}


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


def test_flight_processor_handles_multiflight_and_empty_result(tmp_path):
    output_dir = tmp_path / "processed"
    processor = FlightProcessor(
        output_dir=output_dir,
        sensors=["a"],
        timestamp_column="time",
        resample_rate="1S",
    )

    multi_flight_path = tmp_path / "multi.parquet"
    df = pd.DataFrame(
        {
            "time": pd.date_range("2024-01-01", periods=3, freq="1s").repeat(2),
            "flight_id": ["x", "x", "x", "y", "y", "y"],
            "a": [None, None, None, None, None, None],
        }
    )
    df.to_parquet(multi_flight_path, engine="pyarrow")

    output = processor.process_flight("y", (multi_flight_path, "flight_id", "y"))
    assert output is None


def test_flight_processor_missing_sensors_returns_none(tmp_path):
    output_dir = tmp_path / "processed"
    processor = FlightProcessor(
        output_dir=output_dir,
        sensors=["missing"],
        timestamp_column="time",
    )

    flight_path = tmp_path / "flight.parquet"
    pd.DataFrame({"time": pd.date_range("2024-01-01", periods=2, freq="1s"), "present": [1, 2]}).to_parquet(
        flight_path, engine="pyarrow"
    )

    output = processor.process_flight("flight", flight_path)
    assert output is None
    assert not (output_dir / "flight.parquet").exists()


def test_flight_processor_returns_result_metadata(tmp_path):
    output_dir = tmp_path / "processed"
    processor = FlightProcessor(
        output_dir=output_dir,
        sensors=["a"],
        timestamp_column="time",
    )

    flight_path = tmp_path / "flight.parquet"
    pd.DataFrame({"time": pd.date_range("2024-01-01", periods=4, freq="1s"), "a": range(4)}).to_parquet(
        flight_path, engine="pyarrow"
    )

    result = processor.process_flight("flight", flight_path)
    assert isinstance(result, FlightProcessResult)
    assert result.length == 4
    assert result.output_path.exists()


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


def test_flight_validator_reports_missing_path(tmp_path):
    missing_path = tmp_path / "does_not_exist"
    validator = FlightValidator(input_path=missing_path)
    report = validator.validate()
    assert report.has_errors()
    assert "does not exist" in report.errors[0]


def test_flight_validator_no_parquet_files(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    validator = FlightValidator(input_path=empty_dir)
    report = validator.validate()
    assert report.has_errors()
    assert "No parquet files" in report.errors[0]


def test_flight_validator_invalid_timestamp_column(tmp_path):
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    file_path = data_dir / "flight.parquet"
    pd.DataFrame({"timestamp": [1, 2, 3], "sensor": [0.1, 0.2, 0.3]}).to_parquet(file_path, engine="pyarrow")

    validator = FlightValidator(input_path=data_dir, timestamp_column="timestamp")
    report = validator.validate()
    assert report.has_errors()
    assert "timestamp" in report.errors[0]


def test_flight_validator_detect_flights_with_id_column(tmp_path):
    file_path = tmp_path / "flights.parquet"
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=4, freq="1s"),
            "flight_id": ["a", "a", "b", "b"],
            "sensor": [1, 2, 3, 4],
        }
    ).to_parquet(file_path, engine="pyarrow")

    validator = FlightValidator(input_path=file_path, flight_id_column="flight_id")
    registry = validator.detect_flights()
    assert set(registry.keys()) == {"a", "b"}
    assert registry["a"] == (file_path, "flight_id", "a")


def test_flight_validator_quality_checks_detect_issues():
    validator = FlightValidator(input_path=Path("/tmp/unused"))
    timestamps = pd.to_datetime(
        [
            "2024-01-01 00:00:00",
            "2024-01-01 00:00:01",
            "2024-01-01 00:00:01",  # duplicate
            "2024-01-01 00:00:03",
            "2024-01-01 00:00:07",
        ]
    )
    df = pd.DataFrame({"timestamp": timestamps, "sensor": [None, None, None, None, None]})

    validator._check_data_quality(df, sensors=["sensor"])

    assert any("only NaN" in err for err in validator.report.errors)
    assert any("duplicate" in warn for warn in validator.report.warnings)
    assert any("Irregular sampling" in warn for warn in validator.report.warnings)

