
import unittest
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path
import pandas as pd
import numpy as np
import pyarrow.dataset as ds

from airtrace.data.ingest.validator import FlightValidator, SensorMetadata, ValidationReport

class TestFlightValidatorCoverage(unittest.TestCase):
    def setUp(self):
        self.input_path = Path("data/raw")
        self.validator = FlightValidator(self.input_path)

    def test_sensor_metadata_dataclass(self):
        meta = SensorMetadata(
            sensors=["s1"],
            timestamp_column="time",
            timestamp_dtype="datetime64[ns]",
            sampling_rate=1.0,
            sampling_std=0.0
        )
        self.assertEqual(meta.sensors, ["s1"])
        self.assertEqual(meta.dtypes, {})

    def test_validation_report(self):
        report = ValidationReport()
        self.assertFalse(report.has_errors())
        report.add_error("error")
        self.assertTrue(report.has_errors())
        report.add_warning("warning")
        self.assertEqual(report.warnings, ["warning"])

    @patch("pathlib.Path.exists", return_value=False)
    def test_validate_path_not_exists(self, mock_exists):
        report = self.validator.validate()
        self.assertTrue(report.has_errors())
        self.assertIn("Path does not exist", report.errors[0])

    @patch("pathlib.Path.exists", return_value=True)
    @patch("airtrace.data.ingest.validator.FlightValidator._get_parquet_files", return_value=[])
    def test_validate_no_files(self, mock_get_files, mock_exists):
        report = self.validator.validate()
        self.assertTrue(report.has_errors())
        self.assertIn("No parquet files found", report.errors[0])

    @patch("pathlib.Path.exists", return_value=True)
    @patch("airtrace.data.ingest.validator.FlightValidator._get_parquet_files")
    @patch("airtrace.data.ingest.validator.FlightValidator._sample_dataframe", side_effect=Exception("Load fail"))
    def test_validate_load_fail(self, mock_sample, mock_get_files, mock_exists):
        mock_get_files.return_value = [Path("f1.parquet")]
        report = self.validator.validate()
        self.assertTrue(report.has_errors())
        self.assertIn("Failed to load", report.errors[0])

    @patch("pathlib.Path.exists", return_value=True)
    @patch("airtrace.data.ingest.validator.FlightValidator._get_parquet_files")
    @patch("airtrace.data.ingest.validator.FlightValidator._sample_dataframe")
    def test_validate_no_timestamp(self, mock_sample, mock_get_files, mock_exists):
        mock_get_files.return_value = [Path("f1.parquet")]
        mock_sample.return_value = pd.DataFrame({"s1": [1, 2, 3]}) # No timestamp
        
        report = self.validator.validate()
        self.assertTrue(report.has_errors())
        self.assertIn("No timestamp column found", report.errors[0])

    @patch("pathlib.Path.exists", return_value=True)
    @patch("airtrace.data.ingest.validator.FlightValidator._get_parquet_files")
    @patch("airtrace.data.ingest.validator.FlightValidator._sample_dataframe")
    def test_validate_no_sensors(self, mock_sample, mock_get_files, mock_exists):
        mock_get_files.return_value = [Path("f1.parquet")]
        # Create DF with timestamp but no numeric cols
        df = pd.DataFrame({"time": pd.date_range("2021-01-01", periods=3)})
        mock_sample.return_value = df
        
        report = self.validator.validate()
        self.assertTrue(report.has_errors())
        self.assertIn("No numeric sensor columns detected", report.errors[0])

    @patch("pathlib.Path.exists", return_value=True)
    @patch("airtrace.data.ingest.validator.FlightValidator._get_parquet_files")
    @patch("airtrace.data.ingest.validator.FlightValidator._sample_dataframe")
    def test_validate_success(self, mock_sample, mock_get_files, mock_exists):
        mock_get_files.return_value = [Path("f1.parquet")]
        df = pd.DataFrame({
            "time": pd.date_range("2021-01-01", periods=10, freq="1s"),
            "s1": np.random.randn(10)
        })
        mock_sample.return_value = df
        
        report = self.validator.validate()
        self.assertFalse(report.has_errors())

    @patch("airtrace.data.ingest.validator.FlightValidator._get_parquet_files", return_value=[])
    def test_detect_schema_no_files(self, mock_get_files):
        with self.assertRaises(ValueError):
            self.validator.detect_schema()

    @patch("airtrace.data.ingest.validator.FlightValidator._get_parquet_files")
    @patch("airtrace.data.ingest.validator.FlightValidator._sample_dataframe")
    def test_detect_schema_no_timestamp(self, mock_sample, mock_get_files):
        mock_get_files.return_value = [Path("f1.parquet")]
        mock_sample.return_value = pd.DataFrame({"s1": [1, 2, 3]})
        with self.assertRaises(ValueError):
            self.validator.detect_schema()

    @patch("airtrace.data.ingest.validator.FlightValidator._get_parquet_files")
    @patch("airtrace.data.ingest.validator.FlightValidator._sample_dataframe")
    def test_detect_schema_success(self, mock_sample, mock_get_files):
        mock_get_files.return_value = [Path("f1.parquet")]
        df = pd.DataFrame({
            "time": pd.date_range("2021-01-01", periods=10, freq="1s"),
            "s1": np.random.randn(10),
            "s2": [1.0] * 9 + [np.nan]
        })
        mock_sample.return_value = df
        
        meta = self.validator.detect_schema()
        self.assertEqual(meta.timestamp_column, "time")
        self.assertIn("s1", meta.sensors)
        self.assertIn("s2", meta.sensors)
        self.assertAlmostEqual(meta.sampling_rate, 1.0)
        self.assertEqual(meta.nan_percentages["s2"], 10.0)

    @patch("airtrace.data.ingest.validator.FlightValidator._get_parquet_files")
    @patch("airtrace.data.ingest.validator.FlightValidator._sample_dataframe")
    def test_detect_schema_timestamp_index(self, mock_sample, mock_get_files):
        mock_get_files.return_value = [Path("f1.parquet")]
        df = pd.DataFrame({
            "s1": np.random.randn(10)
        }, index=pd.date_range("2021-01-01", periods=10, freq="1s"))
        df.index.name = "timestamp"
        mock_sample.return_value = df
        
        meta = self.validator.detect_schema()
        self.assertEqual(meta.timestamp_column, "timestamp")

    def test_detect_flights_single_file(self):
        with patch("pathlib.Path.is_file", return_value=True):
            with patch("pathlib.Path.is_dir", return_value=False):
                with patch("airtrace.data.ingest.validator.FlightValidator._get_parquet_files", return_value=[Path("f1.parquet")]):
                    flights = self.validator.detect_flights()
                    self.assertEqual(flights, {"f1": Path("f1.parquet")})

    def test_detect_flights_directory(self):
        self.validator.input_path = Path("data") # is_dir = True
        with patch("pathlib.Path.is_file", return_value=False):
            with patch("pathlib.Path.is_dir", return_value=True):
                with patch("airtrace.data.ingest.validator.FlightValidator._get_parquet_files", return_value=[Path("f1.parquet"), Path("f2.parquet")]):
                    flights = self.validator.detect_flights()
                    self.assertEqual(len(flights), 2)
                    self.assertEqual(flights["f1"], Path("f1.parquet"))

    def test_detect_flights_grouped(self):
        self.validator.flight_id_column = "fid"
        self.validator.input_path = Path("flights.parquet")
        
        with patch("pathlib.Path.is_file", return_value=True):
            with patch("pathlib.Path.is_dir", return_value=False):
                with patch("airtrace.data.ingest.validator.FlightValidator._get_parquet_files", return_value=[Path("flights.parquet")]):
                    df = pd.DataFrame({"fid": ["f1", "f1", "f2"]})
                    with patch("pandas.read_parquet", return_value=df):
                        flights = self.validator.detect_flights()
                        self.assertEqual(len(flights), 2)
                        self.assertEqual(flights["f1"], (Path("flights.parquet"), "fid", "f1"))

    def test_get_parquet_files_file(self):
        validator = FlightValidator(Path("file.parquet"))
        with patch("pathlib.Path.is_file", return_value=True):
            files = validator._get_parquet_files()
            self.assertEqual(len(files), 1)

    def test_get_parquet_files_file_invalid_ext(self):
        validator = FlightValidator(Path("file.txt"))
        with patch("pathlib.Path.is_file", return_value=True):
            files = validator._get_parquet_files()
            self.assertEqual(len(files), 0)

    def test_get_parquet_files_dir(self):
        validator = FlightValidator(Path("dir"))
        with patch("pathlib.Path.is_file", return_value=False):
            with patch("pathlib.Path.is_dir", return_value=True):
                with patch("pathlib.Path.glob", side_effect=[[Path("1.parquet")], [Path("2.pq")]]):
                    files = validator._get_parquet_files()
                    self.assertEqual(len(files), 2)

    def test_get_parquet_files_invalid(self):
        validator = FlightValidator(Path("invalid"))
        with patch("pathlib.Path.is_file", return_value=False):
            with patch("pathlib.Path.is_dir", return_value=False):
                files = validator._get_parquet_files()
                self.assertEqual(len(files), 0)

    def test_validate_timestamp_specified_missing(self):
        validator = FlightValidator(Path("."), timestamp_column="missing")
        df = pd.DataFrame({"other": [1]})
        col = validator._validate_timestamp(df)
        self.assertIsNone(col)
        self.assertIn("Specified timestamp column", validator.report.errors[0])

    def test_detect_timestamp_ambiguous(self):
        df = pd.DataFrame({
            "t1": pd.date_range("2021", periods=1),
            "t2": pd.date_range("2022", periods=1)
        })
        # Should pick first if no common names match
        col = self.validator._detect_timestamp_column(df)
        self.assertEqual(col, "t1")

    def test_detect_timestamp_ambiguous_common(self):
        df = pd.DataFrame({
            "other_time": pd.date_range("2021", periods=1),
            "timestamp": pd.date_range("2022", periods=1)
        })
        col = self.validator._detect_timestamp_column(df)
        self.assertEqual(col, "timestamp")

    def test_detect_sensors_exclude_id(self):
        validator = FlightValidator(Path("."), flight_id_column="fid")
        df = pd.DataFrame({"s1": [1], "fid": [2], "time": pd.date_range("2021", periods=1)})
        sensors = validator._detect_sensors(df, "time")
        self.assertEqual(sensors, ["s1"])

    def test_check_data_quality_issues(self):
        df = pd.DataFrame({
            "s1": [np.nan, np.nan], # All NaN
            "s2": [1.0, np.nan],    # >20% NaN
            "time": pd.to_datetime(["2021-01-01 00:00:00", "2021-01-01 00:00:00"]) # Duplicate
        })
        self.validator._check_data_quality(df, ["s1", "s2"])
        
        self.assertTrue(any("only NaN values" in e for e in self.validator.report.errors))
        self.assertTrue(any("missing values" in w for w in self.validator.report.warnings))
        self.assertTrue(any("duplicate timestamps" in w for w in self.validator.report.warnings))

    def test_check_data_quality_irregular(self):
        # 0s, 1s, 10s -> median ~1, std high
        df = pd.DataFrame({
            "time": pd.to_datetime(["2021-01-01 00:00:00", "2021-01-01 00:00:01", "2021-01-01 00:00:10"])
        })
        self.validator._check_data_quality(df, [])
        self.assertTrue(any("Irregular sampling detected" in w for w in self.validator.report.warnings))

    @patch("pyarrow.dataset.dataset")
    def test_sample_dataframe_success(self, mock_ds):
        mock_scanner = MagicMock()
        mock_ds.return_value.scanner.return_value = mock_scanner
        
        batch1 = MagicMock()
        batch1.__len__.return_value = 100
        batch1.to_pandas.return_value = pd.DataFrame({"a": range(100)})
        
        mock_scanner.to_batches.return_value = [batch1]
        
        df = self.validator._sample_dataframe(Path("test.parquet"), sample_rows=100)
        self.assertEqual(len(df), 100)

    @patch("pyarrow.dataset.dataset")
    def test_sample_dataframe_partial(self, mock_ds):
        mock_scanner = MagicMock()
        mock_ds.return_value.scanner.return_value = mock_scanner
        
        batch1 = MagicMock()
        batch1.__len__.return_value = 100
        batch1.slice.return_value.to_pandas.return_value = pd.DataFrame({"a": range(50)})
        
        mock_scanner.to_batches.return_value = [batch1]
        
        df = self.validator._sample_dataframe(Path("test.parquet"), sample_rows=50)
        self.assertEqual(len(df), 50)
        batch1.slice.assert_called_with(0, 50)

    @patch("pyarrow.dataset.dataset", side_effect=Exception("DS fail"))
    @patch("pandas.read_parquet")
    def test_sample_dataframe_fallback(self, mock_read, mock_ds):
        mock_read.return_value = pd.DataFrame({"a": range(100)})
        df = self.validator._sample_dataframe(Path("test.parquet"), sample_rows=10)
        self.assertEqual(len(df), 10)
        mock_read.assert_called()

    @patch("pyarrow.dataset.dataset", side_effect=Exception("DS fail"))
    @patch("pandas.read_parquet", side_effect=Exception("Pandas fail"))
    def test_sample_dataframe_fail_all(self, mock_read, mock_ds):
        with self.assertRaises(Exception):
            self.validator._sample_dataframe(Path("test.parquet"))

    @patch("pyarrow.dataset.dataset")
    def test_sample_dataframe_full(self, mock_ds):
        mock_scanner = MagicMock()
        mock_ds.return_value.scanner.return_value = mock_scanner
        mock_scanner.to_table.return_value.to_pandas.return_value = pd.DataFrame({"a": range(200)})
        
        df = self.validator._sample_dataframe(Path("test.parquet"), sample_rows=0)
        self.assertEqual(len(df), 200)
        mock_scanner.to_table.assert_called()

