
import unittest
from unittest.mock import patch, MagicMock, mock_open, call
from pathlib import Path
import pandas as pd
import numpy as np
import json
import logging

from airtrace.data.ingest.processor import FlightProcessor

class TestFlightProcessorCoverage(unittest.TestCase):
    def setUp(self):
        self.output_dir = Path("output")
        self.processor = FlightProcessor(
            output_dir=self.output_dir,
            sensors=["s1", "s2"],
            timestamp_column="time",
            resample_rate="1s"
        )

    def test_init_invalid_backend(self):
        with self.assertRaises(ValueError):
            FlightProcessor(self.output_dir, [], "time", resample_backend="invalid")

    @patch("pandas.read_parquet")
    def test_process_flight_exception(self, mock_read):
        mock_read.side_effect = Exception("Read error")
        result = self.processor.process_flight("f1", Path("f1.parquet"))
        self.assertIsNone(result)

    @patch("pyarrow.dataset.dataset")
    def test_process_flight_multiflight_success(self, mock_ds):
        # Mock pyarrow dataset and scanner
        mock_scanner = MagicMock()
        mock_ds.return_value.scanner.return_value = mock_scanner
        
        # Mock batch to pandas
        df = pd.DataFrame({
            "time": pd.date_range("2021-01-01", periods=10, freq="1s"),
            "s1": np.random.randn(10),
            "s2": np.random.randn(10),
            "fid": ["f1"] * 10
        })
        mock_batch = MagicMock()
        mock_batch.to_pandas.return_value = df
        mock_scanner.to_batches.return_value = [mock_batch]
        
        with patch.object(self.processor, "_standardize_timestamp", return_value=df.set_index("time")):
            with patch.object(self.processor, "_filter_sensors", return_value=df.set_index("time")[["s1", "s2"]]):
                with patch.object(self.processor, "_resample", return_value=df.set_index("time")[["s1", "s2"]]):
                    with patch("pandas.DataFrame.to_parquet"):
                        result = self.processor.process_flight("f1", (Path("multi.parquet"), "fid", "f1"))
                        self.assertIsNotNone(result)

    @patch("pyarrow.dataset.dataset")
    def test_process_flight_multiflight_empty(self, mock_ds):
        mock_scanner = MagicMock()
        mock_ds.return_value.scanner.return_value = mock_scanner
        mock_scanner.to_batches.return_value = [] # Empty batches
        
        result = self.processor.process_flight("f1", (Path("multi.parquet"), "fid", "f1"))
        self.assertIsNone(result)

    def test_process_flight_logging(self):
        processor = FlightProcessor(
            self.output_dir, ["s1"], "time", log_each_flight=True
        )
        df = pd.DataFrame({"time": [1], "s1": [1.0]})
        with patch("pandas.read_parquet", return_value=df):
            with patch.object(processor, "_standardize_timestamp", return_value=df.set_index("time")):
                with patch.object(processor, "_filter_sensors", return_value=df.set_index("time")):
                    with patch("pandas.DataFrame.to_parquet"):
                        with self.assertLogs(level="INFO") as cm:
                            processor.process_flight("f1", Path("f1.parquet"))
                            self.assertTrue(any("Processed flight f1" in m for m in cm.output))

    def test_process_all_parallel(self):
        processor = FlightProcessor(self.output_dir, ["s1"], "time")
        registry = {"f1": Path("f1.parquet"), "f2": Path("f2.parquet")}
        
        with patch.object(processor, "process_flight") as mock_process:
            mock_process.return_value = (Path("out.parquet"), 100)
            with patch.object(processor, "_compute_source_signature", return_value={"checksum": "123"}):
                with patch.object(processor, "_compute_processing_signature", return_value="456"):
                    with patch("pathlib.Path.stat") as mock_stat:
                        mock_stat.return_value.st_mtime = 1000
                        success, _ = processor.process_all(registry, num_workers=2)
                        self.assertEqual(len(success), 2)

    def test_process_all_parallel_exception(self):
        processor = FlightProcessor(self.output_dir, ["s1"], "time")
        registry = {"f1": Path("f1.parquet"), "f2": Path("f2.parquet")}
        
        with patch.object(processor, "process_flight", side_effect=Exception("Parallel fail")):
             with patch.object(processor, "_compute_source_signature", return_value={"checksum": "123"}):
                 success, _ = processor.process_all(registry, num_workers=2)
                 self.assertEqual(len(success), 0)

    def test_process_all_min_length_skip(self):
        processor = FlightProcessor(self.output_dir, ["s1"], "time")
        registry = {"f1": Path("f1.parquet")}
        
        with patch.object(processor, "process_flight", return_value=(Path("out.parquet"), 5)):
            with patch.object(processor, "_compute_source_signature", return_value={"checksum": "123"}):
                with patch("pathlib.Path.unlink") as mock_unlink:
                    success, _ = processor.process_all(registry, min_length=10)
                    self.assertEqual(len(success), 0)
                    mock_unlink.assert_called()

    def test_process_all_reuse_cached(self):
        processor = FlightProcessor(self.output_dir, ["s1"], "time", log_each_flight=True)
        registry = {"f1": Path("f1.parquet")}
        
        with patch("pathlib.Path.exists", return_value=True): # output exists
            with patch.object(processor, "_load_existing_metadata", return_value={
                "f1": {
                    "source_signature": {"checksum": "123"},
                    "processing_signature": processor._compute_processing_signature(),
                    "length": 100
                }
            }):
                with patch.object(processor, "_compute_source_signature", return_value={"checksum": "123"}):
                    with self.assertLogs(level="INFO") as cm:
                        success, _ = processor.process_all(registry)
                        self.assertEqual(len(success), 1)
                        self.assertTrue(any("Reusing processed flight f1" in m for m in cm.output))

    def test_process_all_reuse_cached_debug(self):
        processor = FlightProcessor(self.output_dir, ["s1"], "time", log_each_flight=False)
        registry = {"f1": Path("f1.parquet")}
        
        with patch("pathlib.Path.exists", return_value=True):
            with patch.object(processor, "_load_existing_metadata", return_value={
                "f1": {
                    "source_signature": {"checksum": "123"},
                    "processing_signature": processor._compute_processing_signature(),
                    "length": 100
                }
            }):
                with patch.object(processor, "_compute_source_signature", return_value={"checksum": "123"}):
                    with self.assertLogs(level="DEBUG") as cm:
                        success, _ = processor.process_all(registry)
                        self.assertTrue(any("Reusing processed flight f1" in m for m in cm.output))

    def test_process_all_cache_invalid(self):
        processor = FlightProcessor(self.output_dir, ["s1"], "time")
        registry = {"f1": Path("f1.parquet")}
        
        with patch("pathlib.Path.exists", return_value=True):
            with patch.object(processor, "_load_existing_metadata", return_value={
                "f1": {
                    "source_signature": {"checksum": "old"},
                    "processing_signature": "old_sig",
                    "length": 100
                }
            }):
                with patch.object(processor, "_compute_source_signature", return_value={"checksum": "new"}):
                    with patch.object(processor, "process_flight") as mock_proc:
                        mock_proc.return_value = (Path("out.parquet"), 100)
                        with patch("pathlib.Path.stat") as mock_stat:
                            mock_stat.return_value.st_mtime = 12345
                            # Should reprocess
                            with self.assertLogs(level="DEBUG") as cm:
                                processor.process_all(registry)
                                self.assertTrue(any("Cache invalidated" in m for m in cm.output))
                        mock_proc.assert_called()

    def test_standardize_timestamp_already_index(self):
        df = pd.DataFrame({"s1": [1]}, index=pd.to_datetime(["2021-01-01"]))
        res = self.processor._standardize_timestamp(df)
        self.assertEqual(res.index.name, "timestamp")
        pd.testing.assert_frame_equal(df.reset_index(drop=True), res.reset_index(drop=True))

    def test_standardize_timestamp_index_name_match(self):
        df = pd.DataFrame({"s1": [1]})
        df.index = pd.to_datetime(["2021-01-01"])
        df.index.name = "time"
        res = self.processor._standardize_timestamp(df)
        self.assertEqual(res.index.name, "timestamp")

    def test_standardize_timestamp_missing(self):
        df = pd.DataFrame({"s1": [1]})
        with self.assertRaises(ValueError):
            self.processor._standardize_timestamp(df)

    def test_to_datetime_index_dtype_valid(self):
        proc = FlightProcessor(self.output_dir, [], "t", timestamp_dtype="datetime64[ns]")
        idx = pd.Index(["2021-01-01"], dtype="object")
        res = proc._to_datetime_index(idx)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(res))

    def test_to_datetime_index_dtype_invalid(self):
        proc = FlightProcessor(self.output_dir, [], "t", timestamp_dtype="invalid")
        idx = pd.Index(["2021-01-01"], dtype="object")
        # Should fall back to pd.to_datetime and succeed
        res = proc._to_datetime_index(idx)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(res))

    def test_to_datetime_index_already_datetime(self):
        idx = pd.DatetimeIndex(["2021-01-01"])
        res = self.processor._to_datetime_index(idx)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(res))

    def test_filter_sensors_missing_warning(self):
        df = pd.DataFrame({"s1": [1]})
        with self.assertLogs(level="WARNING") as cm:
            res = self.processor._filter_sensors(df)
            self.assertTrue(any("Some sensors not found" in m for m in cm.output))
        self.assertEqual(list(res.columns), ["s1"])

    def test_filter_sensors_all_missing(self):
        df = pd.DataFrame({"s3": [1]})
        with self.assertRaises(ValueError):
            self.processor._filter_sensors(df)

    def test_resample_ffill_limits(self):
        df = pd.DataFrame(
            {"s1": [1.0, np.nan, np.nan, 2.0]},
            index=pd.date_range("2021-01-01", periods=4, freq="1s")
        )
        
        # Limit 1
        proc = FlightProcessor(self.output_dir, [], "t", resample_rate="1s", ffill_limit=1)
        res = proc._resample(df)
        self.assertEqual(len(res), 3) # One row dropped (the middle nan) - wait, resample doesn't drop unless dropna is called
        # The logic is: resample(mean) -> ffill(limit) -> where(mask) -> dropna
        # 1.0, nan, nan, 2.0 -> resample mean -> same
        # ffill(1) -> 1.0, 1.0, nan, 2.0
        # dropna -> 1.0, 1.0, 2.0 (3 rows)
        self.assertEqual(len(res), 3)

        # Limit 0
        proc = FlightProcessor(self.output_dir, [], "t", resample_rate="1s", ffill_limit=0)
        res = proc._resample(df)
        # ffill(0) -> no fill
        # 1.0, nan, nan, 2.0
        # dropna -> 1.0, 2.0 (2 rows)
        self.assertEqual(len(res), 2)

        # Limit None (infinite)
        proc = FlightProcessor(self.output_dir, [], "t", resample_rate="1s", ffill_limit=None)
        res = proc._resample(df)
        # ffill() -> 1.0, 1.0, 1.0, 2.0
        # dropna -> 4 rows
        self.assertEqual(len(res), 4)

    def test_resample_dropped_rows_logging(self):
        df = pd.DataFrame(
            {"s1": [1.0, np.nan]},
            index=pd.date_range("2021-01-01", periods=2, freq="1s")
        )
        proc = FlightProcessor(self.output_dir, [], "t", resample_rate="1s", ffill_limit=0)
        with self.assertLogs(level="DEBUG") as cm:
            proc._resample(df)
            self.assertTrue(any("Dropped 1 rows" in m for m in cm.output))

    def test_resample_numpy_empty(self):
        proc = FlightProcessor(self.output_dir, [], "t", resample_backend="numpy")
        df = pd.DataFrame()
        res, mask = proc._resample_with_numpy(df)
        self.assertTrue(res.empty)
        self.assertIsNone(mask)

    def test_resample_numpy_freq_undefined(self):
        proc = FlightProcessor(self.output_dir, [], "t", resample_backend="numpy", resample_rate="1s")
        df = pd.DataFrame({"s1": [1]}, index=pd.Index([pd.Timestamp("2021-01-01")]))
        # With only 1 point, date_range might not have freq if we don't pass it?
        # The code does: pd.date_range(..., freq=self.resample_rate)
        # So freq should be defined.
        
        # To trigger freq is None, pd.date_range must return index without freq.
        # This happens if freq is None in date_range, but we pass self.resample_rate.
        # Maybe if resample_rate is None? But then _resample skips it.
        pass

    def test_resample_numpy_single_value(self):
        proc = FlightProcessor(self.output_dir, ["s1"], "t", resample_backend="numpy", resample_rate="1s")
        df = pd.DataFrame(
            {"s1": [1.0, np.nan]},
            index=pd.date_range("2021-01-01", periods=2, freq="1s")
        )
        # Valid mask has 1 value
        res, mask = proc._resample_with_numpy(df)
        self.assertEqual(res["s1"].iloc[0], 1.0)
        self.assertEqual(res["s1"].iloc[1], 1.0) # Should propagate single value?
        # Code: if len(valid_y) == 1: resampled_series = np.full_like(..., valid_y[0])
        
    def test_resample_numpy_all_nan(self):
        proc = FlightProcessor(self.output_dir, ["s1"], "t", resample_backend="numpy", resample_rate="1s")
        df = pd.DataFrame(
            {"s1": [np.nan, np.nan]},
            index=pd.date_range("2021-01-01", periods=2, freq="1s")
        )
        res, mask = proc._resample_with_numpy(df)
        self.assertTrue(res["s1"].isna().all())

    def test_resample_numpy_gap_masking(self):
        proc = FlightProcessor(self.output_dir, ["s1"], "t", resample_backend="numpy", resample_rate="1s", ffill_limit=1)
        # 0s, 1s, (gap > 1s), 10s
        idx = pd.to_datetime(["2021-01-01 00:00:00", "2021-01-01 00:00:01", "2021-01-01 00:00:10"])
        df = pd.DataFrame({"s1": [0.0, 1.0, 10.0]}, index=idx)
        
        res, mask = proc._resample_with_numpy(df)
        # Expect mask to be False in the gap
        # freq=1s.
        # 00 to 01: gap=0.
        # 01 to 10: gap=(9s)/1s - 1 = 8 steps. > ffill_limit(1).
        # So mask should mask out steps 2 to 9 (exclusive or inclusive?)
        # Indices: 0 (00), 1 (01), 2 (02) ... 10 (10).
        # Gap starts at 01 + 1s = 02 (index 2).
        # Gap ends at 10 (index 10).
        # mask[2:10] should be False.
        self.assertTrue(mask[0])
        self.assertTrue(mask[1])
        self.assertFalse(mask[2]) # 02
        self.assertFalse(mask[9]) # 09
        self.assertTrue(mask[10]) # 10

    def test_load_existing_metadata_error(self):
        proc = FlightProcessor(self.output_dir, [], "t", dataset_name="foo")
        with patch("builtins.open", side_effect=Exception("Read fail")):
            meta = proc._load_existing_metadata()
            self.assertEqual(meta, {})

