
import unittest
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path
import numpy as np
import pandas as pd
import json

from airtrace.data.ingest.indexer import WindowIndexer

class TestWindowIndexerCoverage(unittest.TestCase):
    def setUp(self):
        self.processed_dir = Path("processed")
        self.metadata_dir = Path("metadata")
        self.indexer = WindowIndexer(
            input_len=10,
            pred_len=5,
            stride=5,
            processed_dir=self.processed_dir,
            metadata_dir=self.metadata_dir
        )

    def test_get_length_from_metadata_success(self):
        with patch("pyarrow.parquet.read_metadata") as mock_read:
            mock_read.return_value.num_rows = 100
            length = WindowIndexer._get_length_from_metadata(Path("f.parquet"))
            self.assertEqual(length, 100)

    def test_get_length_from_metadata_fallback(self):
        with patch("pyarrow.parquet.read_metadata", side_effect=Exception("Read fail")):
            with patch("pandas.read_parquet", return_value=pd.DataFrame(index=range(50))):
                length = WindowIndexer._get_length_from_metadata(Path("f.parquet"))
                self.assertEqual(length, 50)

    def test_get_length_from_metadata_fail_all(self):
        with patch("pyarrow.parquet.read_metadata", side_effect=Exception("Read fail")):
            with patch("pandas.read_parquet", side_effect=Exception("Pandas fail")):
                length = WindowIndexer._get_length_from_metadata(Path("f.parquet"))
                self.assertIsNone(length)

    def test_get_flight_length_cached(self):
        self.indexer._length_cache["f.parquet"] = 123
        length = self.indexer._get_flight_length(Path("f.parquet"))
        self.assertEqual(length, 123)

    def test_get_flight_length_uncached(self):
        with patch.object(WindowIndexer, "_get_length_from_metadata", return_value=100):
            length = self.indexer._get_flight_length(Path("f.parquet"))
            self.assertEqual(length, 100)
            self.assertEqual(self.indexer._length_cache["f.parquet"], 100)

    def test_process_flight_file_not_found(self):
        args = ("f1", Path("proc"), 15, 5)
        with patch("pathlib.Path.exists", return_value=False):
            res = WindowIndexer._process_flight(args)
            self.assertIsNone(res)

    def test_process_flight_sequential_file_not_found(self):
        with patch("pathlib.Path.exists", return_value=False):
            res = self.indexer._process_flight_sequential("f1")
            self.assertIsNone(res)

    def test_process_flight_too_short(self):
        args = ("f1", Path("proc"), 15, 5)
        with patch("pathlib.Path.exists", return_value=True):
            with patch.object(WindowIndexer, "_get_length_from_metadata", return_value=10):
                res = WindowIndexer._process_flight(args)
                self.assertIsNone(res)

    def test_process_flight_sequential_too_short(self):
        with patch("pathlib.Path.exists", return_value=True):
            with patch.object(self.indexer, "_get_flight_length", return_value=10):
                res = self.indexer._process_flight_sequential("f1")
                self.assertIsNone(res)

    def test_process_flight_success(self):
        args = ("f1", Path("proc"), 15, 5)
        # Length 20. Total len 15. Stride 5.
        # Starts: 0, 5. Ends: 15, 20.
        with patch("pathlib.Path.exists", return_value=True):
            with patch.object(WindowIndexer, "_get_length_from_metadata", return_value=20):
                fid, starts, ends = WindowIndexer._process_flight(args)
                self.assertEqual(fid, "f1")
                np.testing.assert_array_equal(starts, [0, 5])
                np.testing.assert_array_equal(ends, [15, 20])

    def test_iter_windows_sequential(self):
        with patch.object(self.indexer, "_process_flight_sequential", return_value=("f1", np.array([0]), np.array([15]))):
            results = list(self.indexer._iter_windows(["f1"], num_workers=1))
            self.assertEqual(len(results), 1)

    def test_iter_windows_parallel(self):
        with patch("multiprocessing.pool.Pool") as mock_pool:
            pool_instance = mock_pool.return_value
            pool_instance.__enter__.return_value = pool_instance
            pool_instance.imap_unordered.return_value = [("f1", np.array([0]), np.array([15]))]
            
            results = list(self.indexer._iter_windows(["f1"], num_workers=2))
            self.assertEqual(len(results), 1)

    def test_create_index_partitioned(self):
        with patch.object(self.indexer, "_iter_windows", return_value=[("f1", np.array([0]), np.array([15]))]):
            with patch("pyarrow.parquet.write_to_dataset") as mock_write:
                self.indexer.create_index(["f1"], "train", partitioned=True, output_path=Path("out"), materialize_dataframe=False)
                mock_write.assert_called()

    def test_create_index_stream_file(self):
        with patch.object(self.indexer, "_iter_windows", return_value=[("f1", np.array([0]), np.array([15]))]):
            with patch("pyarrow.parquet.ParquetWriter") as mock_writer:
                self.indexer.create_index(["f1"], "train", partitioned=False, output_path=Path("out.parquet"), materialize_dataframe=False)
                mock_writer.return_value.write_table.assert_called()

    def test_create_index_materialize_empty(self):
        with patch.object(self.indexer, "_iter_windows", return_value=[]):
            df = self.indexer.create_index(["f1"], "train", materialize_dataframe=True)
            self.assertTrue(df.empty)
            self.assertEqual(list(df.columns), ["flight_id", "start_idx", "end_idx"])

    def test_create_index_return_count(self):
        with patch.object(self.indexer, "_iter_windows", return_value=[("f1", np.array([0, 5]), np.array([15, 20]))]):
            df, count = self.indexer.create_index(["f1"], "train", materialize_dataframe=True, return_count=True)
            self.assertEqual(count, 2)
            self.assertEqual(len(df), 2)

    def test_create_all_indices_reuse(self):
        with patch.object(self.indexer, "_can_reuse_indices", return_value=True):
            with patch.object(self.indexer, "_load_metadata", return_value={"window_counts": {"train": 10}}):
                with patch("pathlib.Path.exists", return_value=True):
                    _, _, _, counts = self.indexer.create_all_indices([], [], [], Path("out"), "ds")
                    self.assertEqual(counts["train"], 10)

    def test_create_all_indices_no_reuse(self):
        with patch.object(self.indexer, "_can_reuse_indices", return_value=False):
            with patch.object(self.indexer, "create_index", return_value=(pd.DataFrame(), 10)):
                with patch("pandas.DataFrame.to_parquet"):
                    with patch("builtins.open", mock_open()): # save metadata
                        train_p, val_p, test_p, counts = self.indexer.create_all_indices(["t"], ["v"], ["ts"], Path("out"), "ds")
                        self.assertEqual(counts["train"], 10)

    def test_processed_signature_empty(self):
        sig = self.indexer._processed_signature(None, [])
        self.assertEqual(sig, "")

    def test_can_reuse_indices_no_meta(self):
        with patch.object(self.indexer, "_load_metadata", return_value={}):
            reuse = self.indexer._can_reuse_indices(Path("meta"), [], [], [], {})
            self.assertFalse(reuse)

    def test_can_reuse_indices_params_mismatch(self):
        meta = {
            "input_len": 999, # Mismatch
            "pred_len": 5, "stride": 5,
            "train_ids": [], "val_ids": [], "test_ids": [],
            "processed_signature": ""
        }
        with patch.object(self.indexer, "_load_metadata", return_value=meta):
            with patch.object(self.indexer, "_processed_signature", return_value=""):
                reuse = self.indexer._can_reuse_indices(Path("meta"), [], [], [], {})
                self.assertFalse(reuse)

    def test_can_reuse_indices_ids_mismatch(self):
        meta = {
            "input_len": 10, "pred_len": 5, "stride": 5,
            "train_ids": ["different"],
            "val_ids": [], "test_ids": [],
            "processed_signature": ""
        }
        with patch.object(self.indexer, "_load_metadata", return_value=meta):
            with patch.object(self.indexer, "_processed_signature", return_value=""):
                reuse = self.indexer._can_reuse_indices(Path("meta"), [], [], [], {})
                self.assertFalse(reuse)

    def test_save_metadata(self):
        with patch("builtins.open", mock_open()) as mock_file:
            with patch("json.dump") as mock_json:
                self.indexer._save_metadata(Path("meta.json"), [], [], [], {}, {})
                mock_json.assert_called()

    def test_load_metadata_fail(self):
        with patch("pathlib.Path.exists", return_value=True):
            with patch("builtins.open", side_effect=Exception("Read fail")):
                res = self.indexer._load_metadata(Path("meta.json"))
                self.assertEqual(res, {})

