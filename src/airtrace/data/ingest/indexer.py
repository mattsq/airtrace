"""
Window index generation for train/val/test splits.
"""

import logging
from multiprocessing import Pool
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


class WindowIndexer:
    """Generates sliding window indices for dataset splits."""

    def __init__(
        self,
        input_len: int,
        pred_len: int,
        stride: int,
        processed_dir: Path,
    ):
        """
        Args:
            input_len: Input sequence length (history)
            pred_len: Prediction sequence length (forecast)
            stride: Sliding window stride
            processed_dir: Directory containing processed flight parquet files
        """
        self.input_len = input_len
        self.pred_len = pred_len
        self.stride = stride
        self.processed_dir = Path(processed_dir)
        self.total_len = input_len + pred_len
        self._length_cache: dict[str, int] = {}

    @staticmethod
    def _get_length_from_metadata(flight_path: Path) -> Optional[int]:
        """Get number of rows from parquet metadata without loading full file."""

        try:
            metadata = pq.read_metadata(flight_path)
            return metadata.num_rows
        except Exception:
            try:
                # columns=[] performs an inexpensive read of schema-only data
                return len(pd.read_parquet(flight_path, columns=[]))
            except Exception as exc:  # pragma: no cover - logged by caller
                logger.error(f"Failed to read metadata for {flight_path}: {exc}")
                return None

    def _get_flight_length(self, flight_path: Path) -> Optional[int]:
        """Return cached flight length, reading parquet metadata when needed."""

        cache_key = flight_path.name
        if cache_key in self._length_cache:
            return self._length_cache[cache_key]

        length = self._get_length_from_metadata(flight_path)
        if length is not None:
            self._length_cache[cache_key] = length
        return length

    @staticmethod
    def _process_flight(
        args: Tuple[str, Path, int, int]
    ) -> Optional[Tuple[str, np.ndarray, np.ndarray]]:
        """Worker-safe helper to generate window boundaries for a single flight."""

        flight_id, processed_dir, total_len, stride = args
        flight_path = processed_dir / f"{flight_id}.parquet"

        if not flight_path.exists():
            logger.warning(f"Flight file not found: {flight_path}, skipping")
            return None

        length = WindowIndexer._get_length_from_metadata(flight_path)
        if length is None:
            return None

        if length < total_len:
            logger.warning(
                f"Flight {flight_id} too short for windowing ({length} < {total_len}), skipping"
            )
            return None

        start_indices = np.arange(0, length - total_len + 1, stride, dtype=np.int64)
        if start_indices.size == 0:
            return None

        end_indices = start_indices + total_len
        return flight_id, start_indices, end_indices

    def _process_flight_sequential(
        self, flight_id: str
    ) -> Optional[Tuple[str, np.ndarray, np.ndarray]]:
        """Sequential version that reuses cached flight lengths."""

        flight_path = self.processed_dir / f"{flight_id}.parquet"

        if not flight_path.exists():
            logger.warning(f"Flight file not found: {flight_path}, skipping")
            return None

        length = self._get_flight_length(flight_path)
        if length is None:
            return None

        if length < self.total_len:
            logger.warning(
                f"Flight {flight_id} too short for windowing ({length} < {self.total_len}), skipping"
            )
            return None

        start_indices = np.arange(0, length - self.total_len + 1, self.stride, dtype=np.int64)
        if start_indices.size == 0:
            return None

        end_indices = start_indices + self.total_len
        return flight_id, start_indices, end_indices

    def _iter_windows(
        self, flight_ids: Iterable[str], num_workers: int
    ) -> Iterable[Tuple[str, np.ndarray, np.ndarray]]:
        """Yield (flight_id, start_indices, end_indices) tuples for flights."""

        if num_workers > 1:
            with Pool(processes=num_workers) as pool:
                args = (
                    (flight_id, self.processed_dir, self.total_len, self.stride)
                    for flight_id in flight_ids
                )
                for result in pool.imap_unordered(WindowIndexer._process_flight, args):
                    if result is not None:
                        yield result
        else:
            for flight_id in flight_ids:
                result = self._process_flight_sequential(flight_id)
                if result is not None:
                    yield result

    @staticmethod
    def _windows_to_table(
        flight_id: str, start_indices: np.ndarray, end_indices: np.ndarray
    ) -> pa.Table:
        """Convert window arrays to a pyarrow Table for parquet writing."""

        return pa.Table.from_arrays(
            [
                pa.array(np.repeat(flight_id, start_indices.size)),
                pa.array(start_indices),
                pa.array(end_indices),
            ],
            names=["flight_id", "start_idx", "end_idx"],
        )

    def create_index(
        self,
        flight_ids: List[str],
        split_name: str,
        *,
        num_workers: int = 1,
        output_path: Optional[Path] = None,
        partitioned: bool = False,
        materialize_dataframe: bool = True,
        return_count: bool = False,
    ) -> pd.DataFrame | Tuple[pd.DataFrame, int]:
        """
        Create window index for a split.

        Args:
            flight_ids: List of flight IDs in this split
            split_name: Name of split (for logging)
            num_workers: Number of worker processes for parallel generation
            output_path: Optional parquet path/directory to stream results to
            partitioned: Write a partitioned parquet dataset (one partition per flight)
            materialize_dataframe: Whether to build a pandas DataFrame in memory
            return_count: Whether to also return the window count

        Returns:
            DataFrame with columns [flight_id, start_idx, end_idx]. When
            ``return_count`` is True, returns a tuple of (DataFrame, num_windows).
        """
        materialize_dataframe = bool(materialize_dataframe)
        index_rows: List[Tuple[str, np.ndarray, np.ndarray]] = []
        total_windows = 0

        parquet_writer: Optional[pq.ParquetWriter] = None
        if output_path is not None:
            output_path = Path(output_path)
            if partitioned:
                output_path.mkdir(parents=True, exist_ok=True)
            else:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                if output_path.exists():
                    output_path.unlink()

        try:
            for flight_id, start_indices, end_indices in self._iter_windows(flight_ids, num_workers):
                total_windows += start_indices.size

                if output_path is not None:
                    table = self._windows_to_table(flight_id, start_indices, end_indices)
                    if partitioned:
                        pq.write_to_dataset(
                            table,
                            root_path=str(output_path),
                            partition_cols=["flight_id"],
                        )
                    else:
                        if parquet_writer is None:
                            parquet_writer = pq.ParquetWriter(
                                where=output_path,
                                schema=table.schema,
                                compression="snappy",
                            )
                        parquet_writer.write_table(table)

                if materialize_dataframe:
                    index_rows.append((flight_id, start_indices, end_indices))
        finally:
            if parquet_writer is not None:
                parquet_writer.close()

        if materialize_dataframe:
            if index_rows:
                flight_column = np.concatenate(
                    [np.repeat(fid, starts.size) for fid, starts, _ in index_rows]
                )
                start_column = np.concatenate([starts for _, starts, _ in index_rows])
                end_column = np.concatenate([ends for _, _, ends in index_rows])
                index_df = pd.DataFrame(
                    {
                        "flight_id": flight_column,
                        "start_idx": start_column,
                        "end_idx": end_column,
                    }
                )
            else:
                index_df = pd.DataFrame(columns=["flight_id", "start_idx", "end_idx"])
        else:
            index_df = pd.DataFrame(columns=["flight_id", "start_idx", "end_idx"])

        logger.info(
            f"{split_name}: {total_windows} windows from {len(flight_ids)} flights"
        )

        if return_count:
            return index_df, total_windows
        return index_df

    def create_all_indices(
        self,
        train_ids: List[str],
        val_ids: List[str],
        test_ids: List[str],
        output_dir: Path,
        dataset_name: str,
        *,
        num_workers: int = 1,
        write_partitioned: bool = False,
        materialize_dataframe: bool = True,
        return_counts: bool = False,
    ) -> Tuple[Path, Path, Path] | Tuple[Tuple[Path, Path, Path], dict[str, int]]:
        """
        Create train/val/test indices.

        Args:
            train_ids: Training flight IDs
            val_ids: Validation flight IDs
            test_ids: Test flight IDs
            output_dir: Directory to save index files
            dataset_name: Dataset name for file naming
            num_workers: Number of worker processes for parallel generation
            write_partitioned: Write indices as a partitioned parquet dataset
            materialize_dataframe: Whether to keep full index DataFrames in memory
            return_counts: Whether to also return split window counts

        Returns:
            Tuple of (train_path, val_path, test_path). When ``return_counts`` is
            True, returns ((train_path, val_path, test_path), counts).
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_suffix = "" if write_partitioned else ".parquet"

        def build_split(
            split_ids: List[str],
            split_name: str,
        ) -> Tuple[pd.DataFrame, Path, int]:
            split_path = output_dir / f"{dataset_name}_{split_name}_index{output_suffix}"
            stream_write = write_partitioned or not materialize_dataframe

            split_index, split_count = self.create_index(
                split_ids,
                split_name,
                num_workers=num_workers,
                output_path=split_path if stream_write else None,
                partitioned=write_partitioned,
                materialize_dataframe=materialize_dataframe,
                return_count=True,
            )

            if materialize_dataframe and not write_partitioned:
                split_index.to_parquet(split_path, index=False, engine="pyarrow")

            return split_index, split_path, split_count

        train_index, train_path, train_count = build_split(train_ids, "train")
        val_index, val_path, val_count = build_split(val_ids, "val")
        test_index, test_path, test_count = build_split(test_ids, "test")

        logger.info(f"Saved window indices to {output_dir}")
        paths = (train_path, val_path, test_path)
        if return_counts:
            return paths, {"train": train_count, "val": val_count, "test": test_count}
        return paths

