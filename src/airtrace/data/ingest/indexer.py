"""
Window index generation for train/val/test splits.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


class WindowIndexer:
    """Generates sliding window indices for dataset splits."""

    def __init__(
        self,
        input_len: int,
        pred_len: int,
        stride: int,
        processed_dir: Path
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

    def create_index(
        self,
        flight_ids: List[str],
        split_name: str
    ) -> pd.DataFrame:
        """
        Create window index for a split.

        Args:
            flight_ids: List of flight IDs in this split
            split_name: Name of split (for logging)

        Returns:
            DataFrame with columns [flight_id, start_idx, end_idx]
        """
        frames = []

        for flight_id in flight_ids:
            flight_path = self.processed_dir / f"{flight_id}.parquet"

            if not flight_path.exists():
                logger.warning(f"Flight file not found: {flight_path}, skipping")
                continue

            T = self._flight_length(flight_path)
            if T is None:
                continue

            if T < self.total_len:
                logger.warning(
                    f"Flight {flight_id} too short for windowing "
                    f"({T} < {self.total_len}), skipping"
                )
                continue

            start_indices = np.arange(0, T - self.total_len + 1, self.stride)
            if len(start_indices) == 0:
                continue

            end_indices = start_indices + self.total_len
            frames.append(
                pd.DataFrame(
                    {
                        "flight_id": np.repeat(flight_id, len(start_indices)),
                        "start_idx": start_indices,
                        "end_idx": end_indices,
                    }
                )
            )

        index_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

        logger.info(
            f"{split_name}: {len(index_df)} windows from {len(flight_ids)} flights"
        )

        return index_df

    def create_all_indices(
        self,
        train_ids: List[str],
        val_ids: List[str],
        test_ids: List[str],
        output_dir: Path,
        dataset_name: str
    ) -> Tuple[Path, Path, Path, int, int, int]:
        """
        Create train/val/test indices.

        Args:
            train_ids: Training flight IDs
            val_ids: Validation flight IDs
            test_ids: Test flight IDs
            output_dir: Directory to save index files
            dataset_name: Dataset name for file naming

        Returns:
            Tuple of (train_path, val_path, test_path, train_len, val_len, test_len)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create indices
        train_index = self.create_index(train_ids, "train")
        val_index = self.create_index(val_ids, "val")
        test_index = self.create_index(test_ids, "test")

        # Save indices
        train_path = output_dir / f"{dataset_name}_train_index.parquet"
        val_path = output_dir / f"{dataset_name}_val_index.parquet"
        test_path = output_dir / f"{dataset_name}_test_index.parquet"

        train_index.to_parquet(train_path, index=False, engine="pyarrow")
        val_index.to_parquet(val_path, index=False, engine="pyarrow")
        test_index.to_parquet(test_path, index=False, engine="pyarrow")

        logger.info(f"Saved window indices to {output_dir}")

        return (
            train_path,
            val_path,
            test_path,
            len(train_index),
            len(val_index),
            len(test_index),
        )

    def _flight_length(self, flight_path: Path) -> Optional[int]:
        """Return flight length using parquet metadata when available."""

        try:
            metadata = pq.ParquetFile(flight_path).metadata
            if metadata is not None:
                return metadata.num_rows
        except Exception as exc:  # pragma: no cover - metadata failures are logged
            logger.debug(f"Could not read metadata for {flight_path}: {exc}")

        try:
            df = pd.read_parquet(flight_path, columns=[])
            return len(df)
        except Exception as exc:  # pragma: no cover - surfacing via logger
            logger.error(f"Failed to load {flight_path}: {exc}, skipping")
            return None
