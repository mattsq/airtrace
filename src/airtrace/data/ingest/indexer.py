"""
Window index generation for train/val/test splits.
"""

from pathlib import Path
from typing import List, Tuple
import pandas as pd
import logging

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
        index_rows = []

        for flight_id in flight_ids:
            flight_path = self.processed_dir / f"{flight_id}.parquet"

            if not flight_path.exists():
                logger.warning(f"Flight file not found: {flight_path}, skipping")
                continue

            # Load flight to get length
            try:
                df = pd.read_parquet(flight_path)
                T = len(df)
            except Exception as e:
                logger.error(f"Failed to load {flight_path}: {e}, skipping")
                continue

            # Check if flight is long enough
            if T < self.total_len:
                logger.warning(
                    f"Flight {flight_id} too short for windowing "
                    f"({T} < {self.total_len}), skipping"
                )
                continue

            # Generate sliding windows
            for start_idx in range(0, T - self.total_len + 1, self.stride):
                end_idx = start_idx + self.total_len
                index_rows.append({
                    "flight_id": flight_id,
                    "start_idx": start_idx,
                    "end_idx": end_idx
                })

        # Create DataFrame
        index_df = pd.DataFrame(index_rows)

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
    ) -> Tuple[Path, Path, Path]:
        """
        Create train/val/test indices.

        Args:
            train_ids: Training flight IDs
            val_ids: Validation flight IDs
            test_ids: Test flight IDs
            output_dir: Directory to save index files
            dataset_name: Dataset name for file naming

        Returns:
            Tuple of (train_path, val_path, test_path)
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

        return train_path, val_path, test_path
