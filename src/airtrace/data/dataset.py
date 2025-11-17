"""PyTorch Dataset implementations for sensor timeseries."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SensorWindowDataset(Dataset):
    """Dataset for windowed sensor timeseries data.

    This dataset loads pre-windowed data based on an index file.
    Each sample returns (x, y, meta) where:
    - x: Input sequence [T_in, D_in]
    - y: Target sequence [T_out, D_out]
    - meta: Dictionary with flight_id, timestamps, etc.
    """

    def __init__(
        self,
        index_df: pd.DataFrame,
        data_store: "DataStore",
        transforms: Optional[Any] = None,
        sensor_names: Optional[List[str]] = None,
        target_sensors: Optional[List[str]] = None,
        window_spec: Optional[Any] = None
    ):
        """Initialize dataset.

        Args:
            index_df: DataFrame with columns [flight_id, start_idx, end_idx]
            data_store: DataStore object for fast access to flight data
            transforms: Optional transform pipeline to apply
            sensor_names: List of sensor names to use as inputs
            target_sensors: List of sensor names to use as targets
            window_spec: WindowSpec object defining window structure
        """
        self.index_df = index_df.reset_index(drop=True)
        self.data_store = data_store
        self.transforms = transforms
        self.sensor_names = sensor_names
        self.target_sensors = target_sensors or sensor_names
        self.window_spec = window_spec

        # Compute dimensions
        self.in_dim = len(sensor_names) if sensor_names else None
        self.out_dim = len(self.target_sensors) if self.target_sensors else None

    def __len__(self) -> int:
        """Return number of windows in dataset."""
        return len(self.index_df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single window.

        Args:
            idx: Index of window

        Returns:
            Dictionary with 'x', 'y', 'meta' keys
        """
        row = self.index_df.iloc[idx]

        # Get all columns needed (sensors + targets)
        all_columns = list(set(self.sensor_names) | set(self.target_sensors))

        # Get full window from data store (no splitting)
        window_data, meta = self.data_store.get_full_window(
            flight_id=row.flight_id,
            start_idx=row.start_idx,
            end_idx=row.end_idx,
            column_names=all_columns
        )

        # Dataset handles splitting using WindowSpec
        if self.window_spec is not None:
            input_len = self.window_spec.input_len
        else:
            # Fallback: use heuristic (this maintains backward compatibility)
            # Assume 90% input, 10% target
            total_len = len(window_data)
            input_len = int(total_len * 0.9)

        # Get column indices for sensors and targets
        sensor_indices = [all_columns.index(s) for s in self.sensor_names]
        target_indices = [all_columns.index(s) for s in self.target_sensors]

        # Split window data
        x = window_data[:input_len, sensor_indices]
        y = window_data[input_len:, target_indices]

        # Apply transforms
        if self.transforms is not None:
            x, y, meta = self.transforms(x, y, meta)

        # Convert to tensors
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()

        return {
            "x": x,
            "y": y,
            "meta": meta
        }


class DataStore:
    """Fast access to flight timeseries data.

    This class provides an interface for loading windowed data from
    processed flight files. It can be backed by different storage formats
    (Parquet, HDF5, zarr, etc.).
    """

    def __init__(self, data_root: Path, format: str = "parquet"):
        """Initialize data store.

        Args:
            data_root: Root directory containing processed data
            format: Data format ('parquet', 'hdf5', etc.)
        """
        self.data_root = Path(data_root)
        self.format = format
        self._cache = {}  # Simple in-memory cache

    def get_full_window(
        self,
        flight_id: str,
        start_idx: int,
        end_idx: int,
        column_names: List[str]
    ) -> tuple:
        """Get a complete window without splitting into input/target.

        This is the new preferred method. The Dataset will handle splitting.

        Args:
            flight_id: Flight identifier
            start_idx: Start index of window
            end_idx: End index of window
            column_names: Column names to extract

        Returns:
            Tuple of (window_array, meta) where:
                window_array: NumPy array [T_window, D]
                meta: Metadata dict
        """
        # Load flight data (with caching)
        if flight_id not in self._cache:
            self._cache[flight_id] = self._load_flight(flight_id)

        flight_data = self._cache[flight_id]

        # Validate columns exist
        missing_cols = set(column_names) - set(flight_data.columns)
        if missing_cols:
            raise ValueError(
                f"Missing columns in flight {flight_id}: {missing_cols}\n"
                f"Available columns: {list(flight_data.columns)}"
            )

        # Extract window using iloc (correct pandas indexing)
        window_df = flight_data.iloc[start_idx:end_idx]

        # Select columns and convert to numpy
        window_array = window_df[column_names].values

        meta = {
            "flight_id": flight_id,
            "start_idx": start_idx,
            "end_idx": end_idx
        }

        return window_array, meta

    def get_window(
        self,
        flight_id: str,
        start_idx: int,
        end_idx: int,
        sensor_names: List[str],
        target_sensors: List[str]
    ) -> tuple:
        """Get a window of data (DEPRECATED - use get_full_window instead).

        This method is kept for backward compatibility but is deprecated.
        Use get_full_window() and handle splitting in the Dataset.

        Args:
            flight_id: Flight identifier
            start_idx: Start index of window
            end_idx: End index of window
            sensor_names: Sensor names for inputs
            target_sensors: Sensor names for targets

        Returns:
            Tuple of (x, y, meta) where:
                x: Input array [T_in, D_in]
                y: Target array [T_out, D_out]
                meta: Metadata dict
        """
        # Get full window
        all_columns = list(set(sensor_names) | set(target_sensors))
        window_array, meta = self.get_full_window(
            flight_id=flight_id,
            start_idx=start_idx,
            end_idx=end_idx,
            column_names=all_columns
        )

        # Use heuristic split (90/10) for backward compatibility
        total_len = len(window_array)
        input_len = int(total_len * 0.9)

        # Get column indices
        sensor_indices = [all_columns.index(s) for s in sensor_names]
        target_indices = [all_columns.index(s) for s in target_sensors]

        # Split
        x = window_array[:input_len, sensor_indices]
        y = window_array[input_len:, target_indices]

        return x, y, meta

    def _load_flight(self, flight_id: str) -> pd.DataFrame:
        """Load flight data from disk.

        Args:
            flight_id: Flight identifier

        Returns:
            DataFrame with sensor timeseries
        """
        if self.format == "parquet":
            file_path = self.data_root / "processed" / f"{flight_id}.parquet"
            return pd.read_parquet(file_path)
        else:
            raise NotImplementedError(f"Format {self.format} not yet implemented")

    def clear_cache(self):
        """Clear the in-memory cache."""
        self._cache.clear()
