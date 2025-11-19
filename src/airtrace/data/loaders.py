"""Data loading utilities for raw → interim → processed pipeline."""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


class RawDataLoader:
    """Load and process raw flight data files.

    Handles the raw → interim transformation:
    - Load raw CSV/Parquet files
    - Clean and validate data
    - Resample to uniform timesteps
    - Align sensors
    """

    def __init__(self, data_root: Path):
        """Initialize loader.

        Args:
            data_root: Root data directory
        """
        self.data_root = Path(data_root)
        self.raw_dir = self.data_root / "raw"
        self.interim_dir = self.data_root / "interim"
        self.interim_dir.mkdir(exist_ok=True, parents=True)

    def load_raw_flight(self, flight_id: str) -> pd.DataFrame:
        """Load raw flight data.

        Args:
            flight_id: Flight identifier

        Returns:
            DataFrame with columns: timestamp, sensor_name, value, ...
        """
        # Try different formats
        for ext in [".parquet", ".csv"]:
            file_path = self.raw_dir / f"{flight_id}{ext}"
            if file_path.exists():
                if ext == ".parquet":
                    return pd.read_parquet(file_path)
                else:
                    return pd.read_csv(file_path)

        raise FileNotFoundError(f"Raw data for {flight_id} not found")

    def process_to_interim(
        self,
        flight_id: str,
        resample_rate: str = "1S",
        sensor_list: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Process raw data to interim format.

        Args:
            flight_id: Flight identifier
            resample_rate: Resampling rate (e.g., '1S' for 1 second)
            sensor_list: Optional list of sensors to keep

        Returns:
            DataFrame with uniform timesteps and aligned sensors
        """
        # Load raw data
        raw_df = self.load_raw_flight(flight_id)

        # Convert to wide format if needed (sensor_name → columns)
        if "sensor_name" in raw_df.columns:
            raw_df = raw_df.pivot(
                index="timestamp",
                columns="sensor_name",
                values="value"
            )

        # Ensure timestamp is the index
        if "timestamp" in raw_df.columns:
            raw_df = raw_df.set_index("timestamp")

        # Ensure timestamp index is DatetimeIndex
        if not isinstance(raw_df.index, pd.DatetimeIndex):
            raw_df.index = pd.to_datetime(raw_df.index)

        # Resample to uniform timesteps
        interim_df = raw_df.resample(resample_rate).mean()

        # Forward fill missing values (within reason)
        interim_df = interim_df.fillna(method="ffill", limit=5)

        # Filter sensors if specified
        if sensor_list is not None:
            available_sensors = [s for s in sensor_list if s in interim_df.columns]
            interim_df = interim_df[available_sensors]

        # Save interim data
        output_path = self.interim_dir / f"{flight_id}.parquet"
        interim_df.to_parquet(output_path)

        return interim_df


class InterimDataProcessor:
    """Process interim data to windowed format.

    Handles interim → processed transformation:
    - Load interim data
    - Create sliding windows
    - Generate index files
    """

    def __init__(self, data_root: Path):
        """Initialize processor.

        Args:
            data_root: Root data directory
        """
        self.data_root = Path(data_root)
        self.interim_dir = self.data_root / "interim"
        self.processed_dir = self.data_root / "processed"
        self.metadata_dir = self.data_root / "metadata"
        self.processed_dir.mkdir(exist_ok=True, parents=True)
        self.metadata_dir.mkdir(exist_ok=True, parents=True)

    def load_interim_flight(self, flight_id: str) -> pd.DataFrame:
        """Load interim flight data.

        Args:
            flight_id: Flight identifier

        Returns:
            DataFrame with uniform timeseries
        """
        file_path = self.interim_dir / f"{flight_id}.parquet"
        return pd.read_parquet(file_path)

    def create_windows(
        self,
        flight_ids: List[str],
        window_spec,
        output_name: str = "train"
    ) -> pd.DataFrame:
        """Create sliding windows for multiple flights.

        Args:
            flight_ids: List of flight IDs to process
            window_spec: WindowSpec object
            output_name: Name for output index file (e.g., 'train', 'val')

        Returns:
            DataFrame with window index
        """
        all_windows = []

        for flight_id in flight_ids:
            # Load interim data
            flight_df = self.load_interim_flight(flight_id)

            # Convert to numpy for windowing
            flight_data = flight_df.values

            # Create windows using spec
            windows_df = window_spec.create_windows(
                data=flight_data,
                flight_id=flight_id
            )

            all_windows.append(windows_df)

        # Combine all windows (handle empty case)
        if all_windows:
            index_df = pd.concat(all_windows, ignore_index=True)
        else:
            # Create empty index with correct schema
            index_df = pd.DataFrame(columns=["flight_id", "start_idx", "end_idx"])

        # Save index
        index_path = self.metadata_dir / f"{output_name}_index.parquet"
        index_df.to_parquet(index_path)

        return index_df
