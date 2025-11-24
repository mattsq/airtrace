"""
Flight data processing - cleaning, resampling, and saving.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class FlightProcessor:
    """Processes flight data into uniform format for AirTrace."""

    def __init__(
        self,
        output_dir: Path,
        sensors: List[str],
        timestamp_column: str,
        resample_rate: Optional[str] = None,
        dataset_name: Optional[str] = None,
        timestamp_dtype: Optional[str] = None,
        resample_backend: str = "pandas",
        ffill_limit: Optional[int] = 5
    ):
        """
        Args:
            output_dir: Directory to save processed flight files
            sensors: List of sensor columns to include
            timestamp_column: Name of timestamp column
            resample_rate: Optional resample rate (e.g., "1S" for 1 second)
            dataset_name: Optional dataset name for auto-naming flights
            timestamp_dtype: Optional dtype detected by the validator for fast datetime handling
            resample_backend: Backend to use for resampling ("pandas" or "numpy")
            ffill_limit: Forward-fill limit for small gaps (None disables limit)
        """
        self.output_dir = Path(output_dir)
        self.sensors = sensors
        self.timestamp_column = timestamp_column
        self.resample_rate = resample_rate
        self.dataset_name = dataset_name
        self.timestamp_dtype = timestamp_dtype
        self.resample_backend = resample_backend
        self.ffill_limit = ffill_limit

        if self.resample_backend not in {"pandas", "numpy"}:
            raise ValueError("resample_backend must be 'pandas' or 'numpy'")

        # Create output directory if needed
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_flight(
        self,
        flight_id: str,
        source: Union[Path, Tuple[Path, str, str]]
    ) -> Optional[Path]:
        """
        Process a single flight.

        Args:
            flight_id: Identifier for this flight
            source: Either a Path to parquet file, or (Path, column_name, value) tuple
                   for filtering multi-flight files

        Returns:
            Path to saved processed file, or None if processing failed
        """
        try:
            # Load flight data
            if isinstance(source, tuple):
                # Multi-flight file - need to filter
                file_path, id_column, id_value = source
                df = pd.read_parquet(file_path, engine="pyarrow")
                df = df[df[id_column] == id_value].copy()
            else:
                # Single flight file
                df = pd.read_parquet(source, engine="pyarrow")

            # Standardize timestamp index
            df = self._standardize_timestamp(df)

            # Filter to sensor columns
            df = self._filter_sensors(df)

            # Resample if requested
            if self.resample_rate:
                df = self._resample(df)

            # Validate minimum length
            if len(df) == 0:
                logger.warning(f"Flight {flight_id} has no data after processing, skipping")
                return None

            # Save to output directory
            output_path = self.output_dir / f"{flight_id}.parquet"
            df.to_parquet(output_path, engine="pyarrow", index=True)

            logger.info(f"Processed flight {flight_id}: {len(df)} timesteps, saved to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Failed to process flight {flight_id}: {e}")
            return None

    def process_all(
        self,
        flight_registry: Dict[str, Union[Path, Tuple[Path, str, str]]],
        min_length: int = 1
    ) -> List[str]:
        """
        Process all flights in registry.

        Args:
            flight_registry: Dictionary mapping flight_id -> source
            min_length: Minimum number of timesteps (flights shorter are skipped)

        Returns:
            List of successfully processed flight IDs
        """
        successful_flights = []

        for flight_id, source in flight_registry.items():
            output_path = self.process_flight(flight_id, source)

            if output_path is not None:
                # Check length
                df = pd.read_parquet(output_path)
                if len(df) >= min_length:
                    successful_flights.append(flight_id)
                else:
                    logger.warning(
                        f"Flight {flight_id} too short ({len(df)} < {min_length}), skipping"
                    )
                    # Remove the file
                    output_path.unlink()

        logger.info(
            f"Successfully processed {len(successful_flights)}/{len(flight_registry)} flights"
        )

        return successful_flights

    def _standardize_timestamp(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize timestamp as index."""

        # If timestamp is already the index, we're done
        if pd.api.types.is_datetime64_any_dtype(df.index):
            df = df.copy()
            df.index = self._to_datetime_index(df.index)
            return df.sort_index()

        # Set timestamp column as index
        if self.timestamp_column in df.columns:
            timestamp_source = df[self.timestamp_column]
        elif df.index.name == self.timestamp_column:
            timestamp_source = df.index
        else:
            raise ValueError(f"Timestamp column '{self.timestamp_column}' not found")

        datetime_index = self._to_datetime_index(timestamp_source)
        df = df.drop(columns=[self.timestamp_column], errors="ignore").copy()
        df.index = datetime_index

        return df.sort_index()

    def _to_datetime_index(self, timestamp_source: Union[pd.Series, pd.Index]) -> pd.DatetimeIndex:
        """Convert timestamp source to a DatetimeIndex using a single fastpath conversion."""

        if self.timestamp_dtype:
            try:
                dtype = pd.api.types.pandas_dtype(self.timestamp_dtype)
                if pd.api.types.is_datetime64_any_dtype(dtype):
                    datetime_index = pd.DatetimeIndex(timestamp_source, copy=False)
                    datetime_index.name = "timestamp"
                    return datetime_index
            except (TypeError, ValueError):
                # Fall back to normal conversion if dtype string is not understood
                pass

        if pd.api.types.is_datetime64_any_dtype(timestamp_source):
            datetime_index = pd.DatetimeIndex(timestamp_source, copy=False)
        else:
            datetime_index = pd.to_datetime(timestamp_source)

        datetime_index.name = "timestamp"
        return datetime_index

    def _filter_sensors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter to only sensor columns."""

        # Find which sensors are actually in the dataframe
        available_sensors = [s for s in self.sensors if s in df.columns]

        if not available_sensors:
            raise ValueError(f"None of the specified sensors found in data: {self.sensors}")

        missing_sensors = set(self.sensors) - set(available_sensors)
        if missing_sensors:
            logger.warning(f"Some sensors not found in data: {missing_sensors}")

        return df[available_sensors]

    def _resample(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resample to uniform time intervals."""

        mask = None

        if self.resample_backend == "numpy":
            df_resampled, mask = self._resample_with_numpy(df)
        else:
            df_resampled = df.resample(self.resample_rate).mean()

        if self.ffill_limit is not None:
            if self.ffill_limit > 0:
                df_resampled = df_resampled.ffill(limit=self.ffill_limit)
        else:
            df_resampled = df_resampled.ffill()

        if mask is not None and not np.all(mask):
            df_resampled = df_resampled.where(mask[:, None], np.nan)

        # Drop any remaining NaN rows
        initial_len = len(df_resampled)
        df_resampled = df_resampled.dropna()
        dropped = initial_len - len(df_resampled)

        if dropped > 0:
            logger.debug(f"Dropped {dropped} rows with NaN after resampling")

        return df_resampled

    def _resample_with_numpy(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
        """Resample using numpy interpolation to reduce pandas overhead."""

        if df.empty:
            return df, None

        target_index = pd.date_range(
            start=df.index[0], end=df.index[-1], freq=self.resample_rate, name=df.index.name
        )

        freq = target_index.freq
        if freq is None:
            raise ValueError("Target index frequency is undefined for numpy resampling")

        freq_ns = int(freq.nanos)

        current_ns = df.index.view("int64")
        target_ns = target_index.view("int64")
        resampled_data = {}

        if self.ffill_limit is None:
            allowed_mask: Optional[np.ndarray] = None
        elif self.ffill_limit == 0:
            allowed_mask = np.isin(target_index.view("int64"), current_ns)
        else:
            allowed_mask = np.ones(len(target_index), dtype=bool)
            # Mask out large gaps between observed points to honor forward-fill limit
            for start_ns, end_ns in zip(current_ns[:-1], current_ns[1:]):
                gap_steps = int((end_ns - start_ns) // freq_ns - 1)
                if gap_steps > self.ffill_limit:
                    gap_start = np.searchsorted(target_index.view("int64"), start_ns + freq_ns)
                    gap_end = np.searchsorted(target_index.view("int64"), end_ns)
                    allowed_mask[gap_start:gap_end] = False

            # Avoid filling beyond the forward-fill limit after the final observation
            final_idx = np.searchsorted(target_index.view("int64"), current_ns[-1])
            trailing_steps = len(target_index) - final_idx - 1
            if trailing_steps > self.ffill_limit:
                allowed_mask[final_idx + self.ffill_limit + 1 :] = False

        for column in df.columns:
            series = df[column].to_numpy()
            valid_mask = np.isfinite(series)

            if not valid_mask.any():
                resampled_data[column] = np.full_like(target_ns, np.nan, dtype=float)
                continue

            valid_x = current_ns[valid_mask]
            valid_y = series[valid_mask]

            if len(valid_y) == 1:
                resampled_series = np.full_like(target_ns, valid_y[0], dtype=float)
            else:
                resampled_series = np.interp(target_ns, valid_x, valid_y)

            resampled_data[column] = resampled_series

        resampled_df = pd.DataFrame(resampled_data, index=target_index)

        return resampled_df, allowed_mask
