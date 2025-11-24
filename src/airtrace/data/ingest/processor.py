"""
Flight data processing - cleaning, resampling, and saving.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
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
        dataset_name: Optional[str] = None
    ):
        """
        Args:
            output_dir: Directory to save processed flight files
            sensors: List of sensor columns to include
            timestamp_column: Name of timestamp column
            resample_rate: Optional resample rate (e.g., "1S" for 1 second)
            dataset_name: Optional dataset name for auto-naming flights
        """
        self.output_dir = Path(output_dir)
        self.sensors = sensors
        self.timestamp_column = timestamp_column
        self.resample_rate = resample_rate
        self.dataset_name = dataset_name

        # Create output directory if needed
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_flight(
        self,
        flight_id: str,
        source: Union[Path, Tuple[Path, str, str]],
        with_length: bool = False,
    ) -> Optional[Union[Path, Tuple[Path, int]]]:
        """
        Process a single flight.

        Args:
            flight_id: Identifier for this flight
            source: Either a Path to parquet file, or (Path, column_name, value) tuple
                   for filtering multi-flight files
            with_length: When True, return the processed length alongside the path

        Returns:
            Path to saved processed file (and optionally its length), or None if processing failed
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
            if with_length:
                return output_path, len(df)
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
            result = self.process_flight(flight_id, source, with_length=True)

            if result is not None:
                output_path, processed_length = result
                if processed_length >= min_length:
                    successful_flights.append(flight_id)
                else:
                    logger.warning(
                        f"Flight {flight_id} too short ({processed_length} < {min_length}), skipping"
                    )
                    # Remove the file
                    output_path.unlink(missing_ok=True)

        logger.info(
            f"Successfully processed {len(successful_flights)}/{len(flight_registry)} flights"
        )

        return successful_flights

    def _standardize_timestamp(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize timestamp as index."""

        # If timestamp is already the index, we're done
        if pd.api.types.is_datetime64_any_dtype(df.index):
            # Ensure sorted
            df = df.sort_index()
            return df

        # Set timestamp column as index
        if self.timestamp_column in df.columns:
            df = df.set_index(self.timestamp_column)
            df.index.name = "timestamp"
        elif df.index.name == self.timestamp_column:
            # Already set, just ensure it's sorted
            pass
        else:
            raise ValueError(f"Timestamp column '{self.timestamp_column}' not found")

        # Ensure sorted
        df = df.sort_index()

        return df

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

        # Resample using mean (for numeric data)
        df_resampled = df.resample(self.resample_rate).mean()

        # Forward-fill small gaps (up to 5 steps)
        df_resampled = df_resampled.ffill(limit=5)

        # Drop any remaining NaN rows
        initial_len = len(df_resampled)
        df_resampled = df_resampled.dropna()
        dropped = initial_len - len(df_resampled)

        if dropped > 0:
            logger.debug(f"Dropped {dropped} rows with NaN after resampling")

        return df_resampled
