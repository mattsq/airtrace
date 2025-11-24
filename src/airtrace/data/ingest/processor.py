"""
Flight data processing - cleaning, resampling, and saving.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import pandas as pd
import pyarrow.dataset as ds

logger = logging.getLogger(__name__)


@dataclass
class FlightProcessResult:
    """Result of a processed flight."""

    flight_id: str
    output_path: Path
    length: int


class FlightProcessor:
    """Processes flight data into uniform format for AirTrace."""

    def __init__(
        self,
        output_dir: Path,
        sensors: List[str],
        timestamp_column: str,
        resample_rate: Optional[str] = None,
        dataset_name: Optional[str] = None,
        forward_fill_limit: int = 5
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
        self.forward_fill_limit = forward_fill_limit

        # Create output directory if needed
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_flight(
        self,
        flight_id: str,
        source: Union[Path, Tuple[Path, str, str]],
        min_length: int = 1
    ) -> Optional[FlightProcessResult]:
        """
        Process a single flight.

        Args:
            flight_id: Identifier for this flight
            source: Either a Path to parquet file, or (Path, column_name, value) tuple
                   for filtering multi-flight files

        Returns:
            Result with path and length, or None if processing failed/too short
        """
        try:
            df = self._load_flight_dataframe(source)

            # Standardize timestamp index
            df = self._standardize_timestamp(df)

            # Filter to sensor columns
            df = self._filter_sensors(df)

            # Resample if requested
            if self.resample_rate:
                df = self._resample(df)

            # Validate minimum length
            if len(df) < min_length:
                logger.warning(
                    f"Flight {flight_id} too short after processing ({len(df)} < {min_length}), skipping"
                )
                return None

            # Save to output directory
            output_path = self.output_dir / f"{flight_id}.parquet"
            df.to_parquet(output_path, engine="pyarrow", index=True)

            length = len(df)
            logger.info(
                f"Processed flight {flight_id}: {length} timesteps, saved to {output_path}"
            )
            return FlightProcessResult(flight_id=flight_id, output_path=output_path, length=length)

        except Exception as e:
            logger.error(f"Failed to process flight {flight_id}: {e}")
            return None

    def process_all(
        self,
        flight_registry: Dict[str, Union[Path, Tuple[Path, str, str]]],
        min_length: int = 1,
        max_workers: Optional[int] = None
    ) -> List[str]:
        """
        Process all flights in registry.

        Args:
            flight_registry: Dictionary mapping flight_id -> source
            min_length: Minimum number of timesteps (flights shorter are skipped)

        Returns:
            List of successfully processed flight IDs
        """
        success_map: Dict[str, bool] = {flight_id: False for flight_id in flight_registry}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.process_flight, flight_id, source, min_length): flight_id
                for flight_id, source in flight_registry.items()
            }

            for future in as_completed(futures):
                flight_id = futures[future]
                try:
                    result = future.result()
                except Exception as exc:  # pragma: no cover - surfaced via logger
                    logger.error(f"Failed to process flight {flight_id}: {exc}")
                    continue

                if result is not None:
                    success_map[flight_id] = True

        successful_flights = [fid for fid in flight_registry if success_map[fid]]

        logger.info(
            f"Successfully processed {len(successful_flights)}/{len(flight_registry)} flights"
        )

        return successful_flights

    def _standardize_timestamp(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize timestamp as index."""

        # If timestamp is already the index, we're done
        if pd.api.types.is_datetime64_any_dtype(df.index):
            return df.sort_index()

        # Set timestamp column as index
        if self.timestamp_column in df.columns:
            df = df.set_index(self.timestamp_column)
            df.index.name = "timestamp"
        elif df.index.name == self.timestamp_column:
            # Already set, just ensure it's sorted
            pass
        else:
            raise ValueError(f"Timestamp column '{self.timestamp_column}' not found")

        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df.sort_index()

        return df[~df.index.to_series().isna()]

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
        df_resampled = df_resampled.ffill(limit=self.forward_fill_limit)

        # Drop any remaining NaN rows
        initial_len = len(df_resampled)
        df_resampled = df_resampled.dropna()
        dropped = initial_len - len(df_resampled)

        if dropped > 0:
            logger.debug(f"Dropped {dropped} rows with NaN after resampling")

        return df_resampled

    def _load_flight_dataframe(
        self, source: Union[Path, Tuple[Path, str, str]]
    ) -> pd.DataFrame:
        """Load a single flight with column pruning and filtering."""

        if isinstance(source, tuple):
            file_path, id_column, id_value = source
            dataset = ds.dataset(file_path)
            columns = self._required_columns({id_column})
            available_columns = [c for c in columns if c in dataset.schema.names]
            table = dataset.to_table(
                columns=available_columns, filter=ds.field(id_column) == id_value
            )
            return table.to_pandas()

        dataset = ds.dataset(source)
        columns = self._required_columns()
        available_columns = [c for c in columns if c in dataset.schema.names]
        table = dataset.to_table(columns=available_columns)
        return table.to_pandas()

    def _required_columns(self, extra: Optional[Set[str]] = None) -> Set[str]:
        """Compute the minimal set of columns needed for processing."""

        columns: Set[str] = set(self.sensors)
        columns.add(self.timestamp_column)

        if extra:
            columns.update(extra)

        return columns
