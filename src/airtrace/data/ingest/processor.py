"""
Flight data processing - cleaning, resampling, and saving.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import pyarrow.dataset as ds
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
        ffill_limit: Optional[int] = 5,
        metadata_dir: Path = Path("data/metadata")
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
            metadata_dir: Directory to store processing metadata for caching
        """
        self.output_dir = Path(output_dir)
        self.sensors = sensors
        self.timestamp_column = timestamp_column
        self.resample_rate = resample_rate
        self.dataset_name = dataset_name
        self.timestamp_dtype = timestamp_dtype
        self.resample_backend = resample_backend
        self.ffill_limit = ffill_limit
        self.metadata_dir = Path(metadata_dir)
        self.metadata_path = (
            self.metadata_dir / f"{self.dataset_name}_processed_meta.json"
            if self.dataset_name
            else None
        )

        if self.resample_backend not in {"pandas", "numpy"}:
            raise ValueError("resample_backend must be 'pandas' or 'numpy'")

        # Create output directory if needed
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_flight(
        self,
        flight_id: str,
        source: Union[Path, Tuple[Path, str, str]]
    ) -> Optional[Tuple[Path, int]]:
        """
        Process a single flight.

        Args:
            flight_id: Identifier for this flight
            source: Either a Path to parquet file, or (Path, column_name, value) tuple
                   for filtering multi-flight files

        Returns:
            Tuple of (output_path, length) or None if processing failed
        """
        try:
            # Load flight data with PyArrow column projection for efficiency
            if isinstance(source, tuple):
                # Multi-flight file - use PyArrow scanner with filter
                file_path, id_column, id_value = source
                columns = list({*self.sensors, self.timestamp_column, id_column})
                dataset = ds.dataset(file_path, format="parquet")
                scanner = dataset.scanner(
                    columns=columns,
                    filter=ds.field(id_column) == id_value,
                    use_threads=True,
                )
                batches = [batch.to_pandas() for batch in scanner.to_batches()]

                if not batches:
                    raise ValueError(
                        f"No rows found for {id_column}={id_value} in {file_path}"
                    )

                df = pd.concat(batches, ignore_index=True)
            else:
                # Single flight file - use column projection
                df = pd.read_parquet(
                    source,
                    columns=list({*self.sensors, self.timestamp_column}),
                    engine="pyarrow"
                )

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
            return output_path, len(df)

        except Exception as e:
            logger.error(f"Failed to process flight {flight_id}: {e}")
            return None

    def process_all(
        self,
        flight_registry: Dict[str, Union[Path, Tuple[Path, str, str]]],
        min_length: int = 1
    ) -> Tuple[List[str], Dict[str, Dict]]:
        """
        Process all flights with automatic caching.

        Args:
            flight_registry: Dictionary mapping flight_id -> source
            min_length: Minimum number of timesteps (flights shorter are skipped)

        Returns:
            Tuple of (successful flight IDs, metadata dict keyed by flight_id)
        """
        successful_flights: List[str] = []
        processed_metadata: Dict[str, Dict] = {}
        existing_metadata = self._load_existing_metadata()

        for flight_id, source in flight_registry.items():
            source_path = source[0] if isinstance(source, tuple) else source
            signature = self._compute_source_signature(Path(source_path))
            output_path = self.output_dir / f"{flight_id}.parquet"

            # Automatic reuse: check if we can reuse existing processed file
            if (
                output_path.exists()
                and self._is_metadata_match(flight_id, signature, existing_metadata)
            ):
                stored = existing_metadata[flight_id]
                length = int(stored.get("length", 0))
                if length >= min_length:
                    logger.info(f"Reusing processed flight {flight_id} (cached)")
                    successful_flights.append(flight_id)
                    processed_metadata[flight_id] = stored
                    continue

            # Process flight
            result = self.process_flight(flight_id, source)
            if result is None:
                continue

            output_path, length = result
            if length < min_length:
                logger.warning(
                    f"Flight {flight_id} too short ({length} < {min_length}), skipping"
                )
                output_path.unlink(missing_ok=True)
                continue

            successful_flights.append(flight_id)
            processed_metadata[flight_id] = {
                "source_signature": signature,
                "length": length,
                "processed_path": str(output_path),
                "processed_mtime": output_path.stat().st_mtime,
            }

        # Save metadata for future runs
        if self.metadata_path:
            self._save_metadata(processed_metadata)

        logger.info(
            f"Successfully processed {len(successful_flights)}/{len(flight_registry)} flights"
        )

        return successful_flights, processed_metadata

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

    def _compute_source_signature(self, source_path: Path) -> Dict:
        """Compute checksum signature for source file."""
        stat = source_path.stat()
        checksum = hashlib.md5()
        checksum.update(str(stat.st_mtime).encode())
        checksum.update(str(stat.st_size).encode())
        return {
            "mtime": stat.st_mtime,
            "size": stat.st_size,
            "checksum": checksum.hexdigest(),
        }

    def _is_metadata_match(
        self, flight_id: str, signature: Dict, existing_metadata: Dict
    ) -> bool:
        """Check if existing metadata matches current source signature."""
        if flight_id not in existing_metadata:
            return False
        stored_sig = existing_metadata[flight_id].get("source_signature", {})
        return stored_sig.get("checksum") == signature.get("checksum")

    def _save_metadata(self, metadata: Dict) -> None:
        """Save processing metadata to JSON file."""
        if not self.metadata_path:
            return
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def _load_existing_metadata(self) -> Dict:
        """Load existing processing metadata if available."""
        if not self.metadata_path or not self.metadata_path.exists():
            return {}
        try:
            with open(self.metadata_path, "r") as f:
                return json.load(f)
        except Exception:
            return {}
