"""
Flight data validation and schema detection.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


@dataclass
class SensorMetadata:
    """Metadata about detected sensors and data characteristics."""

    sensors: List[str]
    timestamp_column: str
    sampling_rate: Optional[float]  # Median seconds between samples
    sampling_std: Optional[float]  # Std dev of time deltas
    dtypes: Dict[str, str] = field(default_factory=dict)
    nan_percentages: Dict[str, float] = field(default_factory=dict)


@dataclass
class ValidationReport:
    """Report of validation checks and warnings."""

    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def has_errors(self) -> bool:
        return len(self.errors) > 0

    def add_error(self, msg: str):
        self.errors.append(msg)

    def add_warning(self, msg: str):
        self.warnings.append(msg)


class FlightValidator:
    """Validates raw flight data and detects schema."""

    def __init__(
        self,
        input_path: Path,
        timestamp_column: Optional[str] = None,
        flight_id_column: Optional[str] = None
    ):
        """
        Args:
            input_path: Path to parquet file or directory
            timestamp_column: Name of timestamp column (auto-detect if None)
            flight_id_column: Name of flight ID column (single flight if None)
        """
        self.input_path = Path(input_path)
        self.timestamp_column = timestamp_column
        self.flight_id_column = flight_id_column
        self.report = ValidationReport()

    def validate(self) -> ValidationReport:
        """Run all validation checks."""

        # Check path exists
        if not self.input_path.exists():
            self.report.add_error(f"Path does not exist: {self.input_path}")
            return self.report

        # Get list of parquet files
        files = self._get_parquet_files()
        if not files:
            self.report.add_error(f"No parquet files found at {self.input_path}")
            return self.report

        # Load sample data from first file
        try:
            sample_df = pd.read_parquet(files[0], engine="pyarrow")
            if len(sample_df) > 1000:
                sample_df = sample_df.iloc[:1000]
        except Exception as e:
            self.report.add_error(f"Failed to load {files[0]}: {e}")
            return self.report

        # Validate timestamp
        timestamp_col = self._validate_timestamp(sample_df)
        if timestamp_col is None:
            return self.report

        # Validate sensors
        sensors = self._detect_sensors(sample_df, timestamp_col)
        if not sensors:
            self.report.add_error("No numeric sensor columns detected")
            return self.report

        # Check data quality
        self._check_data_quality(sample_df, sensors)

        return self.report

    def detect_schema(self) -> SensorMetadata:
        """Detect sensor schema and metadata."""

        files = self._get_parquet_files()
        if not files:
            raise ValueError("No parquet files found")

        # Load first file
        df = pd.read_parquet(files[0], engine="pyarrow")

        # Detect timestamp
        timestamp_col = self._detect_timestamp_column(df)
        if timestamp_col is None:
            raise ValueError("No timestamp column found")

        # Set timestamp as index if needed
        if timestamp_col != df.index.name:
            df = df.set_index(timestamp_col)

        # Detect sensors
        sensors = self._detect_sensors(df, timestamp_col)

        # Compute sampling statistics
        time_deltas = df.index.to_series().diff().dt.total_seconds().dropna()
        sampling_rate = float(time_deltas.median()) if len(time_deltas) > 0 else None
        sampling_std = float(time_deltas.std()) if len(time_deltas) > 0 else None

        # Compute NaN percentages
        nan_percentages = {}
        for sensor in sensors:
            if sensor in df.columns:
                nan_pct = float(df[sensor].isna().sum() / len(df) * 100)
                nan_percentages[sensor] = nan_pct

        # Get dtypes
        dtypes = {sensor: str(df[sensor].dtype) for sensor in sensors if sensor in df.columns}

        return SensorMetadata(
            sensors=sensors,
            timestamp_column=timestamp_col,
            sampling_rate=sampling_rate,
            sampling_std=sampling_std,
            dtypes=dtypes,
            nan_percentages=nan_percentages
        )

    def detect_flights(self) -> Dict[str, Path]:
        """
        Identify individual flights from input.

        Returns:
            Dictionary mapping flight_id -> file path or (file path, group key)
        """
        files = self._get_parquet_files()
        flight_registry = {}

        if self.input_path.is_dir():
            # Each file is a separate flight
            for i, file_path in enumerate(files):
                flight_id = file_path.stem  # Use filename without extension
                flight_registry[flight_id] = file_path

        elif self.flight_id_column is not None:
            # Single file with flight ID column - need to group
            df = pd.read_parquet(files[0], columns=[self.flight_id_column])
            flight_ids = df[self.flight_id_column].unique()

            for flight_id in flight_ids:
                # Store (file, flight_id_column, flight_id_value) tuple
                flight_registry[str(flight_id)] = (files[0], self.flight_id_column, flight_id)

        else:
            # Single file, single flight
            flight_id = files[0].stem
            flight_registry[flight_id] = files[0]

        return flight_registry

    def _get_parquet_files(self) -> List[Path]:
        """Get list of parquet files from input path."""
        if self.input_path.is_file():
            if self.input_path.suffix.lower() in [".parquet", ".pq"]:
                return [self.input_path]
            else:
                return []
        elif self.input_path.is_dir():
            parquet_files = list(self.input_path.glob("*.parquet"))
            parquet_files.extend(self.input_path.glob("*.pq"))
            return sorted(parquet_files)
        else:
            return []

    def _detect_timestamp_column(self, df: pd.DataFrame) -> Optional[str]:
        """Detect the timestamp column."""

        # Check if index is datetime
        if pd.api.types.is_datetime64_any_dtype(df.index):
            return df.index.name or "timestamp"

        # Check user-specified column
        if self.timestamp_column:
            if self.timestamp_column in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[self.timestamp_column]):
                    return self.timestamp_column
                else:
                    return None
            else:
                return None

        # Search for datetime columns
        datetime_cols = [
            col for col in df.columns
            if pd.api.types.is_datetime64_any_dtype(df[col])
        ]

        if len(datetime_cols) == 1:
            return datetime_cols[0]
        elif len(datetime_cols) > 1:
            # Ambiguous - prefer common names
            for common_name in ["timestamp", "time", "datetime", "date"]:
                if common_name in datetime_cols:
                    return common_name
            return datetime_cols[0]  # Default to first

        return None

    def _validate_timestamp(self, df: pd.DataFrame) -> Optional[str]:
        """Validate timestamp column exists."""

        timestamp_col = self._detect_timestamp_column(df)

        if timestamp_col is None:
            if self.timestamp_column:
                self.report.add_error(
                    f"Specified timestamp column '{self.timestamp_column}' not found or not datetime type"
                )
            else:
                self.report.add_error(
                    "No timestamp column found. Use --timestamp-column or ensure index is datetime"
                )

        return timestamp_col

    def _detect_sensors(self, df: pd.DataFrame, timestamp_col: str) -> List[str]:
        """Detect sensor columns (numeric, excluding metadata)."""

        # Get all numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Exclude flight ID column if specified
        if self.flight_id_column and self.flight_id_column in numeric_cols:
            numeric_cols.remove(self.flight_id_column)

        # Exclude timestamp if it's a column (not index)
        if timestamp_col in numeric_cols:
            numeric_cols.remove(timestamp_col)

        return numeric_cols

    def _check_data_quality(self, df: pd.DataFrame, sensors: List[str]):
        """Check for data quality issues."""

        # Check for all-NaN sensors
        for sensor in sensors:
            if sensor in df.columns:
                if df[sensor].isna().all():
                    self.report.add_error(f"Sensor '{sensor}' contains only NaN values")

                # Warn if >20% NaN
                nan_pct = df[sensor].isna().sum() / len(df) * 100
                if nan_pct > 20:
                    self.report.add_warning(
                        f"Sensor '{sensor}' has {nan_pct:.1f}% missing values"
                    )

        # Check for duplicate timestamps
        timestamp_col = self._detect_timestamp_column(df)
        if timestamp_col:
            if timestamp_col == df.index.name or pd.api.types.is_datetime64_any_dtype(df.index):
                timestamps = df.index
            else:
                timestamps = df[timestamp_col]

            if timestamps.duplicated().any():
                n_dupes = timestamps.duplicated().sum()
                self.report.add_warning(f"Found {n_dupes} duplicate timestamps")

        # Check for irregular sampling
        if timestamp_col:
            if timestamp_col == df.index.name or pd.api.types.is_datetime64_any_dtype(df.index):
                time_series = df.index.to_series()
            else:
                time_series = df[timestamp_col]

            deltas = time_series.diff().dt.total_seconds().dropna()
            if len(deltas) > 0:
                median_delta = deltas.median()
                std_delta = deltas.std()

                # If std is >30% of median, sampling is irregular
                if std_delta > 0.3 * median_delta:
                    self.report.add_warning(
                        f"Irregular sampling detected (median={median_delta:.2f}s, std={std_delta:.2f}s). "
                        "Consider using --resample-rate"
                    )
