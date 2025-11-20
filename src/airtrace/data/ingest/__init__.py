"""
Data ingestion module for converting raw parquet files to AirTrace format.

This module provides tools to:
- Validate raw sensor data
- Process flights into uniform format
- Generate window indices for train/val/test splits
- Auto-generate dataset configuration YAML files
"""

from .validator import FlightValidator, ValidationReport, SensorMetadata
from .processor import FlightProcessor
from .indexer import WindowIndexer
from .config_gen import ConfigGenerator

__all__ = [
    "FlightValidator",
    "ValidationReport",
    "SensorMetadata",
    "FlightProcessor",
    "WindowIndexer",
    "ConfigGenerator",
]
