"""
Data ingestion CLI for converting raw parquet files to AirTrace format.

Usage:
    airtrace ingest <input_path> --dataset-name <name> [options]
"""

import sys
import argparse
import airtrace
from pathlib import Path
import numpy as np
import logging

from airtrace.data.ingest import (
    FlightValidator,
    FlightProcessor,
    WindowIndexer,
    ConfigGenerator,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Ingest raw parquet sensor data into AirTrace format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single flight file
  airtrace ingest data/my_flight.parquet --dataset-name qantas_737

  # Directory of flights
  airtrace ingest data/raw/qantas/ --dataset-name qantas_fleet

  # With custom window parameters
  airtrace ingest data/flights.parquet --dataset-name airbus \\
    --input-len 512 --pred-len 64 --stride 16

  # With resampling
  airtrace ingest data/flights.parquet --dataset-name boeing \\
    --resample-rate 1S --target-sensors fuel_flow,mach,n1
        """
    )

    # Required arguments
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to parquet file or directory of parquet files"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        help="Name for the dataset (e.g., 'qantas_737')"
    )

    # Window parameters
    parser.add_argument(
        "--input-len",
        type=int,
        default=256,
        help="Input sequence length (default: 256)"
    )
    parser.add_argument(
        "--pred-len",
        type=int,
        default=32,
        help="Prediction sequence length (default: 32)"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=32,
        help="Sliding window stride (default: 32)"
    )

    # Index generation
    parser.add_argument(
        "--index-workers",
        type=int,
        default=1,
        help="Number of worker processes for window index generation",
    )
    parser.add_argument(
        "--index-partitioned",
        action="store_true",
        help="Write window indices as partitioned parquet datasets",
    )
    parser.add_argument(
        "--stream-index-writes",
        action="store_true",
        help=(
            "Stream window indices directly to parquet instead of keeping the full DataFrame"
        ),
    )

    # Sensor configuration
    parser.add_argument(
        "--target-sensors",
        type=str,
        default=None,
        help="Comma-separated list of sensors to predict (default: all sensors)"
    )

    # Data splitting
    parser.add_argument(
        "--split",
        type=str,
        default="0.7,0.15,0.15",
        help="Train/val/test split ratios (default: '0.7,0.15,0.15')"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits (default: 42)"
    )

    # Data processing
    parser.add_argument(
        "--resample-rate",
        type=str,
        default=None,
        help="Resample to uniform rate (e.g., '1S' for 1 second)"
    )
    parser.add_argument(
        "--resample-backend",
        type=str,
        default="pandas",
        choices=["pandas", "numpy"],
        help="Backend to use when resampling (default: pandas)",
    )
    parser.add_argument(
        "--ffill-limit",
        type=int,
        default=5,
        help="Forward-fill limit for small gaps during resampling (set to 0 to disable)",
    )
    parser.add_argument(
        "--timestamp-column",
        type=str,
        default=None,
        help="Name of timestamp column (auto-detect if not specified)"
    )
    parser.add_argument(
        "--flight-id-column",
        type=str,
        default=None,
        help="Column name for flight IDs in multi-flight files"
    )

    # Control flags
    parser.add_argument(
        "--skip-processed",
        action="store_true",
        help="Skip creating processed files (assume they exist)"
    )
    parser.add_argument(
        "--skip-config",
        action="store_true",
        help="Skip creating YAML config file"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs without creating files"
    )

    return parser.parse_args()


def parse_split_ratios(split_str: str):
    """Parse split ratio string."""
    try:
        ratios = [float(x) for x in split_str.split(",")]
        if len(ratios) != 3:
            raise ValueError("Must provide exactly 3 ratios")
        if not np.isclose(sum(ratios), 1.0):
            raise ValueError(f"Ratios must sum to 1.0 (got {sum(ratios)})")
        return ratios
    except Exception as e:
        logger.error(f"Invalid split ratios '{split_str}': {e}")
        sys.exit(1)


def split_flights(flight_ids, split_ratios, seed):
    """Split flights into train/val/test."""
    train_ratio, val_ratio, test_ratio = split_ratios
    n_flights = len(flight_ids)

    if n_flights < 3:
        logger.warning(f"Only {n_flights} flights - some splits may be empty")

    n_train = int(n_flights * train_ratio)
    n_val = int(n_flights * val_ratio)
    n_test = n_flights - n_train - n_val

    # Shuffle with seed
    rng = np.random.RandomState(seed)
    shuffled_ids = rng.permutation(flight_ids)

    train_ids = shuffled_ids[:n_train].tolist()
    val_ids = shuffled_ids[n_train:n_train+n_val].tolist()
    test_ids = shuffled_ids[n_train+n_val:].tolist()

    return train_ids, val_ids, test_ids


def print_summary(
    dataset_name,
    train_ids,
    val_ids,
    test_ids,
    sensors,
    target_sensors,
    input_len,
    pred_len,
    stride,
    train_windows,
    val_windows,
    test_windows
):
    """Print summary report."""
    print("\n" + "=" * 60)
    print(f"Dataset Ingestion Complete: {dataset_name}")
    print("=" * 60)

    print("\nFiles Created:")
    print(f"  {len(train_ids) + len(val_ids) + len(test_ids)} processed flight files (data/processed/)")
    print(f"  3 window index files (data/metadata/)")
    print(f"  1 config file (src/airtrace/configs/data/{dataset_name}.yaml)")

    print("\nData Summary:")
    print(f"  Total flights: {len(train_ids) + len(val_ids) + len(test_ids)}")
    print(f"  Train flights: {len(train_ids)} ({train_windows} windows)")
    print(f"  Val flights: {len(val_ids)} ({val_windows} windows)")
    print(f"  Test flights: {len(test_ids)} ({test_windows} windows)")

    print(f"\nSensors ({len(sensors)}):")
    sensor_str = ", ".join(sensors)
    print(f"  {sensor_str}")

    print("\nWindow Configuration:")
    print(f"  Input length: {input_len} timesteps")
    print(f"  Prediction length: {pred_len} timesteps")
    print(f"  Stride: {stride} timesteps")

    print(f"\nTarget Sensors ({len(target_sensors)}):")
    target_str = ", ".join(target_sensors)
    print(f"  {target_str}")

    print("\nNext Steps:")
    print("  1. Verify config: src/airtrace/configs/data/{}.yaml".format(dataset_name))
    print("  2. Run training:")
    print(f"     airtrace train data={dataset_name} model=gru_ar task.name=one_step")
    print("\n" + "=" * 60 + "\n")


def ingest_dataset(args):
    """Main ingestion pipeline."""

    # Parse arguments
    input_path = Path(args.input_path)
    dataset_name = args.dataset_name
    split_ratios = parse_split_ratios(args.split)

    logger.info(f"Starting ingestion: {input_path} -> {dataset_name}")

    # Step 1: Validate input
    logger.info("\n[Step 1/5] Validating input data...")
    validator = FlightValidator(
        input_path,
        timestamp_column=args.timestamp_column,
        flight_id_column=args.flight_id_column
    )

    validation_report = validator.validate()

    # Print warnings
    if validation_report.warnings:
        logger.warning("\nWarnings:")
        for warning in validation_report.warnings:
            logger.warning(f"  {warning}")

    # Print errors and exit if any
    if validation_report.has_errors():
        logger.error("\nErrors:")
        for error in validation_report.errors:
            logger.error(f"  {error}")
        logger.error("\nValidation failed. Please fix errors and try again.")
        sys.exit(1)

    # Detect schema
    sensor_metadata = validator.detect_schema()
    flight_registry = validator.detect_flights()

    logger.info(f"  Found {len(flight_registry)} flights")
    logger.info(f"  Detected {len(sensor_metadata.sensors)} sensors")
    if sensor_metadata.sampling_rate:
        logger.info(f"  Sampling rate: {sensor_metadata.sampling_rate:.2f}s (std: {sensor_metadata.sampling_std:.2f}s)")

    # Dry run exit
    if args.dry_run:
        logger.info("\n[DRY RUN] Would create:")
        logger.info(f"  {len(flight_registry)} files in data/processed/")
        logger.info(f"  3 index files in data/metadata/")
        logger.info(f"  1 config in src/airtrace/configs/data/{dataset_name}.yaml")
        logger.info("\nRun without --dry-run to create files.")
        return

    # Step 2: Split flights
    logger.info("\n[Step 2/5] Splitting flights into train/val/test...")
    all_flight_ids = list(flight_registry.keys())
    train_ids, val_ids, test_ids = split_flights(all_flight_ids, split_ratios, args.seed)
    logger.info(f"  Train: {len(train_ids)} flights")
    logger.info(f"  Val: {len(val_ids)} flights")
    logger.info(f"  Test: {len(test_ids)} flights")

    # Step 3: Process flights
    if not args.skip_processed:
        logger.info("\n[Step 3/5] Processing flights...")
        processor = FlightProcessor(
            Path("data/processed"),
            sensor_metadata.sensors,
            sensor_metadata.timestamp_column,
            resample_rate=args.resample_rate,
            dataset_name=dataset_name,
            timestamp_dtype=sensor_metadata.timestamp_dtype,
            resample_backend=args.resample_backend,
            ffill_limit=args.ffill_limit,
        )

        # Process with minimum length requirement
        min_length = args.input_len + args.pred_len
        processed_ids = processor.process_all(flight_registry, min_length=min_length)

        # Update splits to remove failed flights
        train_ids = [id for id in train_ids if id in processed_ids]
        val_ids = [id for id in val_ids if id in processed_ids]
        test_ids = [id for id in test_ids if id in processed_ids]

        logger.info(f"  Successfully processed {len(processed_ids)} flights")
    else:
        logger.info("\n[Step 3/5] Skipping flight processing (--skip-processed)")

    # Step 4: Create window indices
    logger.info("\n[Step 4/5] Creating window indices...")

    # Determine target sensors
    if args.target_sensors:
        target_sensors = args.target_sensors.split(",")
        # Validate all targets are in detected sensors
        invalid = set(target_sensors) - set(sensor_metadata.sensors)
        if invalid:
            logger.error(f"Target sensors not found in data: {invalid}")
            sys.exit(1)
    else:
        target_sensors = sensor_metadata.sensors

    indexer = WindowIndexer(
        args.input_len,
        args.pred_len,
        args.stride,
        Path("data/processed")
    )

    (train_path, val_path, test_path), window_counts = indexer.create_all_indices(
        train_ids,
        val_ids,
        test_ids,
        Path("data/metadata"),
        dataset_name,
        num_workers=args.index_workers,
        write_partitioned=args.index_partitioned,
        materialize_dataframe=not args.stream_index_writes,
        return_counts=True,
    )

    train_windows = window_counts["train"]
    val_windows = window_counts["val"]
    test_windows = window_counts["test"]

    # Step 5: Generate config
    if not args.skip_config:
        logger.info("\n[Step 5/5] Generating config file...")
        
        # Get package config directory
        import airtrace
        pkg_path = Path(airtrace.__file__).parent
        config_path = pkg_path / "configs" / "data" / f"{dataset_name}.yaml"
        
        config_gen = ConfigGenerator(
            dataset_name,
            sensor_metadata.sensors,
            args.input_len,
            args.pred_len,
            args.stride,
            target_sensors
        )

        config_gen.generate(
            config_path,
            n_flights=len(train_ids) + len(val_ids) + len(test_ids),
            input_path=str(input_path),
            force=args.force
        )
    else:
        logger.info("\n[Step 5/5] Skipping config generation (--skip-config)")

    # Print summary
    print_summary(
        dataset_name,
        train_ids,
        val_ids,
        test_ids,
        sensor_metadata.sensors,
        target_sensors,
        args.input_len,
        args.pred_len,
        args.stride,
        train_windows,
        val_windows,
        test_windows
    )


def main():
    """Main entry point."""
    args = parse_args()
    try:
        ingest_dataset(args)
    except KeyboardInterrupt:
        logger.info("\nIngestion cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nIngestion failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
