"""Generate synthetic aircraft sensor dataset.

This script generates a complete synthetic dataset with train/val/test splits
that can be used for experimentation and testing.

Usage:
    # Generate using the default synthetic_cruise config
    airtrace-generate-synthetic

    # Generate with a specific data config
    airtrace-generate-synthetic data=synthetic_cruise

    # Override specific generation parameters
    airtrace-generate-synthetic data=synthetic_cruise data.generation.n_flights=50

    # Use a different seed
    airtrace-generate-synthetic data=synthetic seed=123
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

from airtrace.data.loaders import InterimDataProcessor, RawDataLoader
from airtrace.data.synthetic import CruiseProfile, create_synthetic_dataset
from airtrace.data.windows import WindowSpec


def generate_from_config(config: DictConfig) -> None:
    """Generate synthetic dataset from Hydra config.

    Args:
        config: Hydra configuration with data.generation section
    """
    data_root = Path(config.data.root)
    dataset_name = config.data.get("dataset_name", "synthetic")

    # Extract generation config
    if not hasattr(config.data, "generation"):
        print(f"Error: Config '{dataset_name}' is missing a 'generation' section.")
        print("\nTo generate synthetic data, use a config with generation parameters, e.g.:")
        print("  airtrace-generate-synthetic data=synthetic_cruise")
        print("\nOr add a 'generation' section to your data config.")
        sys.exit(1)

    gen_cfg = config.data.generation
    profile_cfg = gen_cfg.cruise_profile

    # Create cruise profile
    profile = CruiseProfile(
        initial_altitude=profile_cfg.initial_altitude,
        initial_mach=profile_cfg.initial_mach,
        initial_weight=profile_cfg.initial_weight,
        cruise_duration=profile_cfg.cruise_duration,
        fuel_flow_base=profile_cfg.fuel_flow_base,
        n1_base=profile_cfg.n1_base,
        sample_rate=profile_cfg.sample_rate,
        altitude_variation=profile_cfg.altitude_variation,
        mach_variation=profile_cfg.mach_variation,
        turbulence_level=profile_cfg.turbulence_level,
        fuel_flow_noise=profile_cfg.fuel_flow_noise,
        mach_noise=profile_cfg.mach_noise,
        altitude_noise=profile_cfg.altitude_noise,
        n1_noise=profile_cfg.n1_noise,
        oat_noise=profile_cfg.oat_noise,
    )

    # Generate flights
    print(f"Generating {gen_cfg.n_flights} synthetic flights for '{dataset_name}'...")
    splits = create_synthetic_dataset(
        data_root=data_root,
        n_flights=gen_cfg.n_flights,
        profile=profile,
        seed=gen_cfg.seed,
        train_val_test_split=tuple(gen_cfg.train_val_test_split),
        flight_id_prefix=dataset_name,
    )

    print("Generated:")
    print(f"  Train: {len(splits['train'])} flights")
    print(f"  Val:   {len(splits['val'])} flights")
    print(f"  Test:  {len(splits['test'])} flights")

    # Process to interim
    print("\nProcessing raw -> interim...")
    raw_loader = RawDataLoader(data_root)
    all_flights = splits["train"] + splits["val"] + splits["test"]

    for flight_id in all_flights:
        raw_loader.process_to_interim(
            flight_id=flight_id, resample_rate="1S", sensor_list=config.data.sensors.use
        )
    print(f"Processed {len(all_flights)} flights to interim format")

    # Copy interim to processed (DataStore loads from processed directory)
    print("\nCopying interim -> processed...")
    processed_dir = data_root / "processed"
    processed_dir.mkdir(exist_ok=True, parents=True)
    interim_dir = data_root / "interim"

    import shutil
    for flight_id in all_flights:
        src = interim_dir / f"{flight_id}.parquet"
        dst = processed_dir / f"{flight_id}.parquet"
        shutil.copy2(src, dst)
    print(f"Copied {len(all_flights)} flights to processed format")

    # Create windows
    print("\nCreating sliding windows...")
    processor = InterimDataProcessor(data_root)

    window_cfg = config.data.window
    window_spec = WindowSpec(
        input_len=window_cfg.input_len,
        pred_len=window_cfg.pred_len,
        stride=window_cfg.stride,
        target_sensors=window_cfg.target_sensors,
    )

    for split_name, flight_ids in splits.items():
        index_df = processor.create_windows(
            flight_ids=flight_ids,
            window_spec=window_spec,
            output_name=f"{dataset_name}_{split_name}",
        )
        print(f"  {split_name}: {len(index_df)} windows")

    print(f"\nSynthetic dataset ready in {data_root}")
    print(f"  Dataset: {dataset_name}")
    print(f"  Config: configs/data/{dataset_name}.yaml")


@hydra.main(version_base=None, config_path="pkg://airtrace.configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point using Hydra config."""
    generate_from_config(cfg)


def cli() -> None:
    """Entry point for the airtrace-generate-synthetic console script.

    This processes sys.argv and delegates to the Hydra main function,
    allowing users to pass Hydra overrides directly on the command line.
    """
    # Default to synthetic_cruise if no data config is specified
    if len(sys.argv) == 1 or not any(arg.startswith("data=") for arg in sys.argv[1:]):
        sys.argv.append("data=synthetic_cruise")

    main()


if __name__ == "__main__":
    cli()
