#!/usr/bin/env python3
"""Generate synthetic aircraft cruise sensor dataset.

This script generates a complete synthetic dataset with train/val/test splits
that can be used for experimentation and testing.

Usage:
    python generate_synthetic_data.py --n-flights 20 --output data/
    python generate_synthetic_data.py --config configs/data/synthetic_cruise.yaml
"""

import argparse
from pathlib import Path
from typing import Optional

import hydra
from omegaconf import DictConfig, OmegaConf

from airtrace.data.loaders import InterimDataProcessor, RawDataLoader
from airtrace.data.synthetic import CruiseProfile, create_synthetic_dataset
from airtrace.data.windows import WindowSpec


def generate_from_config(config: DictConfig):
    """Generate synthetic dataset from Hydra config.

    Args:
        config: Hydra configuration with data.generation section
    """
    data_root = Path(config.data.root)

    # Extract generation config
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
    print(f"Generating {gen_cfg.n_flights} synthetic cruise flights...")
    splits = create_synthetic_dataset(
        data_root=data_root,
        n_flights=gen_cfg.n_flights,
        profile=profile,
        seed=gen_cfg.seed,
        train_val_test_split=tuple(gen_cfg.train_val_test_split)
    )

    print(f"Generated:")
    print(f"  Train: {len(splits['train'])} flights")
    print(f"  Val:   {len(splits['val'])} flights")
    print(f"  Test:  {len(splits['test'])} flights")

    # Process to interim
    print("\nProcessing raw → interim...")
    raw_loader = RawDataLoader(data_root)
    all_flights = splits['train'] + splits['val'] + splits['test']

    for flight_id in all_flights:
        raw_loader.process_to_interim(
            flight_id=flight_id,
            resample_rate="1S",
            sensor_list=config.data.sensors.use
        )
    print(f"Processed {len(all_flights)} flights to interim format")

    # Create windows
    print("\nCreating sliding windows...")
    processor = InterimDataProcessor(data_root)

    window_cfg = config.data.window
    window_spec = WindowSpec(
        input_len=window_cfg.input_len,
        pred_len=window_cfg.pred_len,
        stride=window_cfg.stride,
        target_sensors=window_cfg.target_sensors
    )

    for split_name, flight_ids in splits.items():
        index_df = processor.create_windows(
            flight_ids=flight_ids,
            window_spec=window_spec,
            output_name=f"synthetic_cruise_{split_name}"
        )
        print(f"  {split_name}: {len(index_df)} windows")

    print(f"\n✓ Synthetic dataset ready in {data_root}")
    print(f"  Use config: configs/data/synthetic_cruise.yaml")


def generate_standalone(
    output_dir: Path,
    n_flights: int = 20,
    duration_hours: float = 1.0,
    seed: Optional[int] = 42
):
    """Generate synthetic dataset with standalone parameters.

    Args:
        output_dir: Output directory for data
        n_flights: Number of flights to generate
        duration_hours: Duration of each cruise in hours
        seed: Random seed
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Create default profile
    profile = CruiseProfile(
        cruise_duration=int(duration_hours * 3600),
    )

    # Generate
    print(f"Generating {n_flights} synthetic cruise flights...")
    splits = create_synthetic_dataset(
        data_root=output_dir,
        n_flights=n_flights,
        profile=profile,
        seed=seed
    )

    print(f"Generated:")
    print(f"  Train: {len(splits['train'])} flights")
    print(f"  Val:   {len(splits['val'])} flights")
    print(f"  Test:  {len(splits['test'])} flights")

    # Process to interim
    print("\nProcessing to interim format...")
    raw_loader = RawDataLoader(output_dir)
    all_flights = splits['train'] + splits['val'] + splits['test']

    for flight_id in all_flights:
        raw_loader.process_to_interim(flight_id=flight_id, resample_rate="1S")

    print(f"\n✓ Synthetic dataset ready in {output_dir}")


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main_hydra(cfg: DictConfig):
    """Main entry point using Hydra config."""
    generate_from_config(cfg)


def main_cli():
    """Main entry point with CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic aircraft cruise sensor dataset"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default="data/",
        help="Output directory for data (default: data/)"
    )
    parser.add_argument(
        "--n-flights",
        type=int,
        default=20,
        help="Number of flights to generate (default: 20)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=1.0,
        help="Duration of each cruise in hours (default: 1.0)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--use-config",
        action="store_true",
        help="Use Hydra config instead of CLI args"
    )

    args = parser.parse_args()

    if args.use_config:
        # Delegate to Hydra
        main_hydra()
    else:
        # Use CLI arguments
        generate_standalone(
            output_dir=args.output,
            n_flights=args.n_flights,
            duration_hours=args.duration,
            seed=args.seed
        )


if __name__ == "__main__":
    main_cli()
