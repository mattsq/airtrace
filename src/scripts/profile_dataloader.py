"""Profile data loader performance."""

import argparse
import time
from pathlib import Path

import torch


def profile_dataloader(data_config_path: str, batch_size: int = 32, num_batches: int = 100):
    """Profile dataloader performance.

    Args:
        data_config_path: Path to data config file
        batch_size: Batch size
        num_batches: Number of batches to profile
    """
    from omegaconf import OmegaConf

    from airtrace.data.datamodule import SensorDataModule

    # Load config
    cfg = OmegaConf.load(data_config_path)

    # Create datamodule
    datamodule = SensorDataModule(
        data_config=cfg.data,
        batch_size=batch_size,
        num_workers=4
    )

    try:
        datamodule.setup()
    except Exception as e:
        print(f"Error setting up datamodule: {e}")
        return

    # Get train loader
    train_loader = datamodule.train_dataloader()

    print(f"Profiling dataloader with batch_size={batch_size}...")
    print(f"Dataset size: {len(datamodule.train_dataset)}")

    # Warmup
    print("Warming up...")
    for i, batch in enumerate(train_loader):
        if i >= 5:
            break

    # Profile
    print(f"\nProfiling {num_batches} batches...")
    start_time = time.time()

    for i, batch in enumerate(train_loader):
        if i >= num_batches:
            break

    elapsed = time.time() - start_time

    # Compute stats
    samples_per_sec = (num_batches * batch_size) / elapsed
    batches_per_sec = num_batches / elapsed

    print(f"\nResults:")
    print(f"  Total time: {elapsed:.2f}s")
    print(f"  Batches/sec: {batches_per_sec:.2f}")
    print(f"  Samples/sec: {samples_per_sec:.2f}")
    print(f"  Time/batch: {elapsed/num_batches*1000:.2f}ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile dataloader")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to data config")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--num-batches", type=int, default=100,
                       help="Number of batches to profile")

    args = parser.parse_args()

    profile_dataloader(args.config, args.batch_size, args.num_batches)
