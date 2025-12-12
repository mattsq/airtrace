"""Debug script to check transform behavior."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np
from omegaconf import OmegaConf
from airtrace.data.datamodule import SensorDataModule
from airtrace.transforms.registry import build_transforms

# Load config
cfg = OmegaConf.load('runs/20251128/medium_20k_fixed/.hydra/config.yaml')

# Build transforms
transform_pipeline = build_transforms(cfg.transforms.pipeline)

# Build data module
datamodule = SensorDataModule(
    data_config=cfg.data,
    transforms=transform_pipeline,
    batch_size=1,
    num_workers=0,
    shuffle=None
)
datamodule.setup()

# Get a sample
test_loader = datamodule.test_dataloader()
batch = next(iter(test_loader))

x = batch['x'][0]  # [T_in, D] - transformed
y = batch['y'][0]  # [T_out, D] - transformed

print(f"Transformed x shape: {x.shape}")
print(f"Transformed y shape: {y.shape}")
print(f"\nTransformed x[-5:, 0] (last 5 altitude values in input):")
print(x[-5:, 0].numpy())
print(f"\nTransformed y[:5, 0] (first 5 altitude values in target):")
print(y[:5, 0].numpy())

# Try inverse transform on concatenated sequence
full_seq = np.concatenate([x.numpy(), y.numpy()], axis=0)
print(f"\nFull transformed sequence shape: {full_seq.shape}")
print(f"Transformed full_seq[123:133, 0] (around boundary):")
print(full_seq[123:133, 0])

# Inverse transform
full_seq_orig, _ = transform_pipeline.inverse(full_seq, None)
print(f"\nInverse transformed shape: {full_seq_orig.shape}")
print(f"Original full_seq[123:133, 0] (around boundary):")
print(full_seq_orig[123:133, 0])

# Check if we can access the original data before transforms
print("\n" + "="*80)
print("Checking original data from dataset...")
print("="*80)

# Access the underlying dataset
test_dataset = datamodule.test_dataset
print(f"\nTest dataset type: {type(test_dataset)}")
print(f"Test dataset has transforms: {hasattr(test_dataset, 'transforms')}")

# Get raw sample
original_transforms = test_dataset.transforms
test_dataset.transforms = None
raw_sample = test_dataset[0]
test_dataset.transforms = original_transforms

raw_x = raw_sample['x'].numpy()
raw_y = raw_sample['y'].numpy()

print(f"\nRaw x shape: {raw_x.shape}")
print(f"Raw y shape: {raw_y.shape}")
print(f"\nRaw x[-5:, 0] (last 5 altitude values in input):")
print(raw_x[-5:, 0])
print(f"\nRaw y[:5, 0] (first 5 altitude values in target):")
print(raw_y[:5, 0])
print(f"\nIs there continuity in raw data? {np.abs(raw_y[0, 0] - raw_x[-1, 0]) < 100}")

# Compare with inverse transformed
print(f"\n" + "="*80)
print("Comparison: Raw vs Inverse Transformed")
print("="*80)
print(f"\nRaw x[-5:, 0]:")
print(raw_x[-5:, 0])
print(f"\nInverse transformed x[-5:, 0]:")
print(full_seq_orig[:len(x)][-5:, 0])
print(f"\nDifference:")
print(raw_x[-5:, 0] - full_seq_orig[:len(x)][-5:, 0])

print(f"\nRaw y[:5, 0]:")
print(raw_y[:5, 0])
print(f"\nInverse transformed y[:5, 0]:")
print(full_seq_orig[len(x):][:5, 0])
print(f"\nDifference:")
print(raw_y[:5, 0] - full_seq_orig[len(x):][:5, 0])

# Get the raw sample (before transforms)
if hasattr(test_dataset, 'dataset'):
    # It's a Subset
    raw_dataset = test_dataset.dataset
    idx = test_dataset.indices[0]
    print(f"Getting raw sample at index {idx}")

    # Temporarily remove transforms
    original_transforms = raw_dataset.transforms
    raw_dataset.transforms = None
    raw_sample = raw_dataset[idx]
    raw_dataset.transforms = original_transforms

    raw_x = raw_sample['x']
    raw_y = raw_sample['y']

    print(f"\nRaw x shape: {raw_x.shape}")
    print(f"Raw y shape: {raw_y.shape}")
    print(f"\nRaw x[-5:, 0] (last 5 altitude values in input):")
    print(raw_x[-5:, 0])
    print(f"\nRaw y[:5, 0] (first 5 altitude values in target):")
    print(raw_y[:5, 0])

    # Compare with inverse transformed
    print(f"\n" + "="*80)
    print("Comparison: Raw vs Inverse Transformed")
    print("="*80)
    print(f"\nRaw x[-5:, 0]:")
    print(raw_x[-5:, 0])
    print(f"\nInverse transformed x[-5:, 0]:")
    print(full_seq_orig[:len(x)][-5:, 0])
    print(f"\nDifference:")
    print(raw_x[-5:, 0] - full_seq_orig[:len(x)][-5:, 0])

    print(f"\nRaw y[:5, 0]:")
    print(raw_y[:5, 0])
    print(f"\nInverse transformed y[:5, 0]:")
    print(full_seq_orig[len(x):][:5, 0])
    print(f"\nDifference:")
    print(raw_y[:5, 0] - full_seq_orig[len(x):][:5, 0])
