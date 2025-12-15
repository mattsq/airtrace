"""Inspect checkpoint to determine actual dimensions."""

from pathlib import Path
import torch

checkpoint_path = Path("runs/20251126/medium_20k/checkpoints/epoch_15.ckpt")

print(f"Loading checkpoint: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

print("\nCheckpoint keys:")
for key in checkpoint.keys():
    print(f"  - {key}")

print("\nModel state dict keys (first 20):")
state_dict = checkpoint["model_state_dict"]
for i, key in enumerate(list(state_dict.keys())[:20]):
    shape = state_dict[key].shape if hasattr(state_dict[key], 'shape') else 'N/A'
    print(f"  {key}: {shape}")

print("\nInferring dimensions from key layers:")
for key, tensor in state_dict.items():
    if "value_embedding.proj.weight" in key:
        print(f"  {key}: {tensor.shape} -> input_dim = {tensor.shape[1]}")
    if "projection.weight" in key:
        print(f"  {key}: {tensor.shape} -> output_dim = {tensor.shape[0]}")
    if "projection.bias" in key:
        print(f"  {key}: {tensor.shape} -> output_dim = {tensor.shape[0]}")

# Check config in checkpoint
if "config" in checkpoint:
    config = checkpoint["config"]
    print("\nConfig data.sensors.use:")
    print(f"  {config.data.sensors.use}")
    print(f"  Count: {len(config.data.sensors.use)}")

    print("\nConfig data.window.target_sensors:")
    print(f"  {config.data.window.target_sensors}")
    print(f"  Count: {len(config.data.window.target_sensors)}")

    print("\nConfig transforms.pipeline:")
    for i, transform in enumerate(config.transforms.pipeline):
        print(f"  {i}: {transform.name}")
        if transform.name == "context":
            print(f"    use_static: {transform.get('use_static', [])}")
