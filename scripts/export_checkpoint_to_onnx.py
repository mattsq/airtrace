"""Export a trained checkpoint to ONNX format."""

from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from airtrace.export.onnx_exporter import ONNXExporter


def main():
    # Paths
    checkpoint_path = Path("runs/20251126/medium_20k/checkpoints/epoch_15.ckpt")
    output_dir = Path("runs/20251126/medium_20k/onnx_export")
    output_path = output_dir / "model.onnx"

    print(f"Loading checkpoint from: {checkpoint_path}")

    # First load checkpoint to inspect dimensions
    import torch
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_state = checkpoint["model_state_dict"]

    # Infer actual dimensions from model weights
    input_dim = model_state["value_embedding.proj.weight"].shape[1]
    output_dim = model_state["projection.weight"].shape[0]

    print(f"Detected dimensions: input_dim={input_dim}, output_dim={output_dim}")
    print(f"  (11 sensors + 2 static features from context transform)")

    # Now we need to manually load and build the model with correct dims
    from airtrace.models.registry import build_model

    config = checkpoint["config"]
    model_config = config.get("model", {})

    # Build model with correct dimensions
    model = build_model(
        config=model_config,
        input_dim=input_dim,
        output_dim=output_dim,
    )

    # Load weights
    model.load_state_dict(model_state)
    model.eval()

    # Create exporter manually since from_checkpoint has dimension inference issues
    exporter = ONNXExporter(
        model=model,
        config=config,
        transform_stats=None  # We'll skip transform stats for now
    )

    print("\nExporting to ONNX format...")

    # Get sequence length from config
    sequence_length = config.data.window.input_len

    # Export the model
    exported_files = exporter.export(
        output_path=output_path,
        end_to_end=False,  # Set to True if you want transforms included
        batch_size=1,
        sequence_length=sequence_length,  # Use actual config value (128)
        opset_version=17,  # Use higher opset version
        verbose=True
    )

    print("\n" + "="*60)
    print("Exported files:")
    for file_type, file_path in exported_files.items():
        print(f"  {file_type}: {file_path}")
    print("="*60)

    # Verify the export
    print("\nVerifying ONNX export...")
    verification_passed = exporter.verify_export(
        onnx_path=output_path,
        end_to_end=False,
        num_samples=5,
        tolerance=1e-5,
        verbose=True
    )

    if verification_passed:
        print("\n[OK] Export successful and verified!")
    else:
        print("\n[WARNING] Export completed but verification failed. Check output differences.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
