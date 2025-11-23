"""ONNX export functionality for AirTrace models."""

import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf

from .end_to_end_model import EndToEndModel, ModelOnlyWrapper
from .transform_wrappers import (
    create_forward_transform_pipeline,
    create_inverse_transform_pipeline,
)


class ONNXExporter:
    """Handles ONNX export of AirTrace models with optional transform handling."""

    def __init__(
        self,
        model: torch.nn.Module,
        config: DictConfig,
        transform_stats: Optional[Dict[str, Any]] = None,
    ):
        """Initialize ONNX exporter.

        Args:
            model: The trained PyTorch model
            config: Hydra configuration used for training
            transform_stats: Optional transform statistics dictionary
        """
        self.model = model
        self.config = config
        self.transform_stats = transform_stats

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path,
        device: str = "cpu",
    ) -> "ONNXExporter":
        """Create exporter from a checkpoint file.

        Args:
            checkpoint_path: Path to the checkpoint file
            device: Device to load the model on

        Returns:
            ONNXExporter instance
        """
        from airtrace.models.registry import build_model
        from airtrace.transforms.registry import build_transforms

        # Load checkpoint (requires full pickle because we persist Hydra config)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Extract config
        if "config" not in checkpoint:
            raise ValueError(
                f"Checkpoint at {checkpoint_path} does not contain config. "
                "Cannot determine model architecture and input dimensions."
            )

        config = checkpoint["config"]

        # Build transform pipeline to get stats
        transform_stats = None
        if "transforms" in config and "pipeline" in config.transforms:
            # Build transforms to access statistics if available
            # Note: We don't fit them, just instantiate to potentially load cached stats
            try:
                transform_pipeline = build_transforms(config.transforms.pipeline)
                if hasattr(transform_pipeline, 'get_stats'):
                    # Try to get stats if they were cached
                    # In practice, we'd need to load from a separate stats file or
                    # the checkpoint if we extend it to store transform stats
                    transform_stats = transform_pipeline.get_stats()
            except Exception as e:
                print(f"Warning: Could not load transform statistics: {e}")
                print("Export will proceed without transform integration.")

        # Determine input/output dimensions
        # Try to get from data config
        data_config = config.get("data", {})

        # Build model
        # For ONNX export, we need to know input_dim and output_dim
        # These should be in the config or we need to infer from data
        model_config = config.get("model", {})

        # Try to extract dimensions from the saved model state
        model_state = checkpoint["model_state_dict"]

        # Infer dimensions from model weights (common patterns)
        input_dim, output_dim = cls._infer_dimensions(model_state, config)

        # Build model with inferred dimensions
        model = build_model(
            config=model_config,
            input_dim=input_dim,
            output_dim=output_dim,
        )

        # Load weights
        model.load_state_dict(model_state)
        model.eval()
        model.to(device)

        return cls(model=model, config=config, transform_stats=transform_stats)

    @staticmethod
    def _infer_dimensions(model_state: Dict[str, torch.Tensor], config: DictConfig) -> Tuple[int, int]:
        """Infer input and output dimensions from model state dict.

        Args:
            model_state: Model state dictionary
            config: Model configuration

        Returns:
            Tuple of (input_dim, output_dim)
        """
        # Try to get from config first
        model_config = config.get("model", {})
        if "input_dim" in model_config and "output_dim" in model_config:
            return model_config.input_dim, model_config.output_dim

        # Try to infer from data config
        data_config = config.get("data", {})
        sensors = data_config.get("sensors", [])
        if sensors:
            # Assuming sensors defines both input and output dimensions
            dim = len(sensors)
            return dim, dim

        # Try to infer from model weights
        # Look for common patterns in layer names
        for key, tensor in model_state.items():
            # Look for first layer input dimension
            if "encoder" in key and "weight" in key and tensor.dim() >= 2:
                # For GRU: weight_ih_l0 has shape [3*hidden_size, input_size]
                # For Linear: weight has shape [out_features, in_features]
                if "weight_ih" in key:  # GRU/LSTM input-hidden weight
                    input_dim = tensor.shape[1]
                elif "embedding" in key or "input" in key:
                    input_dim = tensor.shape[1]
                else:
                    continue

                # Now find output dimension
                for out_key, out_tensor in model_state.items():
                    if "decoder" in out_key and "weight" in out_key:
                        output_dim = out_tensor.shape[0]
                        return input_dim, output_dim

                # If no decoder found, assume same as input
                return input_dim, input_dim

        # Default fallback
        print("Warning: Could not infer dimensions from checkpoint. Using default 15.")
        return 15, 15

    def export(
        self,
        output_path: Path,
        end_to_end: bool = False,
        batch_size: int = 1,
        sequence_length: Optional[int] = None,
        opset_version: int = 14,
        verbose: bool = True,
    ) -> Dict[str, Path]:
        """Export model to ONNX format.

        Args:
            output_path: Path where the ONNX model will be saved
            end_to_end: If True, export with preprocessing and postprocessing
            batch_size: Batch size for dummy input (use 1 for dynamic)
            sequence_length: Sequence length for input (if None, inferred from config)
            opset_version: ONNX opset version
            verbose: Whether to print export info

        Returns:
            Dictionary with paths to exported files
        """
        output_path = Path(output_path)
        output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Infer input shape from config if not provided
        if sequence_length is None:
            data_config = self.config.get("data", {})
            sequence_length = data_config.get("window_size_in", 100)

        # Get model dimensions
        input_dim = self.model.input_dim
        output_dim = self.model.output_dim

        if verbose:
            print(f"\nONNX Export Configuration:")
            print(f"  Output path: {output_path}")
            print(f"  End-to-end: {end_to_end}")
            print(f"  Input shape: [{batch_size}, {sequence_length}, {input_dim}]")
            print(f"  Output dim: {output_dim}")
            print(f"  Opset version: {opset_version}")

        # Create dummy input
        dummy_input = torch.randn(batch_size, sequence_length, input_dim)

        # Prepare model for export
        if end_to_end and self.transform_stats:
            # Create forward transform pipeline for preprocessing
            forward_transforms = create_forward_transform_pipeline(self.transform_stats)

            # Create inverse transform pipeline for postprocessing
            inverse_transforms = create_inverse_transform_pipeline(self.transform_stats)

            # Wrap model with transforms
            export_model = EndToEndModel(
                model=self.model,
                preprocess=forward_transforms,
                postprocess=inverse_transforms,
            )

            if verbose:
                print("  Mode: End-to-end (preprocessing + model + postprocessing)")
        else:
            # Export model only, wrapped to return tensor instead of dict
            export_model = ModelOnlyWrapper(self.model)

            if verbose:
                print("  Mode: Model only")

        export_model.eval()

        # Dynamic axes for variable batch size and sequence length
        dynamic_axes = {
            'input': {0: 'batch_size', 1: 'sequence_length'},
            'output': {0: 'batch_size', 1: 'output_length'},
        }

        # Export to ONNX
        if verbose:
            print("\nExporting to ONNX...")

        try:
            torch.onnx.export(
                export_model,
                dummy_input,
                str(output_path),
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes=dynamic_axes,
                verbose=False,
            )

            if verbose:
                print(f"✓ Model exported to: {output_path}")

        except Exception as e:
            print(f"Error during ONNX export: {e}")
            raise

        # Save additional metadata
        exported_files = {"onnx_model": output_path}

        # Save transform statistics if available and not in end-to-end mode
        if self.transform_stats and not end_to_end:
            stats_path = output_path.with_suffix('.transform_stats.json')
            self._save_transform_stats(stats_path)
            exported_files["transform_stats"] = stats_path

            if verbose:
                print(f"✓ Transform statistics saved to: {stats_path}")

        # Save config
        config_path = output_path.with_suffix('.config.yaml')
        self._save_config(config_path)
        exported_files["config"] = config_path

        if verbose:
            print(f"✓ Configuration saved to: {config_path}")

        # Save metadata
        metadata_path = output_path.with_suffix('.metadata.json')
        metadata = {
            "input_shape": [batch_size, sequence_length, input_dim],
            "output_dim": output_dim,
            "end_to_end": end_to_end,
            "has_transforms": self.transform_stats is not None,
            "opset_version": opset_version,
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        exported_files["metadata"] = metadata_path

        if verbose:
            print(f"✓ Metadata saved to: {metadata_path}")
            print(f"\n✓ Export complete! {len(exported_files)} files created.")

        return exported_files

    def _save_transform_stats(self, path: Path) -> None:
        """Save transform statistics to JSON file.

        Args:
            path: Path to save the statistics
        """
        if not self.transform_stats:
            return

        # Convert numpy arrays to lists for JSON serialization
        serializable_stats = {}
        for transform_name, stats in self.transform_stats.items():
            serializable_stats[transform_name] = {}
            for key, value in stats.items():
                if isinstance(value, np.ndarray):
                    serializable_stats[transform_name][key] = value.tolist()
                elif isinstance(value, (np.integer, np.floating)):
                    serializable_stats[transform_name][key] = value.item()
                else:
                    serializable_stats[transform_name][key] = value

        with open(path, 'w') as f:
            json.dump(serializable_stats, f, indent=2)

    def _save_config(self, path: Path) -> None:
        """Save configuration to YAML file.

        Args:
            path: Path to save the configuration
        """
        config_str = OmegaConf.to_yaml(self.config, resolve=True)
        with open(path, 'w') as f:
            f.write(config_str)

    def verify_export(
        self,
        onnx_path: Path,
        end_to_end: bool = False,
        num_samples: int = 5,
        tolerance: float = 1e-5,
        verbose: bool = True,
    ) -> bool:
        """Verify ONNX export by comparing outputs with PyTorch model.

        Args:
            onnx_path: Path to the exported ONNX model
            end_to_end: Whether the export was end-to-end (with transforms)
            num_samples: Number of random samples to test
            tolerance: Numerical tolerance for comparison
            verbose: Whether to print verification results

        Returns:
            True if verification passed, False otherwise
        """
        try:
            import onnxruntime as ort
        except ImportError:
            print("Warning: onnxruntime not installed. Skipping verification.")
            print("Install with: pip install onnxruntime")
            return False

        if verbose:
            print("\nVerifying ONNX export...")

        # Load ONNX model
        ort_session = ort.InferenceSession(str(onnx_path))

        # Get input shape from model
        input_dim = self.model.input_dim
        data_config = self.config.get("data", {})
        sequence_length = data_config.get("window_size_in", 100)

        # Create the same wrapped model that was exported for fair comparison
        if end_to_end and self.transform_stats:
            # Recreate the end-to-end wrapped model
            forward_transforms = create_forward_transform_pipeline(self.transform_stats)
            inverse_transforms = create_inverse_transform_pipeline(self.transform_stats)
            pytorch_model = EndToEndModel(
                model=self.model,
                preprocess=forward_transforms,
                postprocess=inverse_transforms,
            )
        else:
            # Use ModelOnlyWrapper (extracts "preds" from dict)
            pytorch_model = ModelOnlyWrapper(self.model)

        pytorch_model.eval()

        # Test with multiple random inputs
        passed = True
        for i in range(num_samples):
            # Generate random input
            dummy_input = torch.randn(1, sequence_length, input_dim)

            # PyTorch inference with wrapped model
            with torch.no_grad():
                torch_output = pytorch_model(dummy_input, None)
                torch_output = torch_output.numpy()

            # ONNX inference
            ort_inputs = {
                'input': dummy_input.numpy(),
            }
            onnx_output = ort_session.run(None, ort_inputs)[0]

            # Compare outputs
            diff = np.abs(torch_output - onnx_output)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)

            if verbose:
                print(f"  Sample {i+1}/{num_samples}: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}", end="")

            if max_diff > tolerance:
                if verbose:
                    print(" ✗ FAILED")
                passed = False
            else:
                if verbose:
                    print(" ✓")

        if verbose:
            if passed:
                print("\n✓ Verification passed! ONNX model outputs match PyTorch model.")
            else:
                print("\n✗ Verification failed! ONNX outputs differ from PyTorch.")

        return passed
