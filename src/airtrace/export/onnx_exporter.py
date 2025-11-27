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
    def _compute_input_dim_with_transforms(base_dim: int, transform_pipeline: list) -> int:
        """Compute input dimension accounting for transforms that add features.

        Args:
            base_dim: Base sensor count
            transform_pipeline: List of transform configs

        Returns:
            Adjusted input dimension
        """
        adjusted_dim = base_dim

        for transform_config in transform_pipeline:
            transform_name = transform_config.get("name", "")

            # ContextTransform adds static features
            if transform_name == "context":
                use_static = transform_config.get("use_static", [])
                adjusted_dim += len(use_static)

            # TemporalFeaturesTransform might add time-based features
            elif transform_name == "temporal_features":
                # Add logic if this transform exists
                pass

        return adjusted_dim

    @staticmethod
    def _infer_from_model_weights(model_state: Dict[str, torch.Tensor]) -> Optional[Tuple[int, int]]:
        """Infer dimensions from model weight shapes.

        Args:
            model_state: Model state dictionary

        Returns:
            Tuple of (input_dim, output_dim) or None if inference fails
        """
        input_dim = None
        output_dim = None

        # Try Informer/Transformer pattern: value_embedding.proj.weight and projection
        if "value_embedding.proj.weight" in model_state:
            input_dim = model_state["value_embedding.proj.weight"].shape[1]

        if "projection.weight" in model_state:
            output_dim = model_state["projection.weight"].shape[0]
        elif "projection.bias" in model_state:
            output_dim = model_state["projection.bias"].shape[0]

        if input_dim is not None and output_dim is not None:
            return input_dim, output_dim

        # Try GRU/LSTM pattern: weight_ih and decoder
        for key, tensor in model_state.items():
            if "weight_ih" in key and tensor.dim() >= 2:
                # GRU: weight_ih_l0 has shape [3*hidden_size, input_size]
                input_dim = tensor.shape[1]
                break

        # Find output dimension from decoder
        for key, tensor in model_state.items():
            if "decoder.weight" in key and tensor.dim() >= 2:
                output_dim = tensor.shape[0]
                break
            elif "decoder.bias" in key and tensor.dim() == 1:
                output_dim = tensor.shape[0]
                break

        if input_dim is not None and output_dim is not None:
            return input_dim, output_dim

        # Try generic encoder/decoder pattern
        for key, tensor in model_state.items():
            if ("encoder" in key or "embedding" in key) and "weight" in key and tensor.dim() >= 2:
                if input_dim is None:
                    input_dim = tensor.shape[1]

        for key, tensor in model_state.items():
            if "decoder" in key and "weight" in key and tensor.dim() >= 2:
                if output_dim is None:
                    output_dim = tensor.shape[0]

        if input_dim is not None and output_dim is not None:
            return input_dim, output_dim

        return None

    @staticmethod
    def _infer_dimensions(model_state: Dict[str, torch.Tensor], config: DictConfig) -> Tuple[int, int]:
        """Infer input and output dimensions from model state dict and config.

        This method implements a multi-step inference strategy:
        1. Check explicit config (most reliable)
        2. Compute from sensors + transform adjustments
        3. Validate against model weights
        4. Raise clear error if inference fails

        Args:
            model_state: Model state dictionary
            config: Model configuration

        Returns:
            Tuple of (input_dim, output_dim)

        Raises:
            ValueError: If dimensions cannot be inferred
        """
        # Step 1: Try explicit config (most reliable)
        model_config = config.get("model", {})
        if "input_dim" in model_config and "output_dim" in model_config:
            return model_config.input_dim, model_config.output_dim

        # Step 2: Compute from sensors + transforms
        data_config = config.get("data", {})
        sensors_config = data_config.get("sensors", {})

        # Handle both old format (list) and new format (dict with 'use' key)
        if isinstance(sensors_config, list):
            sensors = sensors_config
        elif isinstance(sensors_config, dict):
            sensors = sensors_config.get("use", [])
        else:
            sensors = []

        if sensors:
            base_dim = len(sensors)
            transforms_config = config.get("transforms", {})
            transform_pipeline = transforms_config.get("pipeline", [])

            # Compute adjusted input dimension (accounts for context transform, etc.)
            input_dim = ONNXExporter._compute_input_dim_with_transforms(base_dim, transform_pipeline)
            output_dim = base_dim  # Output is usually the base sensors

            # Step 3: Validate against model weights
            weight_dims = ONNXExporter._infer_from_model_weights(model_state)
            if weight_dims:
                weight_input, weight_output = weight_dims

                if weight_input != input_dim:
                    print(f"Warning: Config suggests input_dim={input_dim}, but model weights suggest {weight_input}")
                    print(f"  Base sensors: {base_dim}, using weight-based dimension")
                    input_dim = weight_input

                if weight_output != output_dim:
                    output_dim = weight_output

            return input_dim, output_dim

        # Step 4: Try inference from weights alone
        weight_dims = ONNXExporter._infer_from_model_weights(model_state)
        if weight_dims:
            return weight_dims

        # Step 5: Fail explicitly instead of silent fallback
        raise ValueError(
            "Could not infer dimensions from checkpoint.\n"
            "  - No input_dim/output_dim in model config\n"
            "  - No sensors list in data config\n"
            "  - Could not infer from model weights\n"
            "\n"
            "Please ensure your checkpoint was saved by AirTrace's training pipeline,\n"
            "or manually specify dimensions using model.input_dim and model.output_dim in config."
        )

    @staticmethod
    def _recommend_opset_version(model: torch.nn.Module) -> int:
        """Recommend ONNX opset version based on model architecture.

        Args:
            model: The PyTorch model

        Returns:
            Recommended opset version (14, 17, or 18)

        Reasoning:
            - Attention-based models (Informer, Transformer, TimeXer): opset 17+
              These models benefit from better attention operator support in newer opsets
            - Recurrent models (GRU, LSTM): opset 14
              Widest compatibility, well-established support for recurrent ops
            - Linear/baseline models: opset 14
              Simple operations, maximize compatibility
            - Default: opset 17
              Balanced choice for most modern models
        """
        model_class = model.__class__.__name__.lower()

        # Attention-based models: use opset 17 for better attention support
        attention_models = ["informer", "transformer", "timexer", "attention"]
        if any(name in model_class for name in attention_models):
            return 17

        # Recurrent models: use opset 14 for widest compatibility
        recurrent_models = ["gru", "lstm", "rnn"]
        if any(name in model_class for name in recurrent_models):
            return 14

        # Linear/baseline models: use opset 14 for compatibility
        baseline_models = ["linear", "baseline", "persistence"]
        if any(name in model_class for name in baseline_models):
            return 14

        # Default: opset 17 (balanced)
        return 17

    def validate(self, end_to_end: bool = False, verbose: bool = True) -> Dict[str, Any]:
        """Validate that the model and config are ready for ONNX export.

        This performs fast pre-flight checks without actually exporting.

        Args:
            end_to_end: Whether validation is for end-to-end export
            verbose: Whether to print validation results

        Returns:
            Dictionary with validation results:
                - 'passed': bool - overall validation status
                - 'checks': dict - individual check results
                - 'warnings': list - non-critical issues
                - 'errors': list - critical issues
        """
        results = {
            'passed': True,
            'checks': {},
            'warnings': [],
            'errors': [],
        }

        if verbose:
            print("ONNX Export Validation")
            print("=" * 60)

        # Check 1: Model has required attributes
        has_input_dim = hasattr(self.model, 'input_dim')
        has_output_dim = hasattr(self.model, 'output_dim')
        results['checks']['model_input_dim'] = has_input_dim
        results['checks']['model_output_dim'] = has_output_dim

        if not has_input_dim:
            results['errors'].append("Model missing input_dim attribute")
            results['passed'] = False

        if not has_output_dim:
            results['errors'].append("Model missing output_dim attribute")
            results['passed'] = False

        if verbose:
            status = "[OK]" if has_input_dim else "[FAILED]"
            print(f"{status} model_input_dim: ", end="")
            if has_input_dim:
                print(f"Model input_dim = {self.model.input_dim}")
            else:
                print("Model missing input_dim attribute")

            status = "[OK]" if has_output_dim else "[FAILED]"
            print(f"{status} model_output_dim: ", end="")
            if has_output_dim:
                print(f"Model output_dim = {self.model.output_dim}")
            else:
                print("Model missing output_dim attribute")

        # Check 2: Config has window_size_in
        data_config = self.config.get("data", {})
        has_window_size = "window_size_in" in data_config
        results['checks']['config_window_size'] = has_window_size

        if verbose:
            status = "[OK]" if has_window_size else "[WARNING]"
            print(f"{status} config_window_size: ", end="")
            if has_window_size:
                print(f"Window size = {data_config['window_size_in']}")
            else:
                print("Missing window_size_in, will need to specify sequence_length")
                results['warnings'].append("Missing window_size_in in config")

        # Check 3: Transform stats available if end-to-end requested
        if end_to_end:
            has_transform_stats = self.transform_stats is not None
            results['checks']['transform_stats'] = has_transform_stats

            if not has_transform_stats:
                results['errors'].append("End-to-end export requires transform_stats")
                results['passed'] = False

            if verbose:
                status = "[OK]" if has_transform_stats else "[FAILED]"
                print(f"{status} transform_stats: ", end="")
                if has_transform_stats:
                    print(f"Statistics available for {len(self.transform_stats)} transforms")
                else:
                    print("No transform statistics (required for end-to-end export)")

        # Check 4: Model in eval mode
        is_eval = not self.model.training
        results['checks']['model_eval_mode'] = is_eval

        if verbose:
            status = "[OK]" if is_eval else "[WARNING]"
            print(f"{status} model_eval_mode: ", end="")
            if is_eval:
                print("Model in eval mode")
            else:
                print("Model in training mode (should use eval)")
                results['warnings'].append("Model should be in eval mode")

        # Check 5: Check for runtime resolvers in config
        config_str = str(self.config)
        has_runtime_resolvers = "${now:" in config_str or "${oc.env:" in config_str
        results['checks']['config_runtime_resolvers'] = not has_runtime_resolvers

        if verbose and has_runtime_resolvers:
            print("[WARNING] config_runtime_resolvers: Config contains runtime resolvers")
            results['warnings'].append("Config has runtime resolvers (${now:...}, ${oc.env:...})")

        # Check 6: Model architecture complexity
        model_class = self.model.__class__.__name__
        complex_models = ["Informer", "Transformer", "TimeXer"]
        is_complex = any(name in model_class for name in complex_models)
        results['checks']['model_architecture'] = model_class

        if verbose and is_complex:
            print(f"[WARNING] model_architecture: {model_class} has complex attention")
            results['warnings'].append(f"{model_class} may have attention-related export issues")

        # Summary
        if verbose:
            print()
            if results['warnings']:
                print("Warnings:")
                for warning in results['warnings']:
                    print(f"  - {warning}")
                print()

            if results['errors']:
                print("Errors:")
                for error in results['errors']:
                    print(f"  - {error}")
                print()

            print("=" * 60)
            if results['passed']:
                print("[OK] Validation passed - export should succeed")
            else:
                print("[FAILED] Validation failed - fix errors before exporting")
            print()

        return results

    def dry_run(self, end_to_end: bool = False, verbose: bool = True) -> bool:
        """Perform a dry-run of ONNX export without actually exporting.

        This tests the full export pipeline including model wrapping and
        forward pass, but doesn't write any files.

        Args:
            end_to_end: Whether to test end-to-end export
            verbose: Whether to print progress

        Returns:
            True if dry-run succeeded, False otherwise
        """
        if verbose:
            print("ONNX Export Dry-Run")
            print("=" * 60)

        # Step 1: Run validation
        if verbose:
            print("Running validation...")
        validation_results = self.validate(end_to_end=end_to_end, verbose=False)

        if not validation_results['passed']:
            if verbose:
                print("[FAILED] Validation failed")
                for error in validation_results['errors']:
                    print(f"  - {error}")
            return False

        if verbose:
            print("[OK] Validation passed")

        # Step 2: Infer input shape
        data_config = self.config.get("data", {})
        sequence_length = data_config.get("window_size_in", 100)

        if verbose:
            print(f"Testing with input shape: [1, {sequence_length}, {self.model.input_dim}]")

        # Step 3: Create dummy input
        try:
            dummy_input = torch.randn(1, sequence_length, self.model.input_dim)
        except Exception as e:
            if verbose:
                print(f"[FAILED] Could not create dummy input: {e}")
            return False

        # Step 4: Test model wrapping
        try:
            if end_to_end and self.transform_stats:
                from .transform_wrappers import create_forward_transform_pipeline, create_inverse_transform_pipeline

                forward_transforms = create_forward_transform_pipeline(self.transform_stats)
                inverse_transforms = create_inverse_transform_pipeline(self.transform_stats)
                test_model = EndToEndModel(
                    model=self.model,
                    preprocess=forward_transforms,
                    postprocess=inverse_transforms,
                )
                if verbose:
                    print("[OK] End-to-end model wrapping successful")
            else:
                test_model = ModelOnlyWrapper(self.model)
                if verbose:
                    print("[OK] Model-only wrapping successful")
        except Exception as e:
            if verbose:
                print(f"[FAILED] Model wrapping failed: {e}")
            return False

        # Step 5: Test forward pass
        try:
            test_model.eval()
            with torch.no_grad():
                output = test_model(dummy_input, None)

            if verbose:
                print(f"[OK] Forward pass successful, output shape: {output.shape}")
        except Exception as e:
            if verbose:
                print(f"[FAILED] Forward pass failed: {e}")
            return False

        # Success
        if verbose:
            print("=" * 60)
            print("[OK] Dry-run successful - export should work!")
            print()

        return True

    def export(
        self,
        output_path: Path,
        end_to_end: bool = False,
        batch_size: int = 1,
        sequence_length: Optional[int] = None,
        opset_version: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict[str, Path]:
        """Export model to ONNX format.

        Args:
            output_path: Path where the ONNX model will be saved
            end_to_end: If True, export with preprocessing and postprocessing
            batch_size: Batch size for dummy input (use 1 for dynamic)
            sequence_length: Sequence length for input (if None, inferred from config)
            opset_version: ONNX opset version (if None, auto-selected based on model type)
            verbose: Whether to print export info

        Returns:
            Dictionary with paths to exported files
        """
        output_path = Path(output_path)
        output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Auto-select opset version if not provided
        if opset_version is None:
            opset_version = self._recommend_opset_version(self.model)
            if verbose:
                print(f"Auto-selected opset version {opset_version} for {self.model.__class__.__name__}")

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
                # Use legacy exporter for better compatibility with complex models
                # and dynamic axes. The new dynamo exporter (torch.export) has
                # stricter tracing requirements that fail with ProbSparse attention
                # and other dynamic control flow patterns.
                dynamo=False,
            )

            if verbose:
                print(f"[OK] Model exported to: {output_path}")

        except Exception as e:
            error_msg = str(e)
            print(f"\n[FAILED] ONNX export failed:")
            print(f"  {type(e).__name__}: {error_msg[:200]}...")

            # Provide context-aware troubleshooting guidance
            if any(keyword in error_msg.lower() for keyword in ["dynamic", "symbolic", "constraint"]):
                print("\nTroubleshooting: Dynamic Shape Issues")
                print("  1. Try exporting with fixed batch_size and sequence_length")
                print("  2. Consider removing dynamic_axes if deployment doesn't need them")
                print("  3. Check if your model has dynamic control flow (if/else on tensor shapes)")
                print(f"\n  Try: export(..., dynamic_axes=None)")

            elif "opset" in error_msg.lower():
                print("\nTroubleshooting: ONNX Opset Version Issues")
                print("  1. Try a different opset version (current: {})".format(opset_version))
                print("  2. Newer opsets (17-18) have better operator support")
                print("  3. Older opsets (13-14) have wider runtime compatibility")
                print(f"\n  Try: export(..., opset_version=17) or export(..., opset_version=13)")

            elif any(keyword in error_msg for keyword in ["prim::", "aten::", "Not implemented"]):
                print("\nTroubleshooting: Unsupported Operations")
                print("  1. Your model contains operations not supported in ONNX")
                print("  2. Try simplifying the model architecture")
                print("  3. Check for custom PyTorch operations that need conversion")
                print(f"\n  Consider: Implementing custom ONNX operators or using torch.jit.script")

            else:
                # Generic error - provide diagnostic information
                print("\nDiagnostic Information:")
                print(f"  Model type: {self.model.__class__.__name__}")
                print(f"  Input shape: {dummy_input.shape}")
                print(f"  Opset version: {opset_version}")
                print(f"  End-to-end mode: {end_to_end}")
                print("\n  If this error persists, please report it with the above information.")

            print()  # Add blank line before re-raising
            raise

        # Save additional metadata
        exported_files = {"onnx_model": output_path}

        # Save transform statistics if available and not in end-to-end mode
        if self.transform_stats and not end_to_end:
            stats_path = output_path.with_suffix('.transform_stats.json')
            self._save_transform_stats(stats_path)
            exported_files["transform_stats"] = stats_path

            if verbose:
                print(f"[OK] Transform statistics saved to: {stats_path}")

        # Save config
        config_path = output_path.with_suffix('.config.yaml')
        self._save_config(config_path)
        exported_files["config"] = config_path

        if verbose:
            print(f"[OK] Configuration saved to: {config_path}")

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
            print(f"[OK] Metadata saved to: {metadata_path}")
            print(f"\n[OK] Export complete! {len(exported_files)} files created.")

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
        # Try to resolve interpolations, but fall back to unresolved if it fails
        # (e.g., if config has runtime resolvers like ${now:...})
        try:
            config_str = OmegaConf.to_yaml(self.config, resolve=True)
        except Exception as e:
            # Runtime resolvers (like ${now:...}) can't be resolved during export
            print(f"Warning: Could not resolve all config interpolations: {type(e).__name__}")
            print("  This is normal if your config uses runtime resolvers like ${{now:...}}")
            print("  Saving unresolved config instead.")
            config_str = OmegaConf.to_yaml(self.config, resolve=False)

        with open(path, 'w', encoding='utf-8') as f:
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
                    print(" [FAILED]")
                passed = False
            else:
                if verbose:
                    print(" [OK]")

        if verbose:
            if passed:
                print("\n[OK] Verification passed! ONNX model outputs match PyTorch model.")
            else:
                print("\n[FAILED] Verification failed! ONNX outputs differ from PyTorch.")

        return passed
