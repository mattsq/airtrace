"""Model-specific export profiles for ONNX export.

This module defines export profiles that contain optimized settings for different
model architectures. Profiles help ensure consistent and successful exports by
providing pre-configured settings for opset versions, dynamic axes, and verification
tolerances.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Any


@dataclass
class ExportProfile:
    """Export profile with model-specific settings.

    Attributes:
        name: Profile name (e.g., "informer", "gru")
        opset_version: Recommended ONNX opset version
        dynamic_axes: Dynamic axes configuration for variable-length inputs
        verification_tolerance: Tolerance for verifying ONNX vs PyTorch outputs
        description: Human-readable description of the profile
        notes: Additional notes or warnings
    """
    name: str
    opset_version: int
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None
    verification_tolerance: float = 1e-5
    description: str = ""
    notes: str = ""


# Default dynamic axes for time-series models
DEFAULT_DYNAMIC_AXES = {
    'input': {0: 'batch_size', 1: 'sequence_length'},
    'output': {0: 'batch_size', 1: 'output_length'},
}


# Model-specific export profiles
PROFILES = {
    # Attention-based models
    "informer": ExportProfile(
        name="informer",
        opset_version=17,
        dynamic_axes=DEFAULT_DYNAMIC_AXES,
        verification_tolerance=1e-4,  # More lenient due to ProbSparse attention
        description="Informer model with ProbSparse attention",
        notes="Uses legacy ONNX exporter (dynamo=False) for better compatibility with "
              "dynamic attention operations. May show TracerWarnings about converting "
              "tensors to Python values.",
    ),

    "transformer": ExportProfile(
        name="transformer",
        opset_version=17,
        dynamic_axes=DEFAULT_DYNAMIC_AXES,
        verification_tolerance=1e-5,
        description="Standard Transformer model with multi-head attention",
        notes="Opset 17+ recommended for better attention operator support.",
    ),

    "timexer": ExportProfile(
        name="timexer",
        opset_version=17,
        dynamic_axes=DEFAULT_DYNAMIC_AXES,
        verification_tolerance=1e-5,
        description="TimeXer time-series model",
        notes="Opset 17+ recommended for attention operations.",
    ),

    # Recurrent models
    "gru": ExportProfile(
        name="gru",
        opset_version=14,
        dynamic_axes=DEFAULT_DYNAMIC_AXES,
        verification_tolerance=1e-6,
        description="GRU (Gated Recurrent Unit) model",
        notes="Opset 14 provides widest compatibility for recurrent operations. "
              "Well-established ONNX support.",
    ),

    "lstm": ExportProfile(
        name="lstm",
        opset_version=14,
        dynamic_axes=DEFAULT_DYNAMIC_AXES,
        verification_tolerance=1e-6,
        description="LSTM (Long Short-Term Memory) model",
        notes="Opset 14 provides widest compatibility for recurrent operations.",
    ),

    "rnn": ExportProfile(
        name="rnn",
        opset_version=14,
        dynamic_axes=DEFAULT_DYNAMIC_AXES,
        verification_tolerance=1e-6,
        description="Basic RNN model",
        notes="Opset 14 recommended for maximum compatibility.",
    ),

    # Convolutional models
    "tcn": ExportProfile(
        name="tcn",
        opset_version=14,
        dynamic_axes=DEFAULT_DYNAMIC_AXES,
        verification_tolerance=1e-6,
        description="Temporal Convolutional Network",
        notes="Simple convolutional operations, wide compatibility.",
    ),

    # Baseline models
    "linear": ExportProfile(
        name="linear",
        opset_version=14,
        dynamic_axes=DEFAULT_DYNAMIC_AXES,
        verification_tolerance=1e-7,
        description="Simple linear model",
        notes="Maximum compatibility, minimal operations.",
    ),

    "persistence": ExportProfile(
        name="persistence",
        opset_version=14,
        dynamic_axes=DEFAULT_DYNAMIC_AXES,
        verification_tolerance=1e-7,
        description="Persistence baseline (last value carried forward)",
        notes="Trivial operations, very tight verification tolerance.",
    ),

    # Default profile
    "default": ExportProfile(
        name="default",
        opset_version=17,
        dynamic_axes=DEFAULT_DYNAMIC_AXES,
        verification_tolerance=1e-5,
        description="Default profile for unknown model types",
        notes="Balanced settings. Opset 17 provides good operator coverage while "
              "maintaining reasonable compatibility.",
    ),
}


def get_profile_for_model(model: Any) -> ExportProfile:
    """Get the best export profile for a model.

    Args:
        model: PyTorch model instance

    Returns:
        ExportProfile with recommended settings

    Example:
        >>> from airtrace.models import InformerModel
        >>> model = InformerModel(...)
        >>> profile = get_profile_for_model(model)
        >>> print(f"Using {profile.name} profile with opset {profile.opset_version}")
    """
    model_class = model.__class__.__name__.lower()

    # Try to match model class name to profile
    for key in PROFILES:
        if key in model_class:
            return PROFILES[key]

    # Return default profile
    return PROFILES["default"]


def list_profiles() -> Dict[str, ExportProfile]:
    """List all available export profiles.

    Returns:
        Dictionary mapping profile names to ExportProfile objects
    """
    return PROFILES.copy()


def get_profile(name: str) -> ExportProfile:
    """Get a specific profile by name.

    Args:
        name: Profile name (e.g., "informer", "gru")

    Returns:
        ExportProfile

    Raises:
        KeyError: If profile name not found
    """
    if name not in PROFILES:
        available = ", ".join(PROFILES.keys())
        raise KeyError(
            f"Profile '{name}' not found. Available profiles: {available}"
        )
    return PROFILES[name]
