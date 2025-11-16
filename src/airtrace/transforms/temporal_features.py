"""Temporal feature transforms for sensor data."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import Transform
from .registry import register


@register("temporal_features")
class TemporalFeaturesTransform(Transform):
    """Add temporal features to input sequences.

    Adds time-of-flight, phase-of-flight, and other temporal encodings
    to help models understand temporal patterns.
    """

    def __init__(
        self,
        use_time_idx: bool = True,
        use_cyclic_time: bool = True,
        use_flight_phase: bool = False,
        time_scale: float = 1.0,
        normalize_time: bool = True
    ):
        """Initialize temporal features transform.

        Args:
            use_time_idx: Add normalized time index (0 to 1 over sequence)
            use_cyclic_time: Add sin/cos encoding of time (for periodic patterns)
            use_flight_phase: Add flight phase encoding (requires phase in meta)
            time_scale: Scale factor for cyclic encoding period
            normalize_time: If True, normalize time index to [0, 1]
        """
        super().__init__()
        self.use_time_idx = use_time_idx
        self.use_cyclic_time = use_cyclic_time
        self.use_flight_phase = use_flight_phase
        self.time_scale = time_scale
        self.normalize_time = normalize_time

        # Flight phase encoding (if used)
        self.phase_encodings = {
            "taxi": 0,
            "takeoff": 1,
            "climb": 2,
            "cruise": 3,
            "descent": 4,
            "approach": 5,
            "landing": 6
        }

    def fit(self, dataset) -> "TemporalFeaturesTransform":
        """Fit transform (no-op for temporal features).

        Args:
            dataset: Dataset to fit on

        Returns:
            self for method chaining
        """
        # Temporal features are computed per sample, no fitting needed
        self.is_fitted = True
        return self

    def _create_temporal_features(
        self,
        seq_len: int,
        meta: Dict[str, Any]
    ) -> np.ndarray:
        """Create temporal feature array.

        Args:
            seq_len: Length of sequence
            meta: Metadata (may contain phase info)

        Returns:
            Temporal features array [seq_len, n_features]
        """
        features = []

        # Time index feature
        if self.use_time_idx:
            time_idx = np.arange(seq_len, dtype=np.float32)
            if self.normalize_time:
                time_idx = time_idx / max(seq_len - 1, 1)
            features.append(time_idx.reshape(-1, 1))

        # Cyclic time encoding (sin/cos)
        if self.use_cyclic_time:
            time_idx = np.arange(seq_len, dtype=np.float32)
            # Scale time to create periodic patterns
            scaled_time = 2 * np.pi * time_idx / (seq_len * self.time_scale)
            sin_time = np.sin(scaled_time).reshape(-1, 1)
            cos_time = np.cos(scaled_time).reshape(-1, 1)
            features.extend([sin_time, cos_time])

        # Flight phase encoding
        if self.use_flight_phase:
            # Get phase from metadata or default to cruise
            phase = meta.get("flight_phase", "cruise")
            phase_id = self.phase_encodings.get(phase, 3)  # Default to cruise
            phase_feature = np.full((seq_len, 1), phase_id, dtype=np.float32)
            features.append(phase_feature)

        if features:
            return np.concatenate(features, axis=1)
        else:
            # Return empty array with correct shape
            return np.zeros((seq_len, 0), dtype=np.float32)

    def __call__(
        self, x: np.ndarray, y: np.ndarray, meta: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Add temporal features.

        Args:
            x: Input sequence [T_in, D_in]
            y: Target sequence [T_out, D_out]
            meta: Metadata dict

        Returns:
            (x_with_temporal, y, meta) where x has additional temporal features
        """
        if not self.is_fitted:
            raise RuntimeError("Transform not fitted. Call fit() first.")

        T_in = x.shape[0]

        # Create temporal features for input sequence
        temporal_features = self._create_temporal_features(T_in, meta)

        # Concatenate to input
        if temporal_features.shape[1] > 0:
            x_with_temporal = np.concatenate([x, temporal_features], axis=1)
            meta["temporal_features_dim"] = temporal_features.shape[1]
        else:
            x_with_temporal = x
            meta["temporal_features_dim"] = 0

        return x_with_temporal, y, meta

    def inverse(self, x: np.ndarray, y: Optional[np.ndarray] = None, meta: Optional[Dict[str, Any]] = None):
        """Remove temporal features (restore original sensors only).

        Args:
            x: Input with temporal features [T, D_in + D_temporal]
            y: Target
            meta: Metadata with temporal_features_dim

        Returns:
            (x_original, y) where x_original is [T, D_in]
        """
        # Remove temporal features from end of feature dimension
        if meta is not None and "temporal_features_dim" in meta:
            temporal_dim = meta["temporal_features_dim"]
            if temporal_dim > 0:
                x = x[:, :-temporal_dim]

        return x, y
