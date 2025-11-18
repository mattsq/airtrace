"""Context feature transforms."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import Transform
from .registry import register


@register("context")
class ContextTransform(Transform):
    """Add context features to input sequences.

    Context features can include:
    - Static features (aircraft type, route length, etc.)
    - Flight plan deltas (planned vs actual fuel, etc.)
    - Environmental variables (wind, temperature, etc.)
    """

    def __init__(
        self,
        use_static: Optional[List[str]] = None,
        use_plan_deltas: bool = False,
        use_env: bool = False
    ):
        """Initialize context transform.

        Args:
            use_static: List of static feature names to include
            use_plan_deltas: Whether to include flight plan deltas
            use_env: Whether to include environmental variables
        """
        super().__init__()
        self.use_static = use_static or []
        self.use_plan_deltas = use_plan_deltas
        self.use_env = use_env

        # Encoding mappings (to be learned during fit)
        self.static_encoders = {}

    @property
    def context_dim(self) -> int:
        """Return the number of context features that will be added."""
        dim = 0
        if self.use_static:
            dim += len(self.use_static)
        # In the future, we might add other context types here
        # if self.use_plan_deltas:
        #     dim += ...
        return dim

    def fit(self, dataset) -> "ContextTransform":
        """Fit transform (learn encodings for categorical features).

        Args:
            dataset: Dataset to fit on

        Returns:
            self for method chaining
        """
        # Build encoding mappings for categorical static features
        if self.use_static:
            # Collect unique values for each static feature
            for feature_name in self.use_static:
                unique_values = set()
                for i in range(min(len(dataset), 1000)):
                    sample = dataset[i]
                    meta = sample["meta"] if isinstance(sample, dict) else sample[2]
                    if feature_name in meta:
                        unique_values.add(meta[feature_name])

                # Create encoding mapping
                self.static_encoders[feature_name] = {
                    val: idx for idx, val in enumerate(sorted(unique_values))
                }

        self.is_fitted = True
        return self

    def __call__(
        self, x: np.ndarray, y: np.ndarray, meta: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Add context features to input sequence.

        Args:
            x: Input sequence [T_in, D_in]
            y: Target sequence [T_out, D_out]
            meta: Metadata containing context information

        Returns:
            (x_with_context, y, meta) where x_with_context is [T_in, D_in + C]
        """
        T_in = x.shape[0]
        context_features = []

        # Add static features (broadcasted to all timesteps)
        if self.use_static:
            for feature_name in self.use_static:
                if feature_name in meta:
                    value = meta[feature_name]

                    # Encode if categorical
                    if feature_name in self.static_encoders:
                        encoded = self.static_encoders[feature_name].get(value, 0)
                    else:
                        # Assume numeric
                        encoded = float(value)
                else:
                    # Use default value (0.0) when metadata is missing
                    encoded = 0.0

                # Broadcast to all timesteps
                feature_array = np.full((T_in, 1), encoded)
                context_features.append(feature_array)

        # Add plan deltas (if available in meta)
        if self.use_plan_deltas and "plan_deltas" in meta:
            plan_deltas = meta["plan_deltas"]  # Expected shape [T_in, D_plan]
            if len(plan_deltas) == T_in:
                context_features.append(plan_deltas)

        # Add environmental variables
        if self.use_env and "env_vars" in meta:
            env_vars = meta["env_vars"]  # Expected shape [T_in, D_env]
            if len(env_vars) == T_in:
                context_features.append(env_vars)

        # Track original sensor dimension before adding context
        meta["original_sensor_dim"] = x.shape[1]

        # Concatenate context to input
        if context_features:
            context_array = np.concatenate(context_features, axis=1)
            x_with_context = np.concatenate([x, context_array], axis=1)
            meta["context_dim"] = context_array.shape[1]
        else:
            x_with_context = x
            meta["context_dim"] = 0

        return x_with_context, y, meta

    def inverse(self, x: np.ndarray, y: np.ndarray = None):
        """Remove context features (return original sensors only).

        Args:
            x: Input with context [T, D_in + C]
            y: Target

        Returns:
            (x_original, y) where x_original is [T, D_in]

        Note:
            This requires knowing the context dimension, which should be
            stored during forward pass. For now, this is a placeholder.
        """
        # This is a simplified version - real implementation would need
        # to know where to split
        return x, y
