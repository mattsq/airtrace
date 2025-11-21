"""End-to-end model wrapper for ONNX export including transforms."""

from typing import Optional

import torch
import torch.nn as nn


class EndToEndModel(nn.Module):
    """Wraps a model with preprocessing and postprocessing transforms for end-to-end inference.

    This module chains:
    1. Input preprocessing (transforms)
    2. Model forward pass
    3. Output postprocessing (inverse transforms)

    This allows exporting a single ONNX file that handles the entire pipeline.
    """

    def __init__(
        self,
        model: nn.Module,
        preprocess: Optional[nn.Module] = None,
        postprocess: Optional[nn.Module] = None,
    ):
        """Initialize end-to-end model.

        Args:
            model: The core prediction model
            preprocess: Optional preprocessing module (transforms)
            postprocess: Optional postprocessing module (inverse transforms)
        """
        super().__init__()
        self.model = model
        self.preprocess = preprocess if preprocess is not None else nn.Identity()
        self.postprocess = postprocess if postprocess is not None else nn.Identity()

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with preprocessing and postprocessing.

        Args:
            x: Input tensor [B, T_in, D_in]
            context: Optional context tensor (not used in preprocessing)

        Returns:
            Predictions in original scale [B, T_out, D_out]
        """
        # Preprocess input
        x_transformed = self.preprocess(x)

        # Model forward pass
        # Most AirTrace models return a dict with "preds" key
        model_output = self.model(x_transformed, context)

        # Extract predictions (handle both dict and tensor outputs)
        if isinstance(model_output, dict):
            preds = model_output["preds"]
        else:
            preds = model_output

        # Postprocess predictions back to original scale
        # Use inverse method for denormalization, fallback to forward if not available
        if hasattr(self.postprocess, 'inverse'):
            preds_original_scale = self.postprocess.inverse(preds)
        else:
            preds_original_scale = self.postprocess(preds)

        return preds_original_scale


class ModelOnlyWrapper(nn.Module):
    """Wrapper that extracts only the predictions tensor from model output.

    AirTrace models return Dict[str, Tensor] with "preds" key, but ONNX
    export works better with direct tensor outputs. This wrapper extracts
    the predictions tensor.
    """

    def __init__(self, model: nn.Module):
        """Initialize model wrapper.

        Args:
            model: The AirTrace model to wrap
        """
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass returning only predictions tensor.

        Args:
            x: Input tensor [B, T_in, D_in]
            context: Optional context tensor

        Returns:
            Predictions tensor [B, T_out, D_out]
        """
        output = self.model(x, context)

        # Extract predictions from dict
        if isinstance(output, dict):
            return output["preds"]
        return output
