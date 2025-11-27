"""Base classes for autoregressive models."""

from abc import ABC, abstractmethod
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.nn.parameter import UninitializedParameter


class ARBaseModel(nn.Module, ABC):
    """Base class for all autoregressive models.

    All models should inherit from this class and implement the forward method.
    The forward method should return a dictionary with at least 'preds' key.
    """

    def __init__(self, input_dim: int, output_dim: int, **kwargs):
        """Initialize base model.

        Args:
            input_dim: Dimension of input features
            output_dim: Dimension of output predictions
            **kwargs: Additional model-specific parameters
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of the model.

        Args:
            x: Input tensor [B, T_in, D_in]
            context: Optional context tensor [B, T_in, C] or [B, C_static]
            **kwargs: Additional model-specific arguments

        Returns:
            Dictionary containing at least:
                - preds: Predictions [B, T_out, D_out]
                - extras: Optional dict with model-specific outputs
                  (hidden states, attention weights, etc.)
        """
        raise NotImplementedError

    def get_num_params(self) -> int:
        """Get total number of parameters (trainable or frozen)."""

        total_params = 0
        for param in self.parameters():
            if isinstance(param, UninitializedParameter):
                # Lazy modules register parameters immediately but defer
                # materialising their shapes until the first forward pass.
                # Accessing ``numel`` on these tensors raises an error, so we
                # skip them here to keep utility methods like ``__repr__``
                # functional before initialisation. Once the module receives
                # a batch the parameter will become a regular ``Parameter``
                # and will be counted normally.
                continue

            total_params += param.numel()

        return total_params

    def reset_parameters(self):
        """Reset model parameters to initial state.

        Override this method for custom initialization.
        """
        for module in self.children():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  input_dim={self.input_dim},\n"
            f"  output_dim={self.output_dim},\n"
            f"  num_params={self.get_num_params():,}\n"
            f")"
        )
