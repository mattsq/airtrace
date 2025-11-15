"""CycleNet: Enhancing Time Series Forecasting through Modeling Periodic Patterns.

Implementation of CycleNet from:
"CycleNet: Enhancing Time Series Forecasting through Modeling Periodic Patterns"
(NeurIPS 2024 Spotlight)

Key innovation:
Residual Cycle Forecasting (RCF) - explicitly models periodic patterns through
learnable recurrent cycles, then predicts on residual components.

Core approach:
1. Learn periodic patterns: Learnable cycles Q[W, D] where W is period length
2. Extract residuals: residual = input - Q[cycle_indices]
3. Predict residuals: Use simple backbone (Linear or MLP)
4. Add cycles back: prediction = predicted_residual + Q[future_cycle_indices]

This is particularly effective for aircraft sensor data with strong periodic patterns
(engine cycles, atmospheric variations, regular oscillations in cruise).

Reference: https://arxiv.org/abs/2409.18479
Code: https://github.com/ACAT-SCUT/CycleNet
"""

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ARBaseModel
from .registry import register


class LearnableRecurrentCycles(nn.Module):
    """Learnable recurrent cycles for capturing periodic patterns.

    Maintains a learnable parameter Q[W, D] representing one complete cycle
    of length W for D channels. Each time step t maps to cycle position t mod W.
    """

    def __init__(self, period_len: int, num_channels: int):
        """Initialize learnable cycles.

        Args:
            period_len: Length of the recurrent cycle (W)
            num_channels: Number of channels/features (D)
        """
        super().__init__()
        self.period_len = period_len
        self.num_channels = num_channels

        # Learnable cycles Q[W, D] - initialized to zeros as per paper
        self.cycles = nn.Parameter(
            torch.zeros(period_len, num_channels)
        )

    def get_cycle_indices(self, seq_len: int, offset: int = 0) -> torch.Tensor:
        """Compute cycle indices for a sequence.

        Args:
            seq_len: Length of sequence to get indices for
            offset: Offset to add to indices (for future predictions)

        Returns:
            Cycle indices [seq_len] where each value is in [0, period_len)
        """
        # Create indices 0, 1, 2, ..., seq_len-1
        indices = torch.arange(seq_len, device=self.cycles.device)
        # Add offset for future predictions
        indices = indices + offset
        # Map to cycle positions via modulo
        cycle_indices = indices % self.period_len
        return cycle_indices

    def forward(
        self,
        seq_len: int,
        offset: int = 0,
        batch_size: int = 1
    ) -> torch.Tensor:
        """Get cycle values for a sequence.

        Args:
            seq_len: Length of sequence
            offset: Offset for indices (0 for input, T_in for predictions)
            batch_size: Batch size to expand to

        Returns:
            Cycle values [B, seq_len, D]
        """
        # Get cycle indices for this sequence
        cycle_indices = self.get_cycle_indices(seq_len, offset)

        # Index into cycles: [seq_len, D]
        cycle_values = self.cycles[cycle_indices]

        # Expand to batch dimension: [B, seq_len, D]
        cycle_values = cycle_values.unsqueeze(0).expand(batch_size, -1, -1)

        return cycle_values


class ResidualBackbone(nn.Module):
    """Backbone network for predicting residuals after cycle removal.

    Supports two architectures:
    - Linear: Single linear layer
    - MLP: Two-layer MLP with activation and dropout
    """

    def __init__(
        self,
        input_len: int,
        pred_len: int,
        num_channels: int,
        backbone_type: str = "mlp",
        hidden_dim: int = 256,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        """Initialize residual backbone.

        Args:
            input_len: Input sequence length
            pred_len: Prediction horizon
            num_channels: Number of channels
            backbone_type: Type of backbone ('linear' or 'mlp')
            hidden_dim: Hidden dimension for MLP
            dropout: Dropout probability
            activation: Activation function ('relu', 'gelu')
        """
        super().__init__()
        self.backbone_type = backbone_type
        self.input_len = input_len
        self.pred_len = pred_len
        self.num_channels = num_channels

        # Get activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        if backbone_type == "linear":
            # Simple linear projection from input to output
            # Flatten input [B, T_in, D] -> [B, T_in * D]
            # Project to output [B, T_out * D]
            self.network = nn.Linear(
                input_len * num_channels,
                pred_len * num_channels
            )
        elif backbone_type == "mlp":
            # Two-layer MLP
            self.network = nn.Sequential(
                nn.Flatten(start_dim=1),  # [B, T_in, D] -> [B, T_in * D]
                nn.Linear(input_len * num_channels, hidden_dim),
                self.activation,
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, pred_len * num_channels)
            )
        else:
            raise ValueError(f"Unknown backbone type: {backbone_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict residuals.

        Args:
            x: Residual input [B, T_in, D]

        Returns:
            Predicted residuals [B, T_out, D]
        """
        B = x.shape[0]

        if self.backbone_type == "linear":
            # Flatten and predict
            x_flat = x.reshape(B, -1)  # [B, T_in * D]
            pred_flat = self.network(x_flat)  # [B, T_out * D]
            pred = pred_flat.reshape(B, self.pred_len, self.num_channels)
        else:  # mlp
            pred_flat = self.network(x)  # [B, T_out * D]
            pred = pred_flat.reshape(B, self.pred_len, self.num_channels)

        return pred


@register("cyclenet")
class CycleNetModel(ARBaseModel):
    """CycleNet: Residual Cycle Forecasting for Time Series.

    Uses explicit periodic pattern modeling through learnable recurrent cycles.
    Decomposes time series into:
    - Cycles: Learnable periodic patterns Q[W, D]
    - Residuals: Non-periodic components (input - cycles)

    Then predicts residuals using a simple backbone (Linear or MLP) and
    adds back the cycles for final predictions.

    Architecture:
    1. Input: [B, T_in, D]
    2. Extract residuals: residual = input - Q[cycle_indices_input]
    3. Predict residuals: pred_residual = Backbone(residual)
    4. Add future cycles: pred = pred_residual + Q[cycle_indices_future]
    5. Output: [B, T_out, D]

    Key advantage: 90% parameter reduction compared to SOTA transformers
    while maintaining competitive accuracy, especially on periodic data.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        pred_len: int = 1,
        period_len: int = 32,
        backbone: str = "mlp",
        hidden_dim: int = 256,
        dropout: float = 0.1,
        activation: str = "gelu",
        **kwargs
    ):
        """Initialize CycleNet model.

        Args:
            input_dim: Dimension of input features (number of sensors)
            output_dim: Dimension of output predictions
            pred_len: Prediction horizon length
            period_len: Length of learnable recurrent cycles (W)
                       CRITICAL: Must match intrinsic periodicity of data
                       For aircraft sensors, consider:
                       - Engine cycle periods
                       - Sampling rate and expected oscillation frequencies
            backbone: Backbone type ('linear' or 'mlp')
            hidden_dim: Hidden dimension for MLP backbone
            dropout: Dropout probability
            activation: Activation function ('relu', 'gelu')
            **kwargs: Additional arguments
        """
        super().__init__(input_dim, output_dim, **kwargs)

        # Validate period_len
        if period_len < 1:
            raise ValueError(f"period_len must be >= 1, got {period_len}")

        self.pred_len = pred_len
        self.period_len = period_len
        self.backbone_type = backbone
        self.hidden_dim = hidden_dim

        # Learnable recurrent cycles
        # Note: We model cycles for all input channels, even if output_dim < input_dim
        # The backbone will handle dimension mapping
        self.learnable_cycles = LearnableRecurrentCycles(
            period_len=period_len,
            num_channels=input_dim
        )

        # For dimension mapping if output_dim != input_dim
        self.needs_projection = (output_dim != input_dim)

        # Backbone for residual prediction
        # Note: Input length will be dynamic, so we use LazyLinear approach
        # or we need to know seq_len in advance
        # For simplicity, we'll create the backbone on first forward pass
        self.backbone = None
        self.backbone_config = {
            "pred_len": pred_len,
            "num_channels": input_dim,  # Backbone predicts all input channels
            "backbone_type": backbone,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "activation": activation
        }

        # Projection layer if output_dim != input_dim
        if self.needs_projection:
            # Project from input channels to output channels
            self.channel_projection = nn.Linear(input_dim, output_dim)

        # Initialize parameters
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier initialization.

        Cycles are initialized to zeros as per the paper.
        """
        for name, p in self.named_parameters():
            if 'cycles' in name:
                # Keep cycles at zero initialization
                nn.init.zeros_(p)
            elif p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _init_backbone_if_needed(self, input_len: int):
        """Initialize backbone on first forward pass.

        Args:
            input_len: Input sequence length
        """
        if self.backbone is None:
            self.backbone = ResidualBackbone(
                input_len=input_len,
                **self.backbone_config
            )
            # Move to same device as model
            self.backbone = self.backbone.to(self.learnable_cycles.cycles.device)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor [B, T_in, D_in]
            context: Optional context tensor (not used in CycleNet)
            **kwargs: Additional arguments

        Returns:
            Dictionary with 'preds' and 'extras'
        """
        B, T_in, D_in = x.shape

        # Initialize backbone on first forward pass
        self._init_backbone_if_needed(T_in)

        # Step 1: Get cycle values for input sequence
        # Offset = 0 for input (starts at position 0 in cycle)
        input_cycles = self.learnable_cycles(
            seq_len=T_in,
            offset=0,
            batch_size=B
        )  # [B, T_in, D_in]

        # Step 2: Extract residuals by removing cycles
        residuals = x - input_cycles  # [B, T_in, D_in]

        # Step 3: Predict future residuals using backbone
        pred_residuals = self.backbone(residuals)  # [B, pred_len, D_in]

        # Step 4: Get cycle values for future sequence
        # Offset = T_in (future starts after input ends)
        future_cycles = self.learnable_cycles(
            seq_len=self.pred_len,
            offset=T_in,
            batch_size=B
        )  # [B, pred_len, D_in]

        # Step 5: Add cycles back to predictions
        preds_full = pred_residuals + future_cycles  # [B, pred_len, D_in]

        # Step 6: Project to output dimension if needed
        if self.needs_projection:
            # Apply projection along the channel dimension
            # [B, pred_len, D_in] -> [B, pred_len, D_out]
            preds = self.channel_projection(preds_full)
        else:
            preds = preds_full

        return {
            "preds": preds,
            "extras": {
                "residuals": residuals,
                "input_cycles": input_cycles,
                "future_cycles": future_cycles,
                "pred_residuals": pred_residuals,
                "learnable_cycles": self.learnable_cycles.cycles.detach().clone()
            }
        }

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  input_dim={self.input_dim},\n"
            f"  output_dim={self.output_dim},\n"
            f"  pred_len={self.pred_len},\n"
            f"  period_len={self.period_len},\n"
            f"  backbone={self.backbone_type},\n"
            f"  hidden_dim={self.hidden_dim},\n"
            f"  num_params={self.get_num_params():,}\n"
            f")"
        )
