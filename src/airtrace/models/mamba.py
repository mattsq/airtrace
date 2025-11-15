"""S-Mamba: Simple Mamba for Time Series Forecasting.

Implementation of S-Mamba from:
"Is Mamba Effective for Time Series Forecasting?" (Wen et al., 2024)
Published in Neurocomputing, 2024
arXiv: https://arxiv.org/abs/2403.11144

Key innovation:
S-Mamba applies the Mamba selective state space model (SSM) to time series
forecasting with a streamlined 4-layer architecture:

1. Linear Tokenization: Embed time series to d_model dimension
2. Mamba Inter-Variate Correlation (VC) Encoding: Bidirectional Mamba block
   to capture relationships between different sensors/variates
3. FFN Temporal Dependency (TD) Encoding: Feed-forward network to extract
   temporal patterns
4. Projection: Map to output dimension and horizon

Advantages:
- Linear complexity O(L) vs. transformers' quadratic O(L²)
- Captures long-range dependencies without degradation
- Efficient for long sequences (aircraft flight data)
- Strong performance on multivariate sensor data

Reference:
    Mamba: Linear-Time Sequence Modeling with Selective State Spaces
    (Gu & Dao, 2023) https://arxiv.org/abs/2312.00752

    Is Mamba Effective for Time Series Forecasting?
    (Wen et al., 2024) https://arxiv.org/abs/2403.11144
"""

from typing import Dict, Optional

import torch
import torch.nn as nn

from .base import ARBaseModel
from .mamba_blocks import BidirectionalMambaBlock, RMSNorm
from .registry import register


@register("mamba")
class MambaModel(ARBaseModel):
    """S-Mamba: Simple Mamba model for time series forecasting.

    Architecture:
        Input [B, T_in, D_in]
            ↓
        1. Linear Tokenization → [B, T_in, d_model]
            ↓
        2. Bidirectional Mamba Block (Inter-Variate Correlation)
            ↓
        3. Feed-Forward Network (Temporal Dependency)
            ↓
        4. Projection → [B, T_out, D_out]

    The bidirectional Mamba block captures correlations between different
    sensors (variates), while the FFN extracts temporal patterns. This
    separation of concerns leads to better performance and interpretability.

    Key features:
    - Linear complexity in sequence length
    - Selective state space mechanism adapts to input
    - Bidirectional processing for inter-variate relationships
    - Efficient for long sequences (100s to 1000s of time steps)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        pred_len: int = 1,
        d_model: int = 512,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        n_layers: int = 1,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
        norm_type: str = "layer",
        **kwargs
    ):
        """Initialize S-Mamba model.

        Args:
            input_dim: Dimension of input features (number of sensors)
            output_dim: Dimension of output predictions
            pred_len: Prediction horizon length (1 for one-step, >1 for multi-step)
            d_model: Model embedding dimension
            d_state: SSM state expansion factor (N in Mamba paper, typically 16)
            d_conv: Local convolution width (typically 4)
            expand: SSM block expansion factor (typically 2)
            n_layers: Number of Mamba layers (default 1 as in S-Mamba paper)
            ffn_dim: Feed-forward network dimension
            dropout: Dropout probability
            norm_type: Normalization type ('layer' or 'rms')
            **kwargs: Additional arguments
        """
        super().__init__(input_dim, output_dim, **kwargs)

        self.pred_len = pred_len
        self.d_model = d_model
        self.d_state = d_state
        self.n_layers = n_layers

        # Layer 1: Linear Tokenization
        # Embed each time series to d_model dimension
        self.tokenization = nn.Linear(input_dim, d_model)

        # Layer 2: Mamba Inter-Variate Correlation (VC) Encoding
        # Bidirectional Mamba blocks to capture relationships between variates
        self.mamba_layers = nn.ModuleList([
            BidirectionalMambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
                norm_type=norm_type,
            )
            for _ in range(n_layers)
        ])

        # Layer 3: Feed-Forward Network (FFN) for Temporal Dependency (TD) Encoding
        # Extract temporal patterns
        if norm_type == "layer":
            self.ffn_norm = nn.LayerNorm(d_model)
        elif norm_type == "rms":
            self.ffn_norm = RMSNorm(d_model)
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}")

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

        # Layer 4: Projection to output
        # Map from (T_in, d_model) to (T_out, output_dim)
        # Two strategies based on whether output_dim == input_dim
        self.output_dim_equals_input = (output_dim == input_dim)

        if self.output_dim_equals_input:
            # Strategy 1: Project temporal dimension per channel
            # Each channel's d_model features → pred_len predictions
            self.projection = nn.Linear(d_model, pred_len)
        else:
            # Strategy 2: Flatten and project to output shape
            # Uses lazy linear to adapt to any input sequence length
            self.projection = nn.Sequential(
                nn.Flatten(start_dim=1),  # [B, T_in, d_model] → [B, T_in * d_model]
                nn.LazyLinear(output_dim * pred_len),
            )

        # Initialize parameters
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier initialization.

        Mamba blocks have their own specialized initialization.
        """
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) and 'mamba' not in name.lower():
                if module.weight.requires_grad:
                    nn.init.xavier_uniform_(module.weight)
                if module.bias is not None and module.bias.requires_grad:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor [B, T_in, D_in]
            context: Optional context tensor (not used in S-Mamba)
            **kwargs: Additional arguments

        Returns:
            Dictionary with 'preds' and 'extras':
                - preds: Predictions [B, T_out, D_out]
                - extras: Dictionary with intermediate outputs for analysis
        """
        B, T_in, D_in = x.shape

        # Layer 1: Linear Tokenization
        # Embed input time series to model dimension
        x = self.tokenization(x)  # [B, T_in, d_model]

        # Layer 2: Mamba Inter-Variate Correlation (VC) Encoding
        # Apply bidirectional Mamba blocks
        mamba_outputs = []
        for mamba_layer in self.mamba_layers:
            x = mamba_layer(x)  # [B, T_in, d_model]
            mamba_outputs.append(x)

        # Layer 3: FFN Temporal Dependency (TD) Encoding
        # Apply feed-forward network with residual connection
        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + residual  # [B, T_in, d_model]

        # Layer 4: Projection to output
        if self.output_dim_equals_input:
            # Project each channel's features to pred_len predictions
            # [B, T_in, d_model] → [B, T_in, pred_len]
            x = self.projection(x)
            # Permute to [B, pred_len, T_in]
            # But we want [B, pred_len, D_out] where D_out = T_in
            # Actually, we want to project d_model → pred_len per variate
            # So the shape is [B, D_in, pred_len] after permute
            preds = x.permute(0, 2, 1)  # [B, pred_len, D_in]
        else:
            # Flatten and project to output shape
            # [B, T_in, d_model] → [B, T_in * d_model] → [B, output_dim * pred_len]
            x = self.projection(x)
            # Reshape to [B, pred_len, output_dim]
            preds = x.reshape(B, self.pred_len, self.output_dim)

        return {
            "preds": preds,
            "extras": {
                "mamba_outputs": mamba_outputs,
                "final_encoding": x,
            }
        }

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  input_dim={self.input_dim},\n"
            f"  output_dim={self.output_dim},\n"
            f"  pred_len={self.pred_len},\n"
            f"  d_model={self.d_model},\n"
            f"  d_state={self.d_state},\n"
            f"  n_layers={self.n_layers},\n"
            f"  num_params={self.get_num_params():,}\n"
            f")"
        )
