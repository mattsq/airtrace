"""S-Mamba: Simple Mamba for Time Series Forecasting.

Implementation based on:
"Is Mamba Effective for Time Series Forecasting?" (arXiv 2403.11144, March 2024)
"MambaTS: Improved Selective State Space Models for Long-term Time Series Forecasting"
(arXiv 2405.16440, May 2024)

Key innovations:
1. State space models with linear complexity O(n) vs Transformer's O(n²)
2. Selective state space mechanism for data-dependent processing
3. Bidirectional Mamba layer for capturing inter-variate correlations
4. Feed-forward network for learning temporal dependencies

This is particularly effective for long aircraft sensor sequences where quadratic
complexity of Transformers becomes prohibitive.

Architecture:
1. Input: [B, T_in, D] time series with D sensors
2. Variate Tokenization: Linear projection for each sensor
3. Bidirectional Mamba Layer: Process forward and backward, then combine
4. Feed-Forward Network: Learn temporal representations
5. Projection: Map to output dimension and horizon

Reference: https://arxiv.org/abs/2403.11144
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ARBaseModel
from .registry import register


class SelectiveStateSpaceBlock(nn.Module):
    """Selective State Space Block - core component of Mamba.

    Implements a simplified version of selective state space models that:
    - Uses data-dependent state transitions (selection mechanism)
    - Operates in linear time complexity O(n)
    - Captures long-range dependencies efficiently

    This is a PyTorch-native implementation that captures the key ideas
    without requiring the mamba-ssm package.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand_factor: int = 2,
        dropout: float = 0.0,
    ):
        """Initialize Selective SSM block.

        Args:
            d_model: Model dimension
            d_state: State dimension (N in SSM literature)
            d_conv: Convolution kernel size for local context
            expand_factor: Expansion factor for inner dimension
            dropout: Dropout probability
        """
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand_factor

        # Input projection (expand dimension)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)

        # 1D convolution for local context (important for time series)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,  # Depthwise convolution
        )

        # Selective SSM parameters
        # These are data-dependent (selective mechanism)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2)  # For B and C matrices
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner)  # For delta (timestep)

        # Initialize A matrix (state transition) - this is fixed per channel
        # A should be negative for stability (uses log parameterization)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).reshape(1, d_state)
        A = A.repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))  # Log space for stability

        # D matrix (skip connection from input to output)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through selective SSM.

        Args:
            x: Input tensor [B, L, D]

        Returns:
            Output tensor [B, L, D]
        """
        B, L, D = x.shape

        # Input projection and split
        x_and_gate = self.in_proj(x)  # [B, L, 2*d_inner]
        x_proj, gate = x_and_gate.chunk(2, dim=-1)  # Each [B, L, d_inner]

        # Apply 1D convolution for local context
        # Conv1d expects [B, C, L]
        x_conv = x_proj.transpose(1, 2)  # [B, d_inner, L]
        x_conv = self.conv1d(x_conv)[..., :L]  # Trim padding, [B, d_inner, L]
        x_conv = x_conv.transpose(1, 2)  # [B, L, d_inner]

        # Apply activation
        x_conv = F.silu(x_conv)

        # Selective SSM: data-dependent parameters
        ssm_params = self.x_proj(x_conv)  # [B, L, 2*d_state]
        B_sel, C_sel = ssm_params.chunk(2, dim=-1)  # Each [B, L, d_state]

        # Compute delta (data-dependent timestep)
        delta = F.softplus(self.dt_proj(x_conv))  # [B, L, d_inner]

        # Simplified selective scan
        # In full Mamba, this uses hardware-efficient parallel scan
        # Here we use a simplified sequential version
        y = self.selective_scan(
            x_conv, delta, self.A_log.exp(), B_sel, C_sel, self.D
        )

        # Gating mechanism (important for Mamba)
        y = y * F.silu(gate)

        # Output projection
        y = self.out_proj(y)
        y = self.dropout(y)

        return y

    def selective_scan(
        self,
        u: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
    ) -> torch.Tensor:
        """Simplified selective scan operation.

        This implements the core state space recurrence:
        h_t = A * h_{t-1} + B * u_t
        y_t = C * h_t + D * u_t

        Args:
            u: Input [B, L, d_inner]
            delta: Timestep [B, L, d_inner]
            A: State transition [d_inner, d_state]
            B: Input matrix [B, L, d_state]
            C: Output matrix [B, L, d_state]
            D: Skip connection [d_inner]

        Returns:
            Output [B, L, d_inner]
        """
        B_batch, L, d_inner = u.shape
        d_state = A.shape[1]

        # Discretization: convert continuous to discrete time
        # deltaA = exp(delta * A)
        deltaA = torch.exp(delta.unsqueeze(-1) * (-A.unsqueeze(0).unsqueeze(0)))
        # [B, L, d_inner, d_state]

        # deltaB = delta * B
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  # [B, L, 1, d_state] * [B, L, d_state, 1]
        deltaB = deltaB.squeeze(2)  # [B, L, d_state]

        # Initialize state
        h = torch.zeros(B_batch, d_inner, d_state, device=u.device, dtype=u.dtype)

        outputs = []

        # Sequential scan (can be parallelized with associative scan)
        for t in range(L):
            # Update state: h = deltaA * h + deltaB * u
            h = deltaA[:, t] * h + deltaB[:, t].unsqueeze(1) * u[:, t].unsqueeze(-1)
            # Compute output: y = C * h + D * u
            y = (C[:, t].unsqueeze(1) * h).sum(dim=-1) + D * u[:, t]
            outputs.append(y)

        # Stack outputs
        y = torch.stack(outputs, dim=1)  # [B, L, d_inner]

        return y


class BidirectionalMamba(nn.Module):
    """Bidirectional Mamba layer for capturing inter-variate correlations.

    Processes the sequence in both forward and backward directions,
    then combines the results. This is particularly effective for
    capturing relationships between different sensors in aircraft data.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand_factor: int = 2,
        dropout: float = 0.0,
    ):
        """Initialize bidirectional Mamba layer.

        Args:
            d_model: Model dimension
            d_state: State dimension
            d_conv: Convolution kernel size
            expand_factor: Expansion factor
            dropout: Dropout probability
        """
        super().__init__()

        # Forward and backward SSM blocks
        self.forward_ssm = SelectiveStateSpaceBlock(
            d_model, d_state, d_conv, expand_factor, dropout
        )
        self.backward_ssm = SelectiveStateSpaceBlock(
            d_model, d_state, d_conv, expand_factor, dropout
        )

        # Combine forward and backward
        self.combine = nn.Linear(d_model * 2, d_model)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through bidirectional Mamba.

        Args:
            x: Input tensor [B, L, D]

        Returns:
            Output tensor [B, L, D]
        """
        # Forward direction
        x_forward = self.forward_ssm(x)

        # Backward direction (flip sequence)
        x_backward = torch.flip(x, dims=[1])
        x_backward = self.backward_ssm(x_backward)
        x_backward = torch.flip(x_backward, dims=[1])  # Flip back

        # Combine
        x_combined = torch.cat([x_forward, x_backward], dim=-1)
        x_out = self.combine(x_combined)

        # Residual connection and normalization
        x_out = self.norm(x + self.dropout(x_out))

        return x_out


@register("smamba")
class SMambaModel(ARBaseModel):
    """S-Mamba: Simple Mamba for Time Series Forecasting.

    State space model with linear complexity, ideal for long aircraft sensor sequences.

    Architecture:
    1. Variate tokenization: Linear embedding for each sensor
    2. Bidirectional Mamba layers: Capture inter-variate correlations
    3. Feed-forward network: Learn temporal patterns
    4. Output projection: Map to prediction horizon

    Key advantages over Transformers:
    - Linear O(n) complexity vs O(n²)
    - Can process much longer sequences
    - Maintains strong performance on long-range dependencies
    - Lower memory footprint

    Particularly effective for:
    - Long aircraft sensor sequences (hours of flight data)
    - Multivariate time series with cross-sensor dependencies
    - Real-time deployment scenarios requiring efficiency
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        pred_len: int = 1,
        d_model: int = 128,
        d_state: int = 16,
        d_conv: int = 4,
        expand_factor: int = 2,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_norm: bool = True,
        **kwargs
    ):
        """Initialize S-Mamba model.

        Args:
            input_dim: Dimension of input features (number of sensors)
            output_dim: Dimension of output predictions
            pred_len: Prediction horizon length (1 for one-step, >1 for multi-step)
            d_model: Model dimension (hidden size)
            d_state: State space dimension (N in SSM literature)
            d_conv: Convolution kernel size for local context
            expand_factor: Expansion factor for SSM inner dimension
            num_layers: Number of bidirectional Mamba layers
            dropout: Dropout probability
            use_norm: Whether to use layer normalization
            **kwargs: Additional arguments
        """
        super().__init__(input_dim, output_dim, **kwargs)

        self.pred_len = pred_len
        self.d_model = d_model
        self.num_layers = num_layers

        # Variate tokenization: embed each sensor to d_model dimension
        # Uses LazyLinear to adapt to any input sequence length
        self.variate_embedding = nn.LazyLinear(d_model)

        # Positional encoding (learnable for time series)
        # We'll use a simple learnable embedding per position
        # Note: Will be initialized on first forward pass
        self.positional_encoding = None

        # Stack of bidirectional Mamba layers
        self.mamba_layers = nn.ModuleList(
            [
                BidirectionalMamba(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand_factor=expand_factor,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        # Feed-forward network for temporal patterns
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

        # Final normalization
        self.norm = nn.LayerNorm(d_model) if use_norm else nn.Identity()

        # Projection to output
        # Map from hidden representation to output predictions
        # Strategy: pool temporal information, then project to output
        self.to_variates = nn.Linear(d_model, input_dim)  # Map back to variate space
        self.to_output = nn.Linear(input_dim, output_dim * pred_len)  # Final projection

        self.output_dim_equals_input = (output_dim == input_dim)

        # Initialize parameters
        self._init_weights()

    def _init_positional_encoding(self, seq_len: int, device: torch.device):
        """Initialize positional encoding lazily based on sequence length.

        Args:
            seq_len: Sequence length
            device: Device to create tensor on
        """
        if self.positional_encoding is None or self.positional_encoding.shape[1] != seq_len:
            # Create learnable positional encodings
            self.positional_encoding = nn.Parameter(
                torch.randn(1, seq_len, self.d_model, device=device) * 0.02
            )

    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for name, p in self.named_parameters():
            if p.dim() > 1 and "variate_embedding" not in name:
                nn.init.xavier_uniform_(p)

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
            Dictionary with 'preds' and 'extras'
        """
        B, T_in, D_in = x.shape

        # Step 1: Variate tokenization
        # Embed each time point to d_model dimension
        # [B, T_in, D_in] -> [B, T_in, D_in, d_model] is not what we want
        # Instead: treat as [B*D_in, T_in] sequence, embed, then reshape
        # Actually, we want to process across time for each variate
        # Let's keep it simple: [B, T, D] -> [B, T, d_model] with shared projection

        # Flatten spatial: [B, T, D] -> [B*T, D]
        x_flat = x.reshape(B * T_in, D_in)
        # Embed: [B*T, D] -> [B*T, d_model]
        x_embed = self.variate_embedding(x_flat)
        # Reshape back: [B*T, d_model] -> [B, T, d_model]
        x_embed = x_embed.reshape(B, T_in, self.d_model)

        # Initialize positional encoding on first forward pass
        if self.positional_encoding is None or self.positional_encoding.shape[1] != T_in:
            self._init_positional_encoding(T_in, x.device)

        # Add positional encoding
        x_embed = x_embed + self.positional_encoding[:, :T_in, :]

        # Step 2: Apply bidirectional Mamba layers
        hidden = x_embed
        for mamba_layer in self.mamba_layers:
            hidden = mamba_layer(hidden)

        # Step 3: Apply feed-forward network
        hidden_ffn = self.ffn(hidden)
        hidden = self.norm(hidden + hidden_ffn)  # Residual connection

        # Step 4: Projection to output
        # Use mean pooling across time to aggregate temporal information
        hidden_pooled = hidden.mean(dim=1)  # [B, d_model]

        # Map back to variate space
        variate_repr = self.to_variates(hidden_pooled)  # [B, D_in]

        # Project to output dimension and prediction horizon
        out_flat = self.to_output(variate_repr)  # [B, output_dim * pred_len]

        # Reshape to [B, pred_len, output_dim]
        preds = out_flat.reshape(B, self.pred_len, self.output_dim)

        return {
            "preds": preds,
            "extras": {
                "hidden_states": hidden,
                "embedded": x_embed,
            }
        }

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  input_dim={self.input_dim},\n"
            f"  output_dim={self.output_dim},\n"
            f"  pred_len={self.pred_len},\n"
            f"  d_model={self.d_model},\n"
            f"  num_layers={self.num_layers},\n"
            f"  num_params={self.get_num_params():,}\n"
            f")"
        )
