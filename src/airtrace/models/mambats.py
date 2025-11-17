"""MambaTS: Improved Selective State Space Models for Long-term Time Series Forecasting.

Reference: https://arxiv.org/abs/2405.16440

Key innovations:
1. Variable Scan along Time (VST): Segments variables into patches and organizes tokens at the same
   timestep in an alternating fashion temporally, allowing the model to capture both temporal and
   cross-variable dependencies efficiently.
2. Temporal Mamba Block (TMB): Removes causal convolution (unnecessary for LTSF) from standard Mamba
   to better suit multivariate time series forecasting.
3. Linear complexity: O(L) global dependency modeling compared to quadratic attention.
4. Achieves SOTA results on long-term forecasting benchmarks.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ARBaseModel
from .registry import register

LOGGER = logging.getLogger(__name__)


class PatchEmbedding(nn.Module):
    """Patch embedding layer for time series data.

    Segments the input time series into patches (subsequences) and projects them into
    an embedding space. This reduces sequence length and creates richer token representations.

    Args:
        input_dim: Number of input channels/variables
        patch_len: Length of each patch (subsequence)
        stride: Stride for creating patches (overlap if stride < patch_len)
        embed_dim: Embedding dimension for each patch
        dropout: Dropout probability applied to embeddings
    """

    def __init__(
        self,
        input_dim: int,
        patch_len: int,
        stride: int,
        embed_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if patch_len <= 0:
            raise ValueError(f"patch_len must be positive, got {patch_len}")
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")

        self.input_dim = input_dim
        self.patch_len = patch_len
        self.stride = stride
        self.embed_dim = embed_dim

        # Linear projection for each patch
        # Input: [B, num_patches, input_dim, patch_len] -> [B, num_patches, input_dim, embed_dim]
        self.patch_proj = nn.Linear(patch_len, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """Apply patch embedding to input time series.

        Args:
            x: Input tensor [B, T, D] where B is batch size, T is sequence length,
               D is number of variables

        Returns:
            Tuple containing:
                - Embedded patches [B, num_patches, D, embed_dim]
                - Number of patches created
        """
        B, T, D = x.shape

        # Calculate number of patches
        num_patches = (T - self.patch_len) // self.stride + 1

        # Create patches using unfold
        # [B, T, D] -> [B, D, T] -> unfold -> [B, D, num_patches, patch_len]
        x_patches = x.transpose(1, 2).unfold(dimension=2, size=self.patch_len, step=self.stride)
        # [B, D, num_patches, patch_len] -> [B, num_patches, D, patch_len]
        x_patches = x_patches.transpose(1, 2)

        # Project patches to embedding space
        # [B, num_patches, D, patch_len] -> [B, num_patches, D, embed_dim]
        patch_embeds = self.patch_proj(x_patches)
        patch_embeds = self.dropout(patch_embeds)

        return patch_embeds, num_patches


class TemporalMambaBlock(nn.Module):
    """Temporal Mamba Block (TMB) without causal convolution.

    Compared to standard Mamba blocks, TMB removes causal convolution as it's not necessary
    for long-term time series forecasting. The model uses selective state-space scans with
    linear complexity for capturing global dependencies.

    Args:
        embed_dim: Embedding dimension
        state_dim: Internal state dimension for selective scan
        expand_factor: Expansion factor for intermediate dimension
        dt_rank: Rank of delta (Δ) parameter
        dropout: Dropout probability
        bidirectional: Whether to use bidirectional scanning
    """

    def __init__(
        self,
        embed_dim: int,
        state_dim: int = 16,
        expand_factor: int = 2,
        dt_rank: Optional[int] = None,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.state_dim = state_dim
        self.expand_dim = embed_dim * expand_factor
        self.bidirectional = bidirectional

        # Auto-compute dt_rank if not specified (typically small, e.g., ceil(embed_dim / 16))
        if dt_rank is None:
            dt_rank = max(1, embed_dim // 16)
        self.dt_rank = dt_rank

        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)

        # Input projection (no causal conv in TMB)
        self.in_proj = nn.Linear(embed_dim, self.expand_dim * 2)  # For input and gate

        # Selective scan parameters (simplified from full Mamba)
        self.x_proj = nn.Linear(self.expand_dim, dt_rank + state_dim * 2)
        self.dt_proj = nn.Linear(dt_rank, self.expand_dim)

        # State space parameters
        # A: [expand_dim, state_dim] - state transition
        self.A = nn.Parameter(torch.randn(self.expand_dim, state_dim))
        # D: [expand_dim] - direct connection (skip)
        self.D = nn.Parameter(torch.ones(self.expand_dim))

        # Output projection
        self.out_proj = nn.Linear(self.expand_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        self._init_parameters()

    def _init_parameters(self) -> None:
        """Initialize parameters following Mamba conventions."""
        # Initialize A to be negative (for stability)
        nn.init.normal_(self.A, mean=0.0, std=0.02)
        with torch.no_grad():
            self.A.data = -torch.exp(self.A.data)

    def _selective_scan(
        self, x: torch.Tensor, delta: torch.Tensor, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor
    ) -> torch.Tensor:
        """Simplified selective scan operation.

        Args:
            x: Input [B, L, expand_dim]
            delta: Time step [B, L, expand_dim]
            A: State transition [expand_dim, state_dim]
            B: Input matrix [B, L, state_dim]
            C: Output matrix [B, L, state_dim]

        Returns:
            Output [B, L, expand_dim]
        """
        B_batch, L, D = x.shape
        N = self.state_dim

        # Discretize A using delta (Euler method)
        # A: [D, N], delta: [B, L, D] -> deltaA: [B, L, D, N]
        deltaA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))

        # deltaB: [B, L, D, N]
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)

        # Initialize state
        h = torch.zeros(B_batch, D, N, device=x.device, dtype=x.dtype)

        outputs = []
        for t in range(L):
            # h = deltaA * h + deltaB * x
            h = deltaA[:, t] * h + deltaB[:, t] * x[:, t].unsqueeze(-1)
            # y = C * h
            y = (C[:, t].unsqueeze(1) * h).sum(dim=-1)  # [B, D]
            outputs.append(y)

        # Stack outputs: [B, L, D]
        output = torch.stack(outputs, dim=1)
        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Temporal Mamba Block.

        Args:
            x: Input [B, L, embed_dim]

        Returns:
            Output [B, L, embed_dim]
        """
        B, L, D = x.shape
        residual = x

        # Normalize
        x = self.norm(x)

        # Input projection (no causal conv)
        x_and_gate = self.in_proj(x)  # [B, L, expand_dim * 2]
        x_input, gate = x_and_gate.chunk(2, dim=-1)  # Each: [B, L, expand_dim]

        # Apply SiLU activation to input
        x_input = F.silu(x_input)

        # Generate selective scan parameters
        x_params = self.x_proj(x_input)  # [B, L, dt_rank + 2*state_dim]
        delta, B_param, C_param = torch.split(
            x_params, [self.dt_rank, self.state_dim, self.state_dim], dim=-1
        )

        # Project delta to expand_dim
        delta = F.softplus(self.dt_proj(delta))  # [B, L, expand_dim]

        # Forward selective scan
        y_forward = self._selective_scan(x_input, delta, self.A, B_param, C_param)

        # Optionally apply bidirectional scan
        if self.bidirectional:
            # Reverse scan
            x_input_rev = torch.flip(x_input, dims=[1])
            delta_rev = torch.flip(delta, dims=[1])
            B_param_rev = torch.flip(B_param, dims=[1])
            C_param_rev = torch.flip(C_param, dims=[1])

            y_backward = self._selective_scan(x_input_rev, delta_rev, self.A, B_param_rev, C_param_rev)
            y_backward = torch.flip(y_backward, dims=[1])

            # Average forward and backward
            y = (y_forward + y_backward) * 0.5
        else:
            y = y_forward

        # Add skip connection (D parameter)
        y = y + x_input * self.D.unsqueeze(0).unsqueeze(0)

        # Gate the output
        y = y * F.silu(gate)

        # Output projection
        output = self.out_proj(y)
        output = self.dropout(output)

        # Residual connection
        return output + residual


class VariableScanEncoder(nn.Module):
    """Encoder with Variable Scan along Time (VST) mechanism.

    VST organizes patches from different variables at the same time step together,
    allowing the model to capture both temporal patterns and cross-variable dependencies
    efficiently through Mamba's linear-complexity selective scan.

    Args:
        embed_dim: Embedding dimension
        state_dim: State dimension for Mamba blocks
        num_layers: Number of Temporal Mamba Blocks
        expand_factor: Expansion factor for Mamba blocks
        dropout: Dropout probability
        bidirectional: Whether to use bidirectional scanning
    """

    def __init__(
        self,
        embed_dim: int,
        state_dim: int = 16,
        num_layers: int = 4,
        expand_factor: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        # Stack of Temporal Mamba Blocks
        self.layers = nn.ModuleList(
            [
                TemporalMambaBlock(
                    embed_dim=embed_dim,
                    state_dim=state_dim,
                    expand_factor=expand_factor,
                    dropout=dropout,
                    bidirectional=bidirectional,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(embed_dim)

    def apply_variable_scan(self, patch_embeds: torch.Tensor) -> torch.Tensor:
        """Apply Variable Scan along Time (VST) to patch embeddings.

        Rearranges patches so that all variables at time t are processed before time t+1.
        This allows the selective scan to capture cross-variable dependencies at each time step.

        Args:
            patch_embeds: [B, num_patches, num_vars, embed_dim]

        Returns:
            Scanned embeddings [B, num_patches * num_vars, embed_dim]
        """
        B, P, V, E = patch_embeds.shape

        # Rearrange to process all variables at each time step together
        # [B, P, V, E] -> [B, P, V, E] -> [B, P*V, E]
        # Each position in sequence: var1_t1, var2_t1, ..., varV_t1, var1_t2, var2_t2, ...
        vst_embeds = patch_embeds.reshape(B, P * V, E)

        return vst_embeds

    def forward(self, patch_embeds: torch.Tensor) -> torch.Tensor:
        """Forward pass through the VST encoder.

        Args:
            patch_embeds: Patch embeddings [B, num_patches, num_vars, embed_dim]

        Returns:
            Encoded features [B, num_patches * num_vars, embed_dim]
        """
        # Apply Variable Scan along Time
        x = self.apply_variable_scan(patch_embeds)

        # Process through Temporal Mamba Blocks
        for layer in self.layers:
            x = layer(x)

        # Final normalization
        x = self.final_norm(x)

        return x


@register("mambats")
class MambaTSModel(ARBaseModel):
    """MambaTS: Improved Selective State Space Model for Long-term Time Series Forecasting.

    MambaTS adapts the Mamba state-space model for time series with two key innovations:
    1. Variable Scan along Time (VST): Organizes patches from different variables at the same
       timestep together, enabling efficient capture of both temporal and cross-variable patterns.
    2. Temporal Mamba Block (TMB): Removes causal convolution from standard Mamba as it's
       unnecessary for forecasting, simplifying the architecture while maintaining performance.

    Key features:
    - Linear O(L) complexity for long sequences (vs quadratic for attention)
    - Patching mechanism for efficiency and better representation
    - Optional bidirectional scanning for improved stability
    - SOTA performance on long-term forecasting benchmarks

    Reference: https://arxiv.org/abs/2405.16440

    Args:
        input_dim: Number of input sensor channels/variables
        output_dim: Number of output prediction channels
        pred_len: Forecast horizon (number of timesteps to predict)
        patch_len: Length of each patch (typical: 16)
        stride: Stride for patch creation (typical: 8, allows 50% overlap)
        embed_dim: Token embedding dimension
        state_dim: Internal state dimension for selective scan
        num_layers: Number of stacked Temporal Mamba Blocks
        expand_factor: Expansion factor for intermediate dimensions in Mamba blocks
        dropout: Dropout probability
        bidirectional_scan: If True, use bidirectional selective scan
        normalize_input: Whether to normalize input sequences
        **kwargs: Additional arguments passed to ARBaseModel
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        pred_len: int = 1,
        patch_len: int = 16,
        stride: int = 8,
        embed_dim: int = 128,
        state_dim: int = 16,
        num_layers: int = 4,
        expand_factor: int = 2,
        dropout: float = 0.1,
        bidirectional_scan: bool = True,
        normalize_input: bool = True,
        **kwargs: Dict,
    ) -> None:
        super().__init__(input_dim, output_dim, **kwargs)

        # Validation
        if pred_len <= 0:
            raise ValueError(f"pred_len must be positive, got {pred_len}")
        if patch_len <= 0:
            raise ValueError(f"patch_len must be positive, got {patch_len}")
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if state_dim <= 0:
            raise ValueError(f"state_dim must be positive, got {state_dim}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if expand_factor <= 0:
            raise ValueError(f"expand_factor must be positive, got {expand_factor}")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError(f"dropout must be in [0, 1], got {dropout}")

        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.embed_dim = embed_dim
        self.normalize_input = normalize_input

        # Patch embedding
        self.patch_embedding = PatchEmbedding(
            input_dim=input_dim,
            patch_len=patch_len,
            stride=stride,
            embed_dim=embed_dim,
            dropout=dropout,
        )

        # Variable Scan Encoder with Temporal Mamba Blocks
        self.encoder = VariableScanEncoder(
            embed_dim=embed_dim,
            state_dim=state_dim,
            num_layers=num_layers,
            expand_factor=expand_factor,
            dropout=dropout,
            bidirectional=bidirectional_scan,
        )

        # Prediction head
        # Pool encoded features and project to output
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, pred_len * output_dim),
        )

        # Normalization buffers (mean and std for input)
        if normalize_input:
            self.register_buffer("input_mean", torch.zeros(1, 1, input_dim))
            self.register_buffer("input_std", torch.ones(1, 1, input_dim))

    def _normalize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normalize input sequence.

        Args:
            x: Input [B, T, D]

        Returns:
            Tuple of (normalized_x, mean, std)
        """
        if not self.normalize_input:
            return x, None, None

        # Compute statistics along time dimension
        mean = x.mean(dim=1, keepdim=True)  # [B, 1, D]
        std = x.std(dim=1, keepdim=True) + 1e-5  # [B, 1, D]

        # Normalize
        x_norm = (x - mean) / std

        return x_norm, mean, std

    def _denormalize(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """Denormalize predictions.

        Args:
            x: Normalized predictions [B, T, D]
            mean: Mean used for normalization [B, 1, D]
            std: Std used for normalization [B, 1, D]

        Returns:
            Denormalized predictions [B, T, D]
        """
        if not self.normalize_input or mean is None or std is None:
            return x

        return x * std + mean

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        **kwargs: Dict,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through MambaTS.

        Args:
            x: Input tensor [B, T, D_in] where B is batch size, T is sequence length,
               D_in is input dimension (number of variables)
            context: Optional context tensor (currently unused)
            **kwargs: Additional arguments (unused, for interface compatibility)

        Returns:
            Dictionary containing:
                - preds: Predictions [B, pred_len, D_out]
                - extras: Dict with 'patch_embeds' and 'encoded_features' for analysis
        """
        del context, kwargs  # Unused

        B, T, D = x.shape

        # Normalize input
        x_norm, mean, std = self._normalize(x)

        # Create patch embeddings
        # [B, T, D] -> [B, num_patches, D, embed_dim]
        patch_embeds, num_patches = self.patch_embedding(x_norm)

        # Encode with Variable Scan along Time
        # [B, num_patches, D, embed_dim] -> [B, num_patches * D, embed_dim]
        encoded = self.encoder(patch_embeds)

        # Pool across all tokens (mean pooling)
        # [B, num_patches * D, embed_dim] -> [B, embed_dim]
        pooled = encoded.mean(dim=1)

        # Generate predictions
        # [B, embed_dim] -> [B, pred_len * output_dim] -> [B, pred_len, output_dim]
        preds = self.head(pooled)
        preds = preds.view(B, self.pred_len, self.output_dim)

        # Denormalize predictions
        preds = self._denormalize(preds, mean, std)

        extras = {
            "patch_embeds": patch_embeds,
            "encoded_features": encoded,
            "num_patches": num_patches,
        }

        return {"preds": preds, "extras": extras}
