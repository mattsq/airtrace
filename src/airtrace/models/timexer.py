"""TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables.

Implementation of TimeXer from:
"TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables"
(NeurIPS 2024)

Key innovations:
1. Dual representation: Patch-level for endogenous, variate-level for exogenous
2. Global endogenous tokens: Bridge between endogenous and exogenous information
3. Explicit exogenous variable handling: Cross-attention mechanism
4. SOTA performance on 12 benchmarks with exogenous features

Reference: https://arxiv.org/abs/2402.19072
Code: https://github.com/thuml/TimeXer
"""

import math
from typing import Dict, Optional

import torch
import torch.nn as nn

from .base import ARBaseModel
from .registry import register


class Patching(nn.Module):
    """Converts time series into patches (subseries segments)."""

    def __init__(self, patch_len: int, stride: int):
        """Initialize patching module.

        Args:
            patch_len: Length of each patch
            stride: Stride between patches
        """
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply patching to input sequence.

        Args:
            x: Input tensor [B, T, D]

        Returns:
            Patched tensor [B, D, N_patches, patch_len]
            where N_patches = (T - patch_len) // stride + 1
        """
        B, T, D = x.shape

        # Permute to [B, D, T] for channel-wise processing
        x = x.permute(0, 2, 1)  # [B, D, T]

        # Use unfold to create patches
        # unfold(dimension, size, step) extracts sliding windows
        patches = x.unfold(dimension=2, size=self.patch_len, step=self.stride)
        # Result: [B, D, N_patches, patch_len]

        return patches

    def get_num_patches(self, seq_len: int) -> int:
        """Calculate number of patches for a given sequence length.

        Args:
            seq_len: Length of input sequence

        Returns:
            Number of patches
        """
        return (seq_len - self.patch_len) // self.stride + 1


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """Initialize positional encoding.

        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding.

        Args:
            x: Input tensor [B, N, D]

        Returns:
            x + positional encoding
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class GlobalEndogenousToken(nn.Module):
    """Global endogenous token that bridges endogenous and exogenous information.

    This learnable token aggregates information across all endogenous patches
    and serves as a bridge for cross-attention with exogenous variables.
    """

    def __init__(self, d_model: int, num_tokens: int = 1):
        """Initialize global endogenous token.

        Args:
            d_model: Model dimension
            num_tokens: Number of global tokens (typically 1)
        """
        super().__init__()
        self.num_tokens = num_tokens
        # Learnable global token
        self.global_token = nn.Parameter(torch.randn(1, num_tokens, d_model))

    def forward(self, batch_size: int) -> torch.Tensor:
        """Generate global tokens for a batch.

        Args:
            batch_size: Batch size

        Returns:
            Global tokens [B, num_tokens, d_model]
        """
        return self.global_token.expand(batch_size, -1, -1)


@register("timexer")
class TimeXerModel(ARBaseModel):
    """TimeXer: Transformer for Time Series Forecasting with Exogenous Variables.

    TimeXer explicitly handles two types of variables:
    1. Endogenous: Sensor readings we want to predict (target variables)
    2. Exogenous: External factors known in advance (flight plan, weather, etc.)

    Architecture:
    1. Endogenous variables: Processed with patch-level representation
       - Patches capture local temporal patterns
       - Self-attention models dependencies across patches
    2. Exogenous variables: Processed with variate-level representation
       - Each exogenous variable embedded separately
       - Captures relationships between different exogenous factors
    3. Global endogenous tokens: Bridge between endogenous and exogenous
       - Learnable tokens that aggregate endogenous information
       - Cross-attention with exogenous variables
    4. Fusion: Combine endogenous and exogenous information for prediction

    Perfect for aircraft sensor forecasting where:
    - Endogenous: Fuel flow, thrust, temperature (what we predict)
    - Exogenous: Flight plan, altitude schedule, weight, weather (what we know)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        exog_dim: int = 0,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        activation: str = "gelu",
        num_global_tokens: int = 1,
        pred_len: int = 1,
        **kwargs,
    ):
        """Initialize TimeXer model.

        Args:
            input_dim: Dimension of endogenous features (sensors to predict)
            output_dim: Dimension of output predictions
            exog_dim: Dimension of exogenous features (external factors)
                     If 0, model operates without exogenous variables
            patch_len: Length of each patch for endogenous variables
            stride: Stride between patches
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer encoder layers
            dim_feedforward: Feedforward dimension
            dropout: Dropout probability
            activation: Activation function ('relu', 'gelu')
            num_global_tokens: Number of global endogenous tokens
            pred_len: Prediction horizon length
            **kwargs: Additional arguments
        """
        super().__init__(input_dim, output_dim, **kwargs)

        self.exog_dim = exog_dim
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.num_global_tokens = num_global_tokens
        self.pred_len = pred_len
        self.has_exog = exog_dim > 0

        # ===== Endogenous processing (patch-level) =====
        # Patching module for endogenous variables
        self.patching = Patching(patch_len, stride)

        # Patch embedding: project patch to d_model dimension
        self.patch_embedding = nn.Linear(patch_len, d_model)

        # Positional encoding for patches
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        # Global endogenous tokens
        self.global_tokens = GlobalEndogenousToken(d_model, num_global_tokens)

        # ===== Exogenous processing (variate-level) =====
        if self.has_exog:
            # Variate embedding for exogenous variables
            # Each exogenous variable is projected to d_model
            self.exog_embedding = nn.Linear(exog_dim, d_model * exog_dim)
            self.exog_dim_actual = exog_dim

        # ===== Transformer layers =====
        # Encoder for endogenous patches
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        self.endogenous_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Cross-attention for global tokens and exogenous variables
        if self.has_exog:
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
            )

        # ===== Prediction head =====
        # Calculate total dimension after processing
        # We aggregate both the global token summary and patch summary for each
        # endogenous channel, then flatten across channels. This results in a
        # feature vector of size ``2 * d_model * input_dim`` regardless of the
        # number of global tokens (they're averaged per channel).
        head_input_dim = 2 * d_model * input_dim

        self.head = nn.Sequential(
            nn.Linear(head_input_dim, dim_feedforward),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, output_dim * pred_len),
        )

        # Initialize parameters
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None, **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: Endogenous input tensor [B, T_in, D_in]
            context: Exogenous variables [B, T_in, D_exog] or [B, D_exog]
                    If None and exog_dim > 0, will use zeros
            **kwargs: Additional arguments

        Returns:
            Dictionary with 'preds' and 'extras'
        """
        B, T_in, D_in = x.shape

        # ===== Step 1: Process endogenous variables (patch-level) =====
        # Create patches: [B, D_in, N_patches, patch_len]
        patches = self.patching(x)
        _, _, N_patches, patch_len = patches.shape

        # Reshape to process all channels in batch
        # [B, D_in, N_patches, patch_len] -> [B*D_in, N_patches, patch_len]
        patches = patches.reshape(B * D_in, N_patches, patch_len)

        # Embed patches: [B*D_in, N_patches, patch_len] -> [B*D_in, N_patches, d_model]
        patch_embeddings = self.patch_embedding(patches)

        # Add positional encoding
        patch_embeddings = self.pos_encoder(patch_embeddings)

        # Generate global tokens for each batch element
        # We need global tokens for each channel separately
        # [B*D_in, num_global_tokens, d_model]
        global_tokens = self.global_tokens(B * D_in)

        # Concatenate global tokens with patch embeddings
        # [B*D_in, num_global_tokens + N_patches, d_model]
        endogenous_input = torch.cat([global_tokens, patch_embeddings], dim=1)

        # Pass through transformer encoder
        # [B*D_in, num_global_tokens + N_patches, d_model]
        endogenous_output = self.endogenous_encoder(endogenous_input)

        # Extract global token representations (first num_global_tokens positions)
        # [B*D_in, num_global_tokens, d_model]
        global_token_output = endogenous_output[:, : self.num_global_tokens, :]

        # Extract patch representations (for aggregation)
        # [B*D_in, N_patches, d_model]
        patch_output = endogenous_output[:, self.num_global_tokens :, :]

        # Reshape back to separate batch and channels
        # [B*D_in, num_global_tokens, d_model] -> [B, D_in, num_global_tokens, d_model]
        global_token_output = global_token_output.reshape(
            B, D_in, self.num_global_tokens, self.d_model
        )

        # [B*D_in, N_patches, d_model] -> [B, D_in, N_patches, d_model]
        patch_output = patch_output.reshape(B, D_in, N_patches, self.d_model)

        # ===== Step 2: Process exogenous variables (if available) =====
        exog_info = None
        if self.has_exog and context is not None:
            # Handle context: could be [B, T, D_exog] or [B, D_exog]
            if context.dim() == 2:
                # [B, D_exog] -> replicate for time dimension
                # Use mean aggregation or last time step
                exog_features = context
            else:
                # [B, T, D_exog] -> aggregate over time (mean)
                exog_features = context.mean(dim=1)  # [B, D_exog]

            # Embed exogenous features
            # [B, D_exog] -> [B, D_exog * d_model]
            exog_embedded = self.exog_embedding(exog_features)
            # Reshape to [B, D_exog, d_model]
            exog_embedded = exog_embedded.reshape(B, self.exog_dim_actual, self.d_model)

            # Cross-attention: global tokens (query) attend to exogenous (key, value)
            # Flatten global tokens: [B, D_in, num_global_tokens, d_model]
            #   -> [B, D_in * num_global_tokens, d_model]
            global_flat = global_token_output.reshape(
                B, D_in * self.num_global_tokens, self.d_model
            )

            # Cross-attention
            # query: global tokens, key/value: exogenous embeddings
            exog_attended, attn_weights = self.cross_attention(
                query=global_flat, key=exog_embedded, value=exog_embedded
            )

            # Add residual connection
            global_flat = global_flat + exog_attended

            # Reshape back: [B, D_in * num_global_tokens, d_model]
            #   -> [B, D_in, num_global_tokens, d_model]
            global_token_output = global_flat.reshape(B, D_in, self.num_global_tokens, self.d_model)

            exog_info = {"exog_embedded": exog_embedded, "cross_attn_weights": attn_weights}

        # ===== Step 3: Aggregate and predict =====
        # Aggregate patch representations per channel (mean pooling)
        # [B, D_in, N_patches, d_model] -> [B, D_in, d_model]
        patch_aggregated = patch_output.mean(dim=2)

        # Aggregate global tokens per channel
        # [B, D_in, num_global_tokens, d_model] -> [B, D_in, d_model]
        global_aggregated = global_token_output.mean(dim=2)

        # Concatenate global and patch information
        # [B, D_in, d_model] + [B, D_in, d_model] -> [B, D_in, 2*d_model]
        # Then flatten: [B, D_in * 2 * d_model]
        combined = torch.cat([global_aggregated, patch_aggregated], dim=2)  # [B, D_in, 2*d_model]
        combined_flat = combined.reshape(B, D_in * 2 * self.d_model)

        # Prediction head
        # [B, D_in * 2 * d_model] -> [B, output_dim * pred_len]
        out = self.head(combined_flat)

        # Reshape to [B, pred_len, output_dim]
        preds = out.reshape(B, self.pred_len, self.output_dim)

        extras = {
            "endogenous_output": endogenous_output,
            "global_tokens": global_token_output,
            "patch_output": patch_output,
            "num_patches": N_patches,
        }

        if exog_info is not None:
            extras.update(exog_info)

        return {"preds": preds, "extras": extras}

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  input_dim={self.input_dim},\n"
            f"  output_dim={self.output_dim},\n"
            f"  exog_dim={self.exog_dim},\n"
            f"  patch_len={self.patch_len},\n"
            f"  stride={self.stride},\n"
            f"  d_model={self.d_model},\n"
            f"  num_global_tokens={self.num_global_tokens},\n"
            f"  pred_len={self.pred_len},\n"
            f"  num_params={self.get_num_params():,}\n"
            f")"
        )
