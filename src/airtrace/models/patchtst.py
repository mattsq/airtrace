"""PatchTST: A Time Series is Worth 64 Words.

Implementation of PatchTST (Patch Time Series Transformer) from:
"A Time Series is Worth 64 Words: Long-term Forecasting with Transformers"
(ICLR 2023)

Key innovations:
1. Patching: Segments time series into subseries-level patches
2. Channel-independence: Each channel processed independently with shared weights
3. Efficient: Quadratic reduction in attention complexity
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .base import ResidualWrapperCompatible
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
    """Positional encoding for transformer."""

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
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding.

        Args:
            x: Input tensor [B, N, D]

        Returns:
            x + positional encoding
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


@register("patchtst")
class PatchTSTModel(ResidualWrapperCompatible):
    """PatchTST: Patch Time Series Transformer.

    Channel-independent approach: Each channel (sensor) is processed
    independently with shared transformer weights, then outputs are
    aggregated.

    Architecture:
    1. Input: [B, T, D] time series
    2. Patching: Convert to [B, D, N_patches, patch_len]
    3. For each channel:
       - Project patches to d_model dimension
       - Add positional encoding
       - Pass through Transformer encoder
       - Aggregate patch representations
    4. Final projection to output dimension
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        activation: str = "gelu",
        pred_len: int = 1,
        **kwargs
    ):
        """Initialize PatchTST model.

        Args:
            input_dim: Dimension of input features (number of sensors)
            output_dim: Dimension of output predictions
            patch_len: Length of each patch
            stride: Stride between patches
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer encoder layers
            dim_feedforward: Feedforward dimension
            dropout: Dropout probability
            activation: Activation function ('relu', 'gelu')
            **kwargs: Additional arguments
        """
        super().__init__(input_dim, output_dim, **kwargs)

        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.pred_len = pred_len

        # Patching module
        self.patching = Patching(patch_len, stride)

        # Patch embedding: project patch to d_model dimension
        # Each patch is a vector of length patch_len
        self.patch_embedding = nn.Linear(patch_len, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        # Transformer encoder (shared across all channels)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Head for prediction
        # Since we process each channel independently, we need to aggregate
        # We'll use the final patch representation from each channel
        # Then project from d_model * input_dim to output_dim
        self.flatten = nn.Flatten(start_dim=1)
        self.head = nn.Linear(d_model * input_dim, output_dim)

        # Initialize parameters
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Encode input windows into flattened patch representations."""

        B, _, D_in = x.shape

        # Step 1: Create patches
        patches = self.patching(x)
        _, _, num_patches, patch_len = patches.shape

        # Step 2: Process each channel independently
        patches = patches.reshape(B * D_in, num_patches, patch_len)

        # Step 3: Embed patches and add positional encoding
        patch_embeddings = self.patch_embedding(patches)
        patch_embeddings = self.pos_encoder(patch_embeddings)

        # Step 4: Pass through transformer encoder
        encoder_output = self.transformer_encoder(patch_embeddings)

        # Step 5: Aggregate patch representations
        channel_representations = encoder_output[:, -1, :]
        channel_representations = channel_representations.reshape(B, D_in, self.d_model)

        # Step 6: Flatten channel representations for decoding
        flattened = channel_representations.reshape(B, D_in * self.d_model)

        extras = {
            "encoder_output": encoder_output,
            "channel_representations": channel_representations,
            "num_patches": num_patches,
            "patch_len": self.patch_len,
        }

        return flattened, extras

    def decode(self, latent: torch.Tensor, pred_len: int) -> torch.Tensor:
        """Decode predictions from flattened patch features."""

        preds = self.head(latent)
        preds = preds.unsqueeze(1).repeat(1, pred_len, 1)
        return preds

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass using shared encode/decode hooks."""

        latent, extras = self.encode(x, context=context)
        preds = self.decode(latent, pred_len=self.pred_len)

        return {"preds": preds, "extras": extras}

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  input_dim={self.input_dim},\n"
            f"  output_dim={self.output_dim},\n"
            f"  patch_len={self.patch_len},\n"
            f"  stride={self.stride},\n"
            f"  d_model={self.d_model},\n"
            f"  num_params={self.get_num_params():,}\n"
            f")"
        )
