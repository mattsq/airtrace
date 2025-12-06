"""Transformer-based autoregressive model."""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .base import ResidualWrapperCompatible
from .registry import register


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
            x: Input tensor [B, T, D]

        Returns:
            x + positional encoding
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


@register("transformer")
class TransformerModel(ResidualWrapperCompatible):
    """Transformer-based autoregressive model.

    Uses causal self-attention for sequence modeling.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        activation: str = "gelu",
        causal: bool = True,
        **kwargs
    ):
        """Initialize transformer model.

        Args:
            input_dim: Dimension of input features
            output_dim: Dimension of output predictions
            d_model: Model dimension
            nhead: Number of attention heads
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            dim_feedforward: Feedforward dimension
            dropout: Dropout probability
            activation: Activation function ('relu', 'gelu')
            causal: Whether to use causal masking
            **kwargs: Additional arguments
        """
        super().__init__(input_dim, output_dim, **kwargs)

        self.d_model = d_model
        self.causal = causal

        # Input embedding
        self.input_projection = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        # Transformer encoder
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
            num_layers=num_encoder_layers
        )

        # Output projection
        self.fc_out = nn.Linear(d_model, output_dim)

        # Initialize parameters
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate causal mask for self-attention.

        Args:
            sz: Sequence length

        Returns:
            Mask tensor [sz, sz]
        """
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask

    def encode(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        del context
        _, seq_len, _ = x.shape

        x_proj = self.input_projection(x)
        x_embed = self.pos_encoder(x_proj)
        mask = (
            self._generate_square_subsequent_mask(seq_len).to(x.device)
            if self.causal
            else None
        )
        encoder_output = self.transformer_encoder(x_embed, mask=mask)
        final_hidden = encoder_output[:, -1, :]
        extras: Dict[str, torch.Tensor] = {
            "encoder_output": encoder_output,
            "hidden": final_hidden,
        }
        return final_hidden, extras

    def decode(self, latent: torch.Tensor, pred_len: int) -> torch.Tensor:
        preds = self.fc_out(latent).unsqueeze(1)
        if pred_len != 1:
            preds = preds.expand(-1, pred_len, -1)
        return preds

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor [B, T_in, D_in]
            context: Optional context tensor
            **kwargs: Additional arguments

        Returns:
            Dictionary with 'preds' and 'extras'
        """
        pred_len = int(kwargs.get("pred_len", 1))
        latent, extras = self.encode(x, context=context)
        preds = self.decode(latent, pred_len)
        extras["representation"] = latent
        return {"preds": preds, "extras": extras}
