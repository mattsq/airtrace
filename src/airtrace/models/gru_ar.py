"""GRU-based autoregressive model."""

from typing import Dict, Optional

import torch
import torch.nn as nn

from .base import ARBaseModel
from .registry import register


@register("gru_ar")
class GRUARModel(ARBaseModel):
    """GRU-based autoregressive model.

    Uses a GRU encoder to process input sequences and a linear
    decoder to make predictions.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
        use_attention: bool = False,
        **kwargs
    ):
        """Initialize GRU AR model.

        Args:
            input_dim: Dimension of input features
            output_dim: Dimension of output predictions
            hidden_size: Hidden state dimension
            num_layers: Number of GRU layers
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional GRU
            use_attention: Whether to use attention mechanism
            **kwargs: Additional arguments
        """
        super().__init__(input_dim, output_dim, **kwargs)

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention

        # GRU encoder
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True
        )

        # Calculate encoder output dimension
        encoder_output_dim = hidden_size * (2 if bidirectional else 1)

        # Optional attention
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=encoder_output_dim,
                num_heads=4,
                dropout=dropout,
                batch_first=True
            )
        else:
            self.attention = None

        # Output projection
        self.fc_out = nn.Linear(encoder_output_dim, output_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

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
        B, T_in, D_in = x.shape

        # Encode input sequence
        encoder_output, hidden = self.gru(x)  # [B, T_in, H*num_directions]

        # Optional attention over encoder outputs
        if self.attention is not None:
            # Self-attention over encoder outputs
            attn_output, attn_weights = self.attention(
                encoder_output, encoder_output, encoder_output
            )
            # Use attended representation
            representation = attn_output[:, -1, :]  # [B, H*num_directions]
        else:
            # Use final hidden state
            if self.bidirectional:
                # Concatenate forward and backward final states
                hidden = hidden.view(self.num_layers, 2, B, self.hidden_size)
                hidden = torch.cat([hidden[-1, 0], hidden[-1, 1]], dim=1)  # [B, H*2]
            else:
                hidden = hidden[-1]  # [B, H]
            representation = hidden

        # Apply dropout
        representation = self.dropout(representation)

        # Predict output
        preds = self.fc_out(representation)  # [B, D_out]

        # Reshape to [B, 1, D_out] for consistency
        preds = preds.unsqueeze(1)

        extras = {
            "encoder_output": encoder_output,
            "hidden": hidden
        }

        if self.attention is not None:
            extras["attention_weights"] = attn_weights

        return {
            "preds": preds,
            "extras": extras
        }
