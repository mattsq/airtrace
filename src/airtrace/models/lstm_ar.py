"""LSTM-based autoregressive model."""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .base import ResidualWrapperCompatible
from .registry import register


@register("lstm_ar")
class LSTMARModel(ResidualWrapperCompatible):
    """LSTM-based autoregressive model.

    Uses an LSTM encoder to process input sequences and a linear
    decoder to make predictions.

    LSTMs can capture longer-term dependencies than GRUs due to their
    cell state mechanism, though they have more parameters and are
    slower to train.
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
        """Initialize LSTM AR model.

        Args:
            input_dim: Dimension of input features
            output_dim: Dimension of output predictions
            hidden_size: Hidden state dimension
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
            use_attention: Whether to use attention mechanism
            **kwargs: Additional arguments
        """
        super().__init__(input_dim, output_dim, **kwargs)

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention

        # LSTM encoder
        self.lstm = nn.LSTM(
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

    def encode(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        del context
        encoder_output, (hidden, cell) = self.lstm(x)

        if self.attention is not None:
            attn_output, attn_weights = self.attention(
                encoder_output, encoder_output, encoder_output
            )
            representation = attn_output[:, -1, :]
            extras: Dict[str, torch.Tensor] = {
                "encoder_output": encoder_output,
                "hidden": hidden,
                "cell": cell,
                "attention_weights": attn_weights,
            }
        else:
            if self.bidirectional:
                hidden = hidden.view(self.num_layers, 2, x.size(0), self.hidden_size)
                hidden = torch.cat([hidden[-1, 0], hidden[-1, 1]], dim=1)
            else:
                hidden = hidden[-1]
            representation = hidden
            extras = {"encoder_output": encoder_output, "hidden": hidden, "cell": cell}

        representation = self.dropout(representation)
        return representation, extras

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
        pred_len = int(kwargs.get("pred_len", 1))
        representation, extras = self.encode(x, context=context)
        preds = self.decode(representation, pred_len)
        extras["representation"] = representation
        return {"preds": preds, "extras": extras}
