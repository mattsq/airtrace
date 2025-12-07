"""iTransformer: Inverted Transformers Are Effective for Time Series Forecasting.

Implementation of iTransformer from:
"iTransformer: Inverted Transformers Are Effective for Time Series Forecasting"
(ICLR 2024 Spotlight)

Key innovation:
Instead of embedding time points as tokens (traditional approach), iTransformer
treats each variate (sensor) as a token and embeds its entire time series.
This allows:
1. Attention mechanism to capture multivariate correlations
2. Feed-forward network to learn variate-specific temporal patterns
3. Better utilization of multivariate dependencies in sensor data

Reference: https://arxiv.org/abs/2310.06625
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.parameter import UninitializedParameter

from .base import ResidualWrapperCompatible
from .registry import register


class VariateEmbedding(nn.Module):
    """Embeds time series of each variate into a fixed-dimensional vector.

    For each variate (sensor), the entire input time series is embedded
    into a d_model dimensional vector.

    Uses LazyLinear to adapt to any input sequence length.
    """

    def __init__(self, d_model: int):
        """Initialize variate embedding.

        Args:
            d_model: Embedding dimension
        """
        super().__init__()
        self.d_model = d_model

        # Linear projection from time series to embedding
        # LazyLinear adapts to any input sequence length
        self.projection = nn.LazyLinear(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed each variate's time series.

        Args:
            x: Input tensor [B, T, D]

        Returns:
            Embedded tensor [B, D, d_model]
            where each of D variates is represented by a d_model vector
        """
        # Permute to [B, D, T] - each variate's time series
        x = x.permute(0, 2, 1)  # [B, D, T]

        # Project each time series to d_model dimension
        # [B, D, T] -> [B, D, d_model]
        x = self.projection(x)

        return x


class LearnablePositionalEncoding(nn.Module):
    """Learnable positional encoding for variates.

    Unlike sinusoidal encoding, this learns position embeddings for each variate.
    Since we're encoding variates (not time points), learnable is preferred.
    """

    def __init__(self, num_variates: int, d_model: int, dropout: float = 0.1):
        """Initialize learnable positional encoding.

        Args:
            num_variates: Number of variates (sensors)
            d_model: Model dimension
            dropout: Dropout probability
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Learnable position embeddings for each variate
        self.position_embeddings = nn.Parameter(
            torch.randn(1, num_variates, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding.

        Args:
            x: Input tensor [B, D, d_model]

        Returns:
            x + positional encoding
        """
        x = x + self.position_embeddings
        return self.dropout(x)


@register("itransformer")
class iTransformerModel(ResidualWrapperCompatible):
    """iTransformer: Inverted Transformer for Time Series Forecasting.

    Architecture inverts traditional transformer design:
    - Variates (sensors) become tokens instead of time points
    - Attention captures cross-variate correlations
    - Feed-forward networks learn temporal patterns per variate

    This is particularly effective for multivariate sensor data where
    physical relationships between sensors are important (e.g., aircraft
    sensors where fuel flow, thrust, Mach number are correlated).

    Architecture flow:
    1. Input: [B, T_in, D] time series with D sensors
    2. Variate Embedding: Each sensor's time series -> d_model vector
       Result: [B, D, d_model]
    3. Positional Encoding: Add variate position information
    4. Transformer Encoder:
       - Attention captures inter-sensor correlations
       - FFN learns sensor-specific temporal representations
    5. Projection Head: Map to output dimension and horizon
       Result: [B, T_out, D_out]
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        pred_len: int = 1,
        horizon: Optional[int] = None,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_norm: bool = True,
        **kwargs
    ):
        """Initialize iTransformer model.

        Args:
            input_dim: Dimension of input features (number of sensors)
            output_dim: Dimension of output predictions
            pred_len: Prediction window length (for data windowing)
            horizon: Actual prediction horizon (defaults to pred_len if not provided)
            d_model: Model dimension (embedding size)
            nhead: Number of attention heads
            num_layers: Number of transformer encoder layers
            dim_feedforward: Feedforward network dimension
            dropout: Dropout probability
            activation: Activation function ('relu', 'gelu')
            use_norm: Whether to use layer normalization in transformer
            **kwargs: Additional arguments
        """
        super().__init__(input_dim, output_dim, **kwargs)

        self.pred_len = pred_len
        self.horizon = horizon if horizon is not None else pred_len
        self.d_model = d_model
        self.num_layers = num_layers

        # Variate embedding: project each sensor's time series to d_model
        # Uses LazyLinear to adapt to any input sequence length
        self.variate_embedding = VariateEmbedding(d_model)

        # Positional encoding for variates
        self.pos_encoder = LearnablePositionalEncoding(
            num_variates=input_dim,
            d_model=d_model,
            dropout=dropout
        )

        # Transformer encoder
        # Attention operates on variates (sensors), capturing correlations
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=use_norm
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model) if use_norm else None
        )

        # Projection head to output
        # Maps from (d_model, input_dim) -> (horizon, output_dim)
        # Two options:
        # 1. If output_dim == input_dim: project each variate to horizon
        # 2. If output_dim != input_dim: need cross-variate projection

        if output_dim == input_dim:
            # Simple case: each input sensor predicts itself
            # Project each variate's d_model representation to horizon values
            self.projection = nn.Linear(d_model, self.horizon)
        else:
            # Complex case: need to map input variates to output variates
            # First flatten, then project to output
            self.projection = nn.Sequential(
                nn.Flatten(start_dim=1),  # [B, D, d_model] -> [B, D*d_model]
                nn.Linear(input_dim * d_model, output_dim * self.horizon)
            )

        self.output_dim_equals_input = (output_dim == input_dim)

        # Initialize parameters
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier initialization.

        Skips UninitializedParameter objects from LazyLinear which are
        materialized on first forward pass.
        """
        for name, p in self.named_parameters():
            # Skip uninitialized parameters (from LazyLinear)
            if isinstance(p, UninitializedParameter):
                continue
            # Skip position embeddings and 1D parameters (biases)
            if p.dim() > 1 and 'position_embeddings' not in name:
                nn.init.xavier_uniform_(p)

    def encode(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        del context, kwargs

        # [B, T_in, D_in] -> [B, D_in, d_model]
        variate_tokens = self.variate_embedding(x)
        variate_tokens = self.pos_encoder(variate_tokens)
        encoder_output = self.transformer_encoder(variate_tokens)

        extras = {
            "encoder_output": encoder_output,
            "variate_tokens": variate_tokens,
        }
        return encoder_output, extras

    def decode(self, latent: torch.Tensor, pred_len: int) -> torch.Tensor:
        if pred_len != self.horizon:
            raise ValueError(
                f"iTransformerModel only supports pred_len={self.horizon}; got {pred_len}"
            )

        if self.output_dim_equals_input:
            out = self.projection(latent)
            return out.permute(0, 2, 1)

        out = self.projection(latent)
        return out.reshape(latent.size(0), self.horizon, self.output_dim)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass using reusable encode/decode hooks."""

        latent, extras = self.encode(x, context=context, **kwargs)
        preds = self.decode(latent, self.horizon)
        return {"preds": preds, "extras": extras}

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
