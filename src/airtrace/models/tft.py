"""Temporal Fusion Transformer implementation for multi-horizon forecasting.

This module implements a simplified yet faithful version of the Temporal Fusion
Transformer (TFT) introduced in:

> Lim et al., "Temporal Fusion Transformers for Interpretable Multi-horizon
> Time Series Forecasting" (2019)

Key interpretability components are included:
- Variable Selection Networks (VSN) for dynamic feature weighting
- Gated Residual Networks (GRN) with contextual conditioning
- Interpretable multi-head attention with exposure of attention weights
- Quantile projection head for probabilistic forecasts

The model supports optional static covariates and known future inputs, enabling
multi-horizon predictions while returning interpretable artifacts in `extras`.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .base import ResidualWrapperCompatible
from .registry import register


class GatedResidualNetwork(nn.Module):
    """Gated Residual Network (GRN) building block.

    Combines a two-layer feedforward network with contextual conditioning,
    gated linear units, and layer normalization.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: Optional[int] = None,
        context_dim: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim or hidden_dim
        self.context_dim = context_dim

        self.dense_1 = nn.Linear(input_dim, hidden_dim)
        self.context_dense = (
            nn.Linear(context_dim, hidden_dim) if context_dim and context_dim > 0 else None
        )
        self.dense_2 = nn.Linear(hidden_dim, self.output_dim)

        self.dropout = nn.Dropout(dropout)
        self.elu = nn.ELU()

        self.gate_proj = nn.Linear(self.output_dim, self.output_dim * 2)
        self.glu = nn.GLU()

        self.skip = nn.Linear(input_dim, self.output_dim) if input_dim != self.output_dim else None
        self.norm = nn.LayerNorm(self.output_dim)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply GRN transformation.

        Args:
            x: Input tensor [..., input_dim]
            context: Optional context tensor [..., context_dim] or [B, context_dim]

        Returns:
            Tensor with shape [..., output_dim]
        """

        residual = x if self.skip is None else self.skip(x)

        hidden = self.dense_1(x)
        if self.context_dense is not None and context is not None:
            if context.dim() == hidden.dim() - 1:
                # [B, C] -> [B, T, C] to match sequence inputs
                context = context.unsqueeze(1).expand(-1, hidden.size(1), -1)
            hidden = hidden + self.context_dense(context)

        hidden = self.elu(hidden)
        hidden = self.dropout(self.dense_2(hidden))

        gated = self.glu(self.gate_proj(hidden))
        return self.norm(residual + gated)


class VariableSelectionNetwork(nn.Module):
    """Variable Selection Network (VSN).

    Learns soft weights over input variables and returns a weighted combination
    of per-variable embeddings alongside the normalized importance scores.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        context_dim: int = 0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_inputs = input_dim
        self.hidden_size = hidden_size
        self.context_dim = context_dim

        self.variable_projections = nn.ModuleList(
            [nn.Linear(1, hidden_size) for _ in range(self.num_inputs)]
        )

        weight_input_dim = self.num_inputs + (context_dim if context_dim else 0)
        self.weight_network = GatedResidualNetwork(
            input_dim=weight_input_dim,
            hidden_dim=hidden_size,
            output_dim=self.num_inputs,
            context_dim=None,
            dropout=dropout,
        )

    def forward(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute variable selection weights and combined representation.

        Args:
            x: Input tensor [B, T, D]
            context: Optional static context [B, C] or [B, T, C]

        Returns:
            Tuple of:
                - combined representation [B, T, hidden_size]
                - attention weights over variables [B, T, D]
        """

        if x.shape[-1] != self.num_inputs:
            raise ValueError(
                f"Expected {self.num_inputs} input features but received {x.shape[-1]}"
            )

        B, T, _ = x.shape
        embeddings = torch.stack(
            [proj(x[..., idx : idx + 1]) for idx, proj in enumerate(self.variable_projections)],
            dim=2,
        )  # [B, T, D, hidden]

        if context is not None and self.context_dim > 0:
            if context.dim() == 2:
                context_expanded = context.unsqueeze(1).expand(-1, T, -1)
            else:
                context_expanded = context
            weight_input = torch.cat([x, context_expanded], dim=-1)
        else:
            weight_input = x

        raw_weights = self.weight_network(weight_input)
        weights = torch.softmax(raw_weights, dim=-1)  # [B, T, D]

        combined = torch.sum(weights.unsqueeze(-1) * embeddings, dim=2)
        return combined, weights


class InterpretableMultiHeadAttention(nn.Module):
    """Multi-head attention returning attention weights for interpretability."""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads, dropout=dropout, batch_first=True
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run attention and return both outputs and weights."""

        output, attn_weights = self.attn(query, key, value, attn_mask=attn_mask)
        return output, attn_weights


@register("tft")
class TemporalFusionTransformer(ResidualWrapperCompatible):
    """Temporal Fusion Transformer for interpretable multi-horizon forecasting."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_size: int = 128,
        lstm_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        quantiles: Optional[List[float]] = None,
        static_input_dim: int = 0,
        known_future_dim: int = 0,
        pred_len: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(input_dim, output_dim, **kwargs)
        if pred_len <= 0:
            raise ValueError("pred_len must be positive")

        self.hidden_size = hidden_size
        self.pred_len = pred_len
        self.quantiles = quantiles or []
        self.static_input_dim = static_input_dim
        self.known_future_dim = known_future_dim

        self.encoder_vsn = VariableSelectionNetwork(
            input_dim=input_dim,
            hidden_size=hidden_size,
            context_dim=static_input_dim,
            dropout=dropout,
        )

        decoder_input_dim = known_future_dim if known_future_dim > 0 else input_dim
        self.decoder_vsn = VariableSelectionNetwork(
            input_dim=decoder_input_dim,
            hidden_size=hidden_size,
            context_dim=static_input_dim,
            dropout=dropout,
        )

        self.encoder_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
        )
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
        )

        self.temporal_attn = InterpretableMultiHeadAttention(hidden_size, num_heads, dropout)

        self.post_attn_grn = GatedResidualNetwork(
            input_dim=hidden_size * 2,
            hidden_dim=hidden_size,
            output_dim=hidden_size,
            context_dim=static_input_dim,
            dropout=dropout,
        )
        self.positionwise_grn = GatedResidualNetwork(
            input_dim=hidden_size,
            hidden_dim=hidden_size,
            output_dim=hidden_size,
            context_dim=static_input_dim,
            dropout=dropout,
        )

        if self.quantiles:
            self.quantile_proj = nn.Linear(hidden_size, output_dim * len(self.quantiles))
        else:
            self.output_proj = nn.Linear(hidden_size, output_dim)

        self.dropout = nn.Dropout(dropout)

    def encode(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        known_future: Optional[torch.Tensor] = None,
        static_covariates: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Encode inputs into fused temporal features.

        Returns pooled fusion tokens alongside interpretability artifacts so the
        residual wrapper can reuse the same representation without re-running
        the variable selection and attention stacks.
        """

        del kwargs
        static_context = static_covariates
        if static_context is None and context is not None and context.dim() == 2:
            static_context = context

        enc_features, enc_importance = self.encoder_vsn(x, static_context)
        enc_features = self.dropout(enc_features)
        enc_output, (h_n, c_n) = self.encoder_lstm(enc_features)

        decoder_input_dim = self.known_future_dim if self.known_future_dim > 0 else self.input_dim
        if known_future is None:
            device = x.device
            known_future = torch.zeros(
                x.size(0), self.pred_len, decoder_input_dim, device=device, dtype=x.dtype
            )
        elif known_future.shape[1] != self.pred_len:
            raise ValueError(
                f"known_future sequence length {known_future.shape[1]} "
                f"does not match pred_len {self.pred_len}"
            )

        dec_features, dec_importance = self.decoder_vsn(known_future, static_context)
        dec_features = self.dropout(dec_features)
        dec_output, _ = self.decoder_lstm(dec_features, (h_n, c_n))

        attn_output, attn_weights = self.temporal_attn(dec_output, enc_output, enc_output)
        fusion = torch.cat([dec_output, attn_output], dim=-1)
        fusion = self.post_attn_grn(fusion, static_context)
        fusion = self.positionwise_grn(fusion, static_context)
        fusion = self.dropout(fusion)

        extras: Dict[str, torch.Tensor] = {
            "encoder_variable_importance": enc_importance.detach(),
            "decoder_variable_importance": dec_importance.detach(),
            "attention_weights": attn_weights.detach(),
        }

        self._cached_quantile_forecast: Optional[torch.Tensor] = None
        return fusion, extras

    def decode(self, latent: torch.Tensor, pred_len: int) -> torch.Tensor:
        if pred_len != self.pred_len:
            raise ValueError(
                f"TemporalFusionTransformer only supports pred_len={self.pred_len}; got {pred_len}"
            )

        if self.quantiles:
            quantile_values = self.quantile_proj(latent)
            quantile_values = quantile_values.view(
                latent.size(0), pred_len, self.output_dim, len(self.quantiles)
            )
            self._cached_quantile_forecast = quantile_values

            median_idx = self.quantiles.index(0.5) if 0.5 in self.quantiles else len(self.quantiles) // 2
            return quantile_values[..., median_idx]

        self._cached_quantile_forecast = None
        return self.output_proj(latent)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        known_future: Optional[torch.Tensor] = None,
        static_covariates: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of the TFT using reusable encode/decode hooks."""

        latent, extras = self.encode(
            x,
            context=context,
            known_future=known_future,
            static_covariates=static_covariates,
            **kwargs,
        )
        preds = self.decode(latent, self.pred_len)

        if self._cached_quantile_forecast is not None:
            extras["quantile_forecast"] = self._cached_quantile_forecast

        return {"preds": preds, "extras": extras}

    def __repr__(self) -> str:
        return (
            f"TemporalFusionTransformer(\n"
            f"  input_dim={self.input_dim},\n"
            f"  output_dim={self.output_dim},\n"
            f"  hidden_size={self.hidden_size},\n"
            f"  pred_len={self.pred_len},\n"
            f"  quantiles={self.quantiles}\n"
            f")"
        )
