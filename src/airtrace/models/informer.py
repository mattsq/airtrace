"""Informer model implementation with ProbSparse attention and distilling encoder."""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .base import ARBaseModel
from .registry import register


class TokenEmbedding(nn.Module):
    """Linear token embedding for time series values."""

    def __init__(self, input_dim: int, d_model: int, dropout: float):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(x))


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class ProbSparseAttention(nn.Module):
    """ProbSparse multi-head self-attention.

    This implementation follows the core idea of Informer: select the
    most informative queries based on query magnitude while using a
    shared context for the remaining positions to reduce computation.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float, factor: int = 5):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head**-0.5
        self.factor = factor

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L_q, _ = query.shape
        L_k = key.shape[1]

        q = self.q_proj(query).view(B, L_q, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(key).view(B, L_k, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(value).view(B, L_k, self.n_heads, self.d_head).transpose(1, 2)

        # Determine number of important queries (u in the paper)
        u = max(1, min(L_q, int(self.factor * math.log(L_k + 1))))

        # Query sparsity measurement: max magnitude along head dimension
        query_importance = torch.max(torch.abs(q), dim=-1).values  # [B, H, L_q]
        topk_indices = torch.topk(query_importance, k=u, dim=2).indices  # [B, H, u]

        # Default context uses averaged values for non-selected positions
        v_mean = v.mean(dim=2, keepdim=True)
        context = v_mean.expand(-1, -1, L_q, -1).clone()  # [B, H, L_q, d_head]

        # Compute attention only for selected queries
        q_selected = q.gather(2, topk_indices.unsqueeze(-1).expand(-1, -1, -1, self.d_head))
        scores = torch.einsum("bhqd,bhkd->bhqk", q_selected, k) * self.scale
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        selected_context = torch.einsum("bhqk,bhkd->bhqd", attn, v)

        # Scatter selected contexts back to full sequence positions
        context.scatter_(
            2,
            topk_indices.unsqueeze(-1).expand_as(selected_context),
            selected_context,
        )

        # Merge heads
        output = context.transpose(1, 2).contiguous().view(B, L_q, self.n_heads * self.d_head)
        output = self.out_proj(output)

        # For interpretability we return sparse attention map on selected queries
        return output, attn


class FeedForward(nn.Module):
    """Position-wise feedforward network."""

    def __init__(self, d_model: int, ff_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class InformerEncoderLayer(nn.Module):
    """Informer encoder layer with ProbSparse attention and distillation support."""

    def __init__(self, d_model: int, n_heads: int, ff_dim: int, dropout: float, factor: int):
        super().__init__()
        self.attn = ProbSparseAttention(d_model, n_heads, dropout, factor)
        self.ff = FeedForward(d_model, ff_dim, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_out, attn_map = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x, attn_map


class InformerEncoder(nn.Module):
    """Informer encoder with optional distilling between layers."""

    def __init__(self, num_layers: int, d_model: int, n_heads: int, ff_dim: int, dropout: float, factor: int, distill: bool):
        super().__init__()
        self.layers = nn.ModuleList([
            InformerEncoderLayer(d_model, n_heads, ff_dim, dropout, factor) for _ in range(num_layers)
        ])
        self.distill = distill and num_layers > 1
        if self.distill:
            self.distill_convs = nn.ModuleList([
                nn.Sequential(
                    nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
                    nn.ELU(),
                    nn.MaxPool1d(kernel_size=2, stride=2),
                )
                for _ in range(num_layers - 1)
            ])
        else:
            self.distill_convs = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        attn_map: Optional[torch.Tensor] = None
        for idx, layer in enumerate(self.layers):
            x, attn_map = layer(x)
            if self.distill and idx < len(self.layers) - 1:
                x = self.distill_convs[idx](x.transpose(1, 2)).transpose(1, 2)
        return x, attn_map


class InformerDecoderLayer(nn.Module):
    """Informer decoder layer with self and cross attention."""

    def __init__(self, d_model: int, n_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ff = FeedForward(d_model, ff_dim, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, memory: torch.Tensor, self_mask: Optional[torch.Tensor]) -> torch.Tensor:
        attn_output, _ = self.self_attn(x, x, x, attn_mask=self_mask)
        x = self.norm1(x + attn_output)

        cross_output, cross_weights = self.cross_attn(x, memory, memory)
        x = self.norm2(x + cross_output)

        ff_out = self.ff(x)
        x = self.norm3(x + ff_out)
        return x


class InformerDecoder(nn.Module):
    """Informer decoder stacking multiple layers."""

    def __init__(self, num_layers: int, d_model: int, n_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList(
            [InformerDecoderLayer(d_model, n_heads, ff_dim, dropout) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor, memory: torch.Tensor, self_mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, self_mask)
        return x


@register("informer")
class InformerModel(ARBaseModel):
    """Informer: ProbSparse self-attention with distilling encoder-decoder.

    References:
        - Zhou et al. 2021, "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting"
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        e_layers: int = 2,
        d_layers: int = 1,
        ff_dim: int = 256,
        factor: int = 5,
        dropout: float = 0.1,
        pred_len: int = 1,
        distill: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(input_dim, output_dim, **kwargs)
        if pred_len <= 0:
            raise ValueError("pred_len must be positive")

        self.pred_len = pred_len
        self.value_embedding = TokenEmbedding(input_dim, d_model, dropout)
        self.position_encoding = PositionalEncoding(d_model, dropout=dropout)

        self.encoder = InformerEncoder(
            num_layers=e_layers,
            d_model=d_model,
            n_heads=nhead,
            ff_dim=ff_dim,
            dropout=dropout,
            factor=factor,
            distill=distill,
        )
        self.decoder = InformerDecoder(
            num_layers=d_layers,
            d_model=d_model,
            n_heads=nhead,
            ff_dim=ff_dim,
            dropout=dropout,
        )
        self.projection = nn.Linear(d_model, output_dim)

    def _generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
        return mask

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, torch.Tensor]:
        B, T_in, _ = x.shape

        enc_input = self.position_encoding(self.value_embedding(x))
        enc_output, attn_map = self.encoder(enc_input)

        # Decoder input: repeat last observed timestep as start token
        start_token = x[:, -1:, :]
        dec_input = start_token.repeat(1, self.pred_len, 1)
        dec_input = self.position_encoding(self.value_embedding(dec_input))

        dec_mask = self._generate_square_subsequent_mask(self.pred_len, x.device)
        dec_output = self.decoder(dec_input, enc_output, dec_mask)

        preds = self.projection(dec_output)

        return {
            "preds": preds,
            "extras": {
                "encoder_output": enc_output,
                "decoder_output": dec_output,
                "sparse_attention": attn_map,
            },
        }

    def __repr__(self) -> str:  # pragma: no cover - readability helper
        return (
            f"InformerModel(\n"
            f"  input_dim={self.input_dim},\n"
            f"  output_dim={self.output_dim},\n"
            f"  pred_len={self.pred_len}\n"
            f")"
        )
