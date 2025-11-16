"""Mamba-2 style selective state-space model."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .base import ARBaseModel
from .registry import register


class ConvInputAdapter(nn.Module):
    """Project raw inputs into the model embedding space."""

    def __init__(self, input_dim: int, embed_dim: int, kernel_size: int) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd for same-padding")
        self.linear = nn.Linear(input_dim, embed_dim)
        padding = kernel_size // 2
        self.depthwise_conv = nn.Conv1d(
            embed_dim,
            embed_dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=embed_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return embedded tokens."""

        tokens = self.linear(x)
        conv = self.depthwise_conv(tokens.transpose(1, 2)).transpose(1, 2)
        return tokens + conv


class Mamba2Block(nn.Module):
    """Simplified Mamba-2 style selective scan block."""

    def __init__(
        self,
        embed_dim: int,
        state_dim: int,
        conv_kernel_size: int,
        chunk_size: int,
        gating_temperature: float,
        dropout: float,
        ff_expansion: int,
        bidirectional: bool,
    ) -> None:
        super().__init__()
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if gating_temperature <= 0:
            raise ValueError("gating_temperature must be positive")
        if conv_kernel_size % 2 == 0:
            raise ValueError("conv_kernel_size must be odd for same-padding")

        self.chunk_size = chunk_size
        self.gating_temperature = gating_temperature
        self.bidirectional = bidirectional

        self.in_norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, state_dim * 2)
        padding = conv_kernel_size // 2
        self.state_conv = nn.Conv1d(
            state_dim,
            state_dim,
            kernel_size=conv_kernel_size,
            padding=padding,
            groups=state_dim,
        )
        self.out_proj = nn.Linear(state_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.ff_norm = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * ff_expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * ff_expansion, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply selective scan followed by gated feed-forward mixing."""

        residual = x
        y = self.in_norm(x)
        state, gate = self.proj(y).chunk(2, dim=-1)
        gate = torch.sigmoid(gate / self.gating_temperature)
        mixed = self._selective_scan(state, gate)
        mixed = self.state_conv(mixed.transpose(1, 2)).transpose(1, 2)
        mixed = self.out_proj(mixed)
        x = residual + self.dropout(mixed)
        x = x + self.ff(self.ff_norm(x))
        return x, mixed

    def _selective_scan(self, values: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """Chunked cumulative scan resembling the Mamba-2 kernel."""

        B, T, D = values.shape
        chunk = min(self.chunk_size, T)
        padded = (chunk - (T % chunk)) % chunk

        if padded > 0:
            pad_values = torch.zeros(B, padded, D, device=values.device, dtype=values.dtype)
            pad_gate = torch.zeros(B, padded, D, device=gate.device, dtype=gate.dtype)
            values = torch.cat([values, pad_values], dim=1)
            gate = torch.cat([gate, pad_gate], dim=1)
        gated = values * gate
        B, T_pad, _ = gated.shape
        num_chunks = T_pad // chunk
        reshaped = gated.view(B, num_chunks, chunk, D)
        cumsums = torch.cumsum(reshaped, dim=2)
        chunk_totals = cumsums[:, :, -1:, :]
        prefix = torch.cumsum(chunk_totals, dim=1) - chunk_totals
        prefix = prefix.expand(-1, -1, chunk, -1)
        forward = cumsums + prefix
        forward = forward.view(B, T_pad, D)[:, :T, :]

        if not self.bidirectional:
            return forward

        rev_values = torch.flip(gated, dims=[1])
        rev = rev_values.view(B, num_chunks, chunk, D)
        rev_cumsum = torch.cumsum(rev, dim=2)
        rev_totals = rev_cumsum[:, :, -1:, :]
        rev_prefix = torch.cumsum(rev_totals, dim=1) - rev_totals
        rev_prefix = rev_prefix.expand(-1, -1, chunk, -1)
        backward = rev_cumsum + rev_prefix
        backward = backward.view(B, T_pad, D)
        backward = torch.flip(backward, dims=[1])[:, T_pad - T :, :]

        return 0.5 * (forward + backward)


@register("mamba2_ar")
class Mamba2ARModel(ARBaseModel):
    """Token-free Mamba-2 inspired selective state-space model."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        pred_len: int = 1,
        embed_dim: int = 256,
        state_dim: int = 256,
        num_layers: int = 8,
        conv_kernel_size: int = 5,
        chunk_size: int = 64,
        gating_temperature: float = 1.0,
        dropout: float = 0.1,
        ff_expansion: int = 2,
        bidirectional: bool = True,
        adapter_kernel_size: int = 5,
        **kwargs,
    ) -> None:
        super().__init__(input_dim, output_dim, **kwargs)
        if pred_len <= 0:
            raise ValueError("pred_len must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if state_dim <= 0:
            raise ValueError("state_dim must be positive")

        self.pred_len = pred_len
        self.embed_dim = embed_dim
        self.output_dim = output_dim

        self.input_adapter = ConvInputAdapter(
            input_dim=input_dim,
            embed_dim=embed_dim,
            kernel_size=adapter_kernel_size,
        )

        self.layers = nn.ModuleList(
            [
                Mamba2Block(
                    embed_dim=embed_dim,
                    state_dim=state_dim,
                    conv_kernel_size=conv_kernel_size,
                    chunk_size=chunk_size,
                    gating_temperature=gating_temperature,
                    dropout=dropout,
                    ff_expansion=ff_expansion,
                    bidirectional=bidirectional,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(embed_dim)
        self.readout = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, pred_len * output_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Compute predictions for the provided sequence."""

        del context  # Unused but part of the interface
        tokens = self.input_adapter(x)
        block_states: List[torch.Tensor] = []
        for layer in self.layers:
            tokens, state = layer(tokens)
            block_states.append(state)

        encoded = self.final_norm(tokens)
        pooled = encoded.mean(dim=1)
        last = encoded[:, -1, :]
        summary = torch.cat([last, pooled], dim=-1)
        preds = self.readout(summary)
        preds = preds.view(x.size(0), self.pred_len, self.output_dim)

        return {
            "preds": preds,
            "extras": {
                "encoded": encoded,
                "block_states": block_states,
            },
        }
