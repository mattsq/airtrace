from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .base import ResidualWrapperCompatible
from .registry import register


class SimpleSelectiveScan(nn.Module):
    """Lightweight selective scan used by S-Mamba blocks.

    This variant keeps a single state vector per time step and applies a
    learnable decay to blend the previous state with the current input. It can
    optionally run bidirectionally to fuse forward and backward context.
    """

    def __init__(
        self,
        state_dim: int,
        bidirectional: bool,
        dropout: float,
        decay_init: float = 0.0,
    ) -> None:
        super().__init__()
        if state_dim <= 0:
            raise ValueError("state_dim must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.state_dim = state_dim
        self.bidirectional = bidirectional
        self.dropout = nn.Dropout(dropout)
        self.decay = nn.Parameter(torch.full((state_dim,), decay_init))

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the scan over the input sequence.

        Args:
            inputs: Tensor of shape [B, T, state_dim]

        Returns:
            Tuple containing:
                - All intermediate states [B, T, state_dim]
                - Final forward state [B, state_dim]
        """

        forward_states, final_state = self._scan_one_direction(inputs)
        if not self.bidirectional:
            return self.dropout(forward_states), final_state

        backward_states, _ = self._scan_one_direction(torch.flip(inputs, dims=[1]))
        backward_states = torch.flip(backward_states, dims=[1])
        fused_states = 0.5 * (forward_states + backward_states)
        return self.dropout(fused_states), final_state

    def _scan_one_direction(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = inputs.shape
        states: List[torch.Tensor] = []
        state = torch.zeros(B, self.state_dim, device=inputs.device, dtype=inputs.dtype)
        decay = torch.sigmoid(self.decay).view(1, -1)

        for step in inputs.unbind(dim=1):
            state = decay * state + step
            states.append(state.unsqueeze(1))

        all_states = torch.cat(states, dim=1)
        return all_states, state


class SMambaBlock(nn.Module):
    """Single S-Mamba block with depthwise convolution and selective scan."""

    def __init__(
        self,
        embed_dim: int,
        state_dim: int,
        ff_expansion: int,
        dropout: float,
        bidirectional: bool,
        conv_kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if conv_kernel_size % 2 == 0:
            raise ValueError("conv_kernel_size must be odd to preserve sequence length")
        if ff_expansion <= 0:
            raise ValueError("ff_expansion must be positive")

        self.pre_norm = nn.LayerNorm(embed_dim)
        self.depthwise_conv = nn.Conv1d(
            embed_dim,
            embed_dim,
            kernel_size=conv_kernel_size,
            padding=conv_kernel_size // 2,
            groups=embed_dim,
        )
        self.state_in = nn.Linear(embed_dim, state_dim)
        self.state_out = nn.Linear(state_dim, embed_dim)
        self.gate_proj = nn.Linear(embed_dim, embed_dim)
        self.scan = SimpleSelectiveScan(
            state_dim=state_dim,
            bidirectional=bidirectional,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.ff_norm = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * ff_expansion),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * ff_expansion, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = x
        y = self.pre_norm(x)
        conv_out = self.depthwise_conv(y.transpose(1, 2)).transpose(1, 2)
        scan_inputs = self.state_in(y + conv_out)
        scan_outputs, final_state = self.scan(scan_inputs)
        gates = torch.sigmoid(self.gate_proj(y))
        mixed = self.state_out(scan_outputs) * gates
        x = residual + self.dropout(mixed)
        x = x + self.dropout(self.ff(self.ff_norm(x)))
        return x, final_state


@register("s_mamba")
class SMambaModel(ResidualWrapperCompatible):
    """Simple Mamba (S-Mamba) model for time series forecasting.

    This implementation follows the "Is Mamba Effective for Time Series"
    architecture: per-variate linear tokenization, bidirectional selective scan
    blocks, and feed-forward mixers for temporal refinement.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        seq_len: int = 60,
        pred_len: int = 1,
        embed_dim: int = 64,
        state_dim: int = 64,
        num_layers: int = 3,
        ff_expansion: int = 2,
        dropout: float = 0.1,
        bidirectional_scan: bool = True,
        conv_kernel_size: int = 3,
        **kwargs,
    ) -> None:
        super().__init__(input_dim, output_dim, **kwargs)
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if pred_len <= 0:
            raise ValueError("pred_len must be positive")
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive")
        if state_dim <= 0:
            raise ValueError("state_dim must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.embed_dim = embed_dim

        self.token_proj = nn.Linear(input_dim, embed_dim)
        self.blocks = nn.ModuleList(
            [
                SMambaBlock(
                    embed_dim=embed_dim,
                    state_dim=state_dim,
                    ff_expansion=ff_expansion,
                    dropout=dropout,
                    bidirectional=bidirectional_scan,
                    conv_kernel_size=conv_kernel_size,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(embed_dim)
        self.temporal_projection = nn.Linear(seq_len, pred_len)
        self.output_projection = nn.Linear(embed_dim, output_dim)

    def encode(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        del context
        if x.size(1) != self.seq_len:
            raise ValueError(
                f"Expected input sequence length {self.seq_len}, got {x.size(1)}"
            )

        out = self.token_proj(x)
        selective_states: List[torch.Tensor] = []
        for block in self.blocks:
            out, state = block(out)
            selective_states.append(state)

        representation = self.final_norm(out)
        extras: Dict[str, torch.Tensor] = {
            "selective_states": selective_states,
            "token_embeddings": out.detach().clone(),
        }
        return representation, extras

    def decode(self, latent: torch.Tensor, pred_len: int) -> torch.Tensor:
        if pred_len != self.pred_len:
            raise ValueError(
                f"S-Mamba pred_len must match configured pred_len={self.pred_len}, got {pred_len}"
            )

        out = latent.transpose(1, 2)
        out = self.temporal_projection(out)
        out = out.transpose(1, 2)
        out = self.output_projection(out)
        return out

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        pred_len = int(kwargs.get("pred_len", self.pred_len))
        latent, extras = self.encode(x, context=context)
        preds = self.decode(latent, pred_len)
        return {"preds": preds, "extras": extras}

