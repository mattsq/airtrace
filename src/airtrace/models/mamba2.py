"""Temporal Mamba-2 inspired selective state-space model."""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .base import ARBaseModel
from .chronos_bolt import LoRALinear
from .registry import register

LOGGER = logging.getLogger(__name__)


class ChunkedSelectiveScan(nn.Module):
    """Simplified chunked selective state-space scan with bidirectional option."""

    def __init__(
        self,
        state_dim: int,
        chunk_length: int,
        bidirectional: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        if chunk_length <= 0:
            raise ValueError("chunk_length must be positive")
        self.state_dim = state_dim
        self.chunk_length = chunk_length
        self.bidirectional = bidirectional
        self.dropout = nn.Dropout(dropout)
        # Diagonal decay term parameterised to stay in (0, 1)
        self.decay = nn.Parameter(torch.zeros(state_dim))

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run chunked selective scan.

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
        bidir_states = 0.5 * (forward_states + backward_states)
        return self.dropout(bidir_states), final_state

    def _scan_one_direction(
        self, inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, S = inputs.shape
        states: List[torch.Tensor] = []
        state = torch.zeros(B, S, device=inputs.device, dtype=inputs.dtype)
        decay = torch.sigmoid(self.decay).view(1, -1)
        for chunk in torch.split(inputs, self.chunk_length, dim=1):
            chunk_states: List[torch.Tensor] = []
            for step in chunk.unbind(dim=1):
                state = decay * state + step
                chunk_states.append(state.unsqueeze(1))
            states.append(torch.cat(chunk_states, dim=1))
        all_states = torch.cat(states, dim=1)
        return all_states, state


class TemporalMamba2Block(nn.Module):
    """Single Temporal Mamba-2 block mixing convolutions and selective scans."""

    def __init__(
        self,
        embed_dim: int,
        state_dim: int,
        conv_kernel_size: int,
        chunk_length: int,
        bidirectional: bool,
        dropout: float,
        ff_expansion: int,
    ) -> None:
        super().__init__()
        if conv_kernel_size % 2 == 0:
            raise ValueError("conv_kernel_size must be odd to preserve sequence length")
        self.pre_norm = nn.LayerNorm(embed_dim)
        self.depthwise_conv = nn.Conv1d(
            embed_dim,
            embed_dim,
            kernel_size=conv_kernel_size,
            padding=conv_kernel_size // 2,
            groups=embed_dim,
        )
        self.conv_proj = nn.Linear(embed_dim, embed_dim)
        self.gate_proj = nn.Linear(embed_dim, embed_dim)
        self.state_in = nn.Linear(embed_dim, state_dim)
        self.state_out = nn.Linear(state_dim, embed_dim)
        self.scan = ChunkedSelectiveScan(
            state_dim=state_dim,
            chunk_length=chunk_length,
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
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = x
        y = self.pre_norm(x)
        conv_out = self.depthwise_conv(y.transpose(1, 2)).transpose(1, 2)
        local_mix = self.conv_proj(conv_out)
        gates = torch.sigmoid(self.gate_proj(y))
        ssm_inputs = self.state_in(y + local_mix)
        scan_out, final_state = self.scan(ssm_inputs)
        selective = self.state_out(scan_out) * gates
        x = residual + self.dropout(selective)
        x = x + self.dropout(self.ff(self.ff_norm(x)))
        return x, final_state


@register("mamba2")
class Mamba2Model(ARBaseModel):
    """Temporal Mamba-2 inspired model with chunked selective scan blocks."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        pred_len: int = 1,
        embed_dim: int = 256,
        state_dim: int = 128,
        num_layers: int = 6,
        conv_kernel_size: int = 5,
        chunk_length: int = 512,
        bidirectional_scan: bool = True,
        dropout: float = 0.1,
        ff_expansion: int = 4,
        adapter_rank: int = 0,
        adapter_alpha: float = 16.0,
        adapter_dropout: float = 0.0,
        freeze_backbone: bool = False,
        train_head: bool = True,
        pretrained_checkpoint: Optional[str] = None,
        strict_checkpoint: bool = False,
        **kwargs: Dict,
    ) -> None:
        super().__init__(input_dim, output_dim, **kwargs)
        if pred_len <= 0:
            raise ValueError("pred_len must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if embed_dim <= 0 or state_dim <= 0:
            raise ValueError("embed_dim and state_dim must be positive")

        self.pred_len = pred_len
        self.embed_dim = embed_dim
        self.freeze_backbone = freeze_backbone
        self.train_head = train_head

        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.layers = nn.ModuleList(
            [
                TemporalMamba2Block(
                    embed_dim=embed_dim,
                    state_dim=state_dim,
                    conv_kernel_size=conv_kernel_size,
                    chunk_length=chunk_length,
                    bidirectional=bidirectional_scan,
                    dropout=dropout,
                    ff_expansion=ff_expansion,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(embed_dim)
        self.head = LoRALinear(
            embed_dim,
            pred_len * output_dim,
            rank=adapter_rank,
            alpha=adapter_alpha,
            dropout=adapter_dropout,
            freeze_base=freeze_backbone and not train_head,
        )

        if pretrained_checkpoint:
            self._load_pretrained_checkpoint(pretrained_checkpoint, strict_checkpoint)

        self._apply_parameter_freeze()

    def _load_pretrained_checkpoint(self, path: str, strict: bool) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        checkpoint = torch.load(path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        missing, unexpected = self.load_state_dict(state_dict, strict=strict)
        if missing or unexpected:
            LOGGER.warning(
                "Loaded checkpoint with missing=%s unexpected=%s", missing, unexpected
            )

    def _apply_parameter_freeze(self) -> None:
        if not self.freeze_backbone:
            return
        for name, param in self.named_parameters():
            if "lora" in name:
                param.requires_grad = True
                continue
            if self.train_head and name.startswith("head"):
                param.requires_grad = True
                continue
            param.requires_grad = False

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        **kwargs: Dict,
    ) -> Dict[str, torch.Tensor]:
        del context, kwargs
        tokens = self.input_proj(x)
        selective_states: List[torch.Tensor] = []
        for layer in self.layers:
            tokens, state = layer(tokens)
            selective_states.append(state)
        tokens = self.final_norm(tokens)
        pooled = tokens.mean(dim=1)
        preds = self.head(pooled).view(-1, self.pred_len, self.output_dim)
        extras = {
            "selective_states": selective_states,
            "embeddings": tokens,
        }
        return {"preds": preds, "extras": extras}
