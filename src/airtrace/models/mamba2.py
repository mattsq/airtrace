"""Temporal Mamba-2 inspired selective state-space model."""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .base import ARBaseModel
from .chronos_bolt import LoRALinear
from airtrace.models.registry import register

LOGGER = logging.getLogger(__name__)


class ChunkedSelectiveScan(nn.Module):
    """Simplified chunked selective state-space scan with bidirectional option.

    Processes sequences in chunks to improve memory efficiency for long contexts
    (e.g., 100k+ tokens). Each chunk maintains a hidden state that carries forward
    information across timesteps with learned decay rates.

    The chunked approach allows:
    - Linear memory scaling with sequence length
    - Hardware-friendly parallelization within chunks
    - Optional bidirectional fusion for better stability

    Args:
        state_dim: Dimension of the internal hidden state
        chunk_length: Number of timesteps to process per chunk (larger = more memory,
                     potentially better parallelization)
        bidirectional: If True, run scan forward and backward, averaging outputs
        dropout: Dropout probability applied to scan outputs
        decay_init: Initial value for decay parameters before sigmoid (default 0.0 → 0.5 decay)
    """

    def __init__(
        self,
        state_dim: int,
        chunk_length: int,
        bidirectional: bool,
        dropout: float,
        decay_init: float = 0.0,
    ) -> None:
        super().__init__()
        if chunk_length <= 0:
            raise ValueError("chunk_length must be positive")
        self.state_dim = state_dim
        self.chunk_length = chunk_length
        self.bidirectional = bidirectional
        self.dropout = nn.Dropout(dropout)
        # Diagonal decay term parameterised to stay in (0, 1) via sigmoid
        # Initialized to decay_init (default 0.0 gives decay=0.5 after sigmoid)
        # Can add small random noise for diversity across dimensions
        self.decay = nn.Parameter(torch.full((state_dim,), decay_init))

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

    def _scan_one_direction(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
    """Single Temporal Mamba-2 block mixing convolutions and selective scans.

    Architecture:
        1. Layer norm → depthwise conv (local mixing)
        2. Gated projection + selective scan (global context)
        3. Residual connection
        4. Feed-forward network with residual

    This combines local temporal patterns (conv) with long-range dependencies
    (selective scan) in a parameter-efficient manner.

    Args:
        embed_dim: Token embedding dimension
        state_dim: Internal state dimension for selective scan
        conv_kernel_size: Kernel size for depthwise conv (must be odd)
        chunk_length: Timesteps per scan chunk
        bidirectional: Enable bidirectional scanning
        dropout: Dropout probability
        ff_expansion: Feed-forward hidden dimension multiplier
        decay_init: Initial value for state decay parameters (default 0.0)
    """

    def __init__(
        self,
        embed_dim: int,
        state_dim: int,
        conv_kernel_size: int,
        chunk_length: int,
        bidirectional: bool,
        dropout: float,
        ff_expansion: int,
        decay_init: float = 0.0,
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
            decay_init=decay_init,
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
        """Forward pass through Mamba-2 block.

        Args:
            x: Input tokens [B, T, embed_dim]

        Returns:
            Tuple of (output tokens [B, T, embed_dim], final state [B, state_dim])
        """
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
    """Temporal Mamba-2 inspired model with chunked selective scan blocks.

    A state-space model designed for long-context aircraft timeseries forecasting.
    Combines selective state-space scans (Mamba-2 style) with local convolutions
    for efficient processing of sequences up to 100k+ tokens.

    Key features:
    - Chunked selective scan for linear memory scaling
    - Optional bidirectional scanning for stability
    - LoRA adapters for parameter-efficient fine-tuning
    - Pretrained checkpoint loading support

    Args:
        input_dim: Number of input sensor channels
        output_dim: Number of output prediction channels
        pred_len: Forecast horizon (number of timesteps to predict)
        embed_dim: Token embedding dimension for the backbone
        state_dim: Internal state dimension for selective scan
        num_layers: Number of stacked Mamba-2 blocks
        conv_kernel_size: Kernel size for local depthwise convolution (must be odd)
        chunk_length: Timesteps per selective scan chunk (larger = more memory usage)
        bidirectional_scan: If True, use bidirectional scan fusion
        dropout: Dropout probability applied throughout the model
        ff_expansion: Feed-forward network hidden dimension multiplier
        decay_init: Initial value for state decay parameters before sigmoid (default 0.0 → 0.5)
        adapter_rank: LoRA rank for lightweight fine-tuning (0 = disabled, typical: 4-16)
        adapter_alpha: LoRA scaling factor (higher = stronger adaptation, typical: 8-32)
        adapter_dropout: Dropout applied to LoRA adapters
        freeze_backbone: If True, freeze all parameters except LoRA and optionally head
        train_head: Keep prediction head trainable when freeze_backbone=True
        pretrained_checkpoint: Optional path to pretrained .pt or .ckpt file
        strict_checkpoint: If True, require exact parameter name match when loading checkpoint
        **kwargs: Additional arguments passed to ARBaseModel
    """

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
        decay_init: float = 0.0,
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
        # Comprehensive parameter validation
        if pred_len <= 0:
            raise ValueError("pred_len must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if embed_dim <= 0 or state_dim <= 0:
            raise ValueError("embed_dim and state_dim must be positive")
        if ff_expansion <= 0:
            raise ValueError("ff_expansion must be positive")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError(f"dropout must be in [0, 1], got {dropout}")
        if not 0.0 <= adapter_dropout <= 1.0:
            raise ValueError(f"adapter_dropout must be in [0, 1], got {adapter_dropout}")
        if adapter_rank < 0:
            raise ValueError(f"adapter_rank must be non-negative, got {adapter_rank}")
        if adapter_alpha <= 0:
            raise ValueError(f"adapter_alpha must be positive, got {adapter_alpha}")
        if conv_kernel_size <= 0:
            raise ValueError(f"conv_kernel_size must be positive, got {conv_kernel_size}")
        if chunk_length <= 0:
            raise ValueError(f"chunk_length must be positive, got {chunk_length}")

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
                    decay_init=decay_init,
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
            LOGGER.warning("Loaded checkpoint with missing=%s unexpected=%s", missing, unexpected)

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
        """Forward pass through the Temporal Mamba-2 model.

        Args:
            x: Input tensor [B, T, D_in] where B is batch size, T is sequence length,
               D_in is input dimension
            context: Optional context tensor (currently unused - Mamba-2's selective scan
                    inherently captures relevant context through its state mechanism)
            **kwargs: Additional arguments (unused, for interface compatibility)

        Returns:
            Dictionary containing:
                - preds: Predictions [B, pred_len, D_out]
                - extras: Dict with 'selective_states' (list of final states per layer)
                         and 'embeddings' (final token representations)
        """
        del context, kwargs  # Context unused: selective scan captures temporal dependencies
        tokens = self.input_proj(x)
        selective_states: List[torch.Tensor] = []
        for layer in self.layers:
            tokens, state = layer(tokens)
            selective_states.append(state)
        tokens = self.final_norm(tokens)
        # Pool the last token embedding, which already resides in ``embed_dim`` space.
        # Using the selective state directly would require an extra projection because it
        # lives in ``state_dim``. Pooling the normalized tokens keeps the prediction head
        # dimensions consistent and avoids shape mismatches during inference/tests.
        pooled = tokens[:, -1, :]
        preds = self.head(pooled).view(-1, self.pred_len, self.output_dim)
        extras = {
            "selective_states": selective_states,
            "embeddings": tokens,
        }
        return {"preds": preds, "extras": extras}
