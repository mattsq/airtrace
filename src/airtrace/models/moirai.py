"""Moirai-style multiresolution selective state-space model."""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ARBaseModel
from .chronos_bolt import LoRALinear
from .registry import register

LOGGER = logging.getLogger(__name__)


class PatchEmbedding(nn.Module):
    """Patchify a time series and project patches into embeddings."""

    def __init__(self, input_dim: int, patch_size: int, embed_dim: int) -> None:
        super().__init__()
        if patch_size <= 0:
            raise ValueError("patch_size must be positive")
        self.patch_size = patch_size
        self.embed = nn.Linear(input_dim * patch_size, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert input [B, T, D] into patch tokens."""
        B, T, D = x.shape
        remainder = T % self.patch_size
        pad_len = (self.patch_size - remainder) % self.patch_size
        if pad_len > 0:
            pad_tensor = torch.zeros(B, pad_len, D, device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad_tensor], dim=1)
        patches = x.unfold(dimension=1, size=self.patch_size, step=self.patch_size)
        patches = patches.contiguous().view(B, -1, self.patch_size * D)
        return self.embed(patches)


class MultiResolutionPatcher(nn.Module):
    """Fuse multiple patch scales into a single token stream."""

    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        patch_scales: Sequence[int],
    ) -> None:
        super().__init__()
        if not patch_scales:
            raise ValueError("patch_scales must be a non-empty sequence")
        self.patch_scales = tuple(int(scale) for scale in patch_scales)
        self.embedders = nn.ModuleList(
            [PatchEmbedding(input_dim, scale, embed_dim) for scale in self.patch_scales]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Return fused tokens and per-scale token streams."""
        scale_tokens: List[torch.Tensor] = []
        lengths: List[int] = []
        for embedder in self.embedders:
            tokens = embedder(x)
            scale_tokens.append(tokens)
            lengths.append(tokens.shape[1])

        target_len = max(lengths)
        aligned = []
        for tokens in scale_tokens:
            if tokens.shape[1] == target_len:
                aligned.append(tokens)
                continue
            resized = F.interpolate(
                tokens.transpose(1, 2),
                size=target_len,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
            aligned.append(resized)

        fused = torch.stack(aligned, dim=0).mean(dim=0)
        return fused, scale_tokens


class SelectiveSSMBlock(nn.Module):
    """Simplified selective state-space layer with gated depthwise convolution."""

    def __init__(
        self,
        embed_dim: int,
        state_dim: int,
        conv_kernel_size: int,
        dropout: float,
        ff_expansion: int,
    ) -> None:
        super().__init__()
        if conv_kernel_size % 2 == 0:
            raise ValueError("conv_kernel_size must be odd for same-padding")
        self.pre_norm = nn.LayerNorm(embed_dim)
        self.state_proj = nn.Linear(embed_dim, state_dim * 2)
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
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply selective SSM mixing and feed-forward refinement."""
        y = self.pre_norm(x)
        state, gate = self.state_proj(y).chunk(2, dim=-1)
        state = torch.tanh(
            self.state_conv(state.transpose(1, 2)).transpose(1, 2)
        )
        gate = torch.sigmoid(gate)
        mixed = self.out_proj(state * gate)
        x = x + self.dropout(mixed)
        x = x + self.dropout(self.ff(self.ff_norm(x)))
        return x, state


@register("moirai")
class MoiraiModel(ARBaseModel):
    """Moirai-style multiresolution selective state-space model."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        pred_len: int = 1,
        embed_dim: int = 256,
        state_dim: int = 256,
        num_layers: int = 6,
        conv_kernel_size: int = 5,
        dropout: float = 0.1,
        ff_expansion: int = 4,
        patch_scales: Sequence[int] = (4, 16),
        max_positions: int = 4096,
        adapter_rank: int = 0,
        adapter_alpha: float = 8.0,
        adapter_dropout: float = 0.05,
        freeze_backbone: bool = False,
        train_head: bool = True,
        pretrained_checkpoint: Optional[str] = None,
        strict_checkpoint: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(input_dim, output_dim, **kwargs)
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if pred_len <= 0:
            raise ValueError("pred_len must be positive")
        if state_dim <= 0:
            raise ValueError("state_dim must be positive")

        # Default prediction horizon stays at 1 so the CI validation script,
        # which builds models without overrides and expects single-step targets,
        # can compute metrics without broadcasting errors. Configs can still
        # override this to multi-step horizons for research experiments.
        self.pred_len = pred_len
        self.embed_dim = embed_dim
        self.state_dim = state_dim
        self.dropout = dropout
        self.freeze_backbone = freeze_backbone
        self.train_head = train_head
        self.patch_scales = tuple(int(scale) for scale in patch_scales)
        self.max_positions = max_positions

        self.patcher = MultiResolutionPatcher(input_dim, embed_dim, self.patch_scales)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_positions, embed_dim))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        self.layers = nn.ModuleList(
            [
                SelectiveSSMBlock(
                    embed_dim=embed_dim,
                    state_dim=state_dim,
                    conv_kernel_size=conv_kernel_size,
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

    def _positional_encoding(self, tokens: torch.Tensor) -> torch.Tensor:
        num_tokens = tokens.shape[1]
        if num_tokens > self.max_positions:
            pos_emb = F.interpolate(
                self.pos_embedding.transpose(1, 2),
                size=num_tokens,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        else:
            pos_emb = self.pos_embedding[:, :num_tokens, :]
        return tokens + pos_emb

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        **kwargs: Dict,
    ) -> Dict[str, torch.Tensor]:
        del context, kwargs
        tokens, scale_tokens = self.patcher(x)
        tokens = self._positional_encoding(tokens)

        ssm_states: List[torch.Tensor] = []
        for layer in self.layers:
            tokens, state = layer(tokens)
            ssm_states.append(state)

        tokens = self.final_norm(tokens)
        pooled = tokens.mean(dim=1)
        preds = self.head(pooled).view(-1, self.pred_len, self.output_dim)

        extras = {
            "multiresolution_tokens": scale_tokens,
            "ssm_states": ssm_states,
            "fused_tokens": tokens,
        }
        return {"preds": preds, "extras": extras}
