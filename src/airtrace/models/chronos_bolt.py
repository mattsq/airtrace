"""Chronos-Bolt style pretrained foundation model."""

from __future__ import annotations

import logging
import math
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ARBaseModel
from .registry import register

LOGGER = logging.getLogger(__name__)


class LoRALinear(nn.Module):
    """Linear layer with optional LoRA adapter."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        rank: int = 0,
        alpha: float = 1.0,
        dropout: float = 0.0,
        freeze_base: bool = False,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.rank = rank
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout) if rank > 0 else None
        self.freeze_base = freeze_base
        if freeze_base:
            for param in self.linear.parameters():
                param.requires_grad = False

        if rank > 0:
            self.lora_down = nn.Linear(in_features, rank, bias=False)
            self.lora_up = nn.Linear(rank, out_features, bias=False)
            nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_up.weight)
            self.scaling = alpha / rank
        else:
            self.lora_down = None
            self.lora_up = None
            self.scaling = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.linear(x)
        if self.rank == 0:
            return base

        assert self.lora_down is not None
        assert self.lora_up is not None
        dropped = self.dropout(x) if self.dropout is not None else x
        lora = self.lora_up(self.lora_down(dropped)) * self.scaling
        return base + lora


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with optional LoRA adapters."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        lora_rank: int = 0,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
        freeze_base: bool = False,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = LoRALinear(
            embed_dim,
            embed_dim,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            freeze_base=freeze_base,
        )
        self.k_proj = LoRALinear(
            embed_dim,
            embed_dim,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            freeze_base=freeze_base,
        )
        self.v_proj = LoRALinear(
            embed_dim,
            embed_dim,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            freeze_base=freeze_base,
        )
        self.out_proj = LoRALinear(
            embed_dim,
            embed_dim,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            freeze_base=freeze_base,
        )
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).reshape(B, T, self.embed_dim)
        output = self.out_proj(context)
        return output, attn_weights


class GatedConvAttentionBlock(nn.Module):
    """Chronos-Bolt inspired block mixing convolutions and attention."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        conv_kernel_size: int,
        dilation: int,
        dropout: float,
        ff_expansion: int,
        lora_rank: int,
        lora_alpha: float,
        lora_dropout: float,
        freeze_base: bool,
    ) -> None:
        super().__init__()
        padding = ((conv_kernel_size - 1) // 2) * dilation
        self.norm1 = nn.LayerNorm(embed_dim)
        self.depthwise_conv = nn.Conv1d(
            embed_dim,
            embed_dim,
            kernel_size=conv_kernel_size,
            padding=padding,
            dilation=dilation,
            groups=embed_dim,
        )
        self.gate_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.attention = MultiHeadSelfAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            freeze_base=freeze_base,
        )

        self.norm3 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * ff_expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * ff_expansion, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = x
        y = self.norm1(x)
        conv_out = self.depthwise_conv(y.transpose(1, 2)).transpose(1, 2)
        gate = torch.sigmoid(self.gate_proj(y))
        x = residual + self.dropout(conv_out * gate)

        residual = x
        y = self.norm2(x)
        attn_out, attn_weights = self.attention(y)
        x = residual + self.dropout(attn_out)

        residual = x
        y = self.norm3(x)
        x = residual + self.dropout(self.ff(y))

        return x, attn_weights


@register("chronos_bolt")
class ChronosBoltModel(ARBaseModel):
    """Chronos-Bolt style foundation model with optional LoRA adapters."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        pred_len: int = 32,
        embed_dim: int = 256,
        patch_size: int = 16,
        patch_stride: Optional[int] = None,
        num_blocks: int = 6,
        num_heads: int = 8,
        dilation_growth: int = 2,
        conv_kernel_size: int = 5,
        dropout: float = 0.1,
        ff_expansion: int = 4,
        max_positions: int = 2048,
        lora_rank: int = 0,
        lora_alpha: float = 8.0,
        lora_dropout: float = 0.05,
        freeze_backbone: bool = False,
        train_head: bool = True,
        pretrained_checkpoint: Optional[str] = None,
        strict_checkpoint: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(input_dim, output_dim, **kwargs)
        self.pred_len = pred_len
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.patch_stride = patch_stride or patch_size
        self.max_positions = max_positions
        self.freeze_backbone = freeze_backbone
        self.train_head = train_head

        self.patch_embed = nn.Conv1d(
            in_channels=input_dim,
            out_channels=embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_stride,
        )
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_positions, embed_dim))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        self.blocks = nn.ModuleList()
        dilation = 1
        for _ in range(num_blocks):
            block = GatedConvAttentionBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                conv_kernel_size=conv_kernel_size,
                dilation=dilation,
                dropout=dropout,
                ff_expansion=ff_expansion,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                freeze_base=freeze_backbone,
            )
            self.blocks.append(block)
            dilation = min(dilation * dilation_growth, 256)

        self.final_norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, pred_len * output_dim),
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
            if self.train_head and (name.startswith("head") or "head." in name):
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
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        del context, kwargs
        if x.shape[1] < self.patch_size:
            raise ValueError("Input sequence is shorter than the patch size")

        patches = self.patch_embed(x.permute(0, 2, 1)).transpose(1, 2)
        tokens = self._positional_encoding(patches)

        attn_maps: List[torch.Tensor] = []
        for block in self.blocks:
            tokens, attn_weights = block(tokens)
            attn_maps.append(attn_weights)

        tokens = self.final_norm(tokens)
        pooled = tokens.mean(dim=1)
        preds = self.head(pooled).view(-1, self.pred_len, self.output_dim)

        extras = {
            "patch_tokens": tokens,
            "attention_maps": attn_maps,
        }
        return {"preds": preds, "extras": extras}
