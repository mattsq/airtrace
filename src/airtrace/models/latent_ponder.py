"""Latent pondering wrapper with adaptive halting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .base import ARBaseModel, ResidualWrapperCompatible
from .registry import build_model, register


class LatentPonderBlock(nn.Module):
    """Lightweight latent update block used during pondering.

    A two-layer MLP with residual connection keeps the computational cost low
    while still allowing the latent state to evolve across ponder steps.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Update the latent state.

        Args:
            h: Current latent state of shape ``[B, H]``.

        Returns:
            Tuple of the next latent state and probes for logging.
        """
        update = self.ff(h)
        h_next = self.norm(h + update)
        probes = {"h_norm": torch.norm(h_next, dim=-1)}
        return h_next, probes


class TRMScratchpadBlock(nn.Module):
    """TRM-style scratchpad update that conditions on the draft answer.

    The block consumes the current latent state ``h_t`` alongside summaries of
    the input window and the evolving draft prediction ``y_t`` to refine the
    scratchpad. Weight tying across ponder steps keeps compute low while still
    allowing the draft to influence future updates.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.x_proj = nn.Linear(input_dim, hidden_dim)
        self.y_proj = nn.Linear(output_dim, hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self, x_context: torch.Tensor, y: torch.Tensor, h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Update the latent scratchpad using both the draft and input context.

        Args:
            x_context: Pooled representation of the input window ``[B, H]``.
            y: Current draft prediction ``[B, pred_len, D]``.
            h: Current latent scratchpad ``[B, H]``.

        Returns:
            Tuple of ``(h_next, y_summary, probes)`` where ``y_summary`` is the
            projected draft summary used by the halting head.
        """

        y_summary = self.y_proj(y.mean(dim=1))
        concat = torch.cat([h, y_summary, x_context], dim=-1)
        h_next = self.norm(h + self.ff(concat))
        probes = {
            "h_norm": torch.norm(h_next, dim=-1),
            "y_norm": torch.norm(y_summary, dim=-1),
        }
        return h_next, y_summary, probes


class TRMRefinementHead(nn.Module):
    """Refine the draft prediction using the updated latent state."""

    def __init__(self, hidden_dim: int, output_dim: int, mode: str, dropout: float) -> None:
        super().__init__()
        if mode not in {"mlp", "base_head"}:
            raise ValueError(f"Unsupported refine_head mode: {mode}")
        self.mode = mode
        if mode == "mlp":
            self.refiner = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )
        else:
            self.refiner = nn.Linear(hidden_dim, output_dim)

    def forward(self, y: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        update = self.refiner(h)
        return y + update.unsqueeze(1)


@dataclass
class TRMState:
    """Coupled draft/latent state for TRM-style pondering."""

    y: torch.Tensor
    h: torch.Tensor


@register("latent_ponder")
class LatentPonderWrapper(ARBaseModel):
    """Wrap an autoregressive model with latent pondering and adaptive halting."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        base_model: Optional[Dict[str, Any]] = None,
        hidden_dim: int = 128,
        max_steps: int = 6,
        min_steps: int = 1,
        ponder_penalty: float = 0.01,
        aux_weight: float = 0.0,
        halt_bias: float = 0.0,
        max_eval_steps: Optional[int] = None,
        dropout: float = 0.1,
        trm_mode: bool = False,
        refine_head: str = "mlp",
        supervision_steps: Optional[Sequence[int]] = None,
        halting_mode: str = "none",  # "none" | "pondernet" | "trm"
        halting_weight: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(input_dim, output_dim, **kwargs)

        base_config = base_model or {"name": "gru_ar", "params": {}}
        self.base_model = build_model(base_config, input_dim=input_dim, output_dim=output_dim)

        self.hidden_dim = hidden_dim
        self.max_steps = max_steps
        self.min_steps = max(min_steps, 1)
        self.ponder_penalty = ponder_penalty
        self.aux_weight = aux_weight
        self.halt_bias = halt_bias
        self.max_eval_steps = max_eval_steps
        self.trm_mode = trm_mode
        self.refine_head_mode = refine_head
        self.supervision_steps = list(supervision_steps) if supervision_steps is not None else None
        assert halting_mode in {"none", "pondernet", "trm"}
        self.halting_mode = halting_mode
        self.halting_weight = halting_weight

        self.encoder = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.GELU(),
        )
        self.ponder_block = LatentPonderBlock(hidden_dim, dropout=dropout)
        halt_in_features = hidden_dim * 2 if trm_mode else hidden_dim
        self.halt_head = nn.Linear(halt_in_features, 1)
        self.halt_head.bias.data.fill_(halt_bias)
        self.decoder = nn.Linear(hidden_dim, output_dim)
        self.trm_block = (
            TRMScratchpadBlock(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, dropout=dropout)
            if trm_mode
            else None
        )
        self.refinement_head = (
            TRMRefinementHead(hidden_dim=hidden_dim, output_dim=output_dim, mode=refine_head, dropout=dropout)
            if trm_mode
            else None
        )

    def _get_effective_max_steps(self) -> int:
        if self.training or self.max_eval_steps is None:
            return self.max_steps
        return min(self.max_steps, self.max_eval_steps)

    def _decode(self, h: torch.Tensor, pred_len: int) -> torch.Tensor:
        decoded = self.decoder(h)  # [B, D]
        return decoded.unsqueeze(1).expand(-1, pred_len, -1)

    def _record_auxiliary(
        self,
        aux: List[torch.Tensor],
        pred: torch.Tensor,
        step: int,
        max_steps: int,
    ) -> None:
        if self.aux_weight <= 0:
            return
        if self.supervision_steps is None:
            aux.append(pred)
            return
        requested = set(self.supervision_steps)
        if (step + 1) in requested:
            aux.append(pred)
        elif -1 in requested and (step + 1 == max_steps):
            aux.append(pred)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        base_preds, base_extras, pred_len = self._initial_prediction(
            x, context=context, **kwargs
        )

        pooled = base_preds.mean(dim=1)
        h = self.encoder(pooled)
        state: Optional[TRMState] = TRMState(y=base_preds, h=h) if self.trm_mode else None
        x_context = self.trm_block.x_proj(x.mean(dim=1)) if self.trm_mode and self.trm_block is not None else None

        halted = torch.zeros(h.shape[0], dtype=torch.bool, device=h.device)
        steps_taken = torch.zeros(h.shape[0], device=h.device)
        final_h = torch.zeros_like(h)
        final_y = torch.zeros_like(base_preds) if self.trm_mode else None
        halt_logits_list: List[torch.Tensor] = []
        step_preds: List[torch.Tensor] = []
        aux_preds = []

        max_steps = self._get_effective_max_steps()

        for step in range(max_steps):
            if self.trm_mode:
                assert self.trm_block is not None
                assert self.refinement_head is not None
                assert state is not None
                assert x_context is not None
                h, y_summary, _ = self.trm_block(x_context, state.y, state.h)
                refined_y = self.refinement_head(state.y, h)
                state = TRMState(y=refined_y, h=h)
                halt_features = torch.cat([h, y_summary], dim=-1)
                current_pred = refined_y
            else:
                h, _ = self.ponder_block(h)
                halt_features = h
                current_pred = self._decode(h, pred_len)

            logit = self.halt_head(halt_features).squeeze(-1)
            halt_logits_list.append(logit)
            step_preds.append(current_pred)

            if self.training and self.halting_mode in {"pondernet", "trm"}:
                decision = torch.zeros_like(logit, dtype=torch.bool)
            elif not self.training and self.halting_mode == "trm":
                prob = torch.sigmoid(logit)
                if step + 1 < self.min_steps:
                    decision = torch.zeros_like(prob, dtype=torch.bool)
                else:
                    decision = prob > 0.5
            else:
                prob = torch.sigmoid(logit)
                if step + 1 < self.min_steps:
                    decision = torch.zeros_like(prob, dtype=torch.bool)
                else:
                    decision = torch.bernoulli(prob).bool()

            active = ~halted
            new_halts = active & (decision | (step + 1 == max_steps))
            if new_halts.any():
                final_h = torch.where(new_halts.unsqueeze(-1), h, final_h)
                if final_y is not None and state is not None:
                    final_y = torch.where(new_halts.view(-1, 1, 1), state.y, final_y)
                steps_taken[new_halts] = step + 1
            halted = halted | new_halts

            self._record_auxiliary(aux_preds, current_pred, step, max_steps)

        if not halted.all():
            final_h = torch.where((~halted).unsqueeze(-1), h, final_h)
            if final_y is not None and state is not None:
                final_y = torch.where((~halted).view(-1, 1, 1), state.y, final_y)
        steps_taken[~halted] = float(max_steps)

        preds = final_y if final_y is not None else self._decode(final_h, pred_len)

        halt_logits = torch.stack(halt_logits_list, dim=1)
        halt_probs = torch.sigmoid(halt_logits)
        step_preds_tensor = torch.stack(step_preds, dim=1)
        ponder_cost = self.ponder_penalty * steps_taken.mean()
        halting_regularizer = halt_probs.clamp_min(1e-6).log().mean().neg()
        ponder_loss = ponder_cost + halting_regularizer
        ponder_loss_entry = (
            ponder_loss if self.halting_mode == "none" else torch.zeros((), device=h.device)
        )

        extras: Dict[str, Any] = {
            "base_extras": base_extras,
            "halt_logits": halt_logits,
            "halt_probs": halt_probs,
            "halt_distribution": halt_probs.detach(),
            "step_preds": step_preds_tensor,
            "ponder_steps": steps_taken.detach(),
            "mean_ponder_steps": steps_taken.mean().detach(),
            "ponder_cost": ponder_cost.detach(),
            "ponder_loss": ponder_loss_entry,
            "ponder_penalty": float(self.ponder_penalty),
            "halting_weight": float(self.halting_weight),
            "halting_mode": self.halting_mode,
            "max_steps_used": float(max_steps),
        }

        if aux_preds:
            extras["aux_preds"] = torch.stack(aux_preds, dim=1)
            extras["aux_weight"] = float(self.aux_weight)

        return {"preds": preds, "extras": extras}

    def _initial_prediction(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Dict[str, Any], int]:
        pred_len = int(kwargs.get("pred_len", getattr(self.base_model, "pred_len", 1)))

        if isinstance(self.base_model, ResidualWrapperCompatible):
            latent, base_extras = self.base_model.encode(x, context=context)
            base_preds = self.base_model.decode(latent, pred_len=pred_len)
            base_extras = {**base_extras, "latent": latent}
            return base_preds, base_extras, pred_len

        base_output = self.base_model(x, context=context, **kwargs)
        base_preds = base_output["preds"]
        return base_preds, base_output.get("extras", {}), base_preds.shape[1]
