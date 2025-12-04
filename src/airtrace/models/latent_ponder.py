"""Latent pondering wrapper with adaptive halting."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from .base import ARBaseModel
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

        self.encoder = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.GELU(),
        )
        self.ponder_block = LatentPonderBlock(hidden_dim, dropout=dropout)
        self.halt_head = nn.Linear(hidden_dim, 1)
        self.halt_head.bias.data.fill_(halt_bias)
        self.decoder = nn.Linear(hidden_dim, output_dim)

    def _get_effective_max_steps(self) -> int:
        if self.training or self.max_eval_steps is None:
            return self.max_steps
        return min(self.max_steps, self.max_eval_steps)

    def _decode(self, h: torch.Tensor, pred_len: int) -> torch.Tensor:
        decoded = self.decoder(h)  # [B, D]
        return decoded.unsqueeze(1).expand(-1, pred_len, -1)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        base_output = self.base_model(x, context=context, **kwargs)
        base_preds = base_output["preds"]
        pred_len = base_preds.shape[1]

        pooled = base_preds.mean(dim=1)
        h = self.encoder(pooled)

        halted = torch.zeros(h.shape[0], dtype=torch.bool, device=h.device)
        steps_taken = torch.zeros(h.shape[0], device=h.device)
        final_h = torch.zeros_like(h)
        halt_probs = []
        aux_preds = []

        max_steps = self._get_effective_max_steps()

        for step in range(max_steps):
            h, _ = self.ponder_block(h)
            logit = self.halt_head(h).squeeze(-1)
            prob = torch.sigmoid(logit)
            halt_probs.append(prob)

            if step + 1 < self.min_steps:
                decision = torch.zeros_like(prob, dtype=torch.bool)
            else:
                decision = torch.bernoulli(prob).bool()

            active = ~halted
            new_halts = active & (decision | (step + 1 == max_steps))
            if new_halts.any():
                final_h[new_halts] = h[new_halts]
                steps_taken[new_halts] = step + 1
            halted = halted | new_halts

            if self.aux_weight > 0:
                aux_preds.append(self._decode(h, pred_len))

        if not halted.all():
            final_h[~halted] = h[~halted]
            steps_taken[~halted] = float(max_steps)

        preds = self._decode(final_h, pred_len)

        halt_distribution = torch.stack(halt_probs, dim=1)
        ponder_cost = self.ponder_penalty * steps_taken.mean()
        # Encourage confident halting while keeping compute small
        halting_regularizer = (halt_distribution.clamp_min(1e-6).log().mean().neg())
        ponder_loss = ponder_cost + halting_regularizer

        extras: Dict[str, Any] = {
            "base_extras": base_output.get("extras", {}),
            "halt_distribution": halt_distribution.detach(),
            "ponder_steps": steps_taken.detach(),
            "mean_ponder_steps": steps_taken.mean().detach(),
            "ponder_cost": ponder_cost.detach(),
            "ponder_loss": ponder_loss,
            "max_steps_used": float(max_steps),
        }

        if aux_preds:
            extras["aux_preds"] = torch.stack(aux_preds, dim=1)
            extras["aux_weight"] = float(self.aux_weight)

        return {"preds": preds, "extras": extras}
