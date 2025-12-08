"""Residual solver with iterative refinement and probabilistic halting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ARBaseModel, ResidualWrapperCompatible
from .registry import build_model, register


@dataclass
class ResidualSolverConfig:
    """Configuration for :class:`ResidualSolver`."""

    input_dim: int
    hidden_dim: int
    output_dim: int
    residual_hidden: int = 128
    update_hidden: int = 128
    n_residual_layers: int = 2
    n_update_layers: int = 2
    max_steps: int = 20
    residual_halt_threshold: float = 0.01
    residual_bonus_logit: float = 1.0
    halt_eps: float = 1e-6
    lambda_step: float = 1.0
    lambda_consistency: float = 0.1
    lambda_compute: float = 0.01
    halt_prior_p: float = 0.2
    halt_bias: float = 0.0


class SolverState(NamedTuple):
    """Container for the solver's iterative state."""

    h: torch.Tensor
    y: torch.Tensor
    r: torch.Tensor
    step: int


def _mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    n_layers: int,
    zero_last: bool = False,
    scale_last: float = 1.0,
) -> nn.Sequential:
    layers: List[nn.Module] = []
    dim = in_dim
    for _ in range(max(n_layers - 1, 0)):
        layers.extend([nn.Linear(dim, hidden_dim), nn.GELU()])
        dim = hidden_dim
    layers.append(nn.Linear(dim, out_dim))
    mlp = nn.Sequential(*layers)
    if zero_last:
        nn.init.zeros_(mlp[-1].weight)
        nn.init.zeros_(mlp[-1].bias)
    else:
        mlp[-1].weight.data.mul_(scale_last)
        mlp[-1].bias.data.mul_(scale_last)
    return mlp


@register("residual_solver")
class ResidualSolver(ARBaseModel):
    """Iteratively refines predictions of a base model with residual updates and learned halting."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        base_model: Optional[Dict[str, Any]] = None,
        hidden_dim: int = 128,
        residual_hidden: int = 128,
        update_hidden: int = 128,
        n_residual_layers: int = 2,
        n_update_layers: int = 2,
        max_steps: int = 20,
        residual_halt_threshold: float = 0.01,
        residual_bonus_logit: float = 1.0,
        halt_eps: float = 1e-6,
        lambda_step: float = 1.0,
        lambda_consistency: float = 0.1,
        lambda_compute: float = 0.01,
        halt_prior_p: float = 0.2,
        halt_bias: float = 0.0,
        **_: Any,
    ) -> None:
        super().__init__(input_dim, output_dim)

        # Config mostly for hyperparameters
        self.config = ResidualSolverConfig(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            residual_hidden=residual_hidden,
            update_hidden=update_hidden,
            n_residual_layers=n_residual_layers,
            n_update_layers=n_update_layers,
            max_steps=max_steps,
            residual_halt_threshold=residual_halt_threshold,
            residual_bonus_logit=residual_bonus_logit,
            halt_eps=halt_eps,
            lambda_step=lambda_step,
            lambda_consistency=lambda_consistency,
            lambda_compute=lambda_compute,
            halt_prior_p=halt_prior_p,
            halt_bias=halt_bias,
        )

        # Build base model
        base_model_cfg = base_model or {"name": "linear_ar", "params": {}}
        self.base_model = build_model(base_model_cfg, input_dim=input_dim, output_dim=output_dim)

        # Components for solver
        self.hidden_update = nn.GRUCell(hidden_dim + output_dim, hidden_dim)
        self.residual_net = _mlp(
            in_dim=hidden_dim + output_dim + hidden_dim,
            hidden_dim=residual_hidden,
            out_dim=output_dim,
            n_layers=n_residual_layers,
            zero_last=True,
        )
        self.update_net = _mlp(
            in_dim=output_dim + hidden_dim,
            hidden_dim=update_hidden,
            out_dim=output_dim,
            n_layers=n_update_layers,
            scale_last=0.1,
        )
        self.step_embedding = nn.Embedding(max_steps, hidden_dim)
        self.halting_head = nn.Linear(hidden_dim + output_dim + 1 + hidden_dim, 1)
        nn.init.constant_(self.halting_head.bias, halt_bias)

        # Adapter / Fallback
        self.adapter: Optional[nn.Module] = None
        self.fallback_encoder: Optional[nn.Linear] = None

        if isinstance(self.base_model, ResidualWrapperCompatible):
            base_hidden = getattr(self.base_model, "hidden_size", getattr(self.base_model, "hidden_dim", None))
            if base_hidden is not None and base_hidden != hidden_dim:
                self.adapter = nn.Linear(base_hidden, hidden_dim)
            elif base_hidden is None:
                # Use LazyLinear if we can't determine the input dimension statically
                self.adapter = nn.LazyLinear(hidden_dim)

        if not isinstance(self.base_model, ResidualWrapperCompatible):
             # Fallback: project input summary to hidden state
             self.fallback_encoder = nn.Linear(input_dim, hidden_dim)

        # Also need to map input x to hidden_dim for the solver loop?
        self.context_proj = nn.Linear(input_dim, hidden_dim)


    def _init_state_from_base(
        self, x: torch.Tensor, context: Optional[torch.Tensor], **kwargs
    ) -> Tuple[SolverState, torch.Tensor, Dict[str, Any]]:

        # 1. Run Base Model
        if isinstance(self.base_model, ResidualWrapperCompatible):
            latent, base_extras = self.base_model.encode(x, context=context)
            pred_len = int(kwargs.get("pred_len", 1))
            y0 = self.base_model.decode(latent, pred_len=pred_len)

            # Adapt latent
            if self.adapter is not None:
                # If LazyLinear, this initializes it. If Linear, it works if dims match.
                h0 = self.adapter(latent)
            elif latent.shape[-1] != self.config.hidden_dim:
                # Should have been caught by __init__ logic, but just in case
                # We can't fix it here easily without defining a module.
                # Assuming this path is rare if logic above is correct.
                # But if base_hidden was detected as None, self.adapter is LazyLinear.
                # If base_hidden was equal, self.adapter is None.
                h0 = latent
            else:
                h0 = latent
        else:
            # Standard model
            out = self.base_model(x, context=context, **kwargs)
            y0 = out["preds"]
            base_extras = out.get("extras", {})

            # Create h0 from input since we can't extract it
            x_summary = x.mean(dim=1)
            h0 = torch.tanh(self.fallback_encoder(x_summary))

        r0 = torch.zeros_like(y0)
        return SolverState(h=h0, y=y0, r=r0, step=0), base_extras

    def _compute_halt_logits(self, state: SolverState, context_features: torch.Tensor) -> torch.Tensor:
        # context_features is the solver's view of X

        # We need h to be [B, hidden]. If h came from sequence model, might be [B, T, H] or [B, H].
        # Assume h is [B, H].
        h_vec = state.h
        if h_vec.dim() > 2:
            h_vec = h_vec.mean(dim=1) # Pooling if latent is sequence

        step_ids = torch.full((state.h.shape[0],), state.step, dtype=torch.long, device=state.h.device)
        step_emb = self.step_embedding(step_ids)

        r_pool = state.r.mean(dim=1)
        r_norm_pool = torch.norm(r_pool, dim=-1, keepdim=True)

        halt_features = torch.cat([h_vec, r_pool, r_norm_pool, step_emb], dim=-1)

        logits = self.halting_head(halt_features).squeeze(-1)

        # Bonus for small residual
        small_residual = (r_norm_pool.squeeze(-1) < self.config.residual_halt_threshold).float()
        logits = logits + self.config.residual_bonus_logit * small_residual
        return logits

    def _solver_step(self, context_features: torch.Tensor, state: SolverState) -> SolverState:
        # Inputs to nets must be pooled if sequence
        y_pool = state.y.mean(dim=1)
        # x_features/context is already [B, H]
        h_vec = state.h if state.h.dim() == 2 else state.h.mean(dim=1)

        # 1. Estimate residual
        # residual_inp = cat([y, x, h])
        residual_inp = torch.cat([y_pool, context_features, h_vec], dim=-1)
        r_est = self.residual_net(residual_inp) # [B, D]

        # Expand r_est back to [B, T, D]
        T_out = state.y.shape[1]
        r_t_broad = r_est.unsqueeze(1).expand(-1, T_out, -1)

        # 2. Compute Update
        # update_net(cat([r, h]))
        update_inp = torch.cat([r_est, h_vec], dim=-1)
        delta_est = self.update_net(update_inp) # [B, D]
        delta = delta_est.unsqueeze(1).expand(-1, T_out, -1)

        y_next = state.y + delta
        r_next = r_t_broad # The residual we estimated

        # 3. Update Hidden State
        # GRUCell(input, hidden)
        # input = cat([x, y_next]) -> [B, H + D]
        h_inp = torch.cat([context_features, y_next.mean(dim=1)], dim=-1)
        h_next = self.hidden_update(h_inp, h_vec)

        return SolverState(h=h_next, y=y_next, r=r_next, step=state.step + 1)

    def _halting_distribution(self, halt_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q = torch.sigmoid(halt_logits)
        cont = 1 - q
        survival = torch.cumprod(
            torch.cat([torch.ones_like(q[:, :1]), cont[:, :-1]], dim=1), dim=1
        )
        p_halt = q * survival
        p_rest = torch.cumprod(cont, dim=1)
        halt_dist = p_halt.clone()
        halt_dist[:, -1] = halt_dist[:, -1] + p_rest[:, -1]
        return halt_dist, survival

    def forward(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None, **kwargs
    ) -> Dict[str, torch.Tensor]:

        # Context features for solver
        x_summary = x.mean(dim=1)
        x_features = torch.tanh(self.context_proj(x_summary))

        # Init state
        state, base_extras = self._init_state_from_base(x, context, **kwargs)

        step_preds: List[torch.Tensor] = []
        residuals: List[torch.Tensor] = []
        halt_logits: List[torch.Tensor] = []

        # Loop
        for _ in range(self.config.max_steps):
            logits = self._compute_halt_logits(state, x_features)
            halt_logits.append(logits)
            step_preds.append(state.y)
            residuals.append(state.r)
            state = self._solver_step(x_features, state)

        halt_logits_tensor = torch.stack(halt_logits, dim=1)
        step_preds_tensor = torch.stack(step_preds, dim=1)  # [B, Steps, T_out, D]
        residual_tensor = torch.stack(residuals, dim=1)

        halt_dist, _ = self._halting_distribution(halt_logits_tensor)

        # Expected prediction
        # halt_dist [B, Steps], step_preds [B, Steps, T_out, D]
        # Weighted sum over Steps dim (dim 1)
        expected_preds = (halt_dist.view(halt_dist.shape[0], halt_dist.shape[1], 1, 1) * step_preds_tensor).sum(dim=1)

        expected_steps = (
            halt_dist * torch.arange(1, self.config.max_steps + 1, device=x.device)
        ).sum(dim=1)

        extras = {
            "base_extras": base_extras,
            "halt_logits": halt_logits_tensor,
            "halt_distribution": halt_dist,
            "step_preds": step_preds_tensor,
            "residuals": residual_tensor,
            "expected_steps": expected_steps,
        }

        return {"preds": expected_preds, "extras": extras}

    @torch.no_grad()
    def inference(self, x: torch.Tensor, halt_threshold: float = 0.9, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        x_summary = x.mean(dim=1)
        x_features = torch.tanh(self.context_proj(x_summary))

        state, _ = self._init_state_from_base(x, None, **kwargs)
        batch = x.shape[0]

        survival = torch.ones(batch, device=x.device)
        mass = torch.zeros(batch, device=x.device)

        # pred_accum: [B, T_out, D]
        pred_accum = torch.zeros_like(state.y)
        steps_taken = torch.full((batch,), self.config.max_steps, device=x.device, dtype=torch.long)

        for step in range(self.config.max_steps):
            logits = self._compute_halt_logits(state, x_features)
            q = torch.sigmoid(logits)
            p_halt = survival * q

            # Accumulate
            pred_accum = pred_accum + p_halt.view(-1, 1, 1) * state.y
            mass = mass + p_halt
            survival = survival * (1 - q)

            just_halted = (mass >= halt_threshold) & (steps_taken == self.config.max_steps)
            steps_taken = torch.where(just_halted, torch.full_like(steps_taken, step + 1), steps_taken)

            if torch.all(mass >= halt_threshold):
                break

            state = self._solver_step(x_features, state)

        if torch.any(mass < 1 - self.config.halt_eps):
            residual_mass = 1 - mass
            pred_accum = pred_accum + residual_mass.view(-1, 1, 1) * state.y
            mass = mass + residual_mass

        final_preds = pred_accum / mass.clamp_min(self.config.halt_eps).view(-1, 1, 1)
        return final_preds, steps_taken


class ResidualSolverLoss(nn.Module):
    """Composite loss for the residual solver."""

    def __init__(self, config: ResidualSolverConfig):
        super().__init__()
        self.config = config

    def forward(
        self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        preds = outputs["preds"]  # [B, T_out, D]
        extras = outputs["extras"]
        halt_dist = extras["halt_distribution"]  # [B, Steps]
        step_preds = extras["step_preds"]  # [B, Steps, T_out, D]
        residuals = extras["residuals"]

        # Targets: [B, T_out, D]

        task_loss = F.mse_loss(preds, targets)

        # per_step_mse: [B, Steps]
        # (step_preds - targets) -> [B, Steps, T_out, D]
        # mse over T_out, D
        diff = step_preds - targets.unsqueeze(1)
        per_step_mse = (diff ** 2).mean(dim=(-1, -2))

        weighted_step_loss = (halt_dist * per_step_mse).sum(dim=1).mean()

        residual_norm = torch.norm(residuals, dim=-1).mean() # Mean over everything
        consistency_loss = residual_norm

        steps = torch.arange(1, halt_dist.shape[1] + 1, device=halt_dist.device, dtype=halt_dist.dtype)
        expected_steps = (halt_dist * steps).sum(dim=1).mean()

        prior = torch.tensor(
            [(1 - self.config.halt_prior_p) ** i * self.config.halt_prior_p for i in range(halt_dist.shape[1])],
            device=halt_dist.device,
            dtype=halt_dist.dtype,
        )
        prior = prior / prior.sum()

        # KL
        # halt_dist [B, Steps]
        # prior [Steps]
        kl_div = (halt_dist * (halt_dist.clamp_min(self.config.halt_eps).log() - prior.unsqueeze(0).log())).sum(dim=1).mean()

        total_loss = (
            task_loss
            + self.config.lambda_step * weighted_step_loss
            + self.config.lambda_consistency * consistency_loss
            + self.config.lambda_compute * expected_steps
            + kl_div
        )

        return {
            "total_loss": total_loss,
            "task_loss": task_loss,
            "weighted_step_loss": weighted_step_loss,
            "consistency_loss": consistency_loss,
            "compute_penalty": expected_steps,
            "halting_kl": kl_div,
        }


__all__ = ["ResidualSolver", "ResidualSolverConfig", "ResidualSolverLoss"]
