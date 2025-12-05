"""Residual solver with iterative refinement and probabilistic halting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ARBaseModel
from .registry import register


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
    """Iteratively refines predictions with residual updates and learned halting."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
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
        **_: Dict[str, object],
    ) -> None:
        super().__init__(input_dim, output_dim)

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

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.initial_decoder = nn.Linear(hidden_dim, output_dim)
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

    def _init_state(self, x: torch.Tensor) -> SolverState:
        x_summary = x.mean(dim=1)
        h0 = torch.tanh(self.input_proj(x_summary))
        y0 = self.initial_decoder(h0)
        r0 = torch.zeros_like(y0)
        return SolverState(h=h0, y=y0, r=r0, step=0)

    def _compute_halt_logits(self, state: SolverState) -> torch.Tensor:
        residual_norm = torch.norm(state.r, dim=-1, keepdim=True)
        step_ids = torch.full_like(residual_norm, state.step, dtype=torch.long)
        step_emb = self.step_embedding(step_ids.squeeze(-1))
        halt_features = torch.cat([state.h, state.r, residual_norm, step_emb], dim=-1)
        logits = self.halting_head(halt_features).squeeze(-1)
        small_residual = (residual_norm.squeeze(-1) < self.config.residual_halt_threshold).float()
        logits = logits + self.config.residual_bonus_logit * small_residual
        return logits

    def _solver_step(self, x_summary: torch.Tensor, state: SolverState) -> SolverState:
        residual_inp = torch.cat([state.y, x_summary, state.h], dim=-1)
        r_t = self.residual_net(residual_inp)
        delta = self.update_net(torch.cat([r_t, state.h], dim=-1))
        y_next = state.y + delta
        h_inp = torch.cat([x_summary, y_next], dim=-1)
        h_next = self.hidden_update(h_inp, state.h)
        return SolverState(h=h_next, y=y_next, r=r_t, step=state.step + 1)

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
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None, **_: Dict[str, object]
    ) -> Dict[str, torch.Tensor]:
        del context
        batch, _, _ = x.shape
        x_summary = x.mean(dim=1)
        state = self._init_state(x)

        step_preds: List[torch.Tensor] = []
        residuals: List[torch.Tensor] = []
        halt_logits: List[torch.Tensor] = []

        for _ in range(self.config.max_steps):
            logits = self._compute_halt_logits(state)
            halt_logits.append(logits)
            step_preds.append(state.y)
            residuals.append(state.r)
            state = self._solver_step(x_summary, state)

        halt_logits_tensor = torch.stack(halt_logits, dim=1)
        step_preds_tensor = torch.stack(step_preds, dim=1)  # [B, T, D]
        residual_tensor = torch.stack(residuals, dim=1)

        halt_dist, _ = self._halting_distribution(halt_logits_tensor)
        expected_preds = (halt_dist.unsqueeze(-1) * step_preds_tensor).sum(dim=1)
        expected_steps = (
            halt_dist * torch.arange(1, self.config.max_steps + 1, device=x.device)
        ).sum(dim=1)

        preds = expected_preds.unsqueeze(1)
        extras = {
            "halt_logits": halt_logits_tensor,
            "halt_distribution": halt_dist,
            "step_preds": step_preds_tensor,
            "residuals": residual_tensor,
            "expected_steps": expected_steps,
        }

        return {"preds": preds, "extras": extras}

    @torch.no_grad()
    def inference(self, x: torch.Tensor, halt_threshold: float = 0.9) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, _, _ = x.shape
        x_summary = x.mean(dim=1)
        state = self._init_state(x)

        survival = torch.ones(batch, device=x.device)
        mass = torch.zeros(batch, device=x.device)
        pred_accum = torch.zeros(batch, self.output_dim, device=x.device)
        steps_taken = torch.full((batch,), self.config.max_steps, device=x.device, dtype=torch.long)

        for step in range(self.config.max_steps):
            logits = self._compute_halt_logits(state)
            q = torch.sigmoid(logits)
            p_halt = survival * q
            pred_accum = pred_accum + p_halt.unsqueeze(-1) * state.y
            mass = mass + p_halt
            survival = survival * (1 - q)

            just_halted = (mass >= halt_threshold) & (steps_taken == self.config.max_steps)
            steps_taken = torch.where(just_halted, torch.full_like(steps_taken, step + 1), steps_taken)

            if torch.all(mass >= halt_threshold):
                break

            state = self._solver_step(x_summary, state)

        if torch.any(mass < 1 - self.config.halt_eps):
            residual_mass = 1 - mass
            pred_accum = pred_accum + residual_mass.unsqueeze(-1) * state.y
            mass = mass + residual_mass

        final_preds = pred_accum / mass.clamp_min(self.config.halt_eps)
        return final_preds.unsqueeze(1), steps_taken


class ResidualSolverLoss(nn.Module):
    """Composite loss for the residual solver."""

    def __init__(self, config: ResidualSolverConfig):
        super().__init__()
        self.config = config

    def forward(
        self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        preds = outputs["preds"]  # [B, 1, D]
        extras = outputs["extras"]
        halt_dist = extras["halt_distribution"]  # [B, T]
        step_preds = extras["step_preds"]  # [B, T, D]
        residuals = extras["residuals"]

        targets_flat = targets.squeeze(1)

        task_loss = F.mse_loss(preds.squeeze(1), targets_flat)

        per_step_mse = ((step_preds - targets_flat.unsqueeze(1)) ** 2).mean(dim=-1)
        weighted_step_loss = (halt_dist * per_step_mse).sum(dim=1).mean()

        residual_norm = torch.norm(residuals, dim=-1)
        consistency_loss = residual_norm.mean()

        steps = torch.arange(1, halt_dist.shape[1] + 1, device=halt_dist.device, dtype=halt_dist.dtype)
        expected_steps = (halt_dist * steps).sum(dim=1).mean()

        prior = torch.tensor(
            [(1 - self.config.halt_prior_p) ** i * self.config.halt_prior_p for i in range(halt_dist.shape[1])],
            device=halt_dist.device,
            dtype=halt_dist.dtype,
        )
        prior = prior / prior.sum()
        kl_div = (halt_dist * (halt_dist.clamp_min(self.config.halt_eps).log() - prior.log())).sum(dim=1).mean()

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
