from typing import Callable

import torch
import torch.nn.functional as F


def trm_halting_loss(
    halt_logits: torch.Tensor,
    step_preds: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """TRM-style halting: q_t predicts 'is current prediction correct?'.

    Args:
        halt_logits: [B, T] halting logits from the wrapper.
        step_preds:  [B, T, pred_len, D] per-step logits from the wrapper.
        targets:     [B, pred_len] integer class targets.
    """
    B, T, pred_len, D = step_preds.shape

    with torch.no_grad():
        step_correct = []
        for t in range(T):
            logits_t = step_preds[:, t]  # [B, pred_len, D]
            y_t = logits_t.argmax(dim=-1)  # [B, pred_len]
            correct_t = (y_t == targets).all(dim=-1).float()  # [B]
            step_correct.append(correct_t)
        step_correct = torch.stack(step_correct, dim=1)  # [B, T]

    q = torch.sigmoid(halt_logits)  # [B, T]
    bce = F.binary_cross_entropy(q, step_correct)

    # Optional small entropy regulariser to avoid degenerate q
    entropy = -(q * (q + eps).log() + (1 - q) * (1 - q + eps).log())
    halting_entropy = entropy.mean()

    return bce + 0.01 * halting_entropy


def pondernet_loss(
    halt_logits: torch.Tensor,
    step_preds: torch.Tensor,
    targets: torch.Tensor,
    base_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ponder_penalty: float,
) -> torch.Tensor:
    """PonderNet-style expected loss + expected steps.

    Args:
        halt_logits: [B, T]
        step_preds:  [B, T, pred_len, D]
        targets:     [B, pred_len]
        base_loss_fn: loss function taking (logits, targets) and returning
                      per-example, per-position loss with reduction="none".
        ponder_penalty: scalar weight for expected number of steps.
    """
    B, T, pred_len, D = step_preds.shape

    s = halt_logits
    q = torch.sigmoid(s)  # q_t = P(halt at t | not halted before)
    c = 1 - q  # c_t = P(continue at t)

    # P(not halted before t): cumprod over previous c's, with leading 1
    c_prefix = torch.cumprod(
        torch.cat(
            [torch.ones(B, 1, device=c.device, dtype=c.dtype), c[:, :-1]],
            dim=1,
        ),
        dim=1,
    )  # [B, T]

    p_halt = q * c_prefix  # [B, T]
    p_rest = c_prefix[:, -1]  # [B]  prob of never halting

    # Per-step task losses
    per_step_loss = []
    for t in range(T):
        logits_t = step_preds[:, t]  # [B, pred_len, D]
        loss_t = base_loss_fn(logits_t, targets)  # expect reduction="none": [B, pred_len]
        # Average over sequence dimension -> [B]
        loss_t = loss_t.mean(dim=list(range(1, loss_t.ndim)))
        per_step_loss.append(loss_t)
    per_step_loss = torch.stack(per_step_loss, dim=1)  # [B, T]

    # Expected task loss
    exp_task_loss = (p_halt * per_step_loss).sum(dim=1).mean()

    # Expected number of steps: E[T] = sum_t p_halt_t * t + P(no halt)*T_max
    steps = torch.arange(1, T + 1, device=s.device, dtype=s.dtype)  # [T]
    exp_steps = (p_halt * steps).sum(dim=1) + p_rest * T
    exp_steps = exp_steps.mean()

    return exp_task_loss + ponder_penalty * exp_steps
