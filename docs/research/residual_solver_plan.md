# Residual Solver with Latent Halting — AirTrace Implementation Plan

This document adapts the proposed residual solver (latent refinement + probabilistic halting) to the AirTrace framework. It maps the design to our registry/config patterns, outlines the code structure, and provides a concrete build checklist for contributors.

## Scope
- Add a new model module that performs iterative residual-based refinement with a learned halting policy.
- Keep everything configurable via Hydra and registered with the model registry.
- Provide training/inference guidance for research experiments (no production integration yet).

## AirTrace-Aligned File Layout
- **Python implementation**: `src/airtrace/models/residual_solver.py`
  - Export `ResidualSolverConfig`, `ResidualSolver`, and `ResidualSolverLoss`.
  - Register the model with `@register("residual_solver")` and ensure the config name matches.
- **Hydra config**: `configs/model/residual_solver.yaml`
  - Define the hyperparameters mirroring `ResidualSolverConfig` (input/output dims resolved from data/task configs where possible).
- **Tests**: `tests/models/test_residual_solver.py`
  - Cover shape contracts, halting distribution normalization, and expected depth behaviour under extreme biases.
- **Docs**: This file (`docs/research/residual_solver_plan.md`).

## Architecture Summary (to implement in `residual_solver.py`)
- **State**: `(h_t, y_t, r_t, step)` tracked via a `NamedTuple`.
- **Components**:
  - `ResidualNetwork`: estimates residual `r_t = R(y_t, x, h_t)`; near-zero init.
  - `UpdateNetwork`: computes refinement `Δy_t = U(r_t, h_t)`; small-step init.
  - `HiddenStateUpdate`: GRUCell over `[x, y_{t+1}]` → `h_{t+1}` for stability.
  - `ResidualAwareHaltingHead`: halting logits from `(h_t, r_t, ||r_t||, step_emb)` plus a small-residual bonus.
- **Solver cell**: executes one refinement step (residual → update → state refresh → halting logits).
- **Unrolled solver**: runs `max_steps`, computes halting distribution `P(T=t)` from per-step Bernoulli logits, returns expected prediction `y_pred`, per-step tensors, and expected steps.
- **Inference**: sequentially updates until cumulative halt probability crosses a threshold; normalizes accumulated predictions when mass is sufficient.
- **Loss**: `ResidualSolverLoss` combining task MSE on expected prediction, per-step weighted MSE, residual norm regularization, and KL to a geometric halting prior.

## Config Surface (Hydra)
Include the following keys (default suggestions in parentheses):
- Core dims: `input_dim`, `hidden_dim`, `output_dim` (set via data/task configs where possible).
- Architecture: `residual_hidden` (128), `update_hidden` (128), `n_residual_layers` (2), `n_update_layers` (2).
- Halting: `max_steps` (20), `residual_halt_threshold` (0.01), `residual_bonus_logit` (1.0), `halt_eps` (1e-6).
- Loss weights: `lambda_step` (1.0), `lambda_consistency` (0.1), `lambda_compute` (0.01).
- Halting prior: `halt_prior_p` (0.2).

Expose runtime overrides via CLI, e.g., `model.max_steps=12 model.lambda_compute=0.02`.

## Integration Checklist
1. **Implement module** (`src/airtrace/models/residual_solver.py`):
   - Add classes above with full type hints and docstrings.
   - Register model with the registry.
   - Ensure `__all__` exports and add to `src/airtrace/models/__init__.py` if required.
2. **Add Hydra config** (`configs/model/residual_solver.yaml`):
   - Defaults section matching registry key `residual_solver`.
   - Parameterize dimensions from data/task configs; set remaining defaults per Config Surface.
3. **Wire into training**:
   - Instantiate via config; optimizer/lr set in `configs/train` (leave optimizer choice to existing training configs).
   - Logging: surface `task_loss`, `weighted_step_loss`, `consistency_loss`, `compute_penalty`, and `expected_steps` in trainer metrics.
4. **Testing** (`tests/models/test_residual_solver.py`):
   - Shapes: ensure outputs `[B, output_dim]` and per-step tensors `[T, B, ...]` with `T = max_steps`.
   - Halting distribution: sums to 1 per batch; handles extreme logits (always halt vs. never halt).
   - Expected steps: decreases when `residual_bonus_logit` encourages early halting.
   - Inference: early-stop threshold respected; fallback to last state when halt mass is tiny.
5. **Docs & Registry updates**:
   - Add a short entry to `README.md` Model Registry when implemented.
   - Update `MEMORY.md` with any integration gotchas.

## Training & Inference Usage (reference snippet)
```python
from airtrace.models.residual_solver import (
    ResidualSolverConfig,
    ResidualSolver,
    ResidualSolverLoss,
)

config = ResidualSolverConfig(
    input_dim=<input_dim>,
    hidden_dim=128,
    output_dim=<output_dim>,
    max_steps=15,
    halt_prior_p=0.2,
    lambda_step=1.0,
    lambda_consistency=0.1,
    lambda_compute=0.01,
)

model = ResidualSolver(config).to(device)
loss_fn = ResidualSolverLoss(config).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

model.train()
outputs = model(x)
losses = loss_fn(outputs, y_true)
losses["total_loss"].backward()
optimizer.step()

model.eval()
with torch.no_grad():
    y_pred, steps_taken = model.inference(x_test, halt_threshold=0.9)
```

## Open Questions / Follow-ups
- Should we add auxiliary supervision on intermediate `y_t` for stability?
- Do we need a configurable minimum step count (force at least `k` iterations before halting)?
- How should compute penalties be surfaced in experiment dashboards (trainer hooks vs. tensorboard scalars)?
