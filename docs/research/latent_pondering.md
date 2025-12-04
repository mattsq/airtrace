# Latent "Pondering" / Chain-of-Thought Design for AirTrace

**Goal**: let AirTrace models perform multi-step internal reasoning in latent space ("pondering") without emitting extra tokens.
This aligns with ACT/PonderNet-style adaptive computation and latent chain-of-thought methods for time series.

## Why This Fits AirTrace
- The framework already exposes recurrent/autoregressive cores via `ARBaseModel` subclasses in `src/airtrace/models/`.
- Hydra configs make it easy to toggle extra computation at test time (`ponder_steps`, `ponder_penalty`).
- Tasks such as multi-step forecasting can benefit from extra latent refinement without changing output shapes or dataloaders.

## Proposed Architecture
1. **Ponder Core (latent update loop)**
   - Add a reusable module (e.g., `LatentPonderBlock`) that takes a hidden state and returns an updated state plus optional probes.
   - Implement it as either:
     - A small GRU/MLP/attention block (cheap to unroll), or
     - A single transformer block reused with weight tying (mirrors "depth recurrence").
   - API sketch:
     ```python
     class LatentPonderBlock(nn.Module):
         def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
             h_next = self.update(h)  # e.g., GRU cell or transformer block
             probes = {"h_norm": h_next.norm(dim=-1)}
             return h_next, probes
     ```

2. **Halting Policy (adaptive depth)**
   - Add a lightweight head on top of the hidden state that predicts a Bernoulli halt probability `p_halt = sigma(W_h h_t + b)`.
   - Train with **PonderNet** loss: encourage small expected depth while matching targets at the sampled halt step.
   - Configurable penalties: `ponder_penalty` (λ for compute), `halt_bias` initialization to control minimum steps.

3. **Wrapper Model**
   - Create `LatentPonderWrapper(ARBaseModel)` in `src/airtrace/models/latent_ponder.py` that wraps any base predictor (e.g., GRU, Transformer):
     - Encode window → initial hidden `h0` (delegates to wrapped model encoder or a small MLP).
     - Loop `t = 0..T-1` applying `LatentPonderBlock`; stop when halt sampled or max steps reached.
     - Decode final `h_t` using the wrapped model's head to produce `preds` with the usual `[B, pred_len, D]` shape.
     - Return extras: halt probabilities, depth used, intermediate probes for logging.

4. **Training Objective**
   - Primary loss: same task loss (e.g., MSE) on the final `preds`.
   - Halting loss: PonderNet KL/regularizer added to the task loss.
   - Optional deep supervision: auxiliary predictions every k steps to stabilise (`aux_weight` config).

5. **Config & Registry**
   - Register the wrapper with `@register("latent_ponder")` and add `configs/model/latent_ponder.yaml`:
     ```yaml
     defaults:
       - base_model: gru_ar
     max_steps: 6
     min_steps: 1
     ponder_penalty: 0.01
     aux_weight: 0.0
     ```
   - The config composes a child model (e.g., `base_model: transformer`) and forwards its params to the wrapped module.
   - Expose runtime overrides via CLI: `model.max_steps=8 model.ponder_penalty=0.02`.

6. **Evaluation & Logging**
   - In `airtrace/training/trainer.py`, log: average ponder steps, halt distribution, task loss vs. compute cost.
   - Add callbacks to cap depth during validation (`max_eval_steps`) and to trace per-batch ponder curves for debugging.

7. **Testing Strategy**
   - Unit tests in `tests/models/test_latent_ponder.py`:
     - Shape contracts: outputs match `[B, pred_len, D]`.
     - Halting behaviour: with `halt_bias` very negative, model should run full `max_steps`; with high bias, exit early.
     - Determinism: seeding yields reproducible depths.
   - Config test: instantiate from YAML and run a single forward on synthetic data.

## Implementation Steps (ordered)
1. Implement `LatentPonderBlock` and `LatentPonderWrapper` under `src/airtrace/models/` with full type hints.
2. Add Hydra config `configs/model/latent_ponder.yaml` composing a base model and ponder hyperparameters.
3. Extend training logs/callbacks to surface halting metrics; guard against NaNs when no halt before `max_steps`.
4. Add tests covering halt policy and shape contracts; seed randomness for deterministic assertions.
5. Update `README.md` Model Registry with the new entry after implementation.

## Variations to Consider
- **Deterministic depth**: skip halting, just unroll `k` steps; useful for ablations.
- **Cache editing**: swap `LatentPonderBlock` for a cache-augmentation module to mirror latent deliberation for LLMs.
- **RL-style objectives**: for control tasks, treat halt depth as part of the policy and optimise reward minus compute.

## Risks & Mitigations
- **Training instability**: mitigate with auxiliary losses and gradient clipping on halt head.
- **Compute overhead**: expose `max_steps`/`min_steps` and penalties so users can bound latency at inference.
- **Interface drift**: keep wrapper outputs identical to base model (`preds` + `extras`) to preserve downstream tooling.
