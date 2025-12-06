# Residual Solver Wrapper Compatibility Review

This memo explores how to re-use the residual solver's iterative refinement and
halting machinery as a **generic wrapper** around existing AirTrace models. It
builds on the latent halting design in the implementation plan and surveys every
registered model for integration difficulty.

## Background
- The existing residual solver design (iterative residual refinement with
  probabilistic halting) is outlined in the plan. It assumes a PyTorch model
  that consumes `(x, context)` and emits `preds` plus optional `extras`.
- A generic wrapper would embed an arbitrary base model, take its predictions as
  the initial proposal, and run refinement/halting steps around it.
- Forward signatures for all registered models were collected via the model
  registry to understand their call patterns before proposing refactors.

## Wrapper assumptions
To make the residual solver act as a drop-in wrapper, base models should:
1. Accept `x: torch.Tensor` with optional `context` and expose any additional
   required kwargs clearly so the wrapper can pass through or precompute them.
2. Return at least `preds: torch.Tensor` with shape `[B, T_out, D_out]` so the
   wrapper can treat them as the initial state for refinement.
3. Be differentiable through their outputs when trained jointly with the
   wrapper's losses, or be explicitly marked as "non-trainable" so the wrapper
   can freeze them and only learn refinement layers.
4. (Optional) Expose lightweight summary features (hidden states, attention
   maps, etc.) in `extras` that the wrapper can use to initialize its hidden
   state without re-encoding `x`.

## Model-by-model compatibility notes
Difficulty legend: **Low** (wrapper works with thin glue), **Medium** (needs
minor interface shims or data plumbing), **High** (non-differentiable or
structurally mismatched).

### Transformer/sequence encoder family (mostly Low)
- **autoformer** — Standard `(x, context=None, **kwargs)` signature; wrap by
  calling once to seed refinement. Difficulty: **Low**.
- **crossformer** — Same interface and output structure; no special kwargs.
  Difficulty: **Low**.
- **cyclenet** — Standard signature; no dynamic inputs beyond `context`.
  Difficulty: **Low**.
- **dlinear**, **nlinear** — Lightweight encoders with `(x, context=None)` and
  MLP heads; easy to use as deterministic proposals. Difficulty: **Low**.
- **fedformer**, **informer**, **iTransformer**, **nonstationary_transformer**,
  **patchtst**, **transformer** — All expose `(x, context=None, **kwargs)` and
  emit dense predictions; wrapper only needs to forward kwargs. Difficulty:
  **Low**.
- **modernTCN**, **tcn**, **tsmixer**, **timemixer**, **timexer** — Temporal
  convolution/mixing stacks with standard interfaces. Difficulty: **Low**.
- **timesnet**, **softs**, **timeMixer** — Transformer-like forecasting heads
  with no extra runtime arguments. Difficulty: **Low**.
- **autoformer/fedformer duplication** — Already covered above; no additional
  work.

### RNN/Seq2Seq models
- **gru_ar**, **lstm_ar** — Single-shot RNN forecasters with `(x, context=None)`;
  can be wrapped directly. Difficulty: **Low**.
- **gru_seq2seq**, **lstm_seq2seq** — Forward requires `target` and `pred_len`
  for teacher forcing. Wrapper must standardize `pred_len` from its own config
  and optionally inject scheduled sampling targets; add a small adapter that
  strips/sets those kwargs. Difficulty: **Medium**.

### Latent halting and refinement
- **residual_solver** — Already implements refinement/halting; becomes the
  reference implementation for the generic wrapper. Difficulty: **Low** (base).
- **latent_ponder** — Existing wrapper around arbitrary `base_model` with
  ponder-style halting. Can be refactored to share wrapper utilities (base model
  factory, step logging) with residual solver so both wrappers align. Difficulty:
  **Medium**.

### Foundation and diffusion-style models
- **lag_llama** — Forward expects `retrieval_bank` and `num_samples` for the
  diffusion sampler. Wrapper needs pass-through hooks for retrieval tensors and
  a mode to disable stochastic sampling during refinement so gradients remain
  stable. Difficulty: **High**.
- **timer** — HuggingFace causal LM backend with multivariate looping; accepts
  `(x, context=None)` but downloads/checkpoints are heavy. Wrapper should allow
  frozen base weights and optional gradient stop. Difficulty: **Medium**.
- **chronos_bolt**, **moirai**, **mamba2**, **mambats**, **frets** — Foundation
  or large-sequence backbones with standard forwards but heavy parameter counts;
  wrapper should optionally freeze them and cache hidden summaries to avoid
  redundant encoding. Difficulty: **Medium**.

### Probabilistic/attention hybrids
- **tft** — Requires `known_future` and optional `static_covariates` in forward
  pass. Wrapper must surface those fields in its own signature and forward them
  unchanged, or precompute zeros to satisfy shape constraints. Difficulty:
  **Medium**.
- **chronos_bolt** — Standard interface; treat like other transformer
  derivatives. Difficulty: **Low**.
- **timeMixer/tsmixer** — (covered above) no extra work. Difficulty: **Low**.

### Linear/regression baselines
- **linear_trend**, **drift**, **mean**, **median**, **persistence**,
  **moving_average**, **zero** — Stateless, non-trainable torch ops returning
  `[B,1,D]`; wrapper can treat them as frozen base proposals and learn only
  refinement. Difficulty: **Low**.
- **linear_ar**, **mlp_ar** — Trainable torch modules with standard signatures;
  straightforward wrapping. Difficulty: **Low**.
- **polynomial_trend**, **exponential_smoothing**, **seasonal_naive**,
  **theta** — Torch/numpy hybrids with simple arithmetic; compatible as frozen
  proposals, though gradients are not meaningful. Difficulty: **Medium** (mark
  as non-trainable inside wrapper).
- **sarima** — Uses statsmodels fitting per batch; non-differentiable and
  relatively slow. Wrapper would need an evaluation-only mode or a cached fit;
  live refinement gradients are infeasible. Difficulty: **High**.
- **var** — Solves batched linear systems on the fly with torch operations;
  differentiable but heavy. Wrapper should freeze parameters and optionally
  detach outputs. Difficulty: **Medium**.

### Other architectures
- **nbeats** — Standard `(x, context=None)`; easy to wrap. Difficulty: **Low**.
- **tsmixer/timemixer** — Covered above as Low.
- **autoformer/fedformer/patchtst** — Covered above as Low.
- **softs** — Standard signature; Low difficulty.
- **dlinear/nlinear** — Already noted as Low.
- **timesnet**, **timeMixer**, **tsmixer** — Low.

## Recommended refactor path
1. Extract a **BaseProposalProtocol** that standardizes `(x, context, **extra)` →
   `preds, extras`, and implement thin adapters for the few models with extra
   kwargs (`gru_seq2seq`, `tft`, `lag_llama`).
2. Teach the wrapper to mark **non-differentiable bases** (classical baselines,
   SARIMA) as frozen and skip compute-penalty terms for their parameters.
3. Share common utilities between **LatentPonderWrapper** and the residual solver
   wrapper (base-model construction, halting metrics, optional gradient
   detachment) to keep behavior consistent across adaptive-halting models.
