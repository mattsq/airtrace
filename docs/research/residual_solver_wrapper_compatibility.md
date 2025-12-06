# Residual Solver Wrapper Compatibility Survey

This note reviews all currently registered AirTrace models for compatibility with a prospective refactor that would make `residual_solver` a **generic wrapper** around arbitrary base models (similar to `latent_ponder`). It synthesizes guidance from the existing plan and review documents and uses programmatic inspection of the model registry to scope the work for each model.

## Methodology
- Reviewed the intent and design constraints in `residual_solver_plan.md` and the implementation observations in `residual_solver_review.md`.
- Enumerated registered models via the registry (`list_models`) and inspected each class signature/source for encoder/decoder structures, heads, and non-default constructor requirements.
- Assigned a **difficulty score (1–5)** for making each model pluggable as a residual-wrapper base:
  - **1**: Already exposes the needed hooks or is itself the target model
  - **2**: Minimal refactor (factor out encoder/backbone and head)
  - **3**: Moderate refactor (split complex forward into encoder/head; add state handoff)
  - **4**: Heavy refactor (non-PyTorch or external dependency constraints)
  - **5**: Impractical without redesign (statistical models that would need full Torch reimplementation)

## Model-by-Model Notes

### Wrapper-ready with light factoring (Difficulty ≤ 3)
- **residual_solver** — Target wrapper; already structured with halting loop and MLP head. Difficulty **1/5** (no change).
- **latent_ponder** — Existing wrapper pattern; expose a shared base-model adapter to align with residual wrapper signature. Difficulty **2/5**.
- **gru_ar**, **lstm_ar** — Clear encoder (`GRU`/`LSTM`) plus linear head; pull encoder into `encode()` and head into `decode()` for reuse. Difficulty **2/5** each.
- **gru_seq2seq**, **lstm_seq2seq** — Explicit encoder/decoder classes already exist; expose `encode` + `decode_step` hooks and reuse teacher-forcing mask inside wrapper loop. Difficulty **2/5** each.
- **tcn**, **moderntcn** — Convolutional backbone with projection head; wrap backbone as encoder and keep 1-step head for refinement. Difficulty **2/5** each.
- **transformer** — Encoder/decoder stacks already separated; surface `encode` and `decode` to feed wrapper loop. Difficulty **3/5**.
- **patchtst**, **timexer**, **timemixer**, **tsmixer** — Patch/token mixer backbones with projection heads; require factoring out token encoder and forecast head. Difficulty **3/5** across the group.
- **timesnet**, **dlinear**, **nlinear** — Lightweight linear/FFT backbones with small heads; expose backbone as encoder and head as decoder. Difficulty **2/5**.
- **softs** — Explicit state-space blocks with projection head; create `encode` for feature embedder and `decode` for linear head. Difficulty **3/5**.
- **nbeats** — Stack/block architecture with final projection; surface block stack as encoder and existing head as decoder. Difficulty **3/5**.
- **frets**, **cyclenet** — Multi-stage CNN/FFT pipelines with final head; split preprocessing/feature modules from head. Difficulty **3/5** each.
- **crossformer**, **autoformer**, **fedformer**, **informer**, **nonstationary_transformer**, **itransformer** — Transformer variants with explicit encoder/decoder or projection head attributes; add thin adapter exposing `encode` and `decode`. Difficulty **3/5** across the family.
- **tft** — Already separates variable selection networks, LSTM encoder, and projection head; expose hidden/state outputs for wrapper reuse. Difficulty **3/5**.
- **mamba2**, **mambats** — Sequence-model backbones with head; need adapter to return hidden state or last token representation plus head. Difficulty **3/5**.

### Foundation or external backbones (Difficulty 4)
- **chronos_bolt**, **timer**, **lag_llama**, **moirai** — Hugging Face–style foundation models with frozen/backbone options; wrapping would require a thin adapter that forwards through their tokenizers/positional steps and reuses the HF head without breaking checkpoint loading. Difficulty **4/5** each.

### Statistical / classical baselines (Difficulty 5)
These models are implemented with NumPy/statsmodels and return predictions without Torch autograd. Making them compatible would require Torch reimplementation or a new adapter that converts inputs/outputs to tensors and threads gradients (where possible):
- **persistence**, **moving_average**, **mean**, **median**, **zero** — Simple deterministic rules; could be wrapped with Torch-only stubs but offer limited benefit. Difficulty **5/5**.
- **linear_trend**, **polynomial_trend**, **drift**, **theta** — Analytic trend fits; would need Torch parameterization to integrate with halting/gradients. Difficulty **5/5**.
- **exponential_smoothing**, **holt_linear_trend**, **holt_winters**, **seasonal_naive** — Stateful smoothing/seasonality from statsmodels; non-Torch and tightly coupled to fit/predict cycles. Difficulty **5/5**.
- **sarima**, **var** — Classical time-series estimators using statsmodels; no Torch-compatible latent state to reuse. Difficulty **5/5**.
- **linear_ar**, **mlp_ar** — PyTorch modules but structured as single-shot forecasters without exposed encoder/head separation; would need refactor to return latent state plus projection. Difficulty **4/5**.

## Recommended adapter surface
For models in the ≤3 group, add a shared mixin/protocol that exposes:
- `encode(x, context=None) -> latent` (or token features)
- `decode(latent) -> preds`
- Optional `init_state(x)` and `step(state)` when a recurrent hidden state is natural (Seq2Seq, RNNs, TFT)

Wrapper can then:
1. Call `encode` once for initialization.
2. Use `decode` for the initial prediction `y_0` and feed subsequent residual updates.
3. Optionally reuse `step` for models that can refine hidden state without re-encoding inputs.

High-difficulty baselines likely warrant a lightweight Torch adapter (no gradients) rather than full integration unless iterative refinement on statistical models is a research goal.

## Progress (2025-02-07)
- Introduced a shared `ResidualWrapperCompatible` mixin that defines `encode`/`decode` hooks expected by a generic residual wrapper.
- Refactored **gru_ar**, **lstm_ar**, and **tcn** to implement the mixin, exposing pooled latent representations and decoder reuse while keeping their original forwards for backward compatibility.
