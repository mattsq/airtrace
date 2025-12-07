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

## Progress (2025-02-08)
- Refactored **transformer** to surface `encode`/`decode` adapters so the encoder stack can be reused by a residual pondering loop while maintaining the existing single-step forward path.
- Refactored **timesnet** to provide wrapper hooks; the decode step enforces the configured `pred_len` to preserve the existing projection head contract while making the backbone reusable.

## Progress (2025-02-09)
- Refactored **dlinear** and **nlinear** to implement `ResidualWrapperCompatible` by threading decomposition/centering signals through an `encode` representation and reusing their projection heads inside `decode`, while preserving configured `pred_len` contracts.
- Updated **tsmixer** to expose normalized mixer outputs via `encode` and reuse the temporal/head projections in `decode`, enabling drop-in residual pondering without altering the existing forward behavior.

## Progress (2025-02-10)
- Refactored **patchtst** to surface patched-token encodings through `encode` and reuse its channel-level projection head in `decode`, enabling repeated residual updates while keeping the single-step default contract.
- Refactored **timexer** to expose the fused endogenous/exogenous token representations via `encode` and gate decoding through the existing prediction head, aligning cross-attention handling with the residual wrapper interface.
- Refactored **timemixer** to provide multi-scale pooled predictions as the wrapper latent and reuse the ensemble projection inside `decode`, ensuring the multi-horizon contract remains enforced.

## Progress (2025-02-11)
- Refactored **moderntcn** to implement `ResidualWrapperCompatible`, exposing the convolutional backbone through `encode` and the projection head via `decode` while preserving its autoregressive single-step default.
- Refactored **softs** to add `encode`/`decode` hooks that thread normalization statistics through the wrapper loop, guarding the configured horizon during decoding and maintaining de-normalized outputs.

## Progress (2025-02-12)
- Refactored **gru_seq2seq** and **lstm_seq2seq** to inherit from `ResidualWrapperCompatible`, exposing the RNN encoders via `encode` and reusing the stepwise decoder logic through `decode` to support residual pondering while retaining teacher-forcing behavior in the standard forward pass.

## Progress (2025-02-13)
- Refactored **nbeats** to implement `ResidualWrapperCompatible`, factoring the block stack computation into reusable `encode`/`decode` hooks so stacked forecasts can be reused by a residual pondering wrapper without changing the default forward outputs.
- Updated **frets** to surface its FFT/iFFT pipeline through `encode` and reuse the temporal/output projections in `decode`, enabling residual refinement on the time-domain latent while preserving the original forecasting contract.

## Progress (2025-02-14)
- Refactored **crossformer** to implement `ResidualWrapperCompatible`, exposing the pooled token representation via `encode` and reusing its projection head inside `decode` while guarding the configured horizon.
- Refactored **informer** to provide wrapper hooks that cache the start token during `encode` and reuse the distilled encoder memory within `decode`, enabling the residual solver loop without altering the ProbSparse decoder flow.

## Progress (2025-02-15)
- Refactored **autoformer** to implement `ResidualWrapperCompatible`, caching the decoder start tokens and encoder trend from `encode` and reusing the seasonal/trend decomposition inside `decode` so the residual solver loop can refine outputs without re-embedding the window.
- Refactored **fedformer** with the same wrapper hooks, aligning its frequency-enhanced decomposition and cached decoder tail with the generic residual solver interface while preserving the configured horizon checks and extras reporting.

## Progress (2025-02-16)
- Refactored **tft**, **mamba2**, **mambats**, **cyclenet**, **itransformer**, and **nonstationary_transformer** to implement `ResidualWrapperCompatible`, exposing reusable encoder/decoder hooks while caching normalization, cycle, and attention artifacts required to reproduce their existing forward contracts under a generic residual pondering wrapper.

## Progress (2025-02-17)
- Updated **latent_ponder** to consume `ResidualWrapperCompatible` bases through shared `encode`/`decode` hooks while retaining the legacy forward fallback, aligning pondering initialization with the generic residual solver wrapper.
- Refactored **linear_ar** and **mlp_ar** to implement `ResidualWrapperCompatible`, flattening window inputs into reusable latents and sharing their projection heads inside `decode` so lightweight baselines can participate in residual pondering loops.

## Progress (2025-02-18)
- Refactored **chronos_bolt**, **lag_llama**, and **moirai** to implement `ResidualWrapperCompatible`, exposing pooled foundation backbones and reusing their projection heads through `encode`/`decode` adapters while guarding normalization, horizon, and retrieval hooks for a generic residual pondering wrapper.

## Progress (2025-02-19)
- Refactored **timer** to inherit `ResidualWrapperCompatible`, ensuring the Hugging Face backbone and input normalization caches are surfaced through `encode`/`decode` so a generic residual pondering wrapper can reuse cached representations while preserving the existing forward contract.

## Progress (2025-02-20)
- Refactored baseline models **persistence**, **moving_average**, **zero**, **linear_trend**, **mean**, **median**, **drift**, **exponential_smoothing**, and **seasonal_naive** to implement `ResidualWrapperCompatible`, exposing their deterministic statistics through `encode`/`decode` so the residual solver wrapper can reuse aligned outputs across pondering steps without altering their single-step default behaviors.
