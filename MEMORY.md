# AirTrace Agent Memory

This file captures **surprising, unexpected, or non-obvious learnings** discovered by AI agents working on AirTrace. Future agents should read this before making significant changes.

## Purpose

When you discover something that:
- Contradicts initial assumptions
- Has subtle implications for how components interact
- Represents a gotcha or edge case
- Would save future agents debugging time
- Reveals hidden dependencies or constraints

**Add it here!** Use this format:

```
## [Date] [Component/Area]: Brief Title

**Discovered by**: [Agent name/session]
**Impact**: [What this affects]

Description of the learning...

**Example/Code Reference**: [file:line if relevant]
```

---

## Agent Learnings

### 2025-11-14: Project Structure: Config-Code Synchronization Required

**Discovered by**: Initial setup agent
**Impact**: All components (models, transforms, tasks, data)

The framework enforces a strict 1:1 mapping between config files and registered Python components. When adding any component:

1. Python file with `@register("component_name")` decorator
2. YAML config file with matching name: `configs/category/component_name.yaml`
3. Base class inheritance (ARBaseModel, Transform, or Task)

Breaking this contract causes Hydra composition failures at runtime, not import time.

**Example**: If you create `src/airtrace/models/lstm.py` with `@register("lstm")`, you MUST create `configs/model/lstm.yaml`.

---

### 2025-11-14: Project Philosophy: Modular Config-Driven Framework

**Discovered by**: Initial setup agent
**Impact**: Overall development approach

AirTrace is fundamentally a **framework for composing experiments**, not a monolithic training script. Key insights:

1. **Hydra composition**: Experiments are built by composing small config files
2. **Registry pattern**: All components discovered via decorator registration
3. **Interface contracts**: Base classes enforce consistent APIs across components
4. **Reproducibility**: Config + seed = deterministic experiment

This means: Don't hardcode, don't bypass registration, don't break interfaces.

---

### 2025-11-14: Data Pipeline: Three-Stage Transformation

**Discovered by**: Initial setup agent
**Impact**: Data loading and preprocessing

Data flows through exactly three stages:

1. **RAW** (`data/raw/`): Immutable source of truth, never modified
2. **INTERIM** (`data/interim/`): Cleaned, resampled timeseries (can be regenerated)
3. **PROCESSED** (`data/processed/`): Windowed tensors ready for DataLoader (cached)

INTERIM and PROCESSED are caches and can be deleted/regenerated. RAW is sacred.

**Gotcha**: If you modify raw data processing logic, you MUST delete `data/interim/` and `data/processed/` to force regeneration. Stale caches cause silent errors.

---

## How to Use This File

### As a Reader (Every Agent Should Do This)

1. **Read this file** before starting significant work
2. Search for keywords related to your task (model, transform, data, etc.)
3. If you find relevant learnings, factor them into your approach

### As a Writer (When You Discover Something)

1. Use the template format above
2. Be specific: Include file paths, line numbers, code examples
3. Explain the **impact** and **why it matters**
4. Keep it concise but complete
5. Add to the bottom of the "Agent Learnings" section

### Examples of What to Log

✅ **DO LOG**:
- "Model forward() expects [B, T, D] but optimizer state assumes [B, D, T]"
- "Config list overrides replace entire list, not merge - affects transform pipelines"
- "Tests require GPU if model uses .cuda() even with torch.device('cpu')"
- "Hydra instantiate() doesn't work with custom __init__ patterns in X"

❌ **DON'T LOG**:
- Obvious things documented in README.md or CLAUDE.md
- Personal preferences or style choices
- Things already clear from type hints or docstrings
- Temporary hacks you plan to fix immediately

---

## Template for New Entries

```markdown
## [YYYY-MM-DD] [Component]: Brief Description

**Discovered by**: [Agent/Session ID]
**Impact**: [What areas of the codebase this affects]

[Detailed explanation of the learning, including:]
- What you expected
- What actually happens
- Why this matters
- How to work with it

**Example/Code Reference**: [file.py:line]

**Related**: [Links to other MEMORY.md entries if relevant]
```

---

## Maintenance Notes

- **Keep this file organized**: Newer entries at the bottom of "Agent Learnings"
- **Remove outdated entries**: If a learning becomes obsolete (e.g., code refactored), strike through and note: `~~OLD: ...~~ [Fixed in commit abc123]`
- **Link to code**: When referencing specific behavior, include file:line references
- **Search before adding**: Check if someone already logged a similar learning

---

### 2025-11-14: Synthetic Data: Physics-Based Cruise Generator

**Discovered by**: Synthetic dataset generator implementation
**Impact**: Data generation, testing, experimentation

A physics-based synthetic data generator was added for aircraft cruise sensor readings. Key insights:

1. **Physical relationships modeled**:
   - Fuel flow ∝ N1 (engine thrust)
   - Weight decreases linearly as fuel burns
   - OAT follows ISA model with altitude
   - N1 depends on weight, altitude, and speed

2. **Follows standard pipeline**: Generated data goes through Raw → Interim → Processed like real data

3. **Reproducibility**: Same seed produces identical data, critical for experiments

4. **Limitations**: Only models stable cruise conditions, NOT takeoff/descent/maneuvers/anomalies

**Usage**:
```python
from airtrace.data.synthetic import create_synthetic_dataset
splits = create_synthetic_dataset(data_root="data/", n_flights=20, seed=42)
```

**Code Reference**: `src/airtrace/data/synthetic.py`, `configs/data/synthetic_cruise.yaml`

**Related**: Data Pipeline learning (three-stage transformation)

---

### 2025-11-14: Models: Baseline Models for Benchmarking

**Discovered by**: Baseline models implementation
**Impact**: Model evaluation, experiment comparisons, research validity

Five simple baseline models were added to establish performance floors for deep learning models:

1. **PersistenceModel**: Repeats last value - surprisingly strong for autocorrelated data
2. **ZeroModel**: Always predicts zero - floor baseline
3. **MeanModel**: Predicts historical mean - good for stationary processes
4. **MovingAverageModel**: Averages recent k values - handles noise
5. **LinearTrendModel**: Linear extrapolation - captures trends

**Key implementation details**:

- **Minimal parameters**: Baselines have ≤ `input_dim × output_dim` params (just projection if dims differ)
- **No training needed**: Predictions computed directly in forward pass, no optimizer required
- **Deterministic**: Same input always produces same output (important for reproducibility)
- **Dimension handling**: When `input_dim != output_dim`, baselines use a learnable linear projection

**Usage pattern**:
```yaml
# configs/model/persistence.yaml
model:
  name: persistence
  params: {}
```

**Why this matters**:
- Deep learning models MUST beat these baselines (especially Persistence) to be useful
- Provides interpretable reference points: if GRU ≈ Persistence, model isn't learning
- Validates experimental setup: if Zero > Persistence, something is wrong with preprocessing

**Code Reference**: `src/airtrace/models/baselines.py`, `configs/model/{persistence,zero,mean,moving_average,linear_trend}.yaml`

**Testing**: Comprehensive tests in `tests/test_models.py` verify correctness (e.g., persistence actually returns last value)

**Documentation**: Full guide at `docs/baseline_models.md` with usage examples and interpretation guidelines

**CI/CD Integration**: The model validation script (`src/scripts/validate_models.py`) was updated to handle models with zero trainable parameters. It detects parameter-free models and skips optimizer creation/backprop, instead computing only forward pass and loss.

---

### 2025-11-15 Models/Moirai: Default `pred_len` must stay at 1 for CI

**Discovered by**: gpt-5-codex session
**Impact**: Model validation script, any user instantiating `MoiraiModel` without overrides

The CI model validation script (`src/scripts/validate_models.py`) builds every registered model with an empty parameter dictionary and feeds single-step targets (`pred_horizon=1`). When `MoiraiModel` defaulted to `pred_len=24`, `nn.MSELoss` silently broadcast the one-step targets during training but the metric computation (`compute_all_metrics(preds.flatten(), targets.flatten())`) crashed with `operands could not be broadcast together with shapes (...)`. Keeping the model's default `pred_len` at 1 prevents this mismatch while configs (e.g., `configs/model/moirai.yaml`) can still set larger horizons explicitly for experiments.

**Example/Code Reference**: `src/airtrace/models/moirai.py`, `src/scripts/validate_models.py`

---

### 2026-03-10 Dependencies: NumPy 2.x breaks PyTorch wheels

**Discovered by**: gpt-5.1-codex session
**Impact**: Package installation, CLI entrypoints

Installing with unconstrained dependencies can pull NumPy 2.x, but current PyTorch wheels are built
against NumPy 1.x. The mismatch triggers import-time errors such as `A module that was compiled
using NumPy 1.x cannot be run in NumPy 2.x` when running `airtrace` commands. Pinning NumPy to
`<2.0.0` in `pyproject.toml` and installing via `uv` keeps environments on a compatible version.

**Example/Code Reference**: `pyproject.toml` dependency pin, README installation note

---

### 2025-11-19 Training Loop: Verbose progress stalls from RandomSampler

**Discovered by**: gpt-5-codex session
**Impact**: CLI training UX, Trainer `verbose_progress`

`SensorDataModule.train_dataloader()` always instantiates the training `DataLoader`
with `shuffle=True`, which means PyTorch attaches a `RandomSampler` that creates a
`torch.randperm(len(dataset))` tensor every time a new epoch begins. When
`train.verbose_progress=true`, the trainer wraps the loader in `tqdm` and waits for
the sampler to finish generating that permutation before the progress bar can
render, so any dataset with tens or hundreds of thousands of windows can appear
to "hang" for a noticeable period at the start of each training/validation epoch.

To confirm this, I replicated the synthetic cruise dataset 200× (≈291k windows)
and measured the time to fetch the first batch with and without shuffling. The
shuffled loader (which builds the `randperm`) took ~0.205s before the first batch
arrived, while the ordered loader started in ~0.069s – the extra delay aligns with
the sampler cost and matches the pause seen before tqdm renders in verbose mode.

**Example/Code Reference**: `src/airtrace/data/datamodule.py` lines 168-193,
shuffle timing script output `89a1ae†L1-L7`

---

### 2025-11-23 Models/Timer: HuggingFace Foundation Model Integration Pattern

**Discovered by**: Claude (Timer implementation session)
**Impact**: Foundation model integration, HuggingFace dependencies, multivariate handling

Timer is the first HuggingFace-based foundation model in AirTrace, establishing patterns for future foundation model integrations:

**Key Integration Learnings**:

1. **HuggingFace requires `trust_remote_code=True`**: Timer (and likely other HF time series models) use custom modeling code that requires explicit trust. This is a security consideration that must be documented clearly for users.

2. **Multivariate handling for univariate models**: Timer is pre-trained on univariate series in S3 (Single-Series Sequence) format. For multivariate aircraft data (6-30 sensors), we process each dimension independently:
   ```python
   # Process each variate separately
   for d in range(D):
       series_d = x[:, :, d]  # [B, T]
       preds_d = timer_backbone.generate(series_d, max_new_tokens=pred_len)
       predictions.append(preds_d)
   # Stack: [B, pred_len, D]
   ```
   This approach maintains compatibility with pre-trained weights while supporting multivariate data.

3. **Normalization is critical**: Timer expects normalized inputs. The model applies per-variate z-score normalization and stores statistics for denormalization. Without normalization, generation can be unstable.

4. **Model size and dependencies**: Timer-base-84M is ~350MB and requires `transformers>=4.40.1`. First run downloads the checkpoint (cached to `~/.cache/huggingface/`). This should be documented for offline/production deployments.

5. **Testing without downloads**: Use mock fixtures (`@patch("airtrace.models.timer.AutoModelForCausalLM")`) to test integration without downloading checkpoints during CI/CD.

**Code Reference**: `src/airtrace/models/timer.py`, `tests/models/test_timer.py`, `docs/models/timer.md`

**Future models**: This pattern applies to TimesFM (Google), MOMENT (CMU), and other HuggingFace time series models.

---

### 2025-12-15 Tasks/Models: Task-Model Shape Compatibility and Broadcasting Errors

**Discovered by**: Claude Sonnet 4.5 (MSE broadcasting investigation session)
**Impact**: OneStepTask, AnomalyTask, all multi-step models (DLinear, NLinear, Transformers, etc.)

**Problem**: Models like DLinear and NLinear output `[B, pred_len, D]` where `pred_len` comes from `${data.window.pred_len}` (typically 16-32 timesteps). When used with OneStepTask, this caused MSE broadcasting warnings because:
- Model output: `[B, pred_len, D]` (e.g., [64, 32, 30])
- Target: `[B, 1, D]` (e.g., [64, 1, 30])
- MSELoss would broadcast and average incorrectly across the `pred_len` dimension

**The Fix**: OneStepTask (and AnomalyTask) now explicitly slice model outputs to first timestep:
```python
preds = output["preds"][:, :1, :]  # Force [B, 1, D]
```

**Models Affected**: 25+ models use `pred_len` parameter configured via `${data.window.pred_len}`:
- **Decomposition**: DLinear, NLinear
- **Transformers**: Informer, Autoformer, FEDFormer, Nonstationary Transformer
- **Patching/Channel-Independent**: PatchTST, iTransformer, Crossformer, Timexer
- **CNN/Hybrid**: TimesNet, TimeMixer, TSMixer, CycleNet, FRETS, ModernTCN
- **Foundation**: Timer, Moirai, Lag-Llama, Chronos-Bolt
- **State Space**: Mamba2, MambaTS, S-Mamba
- **Advanced**: TFT, N-BEATS, Latent Ponder, SOFTS

All of these are compatible with OneStepTask after the fix (commit 62ba942, PR #197).

**Key Learnings**:

1. **Models are task-agnostic**: They output their configured `pred_len` regardless of task type
   - DLinear with `pred_len=32` will always output [B, 32, D]
   - Task is responsible for adapting model outputs to its requirements

2. **Tasks handle shape slicing**:
   - **OneStepTask**: Slices to `[:, :1, :]` (first timestep)
   - **MultiStepTask**: Uses `[:, -1:, :]` (last timestep) in autoregressive loop
   - **AnomalyTask**: Slices to `[:, 0:1, :]` (first timestep)

3. **Config coupling**: `pred_len: ${data.window.pred_len}` couples model to data config
   - This is intentional for multi-step tasks
   - OneStepTask now safely handles any `pred_len` value via slicing

4. **Silent failures prevented**: Added `_validate_pred_shape()` method to Task base class
   - Validates predictions match targets before loss computation
   - Provides clear error messages if shapes mismatch
   - Helps catch configuration errors early

**Testing**: Comprehensive regression and integration tests added:
- `tests/test_one_step_shape_bug.py` - Reproduces original bug and verifies fix
- `tests/test_dlinear_one_step_integration.py` - Tests DLinear/NLinear with various `pred_len` values
- Both test suites ensure no MSE broadcasting warnings occur

**Code Reference**:
- Fix: `src/airtrace/tasks/one_step.py:47`
- Validation: `src/airtrace/tasks/base.py:166-196`
- Tests: `tests/test_one_step_shape_bug.py`, `tests/test_dlinear_one_step_integration.py`

**Related**: Moirai `pred_len` learning (similar shape mismatch issue in model validation)

---

## Current State

**Total learnings**: 10
**Last updated**: 2025-12-15
**Most active areas**: Project structure, configuration system, data pipeline, synthetic data, baseline models, model validation, dependency management, foundation model integration, task-model compatibility

---

*This file is a living document. Every agent contributes to making AirTrace easier to work with.*
