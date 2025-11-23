# AirTrace Agent Guide

**Welcome, AI Agent!** This guide helps you navigate AirTrace and work effectively within its framework.

## What is AirTrace?

AirTrace is a **modular, config-driven framework** for autoregressive timeseries modeling on aircraft sensor data. The core philosophy is:

1. **Config-driven**: Everything is defined through YAML configs, composed via Hydra
2. **Modular**: Models, transforms, tasks are pluggable components with registered interfaces
3. **Reproducible**: One config + one seed = one deterministic experiment
4. **Type-safe**: Heavy use of type hints, validated with mypy

## Python Environment Setup

**CRITICAL**: This project uses a local virtual environment managed by `uv`.

### Using the Virtual Environment

**ALWAYS use the virtual environment** for all Python commands:

```bash
# On Windows (Git Bash/MINGW):
source .venv/Scripts/activate

# On Linux/macOS:
source .venv/bin/activate
```

**OR** prefix Python commands with the venv path:

```bash
# Windows:
.venv/Scripts/python -m pytest tests/
.venv/Scripts/python -m airtrace.cli train exp=exp_001

# Linux/macOS:
.venv/bin/python -m pytest tests/
.venv/bin/python -m airtrace.cli train exp=exp_001
```

### Package Management with uv

**ALWAYS use `uv` for package management** (NOT pip):

```bash
# Install/update dependencies (use --link-mode=copy on Windows):
uv pip install -e ".[dev]" --link-mode=copy

# Add a new dependency:
# 1. Edit pyproject.toml
# 2. Run: uv pip install -e ".[dev]" --link-mode=copy

# Sync environment (after pulling changes):
uv pip install -e ".[dev]" --link-mode=copy
```

### Important Notes

- **DO NOT** use system Python or create new venvs
- **DO NOT** use `pip` directly - always use `uv pip`
- On Windows with OneDrive/cloud-synced drives, **always use `--link-mode=copy`**
- The venv is at `.venv/` in the project root (gitignored)
- If `.venv/` is missing or broken, recreate it:
  ```bash
  uv venv .venv
  uv pip install -e ".[dev]" --link-mode=copy
  ```

## Quick Orientation

```
airtrace/
├── configs/          # YAML configs composed via Hydra (THE BRAIN)
│   ├── data/        # Dataset configs (window sizes, sensors)
│   ├── model/       # Model architectures (GRU, TCN, Transformer)
│   ├── transforms/  # Data preprocessing pipelines
│   ├── task/        # Prediction objectives (one-step, multi-step, anomaly)
│   ├── train/       # Training hyperparameters
│   └── exp/         # Complete experiment definitions
├── src/airtrace/    # Python implementation
│   ├── data/        # Dataset loaders, windowing logic
│   ├── transforms/  # Transform implementations (scaling, differencing)
│   ├── models/      # Model implementations (must inherit ARBaseModel)
│   ├── tasks/       # Task implementations (must inherit Task)
│   ├── training/    # Training loop, callbacks
│   ├── evaluation/  # Metrics, evaluation loops
│   └── viz/         # Visualization utilities
├── data/            # Data files (gitignored, ephemeral)
│   ├── raw/         # Original flight logs
│   ├── interim/     # Cleaned timeseries
│   ├── processed/   # Windowed tensors
│   └── metadata/    # Sensor metadata
├── tests/           # pytest tests
├── notebooks/       # Jupyter analysis notebooks
└── docs/            # Extended documentation
```

## Core Framework Principles

### 1. The Config-Code Contract

**RULE**: Config structure mirrors code structure. When you see:

```yaml
# configs/exp/my_experiment.yaml
defaults:
  - override /data: qantas_737
  - override /model: gru_ar
  - override /transforms: zscore_diff
  - override /task: one_step
```

This means:
- `configs/data/qantas_737.yaml` exists
- `src/airtrace/data/` has a registered dataset loader
- `configs/model/gru_ar.yaml` exists
- `src/airtrace/models/` has a registered "gru_ar" model class
- And so on...

**Never break this contract.** If you add a model, you need BOTH:
1. Python implementation in `src/airtrace/models/your_model.py`
2. Config in `configs/model/your_model.yaml`

### 2. The Registration Pattern

All components use a registration decorator:

```python
from airtrace.registry import register

@register("my_model")
class MyModel(ARBaseModel):
    def __init__(self, input_dim, hidden_dim, ...):
        ...
```

**RULE**: The string in `@register("my_model")` must match the config filename `configs/model/my_model.yaml`.

### 3. Base Classes Define Interfaces

- **Models**: Inherit from `ARBaseModel` (see `src/airtrace/models/base.py`)
- **Transforms**: Inherit from `Transform` (see `src/airtrace/transforms/base.py`)
- **Tasks**: Inherit from `Task` (see `src/airtrace/tasks/base.py`)

**DO NOT** bypass these interfaces. They ensure:
- Consistent `forward()` signatures
- Proper `fit()` and `transform()` for stateful transforms
- Standardized loss computation and metrics

### 4. Data Pipeline Stages

```
RAW → INTERIM → PROCESSED → DATALOADER
```

1. **Raw** (`data/raw/`): Original CSVs/Parquet from aircraft, untouched
2. **Interim** (`data/interim/`): Cleaned, resampled, aligned timeseries per flight
3. **Processed** (`data/processed/`): Sliding windows [T_in, D] → [T_out, D_target]
4. **DataLoader**: PyTorch DataLoader with batching, shuffling

**RULE**: Never modify raw data. Interim/processed are cache directories and can be regenerated.

### 5. Experiment Naming Convention

Experiments follow: `exp_NNN_MODEL_TRANSFORM_TASK`

Example: `exp_001_gru_zscore_onestep`
- `001`: Experiment number (incremental)
- `gru`: Model type
- `zscore`: Transform type
- `onestep`: Task type

**RULE**: Experiment configs live in `configs/exp/`. Keep them organized and numbered.

## Navigation Cheatsheet

### "Where do I find...?"

| What | Where | Key Files |
|------|-------|-----------|
| Model definitions | `src/airtrace/models/` | `base.py`, `gru.py`, `tcn.py`, `transformer.py` |
| Model configs | `configs/model/` | `gru_ar.yaml`, `tcn.yaml` |
| Data loading logic | `src/airtrace/data/` | `dataset.py`, `windowing.py` |
| Transform implementations | `src/airtrace/transforms/` | `base.py`, `scaling.py`, `differencing.py` |
| Training loop | `src/airtrace/training/` | `trainer.py`, `callbacks.py` |
| Evaluation metrics | `src/airtrace/evaluation/` | `metrics.py` |
| Entry point (CLI) | `src/scripts/` | `train.py`, `eval.py` |
| Dependencies | Root | `pyproject.toml` |
| Documentation | `docs/` | `index.md`, `architecture.md`, `data_format.md` |
| Model-specific docs | `docs/models/` | `chronos_bolt.md`, `softs_*.md` |
| Research/proposals | `docs/research/` | `model_proposals.md` |
| Archived docs | `docs/archive/` | Completed implementation reports |

### "How do I...?"

**Add a new model:**
1. Create `src/airtrace/models/my_model.py`
2. Inherit `ARBaseModel`, implement `__init__`, `forward`
3. Add `@register("my_model")` decorator
4. Create `configs/model/my_model.yaml` with hyperparameters
5. Add tests in `tests/models/test_my_model.py`
6. **Update the Model Registry section in `README.md`** with your model's details (name, class, description)
7. Update `docs/architecture.md` if novel

**Add a new transform:**
1. Create `src/airtrace/transforms/my_transform.py`
2. Inherit `Transform`, implement `fit`, `transform`, `inverse_transform`
3. Add `@register("my_transform")` decorator
4. Create `configs/transforms/my_pipeline.yaml` using it
5. Add tests in `tests/transforms/test_my_transform.py`

**Run an experiment:**
```bash
airtrace train exp=exp_001_gru_zscore
```

**Override from CLI:**
```bash
airtrace train exp=exp_001_gru_zscore model.hidden_dim=256 train.epochs=100
```

**Add a dependency:**
```bash
# 1. Edit pyproject.toml to add the dependency
# 2. Reinstall with uv (use --link-mode=copy on Windows):
uv pip install -e ".[dev]" --link-mode=copy
```

## Common Pitfalls (DON'T DO THIS)

### ❌ Hardcoding paths
```python
# BAD
data = pd.read_csv("/home/user/airtrace/data/raw/flight.csv")
```

```python
# GOOD
from pathlib import Path
data_dir = Path(cfg.data.raw_dir)  # From Hydra config
data = pd.read_csv(data_dir / "flight.csv")
```

### ❌ Breaking the config-code contract
```python
# BAD: Adding a model without a config
@register("my_model")
class MyModel(ARBaseModel):
    ...
# (but no configs/model/my_model.yaml)
```

### ❌ Skipping type hints
```python
# BAD
def process_window(data):
    return data.mean()
```

```python
# GOOD
def process_window(data: torch.Tensor) -> torch.Tensor:
    return data.mean(dim=0)
```

### ❌ Mutating raw data
```python
# BAD
df = pd.read_csv("data/raw/flight.csv")
df["mach"] = df["mach"].fillna(0)  # Mutating in place
df.to_csv("data/raw/flight.csv")  # NEVER overwrite raw!
```

### ❌ Ignoring the registry
```python
# BAD: Direct import instead of registry lookup
from airtrace.models.gru import GRUModel
model = GRUModel(...)
```

```python
# GOOD: Using the registry (allows config-driven instantiation)
from airtrace.registry import get_model
model = get_model(cfg.model.name, **cfg.model.params)
```

### ❌ Adding a model without updating README
```python
# BAD: Adding a new model but forgetting to update README.md
@register("my_new_model")
class MyNewModel(ARBaseModel):
    ...
# (but README.md Model Registry section is not updated)
```

```markdown
# GOOD: After adding a model, update README.md Model Registry section
# Add an entry to the appropriate table with:
#   - Model name (registry key, e.g., "my_new_model")
#   - Class name (e.g., "MyNewModel")
#   - Brief description of what the model does
#   - Whether it's trainable (for baseline models)
```

## Development Workflow

1. **Before starting**: Read `docs/architecture.md` and relevant component docs
2. **Make changes**: Follow the patterns above
3. **Test**: `pytest tests/` (all tests must pass)
4. **Format**: `black src/ tests/` (code must be formatted)
5. **Lint**: `ruff src/ tests/` (no lint errors)
6. **Type check**: `mypy src/` (must pass type checking)
7. **Document**: Update docstrings, add to `docs/` if needed
8. **Commit**: Clear, descriptive messages

### Testing Best Practices

- Aim for **end-to-end or integration-level coverage** when possible so tests validate real behavior instead of assumptions.
- **Never skip, mock, simple-assert, or monkey patch** when a more thorough test can exercise the actual code path and validate the intended outcome.
- Prefer **realistic fixtures and data** over synthetic shortcuts; mirror the way configs and registries are used in production.
- Keep assertions **behavior-focused** (outputs, side effects, invariants) rather than implementation details.
- When you must isolate dependencies, **justify the mock** in a comment and keep the surface area minimal.

## Memory and Learnings

**IMPORTANT**: When you discover something surprising, unexpected, or non-obvious about this codebase that future agents should know, **add it to `MEMORY.md`**.

Examples of what to log:
- "The windowing logic assumes fixed-length flights; variable-length requires padding in dataset.py:145"
- "GRU models expect [batch, seq, features] but TCN expects [batch, features, seq] - conversion happens in task.py:67"
- "Config overrides don't merge lists, they replace them entirely"
- "The transform pipeline applies in order: scaling → differencing → context, inverse goes backward"

See `MEMORY.md` for the running list of agent learnings.

## When In Doubt

1. **Check the existing code**: Grep for similar patterns
2. **Read the base classes**: They define the interface contracts
3. **Look at configs**: They show how components compose
4. **Check tests**: They demonstrate expected usage
5. **Read docs/architecture.md**: Explains the design philosophy
6. **Ask the user**: If something is genuinely ambiguous

## Project Standards

- **Python**: 3.9+
- **Style**: Black (line length 100)
- **Linting**: Ruff
- **Type checking**: mypy (strict mode)
- **Testing**: pytest with >=80% coverage target
- **Config**: Hydra 1.3+
- **ML Framework**: PyTorch 2.0+
- **Data**: Pandas, NumPy for preprocessing; PyTorch tensors for models

## File Creation Rules

**Only create files in these locations:**

- ✅ `src/airtrace/*/` - Python modules (with `__init__.py` updates)
- ✅ `configs/*/` - YAML configs
- ✅ `tests/*/` - Test files (named `test_*.py`)
- ✅ `notebooks/` - Jupyter notebooks (`.ipynb`)
- ✅ `docs/` - Documentation (`.md`)
- ❌ **NEVER** create files in `data/` (ephemeral, gitignored)
- ❌ **NEVER** create files at the root unless absolutely necessary

## Summary: The Three Laws of AirTrace

1. **Config is King**: Every component has a config. No magic values.
2. **Interfaces are Sacred**: Inherit from base classes, implement required methods.
3. **Reproducibility First**: Same config + same seed = same result, always.

---

*For other AI agents: See `GEMINI.md` or `AGENTS.md` - they point here.*
*For discovered insights: See `MEMORY.md`*
*For architecture details: See `docs/architecture.md`*
