# Architecture

This document describes the AirTrace architecture and design principles.

## Design Principles

1. **Modularity**: Components can be swapped via configuration
2. **Extensibility**: Easy to add new models, transforms, tasks
3. **Reproducibility**: Config + seed = deterministic experiment
4. **Composability**: Mix and match components freely

## Core Abstractions

### 1. Transforms

**Base Class:** `Transform`

Transforms operate on `(x, y, meta)` tuples:
- `x`: Input sequence `[T_in, D_in]`
- `y`: Target sequence `[T_out, D_out]`
- `meta`: Metadata dictionary

**Key Methods:**
- `fit(dataset)`: Learn parameters (e.g., scaling statistics)
- `__call__(x, y, meta)`: Apply transform
- `inverse(x, y)`: Reverse transform (if applicable)

**Registration:**
```python
from airtrace.transforms import register, Transform

@register("my_transform")
class MyTransform(Transform):
    def __init__(self, param1, param2):
        ...
```

### 2. Models

**Base Class:** `ARBaseModel`

All models inherit from `torch.nn.Module` and implement:

```python
def forward(self, x, context=None) -> Dict[str, torch.Tensor]:
    """
    Args:
        x: [B, T_in, D_in]
        context: Optional [B, C] - Static features added by ContextTransform
                 (use transforms=zscore_diff_with_context to enable)

    Returns:
        {
            "preds": [B, T_out, D_out],
            "extras": {...}
        }
    """
```

**Available Models:**

Advanced models:
- `GRUARModel`: GRU-based encoder
- `TCNModel`: Temporal convolutional network
- `TransformerModel`: Causal transformer

Baseline models (for comparison):
- `PersistenceModel`: Repeats the last observed value (naive forecast)
- `ZeroModel`: Always predicts zero
- `MeanModel`: Predicts the historical mean
- `MovingAverageModel`: Predicts the average of recent values
- `LinearTrendModel`: Fits a linear trend and extrapolates

### 3. Tasks

**Base Class:** `Task`

Tasks define what "prediction" means:

```python
class Task:
    def training_step(self, batch, model) -> Dict[str, float]:
        """Compute loss and metrics for training."""

    def validation_step(self, batch, model) -> Dict[str, float]:
        """Compute loss and metrics for validation."""
```

**Available Tasks:**
- `OneStepTask`: Predict `x[t+1]` from `x[:t]`
- `MultiStepTask`: Predict `x[t+1:t+K]` with autoregression
- `AnomalyTask`: Likelihood-based anomaly scoring

## Data Flow

```
Raw Data
   ↓
RawDataLoader → Interim Data
   ↓
InterimDataProcessor → Processed Data + Index
   ↓
SensorWindowDataset
   ↓
DataLoader → Batches
   ↓
Transform Pipeline
   ↓
Model → Predictions
   ↓
Task → Loss + Metrics
```

## Training Loop

```
Trainer
  ├── Model
  ├── Task
  ├── DataLoaders
  ├── Optimizer
  ├── Scheduler
  └── Callbacks

For each epoch:
  1. Train on train_loader
  2. Validate on val_loader
  3. Update learning rate
  4. Check early stopping
  5. Save checkpoints
```

## Registry Pattern

All major components use the registry pattern:

```python
# Registration
@register("component_name")
class MyComponent:
    ...

# Building from config
component = build_component(config)
```

This enables config-driven experimentation:

```yaml
model:
  name: gru_ar
  params:
    hidden_size: 128
```

## Extension Points

### Adding a New Model

1. Create `src/airtrace/models/my_model.py`
2. Inherit from `ARBaseModel`
3. Register with `@register("my_model")`
4. Implement `forward()`
5. Create config `configs/model/my_model.yaml`

### Adding a New Transform

1. Create `src/airtrace/transforms/my_transform.py`
2. Inherit from `Transform`
3. Register with `@register("my_transform")`
4. Implement `fit()` and `__call__()`
5. Add to pipeline in `configs/transforms/`

### Adding a New Task

1. Create `src/airtrace/tasks/my_task.py`
2. Inherit from `Task`
3. Register with `@register("my_task")`
4. Implement `training_step()` and `validation_step()`
5. Create config `configs/task/my_task.yaml`

## Configuration Hierarchy

```
config.yaml (base)
  ↓
defaults:
  - data: qantas_737
  - model: gru_ar
  - transforms: zscore_diff  # or zscore_diff_with_context for static features
  - task: one_step
  - train: default
  ↓
exp/exp_001.yaml (overrides)
  ↓
Command line (final overrides)
```

Hydra composes these into a single config at runtime.
