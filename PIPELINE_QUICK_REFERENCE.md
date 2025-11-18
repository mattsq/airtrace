# AirTrace Training Pipeline - Quick Reference

## Command to Training Execution

```
$ airtrace train exp=exp_001_gru_zscore
```

1. **cli.py:cli()** → Parse CLI args
2. **cli.py:prepare_hydra_overrides()** → Convert to Hydra format
3. **@hydra.main()** → Compose YAML configs
4. **cli.py:main()** → Route to train() or eval()
5. **cli.py:train()** → Orchestrate pipeline

---

## 10 Key Files & Their Roles

| File | Role | Lines |
|------|------|-------|
| `src/airtrace/cli.py` | Entry point, orchestrator | 35-137 (train fn) |
| `src/airtrace/configs/config.yaml` | Base config structure | - |
| `src/airtrace/data/datamodule.py` | Data loading orchestration | 21-234 |
| `src/airtrace/data/dataset.py` | Window slicing, caching | 13-260 |
| `src/airtrace/data/windows.py` | WindowSpec (sliding window) | 10-85 |
| `src/airtrace/transforms/registry.py` | Build transforms from config | 36-65 |
| `src/airtrace/models/registry.py` | Build model from config | 36-65 |
| `src/airtrace/tasks/registry.py` | Build task from config | 36-58 |
| `src/airtrace/training/trainer.py` | Main training loop | 273-323 |
| `src/airtrace/training/trainer.py` | train_epoch, validate_epoch | 138-233 |

---

## The 8-Phase Pipeline

### Phase 1: Hydra Config Composition
```
config.yaml + exp_001_gru_zscore.yaml + CLI overrides
    ↓
Merged DictConfig (cfg)
```

**Resulting config has:**
- `cfg.data`: Data loading config
- `cfg.model`: Model architecture
- `cfg.transforms`: Transform pipeline
- `cfg.task`: Task definition
- `cfg.train`: Training hyperparameters

### Phase 2: Build Components from Config
```python
transform_pipeline = build_transforms(cfg.transforms.pipeline)
datamodule = SensorDataModule(cfg.data, transforms_pipeline, ...)
model = build_model(cfg.model, input_dim, output_dim)
task = build_task(cfg.task)
```

### Phase 3: Setup Data & Fit Transforms
```
datamodule.setup()
    ├─ Load index parquets (train, val, test)
    ├─ Create SensorWindowDataset for each split
    ├─ Fit transforms on training data (with caching)
    └─ Compute in_dim, out_dim from sensor names
```

### Phase 4: Create DataLoaders
```python
train_loader = datamodule.train_dataloader()  # shuffle=True
val_loader = datamodule.val_dataloader()      # shuffle=False
```

### Phase 5: Per-Sample Data Processing
```
For each batch:
    SensorWindowDataset.__getitem__(idx)
        ├─ Load window from DataStore [cached]
        ├─ Split into x [T_in, D] and y [T_out, D]
        ├─ Apply transforms (ZScore → Diff → Context)
        └─ Return {"x": tensor, "y": tensor, "meta": dict}
```

### Phase 6: Training Loop
```python
trainer = Trainer(model, task, cfg, train_loader, val_loader)
trainer.train()
    ├─ For epoch in range(epochs):
    │   ├─ train_epoch()
    │   │   └─ For batch in train_loader:
    │   │       ├─ task.training_step(batch, model)
    │   │       ├─ loss.backward()
    │   │       ├─ optimizer.step()
    │   │       └─ Log metrics
    │   ├─ validate_epoch()
    │   │   └─ For batch in val_loader:
    │   │       └─ task.validation_step(batch, model)
    │   ├─ scheduler.step()
    │   ├─ save_checkpoint(val_loss)
    │   └─ Check early stopping
    └─ Return (best checkpoint saved)
```

### Phase 7: Task Execution
```
task.training_step(batch, model):
    x = batch["x"]         # [B, T_in, D_in]
    y = batch["y"]         # [B, T_out, D_out]
    
    output = model(x, meta=batch["meta"])
    preds = output["preds"]  # [B, T_out, D_out]
    
    loss = loss_fn(preds, targets)
    metrics = compute_metrics(preds, targets)
    
    return {"loss": loss, "rmse": float, "mae": float, ...}
```

### Phase 8: Checkpointing
```
save_checkpoint(val_loss, is_best=True)
    ├─ Save to {log_dir}/checkpoints/best.ckpt
    ├─ Save to {log_dir}/checkpoints/epoch_N.ckpt
    └─ Keep only top_k checkpoints
```

---

## Data Flow: From File to Model

```
data/processed/{flight_id}.parquet  [Raw timeseries, T×D]
    ↓ (DataStore LRU cache)
In-memory flight data
    ↓ (Window slicing)
[T_in+T_out, D] window
    ↓ (Split by WindowSpec.input_len)
x: [256, 5]  y: [32, 2]
    ↓ (Transforms)
x: [256, 7]  y: [32, 2]  (after context features added)
    ↓ (Batch collation)
x: [B, 256, 7]  y: [B, 32, 2]
    ↓ (Model forward)
preds: [B, T_pred, D_out]
    ↓ (Task loss)
loss: scalar
```

---

## Configuration Examples

### Running a specific experiment
```bash
airtrace train exp=exp_001_gru_zscore
```

### Override hyperparameters
```bash
airtrace train exp=exp_001_gru_zscore \
  model.params.hidden_size=256 \
  train.batch_size=32 \
  train.epochs=100
```

### Data validation only
```bash
airtrace train exp=exp_001_gru_zscore --data-check
```

### Resume from checkpoint
```bash
airtrace train exp=exp_001_gru_zscore \
  --checkpoint runs/20240516/exp_001/checkpoints/best.ckpt
```

---

## Key Design Patterns

### 1. Registry Pattern
All extensible components use a registry with @register() decorator:

```python
@register("zscore")
class ZScoreTransform(Transform):
    def fit(self, dataset): ...
    def __call__(self, x, y, meta): ...
```

Building from config:
```python
transform = build_transforms([{"name": "zscore", "per_sensor": true}])
```

### 2. Config-Code Contract
Every YAML config corresponds to a Python class:

```
configs/model/gru_ar.yaml ←→ models/gru_ar.py:GRUARModel
configs/task/one_step.yaml ←→ tasks/one_step.py:OneStepTask
configs/transforms/zscore_diff.yaml ←→ transforms/scaling.py:ZScoreTransform
```

### 3. Dataclass with LRU Cache
Flight data cached in memory for fast repeated access:

```python
class DataStore:
    def __init__(self, data_root):
        self._load_flight = functools.lru_cache(maxsize=128)(self._load_flight)
```

### 4. Stateful Transform Fitting
Transforms fit once on training data, then applied identically to all splits:

```python
# In setup():
if cached_stats exists:
    transforms.set_stats(cached_stats)  # Load from cache
else:
    transforms.fit(train_dataset)       # Fit from scratch
    save_transform_stats(stats)         # Cache for future runs

# Then applied per-sample during __getitem__
x, y, meta = transforms(x, y, meta)
```

---

## Output Directory Structure

After running training:

```
runs/
└── 20240516/
    └── exp_name/
        ├── .hydra/
        │   ├── config.yaml          ← Resolved config
        │   ├── hydra.yaml
        │   └── overrides.yaml
        ├── checkpoints/
        │   ├── best.ckpt            ← Best validation loss
        │   ├── epoch_0.ckpt
        │   └── epoch_1.ckpt
        └── events.out.tfevents*     ← TensorBoard logs
                                        (train/loss, val/loss, etc.)
```

---

## Common Debugging Points

### Issue: Data validation fails
**Check:** `data/metadata/` has parquet index files matching config

### Issue: Transform fitting fails
**Check:** Training dataset has at least 1 sample; transforms have valid fit() methods

### Issue: Model forward pass fails
**Check:** input_dim matches (sensors + context features), output_dim matches target_sensors

### Issue: Task training_step fails
**Check:** Model returns dict with "preds" key; preds shape matches expected output

### Issue: Training is slow
**Check:** num_workers setting in datamodule; DataStore cache size; GPU memory

---

## Important Notes

1. **Window Spec**: Controls how raw flight data is split into [T_in, T_out] windows
2. **Transform Stats**: Computed only on training data to prevent data leakage
3. **Early Stopping**: Based on validation loss with configurable patience
4. **Checkpointing**: Saves top-k checkpoints (default k=3) and always saves best.ckpt
5. **Gradient Clipping**: Enabled by default (max_norm=1.0)
6. **LR Scheduling**: Cosine annealing (default) or step decay
7. **Reproducibility**: Set by seed in config; affects random splits, dropout, etc.

