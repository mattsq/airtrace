# AirTrace Training Pipeline Architecture

## 1. ENTRY POINT: Command Invocation

### Console Script Entry
**File:** `/home/user/airtrace/src/airtrace/cli.py` (lines 415-420)

```
airtrace train [flags] [hydra_overrides]
        ↓
cli() function
    ↓
prepare_hydra_overrides(sys.argv[1:])  [Lines 294-313]
    - Parses CLI arguments (--data-check, --dry-run, --checkpoint)
    - Converts to Hydra overrides format
    ↓
@hydra.main() decorator instantiates Hydra
    ↓
main(cfg: DictConfig) [Lines 398-412]
    - Routes to train() or evaluate() based on cfg.mode
    ↓
train(cfg: DictConfig) [Lines 35-137]
    - Main orchestrator function
```

**pyproject.toml Configuration:**
```
[project.scripts]
airtrace = "airtrace.cli:cli"
```

---

## 2. CONFIGURATION COMPOSITION (Hydra)

### Config Hierarchy
**Root Config:** `/home/user/airtrace/src/airtrace/configs/config.yaml`

```yaml
defaults:
  - data: qantas_737              # Data loading config
  - model: gru_ar                 # Model architecture
  - transforms: zscore_diff       # Data preprocessing pipeline
  - task: one_step                # Prediction objective
  - train: default                # Training hyperparameters
  - _self_

mode: train                        # "train" or "eval"
seed: 42                          # Reproducibility
log_dir: runs/${now:%Y%m%d}/${exp_name}
checkpoint: null
```

### Experiment Config Example
**File:** `/home/user/airtrace/src/airtrace/configs/exp/exp_001_gru_zscore.yaml`

```yaml
defaults:
  - override /data: qantas_737
  - override /model: gru_ar
  - override /transforms: zscore_diff
  - override /task: one_step
  - override /train: default

exp_name: "gru_zscore_one_step"
seed: 123
```

**Composition Result:**
```
cfg = {
    mode: "train",
    seed: 123,
    exp_name: "gru_zscore_one_step",
    data: {...},           # from qantas_737.yaml
    model: {...},          # from gru_ar.yaml
    transforms: {...},     # from zscore_diff.yaml
    task: {...},           # from one_step.yaml
    train: {...},          # from default.yaml
    log_dir: "runs/20240516/gru_zscore_one_step",
    checkpoint: null
}
```

---

## 3. DATA LOADING PIPELINE

### Phase: Setup & Validation

**File:** `/home/user/airtrace/src/airtrace/cli.py:train()` (lines 51-78)

```python
# Step 1: Validate data assets exist
missing_assets = _missing_data_assets(cfg.data, require_test=False)
if missing_assets:
    sys.exit(1)  # Require index parquet files
```

**Expected data structure:**
```
data/
├── processed/
│   └── {flight_id}.parquet     # Preprocessed timeseries [T, D]
├── metadata/
│   ├── train_index.parquet     # Index: [flight_id, start_idx, end_idx]
│   ├── val_index.parquet
│   ├── test_index.parquet
│   └── q400_flight_metadata.csv # Static metadata
```

### Phase: DataModule Creation

**File:** `/home/user/airtrace/src/airtrace/data/datamodule.py` (lines 21-234)

```python
# Step 2: Create DataModule
datamodule = SensorDataModule(
    data_config=cfg.data,
    transforms=transform_pipeline,
    batch_size=cfg.train.batch_size,
    num_workers=cfg.train.num_workers
)

# Step 3: Setup datasets
datamodule.setup()
```

**DataModule.__init__()** (lines 28-66):
- Reads data config (root path, sensor names, window specs)
- Creates WindowSpec from config
- Initializes DataStore (flight data loader with LRU cache)

**DataModule.setup()** (lines 73-156):
1. Reads index parquets (train, val, test)
2. Creates SensorWindowDataset for each split
3. **Fits transform pipeline on training data** (lines 99-133)
4. Computes in_dim and out_dim from sensor names

### Data Store (Caching Layer)

**File:** `/home/user/airtrace/src/airtrace/data/dataset.py` (lines 116-260)

```python
class DataStore:
    def __init__(self, data_root: Path, format: str = "parquet"):
        self._load_flight = functools.lru_cache(maxsize=128)(self._load_flight)
        # Caches up to 128 flight files in memory
    
    def get_full_window(self, flight_id, start_idx, end_idx, column_names):
        # Loads flight once, caches subsequent accesses
        flight_data = self._load_flight(flight_id)  # Cached!
        window_df = flight_data.iloc[start_idx:end_idx]
        return window_df[column_names].values, meta
```

### Window Specification

**File:** `/home/user/airtrace/src/airtrace/data/windows.py`

```python
@dataclass
class WindowSpec:
    input_len: int              # T_in (e.g., 256)
    pred_len: int               # T_out (e.g., 32)
    stride: int                 # Sliding window stride
    target_sensors: List[str]   # Output features
    
    @property
    def total_len(self) -> int:
        return input_len + pred_len
```

**Config Example:**
```yaml
# From qantas_737.yaml
data:
  window:
    input_len: 256      # 256 timesteps of history
    pred_len: 32        # Predict 32 timesteps ahead
    stride: 32          # Move window by 32 timesteps
    target_sensors: ["fuel_flow", "mach"]
  
  sensors:
    use: ["fuel_flow", "mach", "altitude", "oat", "n1"]
```

### Dataset Implementation

**File:** `/home/user/airtrace/src/airtrace/data/dataset.py` (lines 13-113)

```python
class SensorWindowDataset(Dataset):
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.index_df.iloc[idx]
        
        # Step 1: Load full window from cache
        window_data, meta = self.data_store.get_full_window(
            flight_id=row.flight_id,
            start_idx=row.start_idx,
            end_idx=row.end_idx,
            column_names=all_columns
        )  # Returns [T_total, D]
        
        # Step 2: Split into input/target using WindowSpec
        input_len = self.window_spec.input_len
        x = window_data[:input_len, sensor_indices]          # [256, 5]
        y = window_data[input_len:, target_indices]          # [32, 2]
        
        # Step 3: Apply transforms
        if self.transforms is not None:
            x, y, meta = self.transforms(x, y, meta)  # Preprocesses data
        
        # Step 4: Convert to tensors
        x = torch.from_numpy(x).float()  # [256, 5]
        y = torch.from_numpy(y).float()  # [32, 2]
        
        return {
            "x": x,      # [T_in, D_in]
            "y": y,      # [T_out, D_out]
            "meta": meta
        }
```

---

## 4. TRANSFORMS PIPELINE

### Transform Building

**File:** `/home/user/airtrace/src/airtrace/cli.py:train()` (lines 57-61)

```python
transform_pipeline = None
if "transforms" in cfg and "pipeline" in cfg.transforms:
    transform_pipeline = build_transforms(cfg.transforms.pipeline)
```

### Build Process

**File:** `/home/user/airtrace/src/airtrace/transforms/registry.py` (lines 36-65)

```python
def build_transforms(config_list: List[Dict[str, Any]]) -> Compose:
    """
    Input: [
        {"name": "zscore", "per_sensor": true, "center": true},
        {"name": "diff", "sensors": ["fuel_flow"], "order": 1},
        {"name": "context", "use_static": ["aircraft_type"]}
    ]
    
    Output: Compose([ZScoreTransform(...), DifferenceTransform(...), ...])
    """
    transforms = []
    for cfg in config_list:
        name = cfg["name"]
        cls = TRANSFORM_REGISTRY[name]  # Look up by name
        params = {k: v for k, v in cfg.items() if k != "name"}
        transforms.append(cls(**params))
    
    return Compose(transforms)
```

### Transform Fitting (Stateful)

**File:** `/home/user/airtrace/src/airtrace/data/datamodule.py` (lines 98-133)

```python
# Fit transforms on training data
if self.transforms is not None:
    # Check cache first
    cached_stats = load_transform_stats(
        cache_path, dataset_name, transform_config, index_hash
    )
    
    if cached_stats is not None:
        # Load from cache
        self.transforms.set_stats(cached_stats)
        print("[INFO] Loaded transform statistics from cache")
    else:
        # Fit from scratch
        print("Fitting transforms on training data...")
        saved_transforms = self.train_dataset.transforms
        self.train_dataset.transforms = None  # Prevent double-application
        
        # Fit: computes mean/std, min/max, etc. on training data
        self.transforms.fit(self.train_dataset)
        
        self.train_dataset.transforms = saved_transforms
        
        # Cache stats for future runs
        stats = self.transforms.get_stats()
        save_transform_stats(stats, cache_path, ...)
```

### Transform Application

**File:** `/home/user/airtrace/src/airtrace/transforms/base.py` (lines 65-147)

```python
class Compose:
    def __call__(self, x, y, meta):
        """Apply all transforms in sequence."""
        for transform in self.transforms:
            x, y, meta = transform(x, y, meta)
        return x, y, meta
```

**Example Transform Pipeline (from config):**

```yaml
transforms:
  pipeline:
    - name: zscore
      per_sensor: true          # Standardize each sensor independently
      center: true
      scale: true
    - name: diff
      sensors: ["fuel_flow"]    # First-order differencing on fuel_flow
      order: 1
    - name: context
      use_static: ["aircraft_type", "route_length"]  # Add static features
```

**Sequence:**
```
Raw input [T, D]
    ↓ (ZScoreTransform)
Standardized [T, D]  (mean=0, std=1 per sensor)
    ↓ (DifferenceTransform)
Differenced [T, D]   (fuel_flow now contains differences)
    ↓ (ContextTransform)
Context-augmented [T, D + C]  (C static features concatenated)
    ↓ to model
```

---

## 5. MODEL & TASK INSTANTIATION

### Model Building

**File:** `/home/user/airtrace/src/airtrace/cli.py:train()` (lines 89-95)

```python
model = build_model(
    config=cfg.model,
    input_dim=datamodule.in_dim,   # Computed from sensors + context
    output_dim=datamodule.out_dim  # Number of target sensors
)
```

**File:** `/home/user/airtrace/src/airtrace/models/registry.py` (lines 36-65)

```python
def build_model(config, input_dim, output_dim) -> ARBaseModel:
    name = config["name"]  # "gru_ar"
    cls = MODEL_REGISTRY[name]  # Look up class
    params = config.get("params", {})
    return cls(input_dim=input_dim, output_dim=output_dim, **params)
```

**Config Example:**
```yaml
model:
  name: gru_ar
  params:
    hidden_size: 128
    num_layers: 2
    dropout: 0.1
    bidirectional: false
    use_attention: false
```

**Model Interface:**

**File:** `/home/user/airtrace/src/airtrace/models/base.py` (lines 11-93)

```python
class ARBaseModel(nn.Module, ABC):
    def forward(
        self,
        x: torch.Tensor,              # [B, T_in, D_in]
        context: Optional[torch.Tensor] = None,  # [B, C]
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Returns dict with at least 'preds' key."""
        return {
            "preds": torch.Tensor,    # [B, T_out, D_out]
            "extras": {...}           # Optional hidden states, attention, etc.
        }
    
    def get_num_params(self) -> int:
        """Count trainable parameters."""
```

### Task Building

**File:** `/home/user/airtrace/src/airtrace/cli.py:train()` (lines 109-111)

```python
task = build_task(cfg.task)
```

**File:** `/home/user/airtrace/src/airtrace/tasks/registry.py` (lines 36-58)

```python
def build_task(config: Dict[str, Any]) -> Task:
    name = config["name"]  # "one_step"
    cls = TASK_REGISTRY[name]
    return cls(config)
```

**Task Interface:**

**File:** `/home/user/airtrace/src/airtrace/tasks/base.py` (lines 9-128)

```python
class Task(ABC):
    def __init__(self, config):
        self.loss_fn = self._build_loss_fn(config.get("loss", "mse"))
        self.metric_names = config.get("metrics", ["rmse", "mae"])
    
    @abstractmethod
    def training_step(batch, model) -> Dict[str, torch.Tensor]:
        """Execute one training step.
        
        Returns: {"loss": scalar_tensor, "rmse": float, "mae": float, ...}
        """
    
    @abstractmethod
    def validation_step(batch, model) -> Dict[str, torch.Tensor]:
        """Execute one validation step (no gradients)."""
    
    def compute_metrics(preds, targets) -> Dict[str, float]:
        """Compute evaluation metrics (rmse, mae, mape, mse)."""
```

**Example Task (One-Step):**

**File:** `/home/user/airtrace/src/airtrace/tasks/one_step.py` (lines 11-78)

```python
@register("one_step")
class OneStepTask(Task):
    def training_step(self, batch, model):
        x = batch["x"]      # [B, 256, 5]
        y = batch["y"]      # [B, 32, 2]
        
        # Forward pass
        output = model(x, meta=batch.get("meta", {}))
        preds = output["preds"]  # [B, 1, 2]
        
        # For one-step, predict only first timestep
        targets = y[:, 0:1, :]  # [B, 1, 2]
        
        # Compute loss
        loss = self.loss_fn(preds, targets)
        
        # Compute metrics
        metrics = self.compute_metrics(preds, targets)
        
        return {"loss": loss, **metrics}
```

---

## 6. DATALOADER CREATION

**File:** `/home/user/airtrace/src/airtrace/cli.py:train()` (lines 114-115)

```python
train_loader = datamodule.train_dataloader()
val_loader = datamodule.val_dataloader()
```

**File:** `/home/user/airtrace/src/airtrace/data/datamodule.py` (lines 158-190)

```python
def train_dataloader(self) -> DataLoader:
    return DataLoader(
        self.train_dataset,
        batch_size=self.batch_size,
        shuffle=True,                   # Shuffle training
        num_workers=self.num_workers,
        pin_memory=True
    )

def val_dataloader(self) -> DataLoader:
    return DataLoader(
        self.val_dataset,
        batch_size=self.batch_size,
        shuffle=False,                  # No shuffle for validation
        num_workers=self.num_workers,
        pin_memory=True
    )
```

---

## 7. TRAINING LOOP

### Trainer Initialization

**File:** `/home/user/airtrace/src/airtrace/cli.py:train()` (lines 118-125)

```python
trainer = Trainer(
    model=model,
    task=task,
    config=cfg,
    train_loader=train_loader,
    val_loader=val_loader
)
```

### Trainer Setup

**File:** `/home/user/airtrace/src/airtrace/training/trainer.py` (lines 16-87)

```python
class Trainer:
    def __init__(self, model, task, config, train_loader, val_loader, device="cuda"):
        self.model = model.to(device)
        self.task = task
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Extract training config
        self.epochs = config.train.epochs
        self.log_every_n_steps = config.train.log_every_n_steps
        
        # Build optimizer & scheduler
        self.optimizer = self._build_optimizer(config.train.optimizer)
        self.scheduler = self._build_scheduler(config.train.scheduler)
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.early_stop_counter = 0
        self.early_stop_patience = config.train.early_stopping.patience
        
        # Checkpointing
        self.checkpoint_dir = Path(config.log_dir) / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Logging
        self.log_dir = Path(config.log_dir)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
```

### Main Training Loop

**File:** `/home/user/airtrace/src/airtrace/training/trainer.py` (lines 273-323)

```python
def train(self):
    """Run full training loop."""
    for epoch in range(self.epochs):
        self.current_epoch = epoch
        
        # 1. TRAINING EPOCH
        train_metrics = self.train_epoch()
        print(f"Epoch {epoch} - Train metrics: {train_metrics}")
        
        # 2. VALIDATION EPOCH
        val_metrics = self.validate_epoch()
        print(f"Epoch {epoch} - Val metrics: {val_metrics}")
        
        # 3. LEARNING RATE SCHEDULING
        if self.scheduler is not None:
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar("lr", current_lr, epoch)
        
        # 4. CHECKPOINTING
        val_loss = val_metrics.get("loss", float('inf'))
        is_best = val_loss < self.best_val_loss
        
        if is_best:
            self.best_val_loss = val_loss
            self.early_stop_counter = 0
        else:
            self.early_stop_counter += 1
        
        self.save_checkpoint(val_loss, is_best=is_best)
        
        # 5. EARLY STOPPING
        if self.early_stop_counter >= self.early_stop_patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break
```

### Training Epoch

**File:** `/home/user/airtrace/src/airtrace/training/trainer.py` (lines 138-197)

```python
def train_epoch(self) -> Dict[str, float]:
    """Train for one epoch."""
    self.model.train()
    metric_sums = {}
    
    for batch in tqdm(self.train_loader):
        # 1. Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}
        
        # 2. Forward pass via task.training_step()
        outputs = self.task.training_step(batch, self.model)
        loss = outputs["loss"]  # Scalar tensor
        
        # 3. Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # 4. Gradient clipping
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip_max_norm
            )
        
        # 5. Optimizer step
        self.optimizer.step()
        
        # 6. Accumulate metrics
        for k, v in outputs.items():
            if k != "loss":
                metric_sums[k] = metric_sums.get(k, 0.0) + v
            else:
                metric_sums[k] = metric_sums.get(k, 0.0) + v.item()
        
        # 7. Log to TensorBoard
        if self.global_step % self.log_every_n_steps == 0:
            for k, v in outputs.items():
                val = v.item() if isinstance(v, torch.Tensor) else v
                self.writer.add_scalar(f"train/{k}", val, self.global_step)
        
        self.global_step += 1
    
    # Compute epoch averages
    epoch_metrics = {k: v / num_batches for k, v in metric_sums.items()}
    return epoch_metrics
```

### Validation Epoch

**File:** `/home/user/airtrace/src/airtrace/training/trainer.py` (lines 199-233)

```python
def validate_epoch(self) -> Dict[str, float]:
    """Validate for one epoch."""
    self.model.eval()
    metric_sums = {}
    
    for batch in tqdm(self.val_loader):
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}
        
        # No gradients needed
        outputs = self.task.validation_step(batch, self.model)
        
        # Accumulate metrics
        for k, v in outputs.items():
            metric_sums[k] = metric_sums.get(k, 0.0) + (v.item() if isinstance(v, torch.Tensor) else v)
    
    # Compute averages and log to TensorBoard
    val_metrics = {k: v / num_batches for k, v in metric_sums.items()}
    for k, v in val_metrics.items():
        self.writer.add_scalar(f"val/{k}", v, self.current_epoch)
    
    return val_metrics
```

### Checkpointing

**File:** `/home/user/airtrace/src/airtrace/training/trainer.py` (lines 235-271)

```python
def save_checkpoint(self, val_loss: float, is_best: bool = False):
    """Save model checkpoint."""
    checkpoint = {
        "epoch": self.current_epoch,
        "model_state_dict": self.model.state_dict(),
        "optimizer_state_dict": self.optimizer.state_dict(),
        "scheduler_state_dict": self.scheduler.state_dict(),
        "val_loss": val_loss,
        "config": self.config
    }
    
    # Save best checkpoint
    if is_best:
        checkpoint_path = self.checkpoint_dir / "best.ckpt"
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved best checkpoint: {checkpoint_path}")
    
    # Manage top-k checkpoints (default: save_top_k=3)
    checkpoint_path = self.checkpoint_dir / f"epoch_{self.current_epoch}.ckpt"
    torch.save(checkpoint, checkpoint_path)
    
    self.saved_checkpoints.append((val_loss, checkpoint_path))
    self.saved_checkpoints.sort(key=lambda x: x[0])
    
    # Remove worst checkpoints
    while len(self.saved_checkpoints) > self.save_top_k:
        _, path_to_remove = self.saved_checkpoints.pop()
        if path_to_remove.exists() and "best" not in path_to_remove.name:
            path_to_remove.unlink()
```

---

## 8. COMPLETE DATA FLOW DIAGRAM

```
┌─────────────────────────────────────────────────────────────────────┐
│ 1. COMMAND INVOCATION                                               │
│ $ airtrace train exp=exp_001_gru_zscore model.hidden_dim=256        │
└──────────────────────┬──────────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────────────┐
│ 2. HYDRA CONFIG COMPOSITION                                         │
│ cfg.yaml + exp_001_gru_zscore.yaml + model overrides                │
│ ↓ Merged Config                                                     │
│ {mode, seed, data, model, transforms, task, train, log_dir, ...}   │
└──────────────────────┬──────────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────────────┐
│ 3. DATA LOADING & WINDOWING                                         │
│                                                                      │
│ ┌─────────────────────────────────────────────────────────────┐    │
│ │ DataModule.setup()                                          │    │
│ │ ├─ Load train/val/test index parquets                       │    │
│ │ ├─ Create SensorWindowDataset for each split               │    │
│ │ │  └─ Associates with DataStore (LRU-cached loader)        │    │
│ │ └─ Create DataLoaders (train: shuffle=True, val: False)    │    │
│ └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│ For each batch during training:                                     │
│ ├─ SensorWindowDataset.__getitem__(idx)                             │
│ │  ├─ Get window indices from index_df[idx]                        │
│ │  ├─ DataStore.get_full_window() [flight cached in memory]       │
│ │  │  └─ Returns [T_total, D] array                                │
│ │  ├─ Split by window_spec.input_len                               │
│ │  │  ├─ x = [T_in, D_in]  (e.g., [256, 5])                       │
│ │  │  └─ y = [T_out, D_out] (e.g., [32, 2])                       │
│ │  └─ Apply transforms (next step)                                  │
│ └─ Return {"x": tensor, "y": tensor, "meta": dict}                 │
└──────────────────────┬──────────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────────────┐
│ 4. TRANSFORM PIPELINE (PER-SAMPLE)                                  │
│                                                                      │
│ Compose([ZScoreTransform, DifferenceTransform, ContextTransform])   │
│                                                                      │
│ For each sample (x, y, meta):                                       │
│ ├─ ZScoreTransform(x, y, meta)                                      │
│ │  ├─ Standardize x: (x - mean) / std [fitted on training set]    │
│ │  └─ Return (x_norm, y, meta)                                     │
│ ├─ DifferenceTransform(x_norm, y, meta)                             │
│ │  ├─ First-order difference on selected sensors                   │
│ │  └─ Return (x_diff, y, meta)                                     │
│ └─ ContextTransform(x_diff, y, meta)                                │
│    ├─ Concatenate static features (aircraft_type, route_length)    │
│    └─ Return (x_augmented, y, meta)                                │
│                                                                      │
│ Final shape: x [T_in, D_in + C_static], y [T_out, D_out]           │
└──────────────────────┬──────────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────────────┐
│ 5. BATCH COLLATION                                                  │
│                                                                      │
│ DataLoader collates N samples into batch:                           │
│ ├─ x: [B, T_in, D_in]      (e.g., [32, 256, 7])                   │
│ ├─ y: [B, T_out, D_out]    (e.g., [32, 32, 2])                    │
│ └─ meta: list of dicts                                              │
└──────────────────────┬──────────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────────────┐
│ 6. FORWARD PASS (Model + Task)                                      │
│                                                                      │
│ For each training batch:                                            │
│ ├─ model.forward(x, meta={...})  [ARBaseModel interface]            │
│ │  └─ Returns {"preds": [B, T_out, D_out], "extras": {...}}        │
│ │                                                                   │
│ ├─ task.training_step(batch, model)  [OneStepTask example]         │
│ │  ├─ preds = model(x)["preds"]      [B, 1, 2] for one-step        │
│ │  ├─ targets = y[:, 0:1, :]         [B, 1, 2] (first timestep)   │
│ │  ├─ loss = MSELoss(preds, targets)                               │
│ │  ├─ metrics = compute_metrics(preds, targets)                    │
│ │  └─ Return {"loss": tensor, "rmse": float, "mae": float, ...}   │
│ └─ Returns outputs dict                                             │
└──────────────────────┬──────────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────────────┐
│ 7. BACKWARD PASS & OPTIMIZATION                                     │
│                                                                      │
│ ├─ loss.backward()                                                  │
│ ├─ torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)   │
│ ├─ optimizer.step()  [Adam, AdamW, or SGD]                         │
│ └─ Scheduler.step() [Cosine annealing or step decay]               │
└──────────────────────┬──────────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────────────┐
│ 8. LOGGING & MONITORING                                             │
│                                                                      │
│ ├─ TensorBoard: train/loss, train/rmse, train/mae, lr per step      │
│ ├─ Console: epoch metrics summary                                   │
│ └─ Checkpoint: save_checkpoint(val_loss, is_best)                   │
│    ├─ best.ckpt (if validation loss improved)                      │
│    ├─ epoch_N.ckpt (if top-k)                                      │
│    └─ Remove old checkpoints (save_top_k=3)                        │
└──────────────────────┬──────────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────────────┐
│ 9. VALIDATION (per epoch)                                           │
│                                                                      │
│ ├─ model.eval()  [Disable dropout, batchnorm updates]               │
│ ├─ For each val batch:                                              │
│ │  └─ task.validation_step(batch, model)  [no_grad context]       │
│ ├─ Compute epoch-level metrics                                      │
│ └─ Check early stopping condition                                   │
└──────────────────────┬──────────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────────────┐
│ 10. EARLY STOPPING & CONCLUSION                                     │
│                                                                      │
│ If val_loss not improved for {patience} epochs:                     │
│ ├─ Break training loop                                              │
│ ├─ Load best.ckpt for evaluation                                    │
│ └─ Print summary & checkpoint paths                                 │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 9. KEY COMPONENT INTERACTIONS

### Configuration Contract
```
Config File Structure ←→ Python Implementation
─────────────────────────────────────────────
configs/model/gru_ar.yaml ←→ src/airtrace/models/gru_ar.py
  name: "gru_ar"             @register("gru_ar")
                             class GRUARModel(ARBaseModel): ...

configs/task/one_step.yaml ←→ src/airtrace/tasks/one_step.py
  name: "one_step"           @register("one_step")
                             class OneStepTask(Task): ...

configs/transforms/zscore_diff.yaml ←→ src/airtrace/transforms/scaling.py
  name: "zscore"             @register("zscore")
                             class ZScoreTransform(Transform): ...
```

### Registry Pattern (Used everywhere)
```python
# Define a component
from airtrace.transforms.registry import register

@register("my_transform")
class MyTransform(Transform):
    def fit(self, dataset): ...
    def __call__(self, x, y, meta): ...

# Build from config
from airtrace.transforms.registry import build_transforms
pipeline = build_transforms([{"name": "my_transform", "param": value}])
```

---

## 10. FILE STRUCTURE REFERENCE

```
├── src/airtrace/
│   ├── cli.py                          ← Entry point (train/eval commands)
│   ├── configs/
│   │   ├── config.yaml                 ← Base config
│   │   ├── exp/                        ← Experiment definitions
│   │   ├── model/                      ← Model architecture configs
│   │   ├── data/                       ← Data loading configs
│   │   ├── transforms/                 ← Transform pipeline configs
│   │   ├── task/                       ← Task definition configs
│   │   └── train/                      ← Training hyperparameter configs
│   ├── data/
│   │   ├── datamodule.py              ← DataModule orchestrator
│   │   ├── dataset.py                 ← SensorWindowDataset & DataStore
│   │   ├── windows.py                 ← WindowSpec (sliding window logic)
│   │   └── loaders.py                 ← Data format loaders
│   ├── models/
│   │   ├── base.py                    ← ARBaseModel interface
│   │   ├── registry.py                ← Model registry & build_model()
│   │   ├── gru_ar.py                  ← GRU implementation
│   │   ├── tcn.py, transformer.py, ... ← Other model architectures
│   │   └── baselines.py               ← Baseline models (Persistence, Mean, etc.)
│   ├── transforms/
│   │   ├── base.py                    ← Transform & Compose interfaces
│   │   ├── registry.py                ← Transform registry & build_transforms()
│   │   ├── scaling.py                 ← ZScoreTransform, MinMaxTransform
│   │   ├── differencing.py            ← DifferenceTransform
│   │   ├── context.py                 ← ContextTransform (static features)
│   │   ├── cache.py                   ← Transform caching utilities
│   │   └── [other transforms]
│   ├── tasks/
│   │   ├── base.py                    ← Task interface
│   │   ├── registry.py                ← Task registry & build_task()
│   │   ├── one_step.py                ← OneStepTask
│   │   ├── multi_step.py              ← MultiStepTask
│   │   └── anomaly.py                 ← AnomalyTask
│   ├── training/
│   │   ├── trainer.py                 ← Trainer main loop
│   │   ├── callbacks.py               ← Training callbacks (if any)
│   │   └── __init__.py                ← Exports set_seed()
│   └── evaluation/
│       ├── eval_runner.py             ← Evaluation orchestrator
│       └── metrics.py                 ← Metric computations

├── data/
│   ├── raw/                           ← Original CSV/Parquet files
│   ├── interim/                       ← Cleaned timeseries
│   ├── processed/                     ← Windowed tensors
│   └── metadata/                      ← Index files & flight metadata

└── runs/
    ├── 20240516/
    │   └── exp_name/
    │       ├── checkpoints/
    │       │   ├── best.ckpt
    │       │   └── epoch_*.ckpt
    │       ├── .hydra/                ← Hydra config snapshot
    │       └── events.out.tfevents... ← TensorBoard logs
```

---

## 11. EXECUTION FLOW SUMMARY

```
START: airtrace train exp=exp_001_gru_zscore
  │
  ├─→ cli() [cli.py:415]
  │    └─→ prepare_hydra_overrides()
  │         └─→ Converts "exp=exp_001_gru_zscore" → Hydra override
  │
  ├─→ @hydra.main() [cli.py:397]
  │    └─→ Loads and merges YAML configs
  │         └─→ Creates cfg: DictConfig
  │
  ├─→ main(cfg) [cli.py:398]
  │    └─→ train(cfg) [cli.py:35]
  │
  ├─→ build_transforms(cfg.transforms.pipeline) [cli.py:60]
  │    └─→ registry.build_transforms()
  │         └─→ Instantiates Compose([...transforms...])
  │
  ├─→ SensorDataModule(cfg.data, transforms, ...) [cli.py:65]
  │    ├─→ __init__(): Set up config & DataStore
  │    └─→ setup(): Create datasets, fit transforms
  │         ├─→ Load index parquets
  │         ├─→ Fit transforms on training data (with caching)
  │         └─→ Create val/test datasets
  │
  ├─→ build_model(cfg.model, in_dim, out_dim) [cli.py:90]
  │    └─→ registry.build_model()
  │         └─→ Instantiates GRUARModel(input_dim, output_dim, ...)
  │
  ├─→ build_task(cfg.task) [cli.py:110]
  │    └─→ registry.build_task()
  │         └─→ Instantiates OneStepTask(cfg.task)
  │
  ├─→ train_loader = datamodule.train_dataloader() [cli.py:114]
  │    └─→ DataLoader(train_dataset, shuffle=True, ...)
  │
  ├─→ val_loader = datamodule.val_dataloader() [cli.py:115]
  │    └─→ DataLoader(val_dataset, shuffle=False, ...)
  │
  ├─→ Trainer(model, task, cfg, ...) [cli.py:119]
  │    └─→ __init__(): Build optimizer, scheduler, logging
  │
  └─→ trainer.train() [cli.py:131]
       │
       └─→ FOR epoch IN range(epochs):
           ├─→ train_epoch()
           │    └─→ FOR batch IN train_loader:
           │         ├─→ task.training_step(batch, model)
           │         ├─→ loss.backward()
           │         ├─→ optimizer.step()
           │         └─→ Log metrics
           │
           ├─→ validate_epoch()
           │    └─→ FOR batch IN val_loader:
           │         └─→ task.validation_step(batch, model)
           │
           ├─→ scheduler.step()
           ├─→ save_checkpoint(val_loss)
           └─→ Check early stopping
       
       └─→ trainer.train() returns
           ├─→ Best checkpoint saved
           └─→ Logs directory: runs/YYYYMMDD/exp_name/

END: Training complete
```

---

## 12. KEY CONFIGURATION FILES FOR COMMON TASKS

### Run a training experiment
```bash
airtrace train exp=exp_001_gru_zscore
```

### Override model hyperparameters
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

### Dry run (no training, just config validation)
```bash
airtrace train exp=exp_001_gru_zscore --dry-run
```

### Resume from checkpoint
```bash
airtrace train exp=exp_001_gru_zscore \
  --checkpoint runs/20240516/exp_001/checkpoints/best.ckpt
```

---

## 13. IMPORTANT NOTES

1. **Config-Code Contract**: Config files in `configs/` must match Python implementations
2. **Registration Pattern**: All components (models, tasks, transforms) use @register() decorator
3. **Stateful Transforms**: Fit on training data only, then applied to all splits
4. **Caching**: Transform stats cached in `data/metadata/` to avoid refitting
5. **DataStore LRU Cache**: Flight files cached in memory (maxsize=128) for fast access
6. **WindowSpec**: Defines input/output window structure; used by dataset during __getitem__
7. **Task Interface**: Handles forward pass, loss computation, and metrics
8. **Early Stopping**: Based on validation loss with configurable patience
9. **Checkpointing**: Saves top-k checkpoints, always saves best.ckpt
10. **TensorBoard**: Training/validation metrics logged to `{log_dir}/events.out.tfevents*`

