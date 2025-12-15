# Scaling Laws Experiment: Expert Critique & Revised Plan

## Executive Summary

Your scaling laws experiment has good foundations but needs refinement. The main issues are:
- **Training duration too short** (10 epochs insufficient for convergence)
- **Original "Large" model is CPU-prohibitive** (3.5M params - would take hours per experiment)
- **Model size naming inconsistency** (POC "Medium" = Full plan "Small")
- **Missing systematic parameter tracking**

**Recommendation**: Use the revised CPU-friendly configuration below, train for 30 epochs with early stopping, and execute in two phases.

---

## Critical Issues with Current Plan

### 1. Insufficient Training Duration ❌

**POC**: 10 epochs with no early stopping
**Problem**: Models need to reach convergence for valid scaling laws. Underfitting gives you "laws" that measure training inefficiency, not true capacity.

**Evidence from deep learning literature**:
- Kaplan et al. (2020) "Scaling Laws for Neural Language Models": Trained models to convergence
- Hestness et al. (2017): "Models must be trained sufficiently to reach their performance ceiling"
- Your own early stopping config: patience=10 suggests 10 epochs isn't enough

**Solution**: Train for 30-50 epochs with early stopping (patience=10). This ensures:
- Small models converge quickly (stop early)
- Large models get enough time
- No wasted compute on already-converged models

### 2. Original "Large" Model is Impractical ❌

```
Original Large: d_model=256, e_layers=3, d_layers=2 → 3,562,507 params
```

**Problems**:
- 3.5M parameters on CPU will be extremely slow (~2-4 hours per experiment)
- 12 experiments × 3 hours = 36+ hours total
- Doesn't fit the "fast, local, CPU-bound" requirement

**Comparison**:
- Your current default Informer config: ~250K params (reasonable for CPU)
- Original Large: 14x bigger than default!

### 3. Parameter Range Analysis

| Configuration | Small | Medium | Large | Range |
|--------------|--------|---------|--------|--------|
| **Original Plan** | 85K | 516K | 3.5M | 41.8x |
| **Proposed CPU-Friendly** | 85K | 292K | 516K | 23.3x |
| **Scaling Laws Best Practice** | - | - | - | 30-100x |

The original plan's range is good (41x), but achieved via an impractical Large model. The proposed plan gets 23x range with all CPU-friendly models (all <550K params).

### 4. Data Range Limitations ⚠️

```
Data sizes: 10% (745) → 100% (7458) = 10x range
```

**Ideal**: 100x range for robust power law fits
**Reality**: Limited by dataset size (998 flights total)

**Mitigation**:
- Could add 5% (373 samples) for 20x range
- Focus on model scaling (easier to vary)
- Accept that data scaling laws will be less robust

### 5. Missing Systematic Tracking ❌

Current plan doesn't specify tracking:
- Exact parameter counts per configuration
- Training time per experiment
- Validation loss trajectory
- Best validation loss + epoch achieved
- Configuration hash for reproducibility

**Solution**: Create a results tracking system (see Implementation Plan below)

---

## Detailed Parameter Analysis

### POC Results (10 epochs, already run)

| Model | d_model | Params | Val Loss (10%) | Val Loss (20%) |
|-------|---------|--------|----------------|----------------|
| Tiny  | 16      | 5,947  | 0.9601         | 0.9569         |
| Small | 32      | 22,123 | 0.9529         | 0.9478         |
| Medium| 64      | 85,195 | 0.9237         | 0.5248         |

**Observations**:
- Only Medium model shows strong improvement with more data (0.92 → 0.52)
- Tiny/Small barely improve (likely underfitting even with more data)
- Suggests models need at least 60-80K params for this task

### Original Full Plan (Analysis)

| Model | d_model | e_layers | Params | CPU Time Est. |
|-------|---------|----------|--------|---------------|
| Small | 64      | 1        | 85K    | ~15 min       |
| Medium| 128     | 2        | 516K   | ~60 min       |
| Large | 256     | 3        | 3.5M   | ~180 min      |

**Issues**:
- "Small" is same as POC "Medium" (confusing naming)
- Large model: 3.5M params is CPU-prohibitive
- Uneven parameter growth: 6x then 7x (should be geometric)

### Proposed CPU-Friendly Plan ✅

| Model | d_model | nhead | e_layers | d_layers | ff_dim | Params | Growth | CPU Time Est. |
|-------|---------|-------|----------|----------|--------|--------|--------|---------------|
| Tiny  | 32      | 2     | 1        | 1        | 64     | 22K    | -      | ~10 min       |
| Small | 64      | 2     | 1        | 1        | 128    | 85K    | 3.9x   | ~20 min       |
| Medium| 96      | 4     | 2        | 1        | 192    | 292K   | 3.4x   | ~40 min       |
| Large | 128     | 4     | 2        | 1        | 256    | 516K   | 1.8x   | ~60 min       |

**Advantages**:
- All models <550K params (CPU-friendly)
- Smooth parameter progression (~3x per step)
- 23.3x total range (good for scaling laws)
- Estimated total time: 8-12 hours for all 16 experiments
- Largest model is same as original Medium (proven CPU-viable)

---

## Recommended Experiment Design

### Configuration Matrix

| Dimension | Levels | Range | Notes |
|-----------|--------|-------|-------|
| **Data** | 10%, 20%, 50%, 100% | 10x | Limited by dataset |
| **Model** | Tiny, Small, Medium, Large | 23x | CPU-friendly |
| **Epochs** | 30 (early stop patience=10) | - | Ensures convergence |
| **Total Experiments** | 16 | - | 4 × 4 grid |

### Why This Design Works

1. **Parameter range (23x)**: Sufficient for power law fitting (Kaplan et al. used 20x-100x)
2. **All CPU-friendly**: Largest model (516K) is reasonable for CPU training
3. **Early stopping**: Prevents wasting compute on converged models
4. **Geometric progression**: Smooth curves for interpolation
5. **Data range (10x)**: Acceptable given dataset constraints

### What Good Scaling Laws Require

From Kaplan et al. (2020), Hestness et al. (2017), and Hoffmann et al. (2022):

✅ **Models trained to convergence** → Using early stopping (patience=10)
✅ **Wide parameter range** → 23x range (Kaplan used 20x-100x)
✅ **Wide data range** → 10x (limited by dataset, acceptable)
✅ **Multiple data points** → 4 model sizes × 4 data sizes = 16 points
✅ **Consistent training** → Same LR, batch size, optimizer across runs
✅ **Validation loss tracking** → Track throughout training
⚠️ **Multiple seeds** → Skipping for speed (can add later if needed)

---

## Execution Strategy (Two-Phase Approach)

### Phase 1: Validation (Fast, ~3-4 hours)

**Goal**: Validate pipeline and check for scaling trends

**Experiments** (8 total):
- Tiny model: 10%, 20%, 50%, 100% data
- Small model: 10%, 20%, 50%, 100% data

**Success criteria**:
1. Training completes without errors
2. Validation loss decreases with more data
3. Larger model (Small) outperforms smaller (Tiny)
4. Early stopping triggers appropriately

**If Phase 1 fails**: Debug before wasting compute on larger models

### Phase 2: Complete Analysis (~5-8 hours)

**Goal**: Complete the scaling law analysis

**Experiments** (8 total):
- Medium model: 10%, 20%, 50%, 100% data
- Large model: 10%, 20%, 50%, 100% data

**Deliverables**:
1. 16 trained models with validation losses
2. Scaling law plots (data scaling & model scaling)
3. Power law exponents (α for data, β for model)
4. Projections for target performance

---

## Implementation Plan

### 1. Create Results Tracking System

```python
# scripts/track_scaling_results.py
import json
import pandas as pd
from pathlib import Path

def log_experiment(exp_name, config, results):
    """Log experiment results to CSV for analysis."""
    log_file = Path("outputs/scaling_laws_results.csv")

    row = {
        "exp_name": exp_name,
        "model_size": config["size"],
        "d_model": config["d_model"],
        "params": config["params"],
        "data_pct": config["data_pct"],
        "data_samples": config["data_samples"],
        "epochs_run": results["epochs_run"],
        "train_loss": results["final_train_loss"],
        "val_loss": results["final_val_loss"],
        "best_val_loss": results["best_val_loss"],
        "best_epoch": results["best_epoch"],
        "training_time_min": results["training_time_min"],
    }

    df = pd.DataFrame([row])
    if log_file.exists():
        df.to_csv(log_file, mode='a', header=False, index=False)
    else:
        df.to_csv(log_file, index=False)
```

### 2. Create Revised Experiment Runner

```powershell
# scripts/run_scaling_laws.ps1

$python = ".venv\Scripts\python"

function Run-Experiment {
    param (
        [string]$data_idx,
        [string]$exp_suffix,
        [int]$d_model,
        [int]$nhead,
        [int]$e_layers,
        [int]$d_layers,
        [int]$ff_dim
    )

    $exp_name = "scaling_$exp_suffix"
    Write-Host "Starting Experiment: $exp_name" -ForegroundColor Cyan

    $start_time = Get-Date

    $cmdArgs = @(
        "-m", "airtrace.cli", "train",
        "model=informer",
        "data=descent_data",
        "data.train_index=$data_idx",
        "model.params.d_model=$d_model",
        "model.params.nhead=$nhead",
        "model.params.e_layers=$e_layers",
        "model.params.d_layers=$d_layers",
        "model.params.ff_dim=$ff_dim",
        "exp_name=$exp_name",
        "train.epochs=30",                    # Increased from 10
        "train.early_stopping.patience=10"    # Enable early stopping
    )

    & $python $cmdArgs

    $elapsed = ((Get-Date) - $start_time).TotalMinutes
    Write-Host "Completed in $([math]::Round($elapsed, 1)) minutes" -ForegroundColor Green
}

# PHASE 1: Tiny + Small models
Write-Host "`n=== PHASE 1: VALIDATION (Tiny + Small Models) ===" -ForegroundColor Yellow

# Tiny model (22K params)
Run-Experiment -data_idx "metadata/descent_train_10pct.parquet" -exp_suffix "tiny_10pct" -d_model 32 -nhead 2 -e_layers 1 -d_layers 1 -ff_dim 64
Run-Experiment -data_idx "metadata/descent_train_20pct.parquet" -exp_suffix "tiny_20pct" -d_model 32 -nhead 2 -e_layers 1 -d_layers 1 -ff_dim 64
Run-Experiment -data_idx "metadata/descent_train_50pct.parquet" -exp_suffix "tiny_50pct" -d_model 32 -nhead 2 -e_layers 1 -d_layers 1 -ff_dim 64
Run-Experiment -data_idx "metadata/descent_train_100pct.parquet" -exp_suffix "tiny_100pct" -d_model 32 -nhead 2 -e_layers 1 -d_layers 1 -ff_dim 64

# Small model (85K params)
Run-Experiment -data_idx "metadata/descent_train_10pct.parquet" -exp_suffix "small_10pct" -d_model 64 -nhead 2 -e_layers 1 -d_layers 1 -ff_dim 128
Run-Experiment -data_idx "metadata/descent_train_20pct.parquet" -exp_suffix "small_20pct" -d_model 64 -nhead 2 -e_layers 1 -d_layers 1 -ff_dim 128
Run-Experiment -data_idx "metadata/descent_train_50pct.parquet" -exp_suffix "small_50pct" -d_model 64 -nhead 2 -e_layers 1 -d_layers 1 -ff_dim 128
Run-Experiment -data_idx "metadata/descent_train_100pct.parquet" -exp_suffix "small_100pct" -d_model 64 -nhead 2 -e_layers 1 -d_layers 1 -ff_dim 128

Write-Host "`n=== PHASE 1 COMPLETE ===" -ForegroundColor Green
Write-Host "Review results before proceeding to Phase 2" -ForegroundColor Yellow
Write-Host "If results look good, run scripts/run_scaling_laws_phase2.ps1" -ForegroundColor Yellow
```

### 3. Create Phase 2 Script

```powershell
# scripts/run_scaling_laws_phase2.ps1

# Same Run-Experiment function...

Write-Host "`n=== PHASE 2: FULL ANALYSIS (Medium + Large Models) ===" -ForegroundColor Yellow

# Medium model (292K params)
Run-Experiment -data_idx "metadata/descent_train_10pct.parquet" -exp_suffix "medium_10pct" -d_model 96 -nhead 4 -e_layers 2 -d_layers 1 -ff_dim 192
Run-Experiment -data_idx "metadata/descent_train_20pct.parquet" -exp_suffix "medium_20pct" -d_model 96 -nhead 4 -e_layers 2 -d_layers 1 -ff_dim 192
Run-Experiment -data_idx "metadata/descent_train_50pct.parquet" -exp_suffix "medium_50pct" -d_model 96 -nhead 4 -e_layers 2 -d_layers 1 -ff_dim 192
Run-Experiment -data_idx "metadata/descent_train_100pct.parquet" -exp_suffix "medium_100pct" -d_model 96 -nhead 4 -e_layers 2 -d_layers 1 -ff_dim 192

# Large model (516K params)
Run-Experiment -data_idx "metadata/descent_train_10pct.parquet" -exp_suffix "large_10pct" -d_model 128 -nhead 4 -e_layers 2 -d_layers 1 -ff_dim 256
Run-Experiment -data_idx "metadata/descent_train_20pct.parquet" -exp_suffix "large_20pct" -d_model 128 -nhead 4 -e_layers 2 -d_layers 1 -ff_dim 256
Run-Experiment -data_idx "metadata/descent_train_50pct.parquet" -exp_suffix "large_50pct" -d_model 128 -nhead 4 -e_layers 2 -d_layers 1 -ff_dim 256
Run-Experiment -data_idx "metadata/descent_train_100pct.parquet" -exp_suffix "large_100pct" -d_model 128 -nhead 4 -e_layers 2 -d_layers 1 -ff_dim 256
```

### 4. Create Analysis Script

```python
# scripts/analyze_scaling_laws.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def fit_power_law(x, y):
    """Fit power law: y = C * x^(-alpha)"""
    log_x = np.log(x)
    log_y = np.log(y)
    coeffs = np.polyfit(log_x, log_y, 1)
    alpha = -coeffs[0]
    C = np.exp(coeffs[1])
    return alpha, C

def main():
    results = pd.read_csv("outputs/scaling_laws_results.csv")

    # Data Scaling: Fix model size, vary data
    print("=== DATA SCALING LAWS ===")
    for model_size in ["tiny", "small", "medium", "large"]:
        subset = results[results["model_size"] == model_size]
        if len(subset) >= 3:
            alpha, C = fit_power_law(subset["data_samples"], subset["best_val_loss"])
            print(f"{model_size.capitalize()}: L = {C:.4f} * D^(-{alpha:.4f})")

    # Model Scaling: Fix data size, vary model
    print("\n=== MODEL SCALING LAWS ===")
    for data_pct in [10, 20, 50, 100]:
        subset = results[results["data_pct"] == data_pct]
        if len(subset) >= 3:
            beta, C = fit_power_law(subset["params"], subset["best_val_loss"])
            print(f"{data_pct}% data: L = {C:.4f} * P^(-{beta:.4f})")

    # Create plots
    # ... (plotting code)
```

---

## Comparison: Current vs. Proposed

| Aspect | Current Plan | Proposed Plan |
|--------|-------------|---------------|
| **Epochs** | 10 (insufficient) | 30 with early stopping |
| **Largest Model** | 3.5M params (CPU-prohibitive) | 516K params (CPU-friendly) |
| **Parameter Range** | 41.8x (good) | 23.3x (acceptable) |
| **Total Experiments** | 12 | 16 |
| **Estimated Time** | 24-48 hours (with 3.5M model) | 8-12 hours (all <550K) |
| **Results Tracking** | None specified | CSV + analysis scripts |
| **Execution Strategy** | All at once | Phased (validate first) |
| **Model Naming** | Inconsistent | Clear progression |

---

## Key Recommendations

### 1. Use Proposed CPU-Friendly Models ✅
- Tiny: 32/2/1/1/64 (22K params)
- Small: 64/2/1/1/128 (85K params)
- Medium: 96/4/2/1/192 (292K params)
- Large: 128/4/2/1/256 (516K params)

### 2. Increase Training Duration ✅
- 30 epochs with early stopping (patience=10)
- Ensures convergence without wasting compute

### 3. Execute in Two Phases ✅
- Phase 1: Tiny + Small (validate pipeline)
- Phase 2: Medium + Large (complete analysis)

### 4. Add Systematic Tracking ✅
- CSV logging of all experiment results
- Parameter counts, training times, losses
- Analysis scripts for power law fitting

### 5. Data Sizes ✅
- Use 10%, 20%, 50%, 100% (10x range)
- Optional: Add 5% for extended range

---

## Expected Outcomes

After completing all 16 experiments, you will be able to:

1. **Plot Data Scaling Laws**: Log-log plot of validation loss vs. dataset size for each model
2. **Plot Model Scaling Laws**: Log-log plot of validation loss vs. parameters for each data size
3. **Extract Power Law Exponents**:
   - α (data scaling): L ∝ D^(-α)
   - β (model scaling): L ∝ P^(-β)
4. **Make Projections**: Estimate required data/model size to reach target performance
5. **Identify Optimal Allocation**: Understand data vs. model size trade-offs

---

## References

- Kaplan et al. (2020). "Scaling Laws for Neural Language Models"
- Hestness et al. (2017). "Deep Learning Scaling is Predictable, Empirically"
- Hoffmann et al. (2022). "Training Compute-Optimal Large Language Models" (Chinchilla)
- Your POC results: Medium model shows strong data scaling (0.92 → 0.52 loss)
