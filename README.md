# AirTrace: Autoregressive Timeseries Modeling for Aircraft Sensors

A modular, config-driven framework for building and evaluating autoregressive models on aircraft sensor timeseries data.

## Overview

AirTrace enables rapid experimentation with different sequence models (RNNs, TCNs, Transformers), data transforms (scaling, differencing, context features), and prediction tasks (one-step, multi-step, anomaly detection) through declarative YAML configurations.

### Key Features

- **Model-agnostic architecture**: Plug in GRU, TCN, Transformer, or custom models through a common interface
- **Composable data transforms**: Mix and match scaling, differencing, and optional context features via config
- **Task abstraction**: Same model for one-step prediction, multi-step forecasting, or anomaly detection
- **Reproducible experiments**: One command = one experiment, tracked by config + seed
- **Hydra configuration**: Override any parameter from command line or config files

## Installation

We recommend using `uv` to create an isolated environment and to ensure dependencies are resolved
with the pinned NumPy version that is compatible with PyTorch wheels.

```bash
# Create and activate a virtual environment
uv venv .venv
source .venv/bin/activate

# Install the latest release
uv pip install airtrace
```

> **Note:** PyTorch binaries are still built against NumPy 1.x. Installing without the pinned
> constraint can pull NumPy 2.x and trigger import-time ABI errors like
> `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x`. The pinned dependency
> in `pyproject.toml` keeps installs on a compatible NumPy version.

## Installation from Source for Development

```bash
# Clone the repository
git clone https://github.com/yourusername/airtrace.git
cd airtrace

# Install in development mode (use --link-mode=copy on Windows)
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Quick Start

This guide walks you through training a timeseries model on your own data in AirTrace.

### Quick Test with Synthetic Data

If you want to test AirTrace before preparing your data:

```bash
# Generate synthetic cruise dataset (creates 20 flights by default)
airtrace-generate-synthetic

# Or generate with a specific config
airtrace-generate-synthetic data=synthetic

# Train a model on synthetic data
airtrace train data=synthetic_cruise model=gru_ar --dry-run  # Verify setup
airtrace train data=synthetic_cruise model=gru_ar train.epochs=10  # Quick training
```

### Working with Your Own Data

#### Step 1: Prepare Your Raw Data

AirTrace expects timeseries data in Parquet or CSV format. Each file should represent one flight or timeseries sequence.

**Required format:**
- One file per flight/sequence: `{flight_id}.parquet` or `{flight_id}.csv`
- **Timestamp as the index** (critical for resampling)
- Sensor value columns (e.g., `fuel_flow`, `mach`, `altitude`)

**Example - saving CSV with timestamp as index:**
```python
import pandas as pd

# Your data with timestamp column
df = pd.DataFrame({
    'timestamp': pd.date_range('2024-01-01', periods=1000, freq='1S'),
    'fuel_flow': [1250.5, 1251.2, ...],
    'mach': [0.82, 0.82, ...],
    'altitude': [35000, 35001, ...],
    'oat': [-45, -45, ...],
    'n1': [85.2, 85.3, ...]
})

# Set timestamp as index before saving
df = df.set_index('timestamp')
df.to_parquet('data/raw/flight_001.parquet')
# or df.to_csv('data/raw/flight_001.csv')
```

**Expected file structure:**
```
timestamp                  fuel_flow  mach  altitude  oat   n1
2024-01-01 00:00:00       1250.5     0.82  35000     -45   85.2
2024-01-01 00:00:01       1251.2     0.82  35001     -45   85.3
...
```

Place your files in `data/raw/`:
```bash
data/raw/flight_001.parquet
data/raw/flight_002.parquet
data/raw/flight_003.parquet
...
```

**⚠️ Important:** If your CSV has timestamp as a regular column (not the index), `process_to_interim` will fail to resample correctly. Always ensure timestamp is the DataFrame index.

#### Step 2: Ingest Your Data (Automated)

**⭐ Recommended:** Use the `airtrace-ingest` CLI to automatically process your data in one command:

```bash
# Ingest from a directory of parquet files
airtrace-ingest data/raw/my_flights/ --dataset-name my_dataset

# Ingest from a single file
airtrace-ingest data/raw/flights.parquet --dataset-name my_dataset

# Customize window parameters and target sensors
airtrace-ingest data/raw/my_flights/ \
  --dataset-name my_dataset \
  --input-len 512 \
  --pred-len 64 \
  --stride 16 \
  --target-sensors fuel_flow,mach,n1

# Resample irregular data to uniform rate
airtrace-ingest data/raw/my_flights/ \
  --dataset-name my_dataset \
  --resample-rate 1S

# Preview without creating files (dry run)
airtrace-ingest data/raw/my_flights/ \
  --dataset-name my_dataset \
  --dry-run
```

**What `airtrace-ingest` does automatically:**
1. ✅ Validates your data (detects sensors, timestamps, sampling rate)
2. ✅ Processes flights (filters sensors, resamples if requested)
3. ✅ Splits into train/val/test (default 70/15/15, customizable with `--split`)
4. ✅ Generates sliding window indices
5. ✅ Creates dataset config YAML → ready for `airtrace train`

**Output:**
```
============================================================
Dataset Ingestion Complete: my_dataset
============================================================

Files Created:
  15 processed flight files (data/processed/)
  3 window index files (data/metadata/)
  1 config file (src/airtrace/configs/data/my_dataset.yaml)

Data Summary:
  Total flights: 15
  Train flights: 10 (3,142 windows)
  Val flights: 2 (673 windows)
  Test flights: 3 (1,089 windows)

Sensors (6):
  fuel_flow, mach, altitude, oat, n1, weight

Next Steps:
  1. Verify config: src/airtrace/configs/data/my_dataset.yaml
  2. Run training:
     airtrace train data=my_dataset model=gru_ar task.name=one_step
============================================================
```

**Advanced options:**
```bash
# Multi-flight file with flight ID column
airtrace-ingest data/all_flights.parquet \
  --dataset-name my_dataset \
  --flight-id-column flight_number

# Custom train/val/test split ratios
airtrace-ingest data/raw/my_flights/ \
  --dataset-name my_dataset \
  --split 0.8,0.1,0.1 \
  --seed 42

# Specify timestamp column name (if not auto-detected)
airtrace-ingest data/raw/my_flights/ \
  --dataset-name my_dataset \
  --timestamp-column time
```

After ingestion, verify your data:
```bash
airtrace train --data-check data=my_dataset
```

### Config Discovery and Management

AirTrace uses a multi-path config discovery system that makes configs portable and persistent across package reinstalls.

**Where configs are searched (priority order):**
1. **Project-local**: `<project>/.airtrace/configs/` (highest priority, for team sharing)
2. **User-level**: `~/.airtrace/configs/` (medium priority, survives reinstalls)
3. **Package**: Built-in configs (lowest priority, e.g., `synthetic_cruise`, `qantas_737`)

**Where airtrace-ingest writes configs:**

By default, configs created by `airtrace-ingest` are written to `~/.airtrace/configs/data/`, ensuring they persist across package reinstalls and are accessible from any directory.

```bash
# Default behavior - writes to ~/.airtrace/configs/data/my_dataset.yaml
airtrace-ingest data/raw/flights/ --dataset-name my_dataset

# Use it from any directory
cd /some/other/project
airtrace train data=my_dataset  # Works! Config is discovered automatically
```

**Custom config locations:**

You can control where configs are written using `--config-dir` or the `AIRTRACE_CONFIG_DIR` environment variable:

```bash
# Write to a specific directory
airtrace-ingest data/raw/flights/ \
  --dataset-name my_dataset \
  --config-dir ./custom_configs/

# Use environment variable (useful in CI/CD)
export AIRTRACE_CONFIG_DIR=/shared/team_configs
airtrace-ingest data/raw/flights/ --dataset-name my_dataset

# Write to project-local directory (share with team via git)
airtrace-ingest data/raw/flights/ \
  --dataset-name my_dataset \
  --config-dir .airtrace/configs/data/

# Add to version control
git add .airtrace/configs/data/my_dataset.yaml
git commit -m "Add team dataset config"
```

**Config priority examples:**

```bash
# If you have configs in multiple locations:
# - Package: ~/.venv/.../airtrace/configs/data/my_dataset.yaml
# - User:    ~/.airtrace/configs/data/my_dataset.yaml
# - Project: ./.airtrace/configs/data/my_dataset.yaml

# Project config takes precedence (overrides user and package)
airtrace train data=my_dataset  # Uses project config if it exists
```

**Inspecting config locations:**

```python
from airtrace.utils.config_paths import get_config_search_paths

# See where AirTrace searches for configs
paths = get_config_search_paths()
for path in paths:
    print(f"Searching: {path}")
```

**Skip to Step 5** if using `airtrace-ingest`. The manual steps below are only needed for advanced customization.

---

#### Alternative: Manual Data Processing

**Step 2 (Manual): Process Data to Interim Format
#### Step 2: Process Data to Interim Format

Convert raw data to clean, aligned timeseries:

```python
from airtrace.data.loaders import RawDataLoader

loader = RawDataLoader("data")

# Process each flight
flight_ids = ["flight_001", "flight_002", "flight_003"]
for flight_id in flight_ids:
    loader.process_to_interim(
        flight_id=flight_id,
        resample_rate="1S",  # 1 second intervals
        sensor_list=["fuel_flow", "mach", "altitude", "oat", "n1"]
    )
```

This creates cleaned files in `data/interim/` with:
- Uniform timesteps (resampled to 1Hz)
- Missing values handled
- Consistent sensor ordering

**If your raw CSVs have timestamp as a column (not index), preprocess them first:**

```python
import pandas as pd
from pathlib import Path

raw_dir = Path("data/raw")
for csv_file in raw_dir.glob("*.csv"):
    # Read with timestamp as column
    df = pd.read_csv(csv_file)

    # Set timestamp as index
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')

    # Save back as parquet with proper index
    output_path = csv_file.with_suffix('.parquet')
    df.to_parquet(output_path)
    print(f"Converted {csv_file.name} -> {output_path.name}")

# Now use the loader on the parquet files
loader = RawDataLoader("data")
for flight_id in ["flight_001", "flight_002"]:
    loader.process_to_interim(flight_id=flight_id, resample_rate="1S")
```

#### Step 3 (Manual): Create Windowed Datasets

Generate sliding windows for model training:

```python
from airtrace.data.loaders import InterimDataProcessor
from airtrace.data.windows import WindowSpec

processor = InterimDataProcessor("data")

# Define window parameters
window_spec = WindowSpec(
    input_len=256,      # 256 timesteps of history
    pred_len=32,        # Predict 32 timesteps ahead
    stride=32,          # Slide by 32 steps (no overlap)
    target_sensors=["fuel_flow", "mach"]  # What to predict
)

# Split your flights
train_flights = ["flight_001", "flight_002"]
val_flights = ["flight_003"]

# Create train windows
processor.create_windows(
    flight_ids=train_flights,
    window_spec=window_spec,
    output_name="train"
)

# Create validation windows
processor.create_windows(
    flight_ids=val_flights,
    window_spec=window_spec,
    output_name="val"
)
```

This creates index files in `data/metadata/`:
- `train_index.parquet` - Training windows
- `val_index.parquet` - Validation windows

#### Step 4 (Manual): Create a Data Configuration

Create a config file for your dataset at `src/airtrace/configs/data/my_dataset.yaml`:

```yaml
# @package _global_

data:
  root: data/
  dataset_name: "my_dataset"
  train_index: "metadata/train_index.parquet"
  val_index: "metadata/val_index.parquet"
  test_index: "metadata/test_index.parquet"  # Optional

  window:
    input_len: 256
    pred_len: 32
    stride: 32
    target_sensors: ["fuel_flow", "mach"]

  sensors:
    use: ["fuel_flow", "mach", "altitude", "oat", "n1"]

  static_features:
    use: []  # Add flight-level features if available
```

Verify your data setup:
```bash
airtrace train --data-check data=my_dataset
```

#### Step 5: Choose Data Transforms

Transforms preprocess your data. Common pipelines:

**Z-score normalization + differencing** (for non-stationary data):
```bash
airtrace train data=my_dataset transforms=zscore_diff
```

**Z-score normalization + differencing + context features** (adds static metadata like aircraft_type):
```bash
airtrace train data=my_dataset transforms=zscore_diff_with_context
```

**Min-max scaling only** (for bounded sensors):
```bash
airtrace train data=my_dataset transforms=minmax_only
```

**Robust scaling** (for outlier-heavy data):
```bash
airtrace train data=my_dataset transforms=robust_scaler
```

**No transforms** (for pre-normalized data):
```bash
airtrace train data=my_dataset transforms=minimal
```

Custom transform pipeline (`src/airtrace/configs/transforms/my_transforms.yaml`):
```yaml
# @package _global_

transforms:
  pipeline:
    - name: zscore
      per_sensor: true
      center: true
      scale: true
    - name: clip
      min_percentile: 1
      max_percentile: 99
```

#### Step 6: Select and Train a Model

Choose a model architecture based on your needs:

**For quick experimentation - Baselines:**
```bash
# Persistence baseline (last value)
airtrace train data=my_dataset model=persistence

# Linear autoregressive
airtrace train data=my_dataset model=linear_ar
```

**For strong performance - Neural architectures:**
```bash
# GRU (good default for timeseries)
airtrace train data=my_dataset model=gru_ar

# Transformer (for long-range dependencies)
airtrace train data=my_dataset model=transformer

# PatchTST (state-of-the-art for many tasks)
airtrace train data=my_dataset model=patchtst

# ModernTCN (efficient convolutional)
airtrace train data=my_dataset model=moderntcn
```

**Full training command with all options:**
```bash
airtrace train \
  data=my_dataset \
  model=gru_ar \
  transforms=zscore_diff \
  task=one_step \
  train.epochs=50 \
  train.batch_size=64 \
  train.lr=0.001
```

**Override model hyperparameters:**
```bash
airtrace train \
  data=my_dataset \
  model=gru_ar \
  model.hidden_dim=256 \
  model.num_layers=3 \
  model.dropout=0.2
```

Training progress is logged to `runs/{date}/{exp_name}/`.

#### Step 7: Evaluate Your Model

After training, evaluate on test data:

```bash
# Evaluate best checkpoint
airtrace eval \
  --checkpoint runs/20241117/my_experiment/checkpoints/best.ckpt \
  data=my_dataset
```

This outputs metrics:
```
================================================================================
Evaluation Results
================================================================================
MSE         : 0.0234
MAE         : 0.1123
RMSE        : 0.1531
Samples     : 1523
================================================================================
```

### Complete Example Workflow

Here's a full end-to-end example:

```bash
# 1. Verify installation
airtrace --version

# 2. Prepare your data (Python script)
python scripts/prepare_my_data.py  # Your preprocessing script

# 3. Validate data
airtrace train --data-check data=my_dataset

# 4. Quick test with baseline
airtrace train data=my_dataset model=persistence train.epochs=1

# 5. Train a real model
airtrace train \
  data=my_dataset \
  model=gru_ar \
  transforms=zscore_diff \
  train.epochs=50 \
  train.batch_size=64

# 6. Evaluate
airtrace eval \
  --checkpoint runs/20241117/gru_zscore_one_step/checkpoints/best.ckpt \
  data=my_dataset
```

### Creating Reusable Experiment Configs

For repeated experiments, create a config at `src/airtrace/configs/exp/my_experiment.yaml`:

```yaml
defaults:
  - override /data: my_dataset
  - override /model: gru_ar
  - override /transforms: zscore_diff
  - override /task: one_step
  - override /train: default

exp_name: "my_gru_experiment"
seed: 42
```

Then run simply:
```bash
airtrace train exp=my_experiment
```

### CLI Reference

```bash
# Help and version
airtrace --help
airtrace --version

# Generate synthetic data
airtrace-generate-synthetic                           # Default config
airtrace-generate-synthetic data=synthetic_cruise     # Specific config
airtrace-generate-synthetic data=synthetic data.generation.n_flights=50
n# Ingest your own data
airtrace-ingest data/raw/my_flights/ --dataset-name my_dataset
airtrace-ingest data/raw/flights.parquet --dataset-name my_dataset --target-sensors fuel_flow,mach,n1
airtrace-ingest data/raw/my_flights/ --dataset-name my_dataset --resample-rate 1S --dry-run

# Validate data only
airtrace train --data-check data=my_dataset

# Dry run (check config without training)
airtrace train --dry-run exp=my_experiment

# Train with overrides
airtrace train model=tcn train.epochs=100 train.batch_size=128

# Evaluate checkpoint
airtrace eval --checkpoint path/to/best.ckpt data=my_dataset

# Resume from checkpoint
airtrace train --checkpoint path/to/checkpoint.ckpt exp=my_experiment
```

### Next Steps

- **Explore models**: See [Model Registry](#model-registry) for 45+ available models
- **Custom components**: See [Adding New Components](#adding-new-components)
- **Advanced features**: Check `docs/architecture.md` for design details
- **Experiment tracking**: Review `docs/experiments.md` for best practices

## Project Structure

```
airtrace/
├── configs/          # Hydra configuration files
│   ├── data/        # Dataset configurations
│   ├── model/       # Model architectures
│   ├── transforms/  # Data transformation pipelines
│   ├── task/        # Prediction tasks
│   ├── train/       # Training hyperparameters
│   └── exp/         # Complete experiment configs
├── data/            # Data storage (not in git)
│   ├── raw/         # Original flight logs
│   ├── interim/     # Cleaned, aligned timeseries
│   ├── processed/   # Windowed tensors
│   └── metadata/    # Sensor definitions, index files
├── src/airtrace/    # Main package
│   ├── data/        # Dataset and data loading
│   ├── transforms/  # Transform implementations
│   ├── models/      # Model implementations
│   ├── tasks/       # Task definitions
│   ├── training/    # Training loop and callbacks
│   ├── evaluation/  # Metrics and evaluation
│   └── viz/         # Visualization utilities
├── notebooks/       # Jupyter notebooks for analysis
├── tests/          # Unit and integration tests
└── docs/           # Documentation
```

## Configuration System

AirTrace uses Hydra for hierarchical configuration. Experiments are defined by composing:

1. **Data**: Which dataset and window configuration to use
2. **Model**: Architecture (GRU, TCN, Transformer, etc.)
3. **Transforms**: Data preprocessing pipeline (scaling, differencing, optional context features)
4. **Task**: Prediction objective (one-step, multi-step, anomaly)
5. **Train**: Optimization hyperparameters

Example experiment config (`configs/exp/exp_001_gru_zscore.yaml`):

```yaml
defaults:
  - override /data: qantas_737
  - override /model: gru_ar
  - override /transforms: zscore_diff_with_context  # Use zscore_diff for no context
  - override /task: one_step
  - override /train: default

exp_name: "gru_zscore_one_step"
seed: 123
```

## Data Pipeline

The data pipeline has three stages:

1. **Raw**: Original flight logs (Parquet/CSV) with `flight_id`, `timestamp`, sensor values
2. **Interim**: Cleaned, resampled, aligned timeseries per flight
3. **Processed**: Sliding windows of `[T_in, D]` inputs → `[T_out, D_target]` outputs

Configure window parameters in data configs:

```yaml
window:
  input_len: 256    # Input sequence length
  pred_len: 32      # Prediction horizon
  stride: 32        # Sliding window stride
  target_sensors: ["fuel_flow", "mach"]
```

### Synthetic Data Generation

For testing and experimentation, AirTrace includes a physics-based synthetic data generator that creates realistic aircraft cruise sensor readings:

```bash
# Generate with default config (20 flights, synthetic_cruise)
airtrace-generate-synthetic

# Generate with specific data config
airtrace-generate-synthetic data=synthetic_cruise

# Override generation parameters
airtrace-generate-synthetic data=synthetic_cruise data.generation.n_flights=50

# Use different random seed
airtrace-generate-synthetic data=synthetic seed=123

# Combine multiple overrides
airtrace-generate-synthetic \
  data=synthetic_cruise \
  data.generation.n_flights=100 \
  data.generation.seed=42
```

**What gets created:**
- `data/raw/` - Raw synthetic flight data
- `data/interim/` - Cleaned, resampled timeseries
- `data/processed/` - Flight data ready for windowing
- `data/metadata/*_index.parquet` - Train/val/test window indices

**Two available configs:**
- `synthetic_cruise` - Long flights (1 hour), realistic cruise parameters
- `synthetic` - Shorter flights (30 min), simpler parameters

The generator produces physically plausible sensor relationships:
- Fuel flow ↔ engine thrust (N1) ↔ aircraft weight
- ISA standard atmosphere temperature model
- Configurable turbulence and noise levels
- Deterministic generation from seed for reproducibility

**Python API (for custom scripts):**
```python
from airtrace.data.synthetic import create_synthetic_dataset
from pathlib import Path

splits = create_synthetic_dataset(
    data_root=Path("data/"),
    n_flights=20,
    seed=42,
    flight_id_prefix="my_synthetic"
)
```

See [Synthetic Data Documentation](docs/synthetic_data.md) for details on the physics model.

## Model Registry

AirTrace includes 52 registered models spanning from simple baselines to sophisticated neural architectures. All models implement the `ARBaseModel` interface and can be composed with any data transform or task configuration.

### Recurrent Neural Networks

| Model Name | Class | Description |
|------------|-------|-------------|
| `gru_ar` | `GRUARModel` | GRU-based autoregressive encoder with optional attention mechanism |
| `lstm_ar` | `LSTMARModel` | LSTM-based autoregressive encoder with cell state for longer-term dependencies |

### Sequence-to-Sequence Models

| Model Name | Class | Description |
|------------|-------|-------------|
| `gru_seq2seq` | `GRUSeq2SeqModel` | GRU encoder-decoder with teacher forcing and optional attention |
| `lstm_seq2seq` | `LSTMSeq2SeqModel` | LSTM encoder-decoder for multi-step forecasting |

### Attention-Based Models

| Model Name | Class | Description |
|------------|-------|-------------|
| `nonstationary_transformer` | `NonStationaryTransformerModel` | Non-stationary Transformer with de-stationary attention blocks and learnable stationarization to handle distribution shifts |
| `transformer` | `TransformerModel` | Transformer with causal self-attention and positional encoding |
| `informer` | `InformerModel` | Informer with ProbSparse attention, distilling encoder, and lightweight decoder for efficient long-horizon forecasting |
| `autoformer` | `AutoformerModel` | Autoformer with series decomposition and auto-correlation attention for long-range dependencies |
| `fedformer` | `FEDformerModel` | FEDformer - Frequency Enhanced Decomposed Transformer with Fourier/Wavelet attention operating in frequency domain for long-term forecasting (ICML 2022) |
| `patchtst` | `PatchTSTModel` | PatchTST - channel-independent patch time series transformer with efficient patching mechanism (ICLR 2023) |
| `itransformer` | `iTransformerModel` | iTransformer - inverted transformer treating variates as tokens for superior multivariate correlation modeling (ICLR 2024 Spotlight) |
| `crossformer` | `CrossformerModel` | Crossformer - two-stage temporal then cross-dimension attention with DSW embeddings for explicit cross-sensor dependency modeling (ICLR 2023) |
| `tft` | `TemporalFusionTransformer` | Temporal Fusion Transformer combining variable selection, LSTM encoders/decoders, and interpretable attention for multi-horizon forecasting |
| `timexer` | `TimeXerModel` | TimeXer - transformer with explicit exogenous variable handling using dual patch-level and variate-level representations with global endogenous tokens bridging endogenous and exogenous information (NeurIPS 2024) |

### Pondering & Wrapper Models

| Model Name | Class | Description |
|------------|-------|-------------|
| `latent_ponder` | `LatentPonderWrapper` | Wraps any base predictor with latent ponder steps, adaptive halting, and optional TRM-style `(y, h)` refinement plus auxiliary supervision for compute-aware refinement |
| `residual_solver` | `ResidualSolver` | Iterative residual-refinement solver with residual-aware halting, stepwise loss weighting, and latent GRU updates for compute–accuracy trade-offs |

### Foundation Models

| Model Name | Class | Description |
|------------|-------|-------------|
| `chronos_bolt` | `ChronosBoltModel` | Chronos-Bolt inspired pretrained foundation model with gated conv-attention blocks, tokenizer-free patching, and optional LoRA adapters for efficient fine-tuning |
| `moirai` | `MoiraiModel` | Moirai-style multiresolution selective state-space model combining hierarchical patching with selective scan blocks and optional LoRA adapters |
| `moment` | `MomentModel` | MOMENT - open-source foundation model with patch-based transformer, masked reconstruction pre-training on Time-series Pile dataset, supports forecasting/classification/anomaly detection with few-shot and zero-shot capabilities (ICML 2024, CMU Auton Lab) |
| `mamba2` | `Mamba2Model` | Temporal Mamba-2 selective state-space model with hardware-aware chunked scans, bidirectional gating, and LoRA-ready forecast head for 100k-token contexts |
| `mambats` | `MambaTSModel` | MambaTS - improved selective state-space model with Variable Scan along Time (VST) and Temporal Mamba Blocks (TMB) for efficient long-term forecasting with linear complexity (arXiv 2024) |
| `s_mamba` | `SMambaModel` | S-Mamba - simple selective state-space model with per-variate tokenization, bidirectional scan blocks, and MLP-based temporal mixing (Neurocomputing 2025) |
| `lag_llama` | `LagLlamaModel` | Retrieval-augmented Lag-Llama-style diffusion forecaster combining patch tokenization, nearest-neighbor memory, and latent diffusion sampling for probabilistic trajectories |
| `timesfm` | `TimesFMModel` | TimesFM - decoder-only patch transformer from Google with patch tokenization, causal attention, and lightweight projection head for efficient long-horizon forecasting (ICML 2024) |
| `timer` | `TimerModel` | Timer - GPT-style pre-trained decoder-only Transformer with zero-shot forecasting capability, trained on 260B time points in Single-Series Sequence (S3) format (ICML 2024) |

### Convolutional Models

| Model Name | Class | Description |
|------------|-------|-------------|
| `tcn` | `TCNModel` | Temporal Convolutional Network with dilated causal convolutions |
| `moderntcn` | `ModernTCNModel` | ModernTCN - modern pure convolution architecture with depthwise separable convolutions, large receptive fields, and improved efficiency over classic TCN (ICLR 2024 Spotlight) |
| `timesnet` | `TimesNetModel` | TimesNet - 2D vision backbone for time series via period-based reshaping, capturing intraperiod and interperiod variations with Inception-style convolutions (ICLR 2023) |

### MLP-Based Models

| Model Name | Class | Description |
|------------|-------|-------------|
| `timemixer` | `TimeMixerModel` | TimeMixer - decomposable multiscale mixing for time series forecasting with MLP architecture (ICLR 2024) |
| `tsmixer` | `TSMixerModel` | TSMixer - all-MLP architecture with alternating time-mixing and feature-mixing operations for efficient multivariate forecasting (KDD 2023) |
| `softs` | `SOFTS` | SOFTS - pure MLP-based multivariate forecaster using STAR (Aggregate-Redistribute) module with stochastic pooling for efficient channel mixing (NeurIPS 2024) |
| `nbeats` | `NBeatsModel` | N-BEATS - residual stack of basis expansion blocks with interpretable trend/seasonality components (ICLR 2020) |
| `nhits` | `NHiTSModel` | N-HiTS - hierarchical interpolation model with multi-resolution pooling stacks for long-horizon forecasting (AAAI 2023) |
| `cyclenet` | `CycleNetModel` | CycleNet - residual cycle forecasting with learnable periodic patterns for extreme efficiency (NeurIPS 2024 Spotlight) |
| `dlinear` | `DLinearModel` | DLinear - decomposition-based linear forecaster with separate trend and seasonal projections |
| `nlinear` | `NLinearModel` | NLinear - mean-adjusted linear projection for non-stationary series |
| `frets` | `FreTSModel` | FreTS - frequency-domain MLP applying learned transformations to low-frequency Fourier coefficients for efficient forecasting (NeurIPS 2023) |

### Baseline Models

Simple, interpretable baselines for comparison. Most are non-trainable (parameter-free).

| Model Name | Class | Trainable | Description |
|------------|-------|-----------|-------------|
| `persistence` | `PersistenceModel` | No | Naive forecast - predicts last observed value |
| `moving_average` | `MovingAverageModel` | No | Mean of recent k values |
| `zero` | `ZeroModel` | No | Always predicts zero (useful for normalized data) |
| `mean` | `MeanModel` | No | Historical mean of input sequence |
| `median` | `MedianModel` | No | Historical median (robust to outliers) |
| `linear_trend` | `LinearTrendModel` | No | Linear trend extrapolation via least squares |
| `polynomial_trend` | `PolynomialTrendModel` | No | Polynomial trend fitting (quadratic, cubic, etc.) |
| `drift` | `DriftModel` | No | Random walk with drift (last value + average change) |
| `exponential_smoothing` | `ExponentialSmoothingModel` | No | Exponentially weighted moving average (EWMA) |
| `holt_linear_trend` | `HoltLinearTrendModel` | No | Holt's double exponential smoothing (level + trend) |
| `holt_winters` | `HoltWintersModel` | No | Holt-Winters triple exponential smoothing with additive/multiplicative seasonality |
| `seasonal_naive` | `SeasonalNaiveModel` | No | Predicts value from previous seasonal cycle |
| `theta` | `ThetaModel` | No | Theta method (M3 competition winner) |
| `sarima` | `SARIMAModel` | No | Seasonal ARIMA leveraging differencing and moving-average residuals via statsmodels |
| `var` | `VARModel` | No | Vector autoregression capturing cross-sensor dynamics with ridge-stabilised fit |
| `linear_ar` | `LinearARModel` | **Yes** | Simple trainable linear autoregressive model |
| `mlp_ar` | `MLPARModel` | **Yes** | Multi-layer perceptron treating window as static features |

### Using Models

All models are instantiated via the registry using config files:

```bash
# Use GRU model
airtrace train model=gru_ar

# Use baseline for comparison
airtrace train model=persistence

# Use transformer with custom parameters
airtrace train model=transformer model.d_model=256 model.nhead=8
```

## Adding New Components

### New Model

1. Implement `ARBaseModel` interface in `src/airtrace/models/your_model.py`
2. Register with `@register("your_model")` decorator
3. Create config in `configs/model/your_model.yaml`
4. **Update the Model Registry table in README.md** with model details

### New Transform

1. Implement `Transform` interface in `src/airtrace/transforms/your_transform.py`
2. Register with `@register("your_transform")` decorator
3. Add to pipeline in `configs/transforms/`

### New Task

1. Implement `Task` interface in `src/airtrace/tasks/your_task.py`
2. Register with `@register("your_task")` decorator
3. Create config in `configs/task/your_task.yaml`

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=airtrace --cov-report=html

# Format code
black src/ tests/

# Lint
ruff src/ tests/

# Type check
mypy src/
```

## License

MIT License - see LICENSE file for details.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{airtrace2025,
  title = {AirTrace: Autoregressive Timeseries Modeling for Aircraft Sensors},
  author = {AirTrace Team},
  year = {2025},
  url = {https://github.com/yourusername/airtrace}
}
```
