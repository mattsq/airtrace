# AirTrace: Autoregressive Timeseries Modeling for Aircraft Sensors

A modular, config-driven framework for building and evaluating autoregressive models on aircraft sensor timeseries data.

## Overview

AirTrace enables rapid experimentation with different sequence models (RNNs, TCNs, Transformers), data transforms (scaling, differencing, context features), and prediction tasks (one-step, multi-step, anomaly detection) through declarative YAML configurations.

### Key Features

- **Model-agnostic architecture**: Plug in GRU, TCN, Transformer, or custom models through a common interface
- **Composable data transforms**: Mix and match scaling, differencing, context features via config
- **Task abstraction**: Same model for one-step prediction, multi-step forecasting, or anomaly detection
- **Reproducible experiments**: One command = one experiment, tracked by config + seed
- **Hydra configuration**: Override any parameter from command line or config files

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/airtrace.git
cd airtrace

# Install in development mode
pip install -e ".[dev]"
```

## Quick Start

### Train a model

```bash
# Run with default config
airtrace train

# Run a specific experiment
airtrace train exp=exp_001_gru_zscore

# Override parameters from command line
airtrace train model=tcn train.epochs=100 train.batch_size=128
```

### Evaluate a model

```bash
airtrace eval exp=exp_001_gru_zscore checkpoint=best.ckpt
```

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
3. **Transforms**: Data preprocessing pipeline (scaling, differencing, context)
4. **Task**: Prediction objective (one-step, multi-step, anomaly)
5. **Train**: Optimization hyperparameters

Example experiment config (`configs/exp/exp_001_gru_zscore.yaml`):

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

For testing and experimentation, AirTrace includes a physics-based synthetic data generator:

```bash
# Generate 20 synthetic cruise flights
python src/scripts/generate_synthetic_data.py --n-flights 20 --output data/

# Or use the Python API
from airtrace.data.synthetic import create_synthetic_dataset
splits = create_synthetic_dataset(
    data_root="data/",
    n_flights=20,
    seed=42
)
```

The generator produces realistic aircraft cruise sensor readings with:
- Physically plausible sensor relationships (fuel flow ↔ thrust ↔ weight)
- ISA temperature model
- Configurable turbulence and noise levels
- Deterministic generation from seed

See [Synthetic Data Documentation](docs/synthetic_data.md) for details.

## Model Registry

AirTrace includes 26 registered models spanning from simple baselines to sophisticated neural architectures. All models implement the `ARBaseModel` interface and can be composed with any data transform or task configuration.

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
| `transformer` | `TransformerModel` | Transformer with causal self-attention and positional encoding |
| `autoformer` | `AutoformerModel` | Autoformer with series decomposition and auto-correlation attention for long-range dependencies |
| `patchtst` | `PatchTSTModel` | PatchTST - channel-independent patch time series transformer with efficient patching mechanism (ICLR 2023) |
| `itransformer` | `iTransformerModel` | iTransformer - inverted transformer treating variates as tokens for superior multivariate correlation modeling (ICLR 2024 Spotlight) |

### Foundation Models

| Model Name | Class | Description |
|------------|-------|-------------|
| `chronos_bolt` | `ChronosBoltModel` | Chronos-Bolt inspired pretrained foundation model with gated conv-attention blocks, tokenizer-free patching, and optional LoRA adapters for efficient fine-tuning |
| `moirai` | `MoiraiModel` | Moirai-style multiresolution selective state-space model combining hierarchical patching with selective scan blocks and optional LoRA adapters |
| `mamba2` | `Mamba2Model` | Temporal Mamba-2 selective state-space model with hardware-aware chunked scans, bidirectional gating, and LoRA-ready forecast head for 100k-token contexts |
| `lag_llama` | `LagLlamaModel` | Retrieval-augmented Lag-Llama-style diffusion forecaster combining patch tokenization, nearest-neighbor memory, and latent diffusion sampling for probabilistic trajectories |

### Convolutional Models

| Model Name | Class | Description |
|------------|-------|-------------|
| `tcn` | `TCNModel` | Temporal Convolutional Network with dilated causal convolutions |

### MLP-Based Models

| Model Name | Class | Description |
|------------|-------|-------------|
| `timemixer` | `TimeMixerModel` | TimeMixer - decomposable multiscale mixing for time series forecasting with MLP architecture (ICLR 2024) |
| `cyclenet` | `CycleNetModel` | CycleNet - residual cycle forecasting with learnable periodic patterns for extreme efficiency (NeurIPS 2024 Spotlight) |

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
