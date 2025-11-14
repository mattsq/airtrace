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

## Adding New Components

### New Model

1. Implement `ARBaseModel` interface in `src/airtrace/models/your_model.py`
2. Register with `@register("your_model")` decorator
3. Create config in `configs/model/your_model.yaml`

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
