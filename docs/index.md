# AirTrace Documentation

Welcome to the AirTrace documentation. AirTrace is a modular framework for autoregressive timeseries modeling of aircraft sensor data.

## Table of Contents

1. [Getting Started](../README.md)
2. [Data Format](data_format.md)
3. [Architecture](architecture.md)
4. [Experiments](experiments.md)

## Quick Links

- [Installation](../README.md#installation)
- [Quick Start](../README.md#quick-start)
- [Configuration System](../README.md#configuration-system)
- [Adding New Components](../README.md#adding-new-components)

## Overview

AirTrace provides:

- **Pluggable Models**: Easily swap between RNN, TCN, Transformer, or custom architectures
- **Composable Transforms**: Mix and match data preprocessing steps via configuration
- **Task Abstraction**: One-step, multi-step, or anomaly detection with the same model
- **Reproducible Experiments**: Config-driven experiments with automatic logging

## Getting Help

- Check the [README](../README.md) for basic usage
- See example configs in `configs/exp/`
- Look at notebooks in `notebooks/` for analysis examples
- Run tests with `pytest tests/`
