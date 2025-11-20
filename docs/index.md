# AirTrace Documentation

Welcome to the AirTrace documentation. AirTrace is a modular framework for autoregressive timeseries modeling of aircraft sensor data.

## Core Documentation

### Getting Started
1. [Installation and Quick Start](../README.md) - Installation, data preparation, and first training run
2. [Data Format](data_format.md) - Data format specifications and requirements
3. [Synthetic Data Generation](synthetic_data.md) - Physics-based synthetic data for testing

### Architecture and Design
4. [Architecture](architecture.md) - Core framework architecture and design philosophy
5. [Baseline Models](baseline_models.md) - Simple baseline models for comparison
6. [Experiments](experiments.md) - Experiment tracking and best practices

### Development
7. [CI/CD Pipeline](ci_cd.md) - Continuous integration and deployment documentation

## Model-Specific Documentation

Detailed documentation for specific model implementations:

- [**Models Directory**](models/) - Model-specific architecture guides and implementation details
  - [Chronos-Bolt Foundation Model](models/chronos_bolt.md)
  - [SOFTS Architecture](models/softs_architecture.md)
  - [SOFTS Implementation Guide](models/softs_implementation.md)
  - [SOFTS Visual Architecture](models/softs_visual.md)

For the complete model registry and usage, see [README - Model Registry](../README.md#model-registry).

## Research and Planning

Forward-looking research documents and proposals:

- [**Research Directory**](research/) - Model proposals and planning materials
  - [Model Proposals](research/model_proposals.md) - Proposals for adding new models (2023-2025 literature)

## Archive

Historical documentation for completed work:

- [**Archive Directory**](archive/) - Completed implementation reports and deprecated guides
  - [Q400 Integration Report](archive/q400_integration_and_datastore_fixes.md)

## Quick Links

### For New Users
- [Installation](../README.md#installation)
- [Quick Start](../README.md#quick-start)
- [Configuration System](../README.md#configuration-system)

### For Developers
- [Adding New Components](../README.md#adding-new-components)
- [Development Workflow](../CLAUDE.md#development-workflow)
- [Testing](../README.md#development)

### For AI Agents
- [Agent Guide (CLAUDE.md)](../CLAUDE.md) - Complete guide for AI agents working on AirTrace
- [Memory and Learnings (MEMORY.md)](../MEMORY.md) - Discovered insights and gotchas

## Overview

AirTrace provides:

- **Pluggable Models**: Easily swap between RNN, TCN, Transformer, or custom architectures
- **Composable Transforms**: Mix and match data preprocessing steps via configuration
- **Task Abstraction**: One-step, multi-step, or anomaly detection with the same model
- **Reproducible Experiments**: Config-driven experiments with automatic logging

## Documentation Structure

```
docs/
├── index.md                    # This file - documentation index
├── architecture.md             # Core architecture
├── data_format.md              # Data specifications
├── synthetic_data.md           # Synthetic data generation
├── baseline_models.md          # Baseline models
├── experiments.md              # Experiment tracking
├── ci_cd.md                    # CI/CD pipeline
├── models/                     # Model-specific documentation
│   ├── README.md
│   ├── chronos_bolt.md
│   ├── softs_architecture.md
│   ├── softs_implementation.md
│   └── softs_visual.md
├── research/                   # Research and proposals
│   ├── README.md
│   └── model_proposals.md
└── archive/                    # Historical documentation
    ├── README.md
    └── q400_integration_and_datastore_fixes.md
```

## Getting Help

- Check the [README](../README.md) for basic usage
- See example configs in `configs/exp/`
- Look at notebooks in `notebooks/` for analysis examples
- Run tests with `pytest tests/`
- For AI agents: Read [CLAUDE.md](../CLAUDE.md) and [MEMORY.md](../MEMORY.md)

## Contributing to Documentation

When adding documentation:

1. **Core docs** (architecture, data, experiments) → `/docs` root
2. **Model-specific guides** → `/docs/models/`
3. **Research and proposals** → `/docs/research/`
4. **Completed work items** → `/docs/archive/`
5. **Update this index** to reference new documents

Keep documentation:
- **Clear and concise** - Focus on essential information
- **Well-organized** - Use headings, lists, and tables
- **Up-to-date** - Update when code changes
- **Cross-referenced** - Link to related documents
