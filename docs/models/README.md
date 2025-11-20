# Model-Specific Documentation

This directory contains detailed documentation for specific models implemented in AirTrace.

## Available Documentation

### Foundation Models

- [**Chronos-Bolt**](chronos_bolt.md) - Pretrained continuous-token foundation model with gated convolution-attention blocks and optional LoRA adapters

### Advanced Architectures

- [**SOFTS Architecture**](softs_architecture.md) - Detailed architecture of the SOFTS (Series-cOre Fused Time Series) model with STAR module
- [**SOFTS Implementation Guide**](softs_implementation.md) - Step-by-step implementation guide for SOFTS
- [**SOFTS Visual Architecture**](softs_visual.md) - Visual diagrams and architecture illustrations for SOFTS

## When to Use Model-Specific Docs

Use these documents when:
- You need detailed architecture information beyond the README
- You're implementing or modifying a specific model
- You want to understand design decisions for a particular model
- You need visual diagrams or implementation guidance

## General Model Documentation

For general model usage and the complete model registry, see:
- [Main README - Model Registry](../../README.md#model-registry)
- [Baseline Models](../baseline_models.md)
- [Architecture Overview](../architecture.md)

## Adding New Model Documentation

When documenting a new model:
1. Create `model_name.md` in this directory
2. Include: architecture overview, key innovations, usage examples, references
3. Add entry to this README
4. Update the main README Model Registry table
