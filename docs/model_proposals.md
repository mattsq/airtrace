# Model Proposals for AirTrace

This document tracks proposals for adding cutting-edge models to AirTrace based on recent literature.

**Date**: 2025-11-15
**Status**: Proposed, awaiting implementation decision

---

## Summary

After reviewing the current model registry (21 models including PatchTST/ICLR 2023 and iTransformer/ICLR 2024) and surveying recent literature, three cutting-edge models are proposed:

1. **TimeMixer** (ICLR 2024) - Primary recommendation
2. **S-Mamba** (2024) - Novel state space model paradigm
3. **ModernTCN** (ICLR 2024 Spotlight) - Modern convolution upgrade

---

## Primary Recommendation: TimeMixer

**Full Title**: TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting
**Venue**: ICLR 2024 Poster
**Paper**: https://arxiv.org/abs/2405.14616
**Code**: https://github.com/kwuking/TimeMixer
**OpenReview**: https://openreview.net/forum?id=7oLshfEIC2

### Key Innovations

1. **Multiscale Decomposition**: Analyzes time series at multiple temporal scales to capture both microscopic (fine-grained) and macroscopic (coarse-grained) patterns
2. **Fully MLP-based**: Achieves SOTA without attention mechanisms, offering computational efficiency
3. **Decomposable Mixing**: Separates seasonal and trend components, then mixes in both fine-to-coarse and coarse-to-fine directions

### Architecture Components

- **Past-Decomposable-Mixing (PDM)**: Decomposes multiscale series and mixes seasonal/trend components bidirectionally
- **Future-Multipredictor-Mixing (FMM)**: Ensembles multiple predictors across scales for robust forecasting

### Why Perfect for Aircraft Sensors

Aircraft sensor data naturally exhibits multiscale behavior:
- **Fine scale** (sub-second): Sensor noise, vibrations, turbulence, sampling artifacts
- **Medium scale** (seconds-minutes): Flight phase transitions, maneuvers, engine adjustments
- **Coarse scale** (minutes-hours): Overall trajectory, fuel consumption trends, weight changes

TimeMixer's explicit multiscale modeling with seasonal/trend decomposition would capture these hierarchical patterns effectively.

### Performance

- **Long-term forecasting**: SOTA performance on multiple benchmarks
- **Short-term forecasting**: SOTA performance with favorable runtime efficiency
- **Computational cost**: Lower than transformer-based models due to MLP architecture

### Complementarity to Existing Models

AirTrace currently has:
- Transformers (vanilla, PatchTST, iTransformer) - all attention-based
- RNNs (GRU, LSTM) - recurrent mechanisms
- TCN - convolutional approach

TimeMixer would add:
- Pure MLP architecture - different inductive bias
- Explicit multiscale modeling - unique capability
- Decomposition-based - complementary to attention/convolution

---

## Alternative 1: S-Mamba

**Full Title**: Is Mamba Effective for Time Series Forecasting?
**Paper**: https://arxiv.org/abs/2403.11144
**Related**: TSMamba, Bi-Mamba4TS, Mamba4Cast
**Year**: 2024

### Key Innovations

1. **Selective State Space Models**: Novel paradigm fundamentally different from RNNs, CNNs, and Transformers
2. **Linear Complexity**: O(n) computational complexity vs O(n²) for transformers
3. **Global Receptive Field**: Despite linear complexity, maintains ability to model long-range dependencies
4. **Hardware-Aware Design**: Efficient GPU implementation

### Variants

- **S-Mamba** (Simple-Mamba): Core architecture for forecasting
- **TSMamba**: Foundation model variant, 15% better than GPT4TS, outperforms PatchTST
- **Bi-Mamba4TS**: Bidirectional variant for improved temporal modeling
- **Mamba4Cast**: Zero-shot forecasting foundation model

### Why for Aircraft Sensors

1. **Long Sequences**: Aircraft flights generate hours of continuous sensor data; linear complexity enables processing without memory explosion
2. **Continuous Dynamics**: State space models naturally represent continuous physical systems (aircraft dynamics, aerodynamics)
3. **Efficiency**: Much lower GPU memory and training time than transformers
4. **State-of-the-Art**: Represents the absolute cutting edge in sequence modeling

### Performance

- TSMamba: +15% over GPT4TS baseline
- Outperforms PatchTST on standard benchmarks
- Maintains performance while reducing computational requirements

### Novelty Factor

State space models are the most recent architectural innovation in deep learning (post-transformers), representing a fundamentally new approach to sequence modeling. Adding Mamba would make AirTrace one of the first time series frameworks to incorporate this paradigm.

---

## Alternative 2: ModernTCN

**Full Title**: ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis
**Venue**: ICLR 2024 Spotlight
**Paper**: https://openreview.net/forum?id=vpJMJerXHU
**Code**: https://github.com/luodhhh/ModernTCN

### Key Innovations

1. **Much Larger Effective Receptive Fields (ERFs)**: Critical improvement over classic TCN
2. **Pure Convolution**: Demonstrates convolutions can match/exceed transformers when properly designed
3. **General Time Series Analysis**: SOTA on 5 tasks (long/short forecasting, imputation, classification, anomaly detection)
4. **Efficiency**: Maintains computational advantages of convolution-based models

### Why for AirTrace

1. **Natural Evolution**: AirTrace already has TCN; ModernTCN is the SOTA upgrade
2. **Proven Architecture**: ICLR 2024 Spotlight recognition validates quality
3. **Smooth Sensor Data**: Convolutions have excellent inductive bias for continuous sensor readings
4. **Production Deployment**: Lower computational cost than transformers for real-world systems

### Technical Improvements Over TCN

- Larger receptive fields through architectural innovations
- Better parameter efficiency
- Improved gradient flow for deeper networks
- Modern training techniques (normalization, activation choices)

### Performance

Achieves state-of-the-art across:
- Long-term forecasting
- Short-term forecasting
- Imputation
- Classification
- Anomaly detection

---

## Alternative 3: Temporal Fusion Transformer (TFT)

**Full Title**: Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting
**Original Paper**: https://arxiv.org/abs/1912.09363 (2019)
**Aviation Application**: "On the Exploration of Temporal Fusion Transformers for Anomaly Detection with Multivariate Aviation Time-Series Data" (MDPI Aerospace 2024)

### Why Despite Being Older

1. **Domain-Specific Success**: 2024 paper demonstrates explicit success on aircraft sensor data for anomaly detection
2. **Interpretability**: Critical for safety-critical applications like aviation
3. **Industry Proven**: Widely deployed in production systems
4. **Unique Capabilities**: Handles diverse input types and provides uncertainty quantification

### Key Features

**Multi-horizon Forecasting**:
- Explicitly designed for predicting multiple future timesteps
- Quantile regression for uncertainty estimation

**Interpretability**:
- Variable selection networks reveal which sensors are important
- Temporal attention shows which time steps matter
- Static covariate encoders handle metadata

**Flexible Input Handling**:
- Time-varying known inputs (flight plan, scheduled altitude)
- Time-varying unknown inputs (actual sensor readings)
- Static metadata (aircraft type, engine model, weather conditions)

### Aircraft-Specific Advantages

From the 2024 aviation paper:
- Successfully detects cascading failures in multivariate sensor data
- Identifies precursor events before catastrophic failures
- Handles sensor readout differences and drift
- Provides interpretable attention weights for failure analysis

---

## Implementation Priority

### Recommended Order

1. **TimeMixer** (Primary)
   - Most recent (ICLR 2024)
   - Best fit for multiscale aircraft sensor data
   - Complementary architecture to existing models
   - Excellent performance-efficiency tradeoff

2. **S-Mamba** (Secondary)
   - Most cutting-edge (novel paradigm)
   - Best for very long sequences
   - Represents future direction of sequence modeling
   - Unique offering in time series frameworks

3. **ModernTCN** (Tertiary)
   - Natural evolution of existing TCN
   - Easiest to implement (similar patterns)
   - Production-friendly (efficient)
   - Proven across all tasks

### Implementation Effort Estimates

**TimeMixer**: Medium
- Core PDM/FMM blocks
- Multiscale downsampling/upsampling
- Decomposition modules
- Estimated: 300-400 lines + tests + config

**S-Mamba**: High
- Selective scan mechanism
- State space model components
- Hardware-aware kernels (may use existing Mamba library)
- Estimated: 400-500 lines + tests + config + dependencies

**ModernTCN**: Low
- Similar to existing TCN
- Enhanced receptive field design
- Modern architectural improvements
- Estimated: 200-300 lines + tests + config

---

## Other Notable Models Considered

### Also Reviewed But Not Recommended (Yet)

1. **TimesNet** (ICLR 2023)
   - 2D vision-based temporal modeling
   - Innovative but less impactful than TimeMixer for this domain

2. **FreTS** (NeurIPS 2023)
   - Frequency-domain MLPs
   - Good for periodic patterns but aircraft sensors are more multiscale

3. **Timer-XL** (ICLR 2025)
   - Very new (just accepted)
   - Long-context unified forecasting
   - Code may not be mature yet; revisit in 6 months

4. **Foundation Models** (Chronos, TimesFM, MOIRAI)
   - Interesting for zero-shot
   - Very large, require significant infrastructure
   - Better as separate research direction

---

## Next Steps

1. **Decision**: Select model(s) to implement
2. **Implementation**: Follow AirTrace patterns
   - Create `src/airtrace/models/{model_name}.py`
   - Create `configs/model/{model_name}.yaml`
   - Create `tests/models/test_{model_name}.py`
   - Update `README.md` Model Registry
3. **Validation**: Run on synthetic data, compare to existing baselines
4. **Documentation**: Update architecture docs with new model

---

## References

- TimeMixer: https://arxiv.org/abs/2405.14616
- S-Mamba: https://arxiv.org/abs/2403.11144
- ModernTCN: https://openreview.net/forum?id=vpJMJerXHU
- TFT Aviation: https://www.mdpi.com/2226-4310/11/8/646
- Time-Series-Library: https://github.com/thuml/Time-Series-Library
