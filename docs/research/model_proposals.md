# Model Proposals for AirTrace

This document tracks proposals for adding models to AirTrace based on recent literature and identified gaps in the model registry.

**Last Updated**: 2025-11-20
**Status**: Active research proposals

---

## Current Status

**Implemented**: 46 models including:
- Modern attention architectures: PatchTST (ICLR 2023), iTransformer (ICLR 2024), Crossformer (ICLR 2023), TimeMixer (ICLR 2024), CycleNet (NeurIPS 2024), TFT, **TimeXer (NeurIPS 2024)**
- Classic transformers: Informer (AAAI 2021), Autoformer (NeurIPS 2021), FEDformer (ICML 2022), Non-stationary Transformer (NeurIPS 2022)
- Foundation models: Chronos-Bolt, Moirai, Mamba2, Lag-Llama, **MambaTS**
- MLP/Basis expansion: N-BEATS (ICLR 2020), DLinear, NLinear, **SOFTS (NeurIPS 2024)**, **FreTS (NeurIPS 2023)**
- RNNs/Seq2Seq: GRU, LSTM variants
- Baseline models: 16 statistical and simple baselines
- Convolutions: TCN, ModernTCN (ICLR 2024 Spotlight), **TimesNet (ICLR 2023)**

**Proposed**: ~9 significant architectures spanning 2023-2025, including latest foundation models and efficient architectures

---

## Priority Tier 1: Latest Foundation Models (ICLR 2025 / ICML 2024)

### 1. Time-MoE (ICLR 2025 Spotlight) ⭐⭐⭐ HIGHEST PRIORITY

**Full Title**: Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts
**Venue**: ICLR 2025 Spotlight (Top 5.1%)
**Paper**: https://arxiv.org/abs/2409.16040
**Code**: https://github.com/Time-MoE/Time-MoE
**OpenReview**: https://openreview.net/forum?id=e1wDDFmlVu

#### Key Innovations

1. **Billion-Scale**: First time series foundation model scaled up to 2.4 billion parameters
2. **Mixture of Experts (MoE)**: Sparse activation of expert networks for computational efficiency
3. **Time-300B Dataset**: Pre-trained on 300+ billion time points across 9 domains
4. **Universal Forecasting**: Arbitrary prediction horizons and context lengths up to 4096
5. **Decoder-Only**: Auto-regressive architecture like GPT for time series

#### Why for AirTrace

**State-of-the-Art Scale**: Time-MoE represents the cutting edge of foundation models:
- Validates scaling laws for time series (more parameters + more data = better performance)
- Sparse MoE design activates only subset of networks per prediction (efficient)
- Significantly outperforms dense models with same activated parameters
- Zero-shot forecasting capability on new datasets/domains

**Aircraft Sensor Applications**:
- Pre-trained on diverse domains → transfer learning to aviation
- Long context (4096) handles entire flight segments
- MoE could specialize experts for different flight phases or sensor types
- Foundation model baseline for fine-tuning on aircraft-specific data

#### Architecture Components

```
Decoder-Only Transformer with MoE:
  ├── Patch Embedding (tokenize time series)
  ├── Positional Encoding (up to 4096 tokens)
  ├── Transformer Blocks (stacked)
  │   ├── Self-Attention
  │   └── Mixture-of-Experts FFN
  │       ├── Router Network (select K experts)
  │       └── Expert Networks (sparse activation)
  └── Forecasting Head (predict next tokens)
```

#### Implementation Guidance

**Effort**: Very High (1500+ lines + infrastructure)

**Challenges**:
- Requires MoE training infrastructure
- Large model size (2.4B parameters for full model, smaller variants available)
- Pre-training dataset (Time-300B) is massive
- May need distributed training support

**Pragmatic Approach**:
1. **Option A**: Use pre-trained checkpoints from HuggingFace for fine-tuning
2. **Option B**: Implement smaller-scale version (e.g., 200M params) for experimentation
3. **Option C**: Wait for community implementations to mature

**Integration**:
- Add as foundation model similar to Chronos-Bolt, Moirai
- Support fine-tuning with LoRA adapters
- Implement zero-shot inference API
- Store router decisions for interpretability

**References**:
- Time-MoE Team (2025): "Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts"

---

### 2. Timer (ICML 2024) ⭐⭐⭐

**Full Title**: Timer: Generative Pre-trained Transformers Are Large Time Series Models
**Venue**: ICML 2024
**Paper**: Available via THUML GitHub
**Code**: https://github.com/thuml/Large-Time-Series-Model
**HuggingFace**: Pre-trained checkpoints available

#### Key Innovations

1. **Generative Pre-training**: GPT-style pre-training for time series
2. **Unified Framework**: Single model for forecasting, imputation, anomaly detection
3. **Zero-Shot Forecasting**: Pre-trained on UTSD (Unified Time Series Dataset)
4. **No Training/GPU Needed**: Out-of-the-box inference on new datasets

#### Why for AirTrace

**Production-Ready Foundation Model**:
- Simple to integrate: pre-trained checkpoints on HuggingFace
- Zero-shot capability: test on aircraft data without training
- General-purpose: works across domains and tasks
- Academic credibility: ICML 2024 acceptance

**Comparison to Time-MoE**:
- Smaller scale (easier to deploy)
- Proven on public benchmarks
- Active open-source community (THUML)

#### Implementation Guidance

**Effort**: Medium (400-500 lines + integration)

**Key Steps**:
1. Load pre-trained checkpoint from HuggingFace
2. Implement AirTrace adapter (convert to ARBaseModel interface)
3. Add patching/tokenization logic
4. Support fine-tuning pipeline
5. Evaluate zero-shot on synthetic data

**Integration**:
```python
@register("timer")
class TimerModel(ARBaseModel):
    def __init__(self, checkpoint="thuml/timer-base", ...):
        self.model = load_pretrained_timer(checkpoint)
        self.patcher = TimePatcher(patch_len=16)

    def forward(self, x):
        patches = self.patcher(x)  # [B, T, D] -> [B, N_patches, D_patch]
        out = self.model(patches)
        return {"preds": self.patcher.inverse(out)}
```

**References**:
- Liu et al. (2024): "Timer: Generative Pre-trained Transformers Are Large Time Series Models"

---

### 3. TimesFM (Google, ICML 2024) ⭐⭐

**Full Title**: A decoder-only foundation model for time-series forecasting
**Venue**: ICML 2024
**Paper**: Google Research Blog
**Code**: https://github.com/google-research/timesfm
**HuggingFace**: google/timesfm-1.0-200m
**Status**: Implemented in AirTrace as `TimesFMModel` (see `configs/model/timesfm.yaml`)

#### Key Innovations

1. **Google-Backed**: 200M parameter Transformer from Google Research
2. **100B Time Points**: Pre-trained on massive real-world corpus
3. **Patch-Based**: Treats patches (groups of time points) as tokens
4. **Outperforms Baselines**: Beats DeepAR and GPT-3-based approaches by >25%

#### Why for AirTrace

**Industrial-Grade**:
- Google-maintained and battle-tested
- Integrated into Google Cloud BigQuery
- Strong zero-shot performance
- Well-documented and supported

**Aircraft Applications**:
- Production-ready for deployment
- Handles variable-length sequences
- Domain-agnostic (tested across industries)

#### Implementation Guidance

**Effort**: Medium (300-400 lines)

**Availability**: Pre-trained model on HuggingFace makes integration straightforward.

**References**:
- Das et al. (2024): "A decoder-only foundation model for time-series forecasting"

---

### 4. MOMENT (ICML 2024) ⭐⭐

**Full Title**: MOMENT: A Family of Open Time-series Foundation Models
**Venue**: ICML 2024
**Paper**: https://arxiv.org/abs/2402.03885
**Code**: https://github.com/moment-timeseries-foundation-model/moment
**HuggingFace**: AutonLab/MOMENT-1-large

#### Key Innovations

1. **Open-Source Foundation Models**: Fully open-source family of pre-trained models
2. **Time-series Pile**: Large diverse collection of public time-series for pre-training
3. **General-Purpose**: Works across forecasting, classification, anomaly detection, imputation
4. **Limited Supervision**: Designed for few-shot and zero-shot scenarios

#### Why for AirTrace

**Open Research**:
- Carnegie Mellon University (Auton Lab) backed
- Fully reproducible with public datasets and code
- Active research community
- Benchmark suite included

**Practical Benefits**:
- Multiple model sizes available
- Pre-training code available (can adapt to aviation data)
- Well-documented and tested
- Academic rigor with practical focus

#### Implementation Guidance

**Effort**: Medium (300-400 lines)

**Resources**:
- Pre-trained models on HuggingFace
- Time-series Pile dataset available
- Comprehensive benchmarks provided

**References**:
- Goswami et al. (2024): "MOMENT: A Family of Open Time-series Foundation Models"

---

## Priority Tier 2: State Space Models (Mamba Variants)

### 5. S-Mamba (Simple-Mamba) ⭐

**Full Title**: Is Mamba Effective for Time Series Forecasting?
**Venue**: Neurocomputing, Volume 619 (February 2025)
**Paper**: https://arxiv.org/abs/2403.11144
**Code**: https://github.com/wzhwzhwzh0921/S-D-Mamba

#### Key Innovations

1. **Simple Design**: Straightforward Mamba application to time series
2. **Tokenization**: Each variate tokenized autonomously via linear layer
3. **Bidirectional Mamba**: Extract inter-variate correlations
4. **FFN for Temporal**: Feed-forward network learns temporal dependencies

#### Why for AirTrace

**Simplicity**:
- Easier to implement than MambaTS (already implemented)
- Good baseline for Mamba-based approaches
- Demonstrates Mamba effectiveness empirically

**Comparison Point**:
- Simpler than AirTrace's current Mamba2 and MambaTS
- Could be faster for some tasks
- Different design philosophy (simple vs. complex)

#### Implementation Guidance

**Effort**: Low-Medium (200-300 lines)

**References**:
- Authors (2024): "Is Mamba Effective for Time Series Forecasting?"

---

## Priority Tier 3: Pure MLP Models

Simple, efficient, non-sequential architectures that surprisingly compete with transformers.

### 6. N-HiTS (AAAI 2023) ⭐

**Full Title**: N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting
**Venue**: AAAI 2023
**Paper**: https://arxiv.org/abs/2201.12886
**Code**: https://github.com/Nixtla/neuralforecast

**Status**: Implemented as `NHiTSModel` (`configs/model/nhits.yaml`).

#### Key Innovations

1. **Hierarchical Interpolation**: Multi-rate data sampling (like TimeMixer but simpler)
2. **Better than N-BEATS**: Improved long-horizon forecasting
3. **MaxPool Downsampling**: Captures patterns at different scales

#### Why for AirTrace

- Evolution of N-BEATS (already implemented) with better long-term performance
- Multi-scale (like TimeMixer) but pure MLP
- Interpretable hierarchical structure

#### Implementation Guidance

**Effort**: Medium (250-350 lines) - Similar to N-BEATS with pooling layers.

**Note**: AirTrace already has N-BEATS; N-HiTS is the improved successor.

**References**:
- Challu et al. (2023): "N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting"

---

## Priority Tier 4: Frequency Domain & Novel Paradigms

### 8. ETSformer (ICML 2023) ⭐

**Full Title**: ETSformer: Exponential Smoothing Transformers for Time-series Forecasting
**Venue**: ICML 2023
**Paper**: https://arxiv.org/abs/2202.01381
**Code**: https://github.com/salesforce/ETSformer

#### Key Innovations

1. **Exponential Smoothing + Transformers**: Combines classical stats with deep learning
2. **Interpretable Latent Components**: Level, growth, seasonality
3. **Frequency Attention**: Attention in Fourier domain

#### Why for Aircraft

- Interpretable components for safety-critical applications
- Combines physics-based priors (smoothing) with learning
- Salesforce-backed implementation

#### Implementation Guidance

**Effort**: Medium-High (400-500 lines) - ETS components + frequency attention.

**References**:
- Woo et al. (2023): "ETSformer: Exponential Smoothing Transformers for Time-series Forecasting"

---

## Priority Tier 5: LLM-Based Models (Future Research)

### 10. TIME-LLM (ICLR 2024) ⭐

**Full Title**: Time-LLM: Time Series Forecasting by Reprogramming Large Language Models
**Venue**: ICLR 2024
**Paper**: https://arxiv.org/abs/2310.01728
**Code**: https://github.com/KimMeen/Time-LLM

#### Key Innovations

1. **LLM Reprogramming**: Uses frozen LLaMA-7B for time series
2. **Patching + Text Prototypes**: Converts time series to "language"
3. **Zero-Shot Transfer**: Leverages LLM's reasoning

#### Why for AirTrace

- Cutting-edge paradigm for time series
- Potential for reasoning about anomalies
- Zero-shot on new aircraft types

#### Challenges

- **Infrastructure**: Requires LLM hosting (7B+ parameters)
- **Integration**: Significant changes to pipeline
- **Compute**: GPU memory intensive

#### Implementation Guidance

**Effort**: Very High (1000+ lines + LLM dependencies + infrastructure)

**Recommendation**: Future research direction. Foundation models (Time-MoE, Timer, TimesFM) are more practical currently.

**References**:
- Jin et al. (2024): "Time-LLM: Time Series Forecasting by Reprogramming Large Language Models"

---

### 11. TEMPO (ICLR 2024) ⭐

**Full Title**: TEMPO: Prompt-based Generative Pre-trained Transformer for Time Series Forecasting
**Venue**: ICLR 2024
**Paper**: https://arxiv.org/abs/2310.04948
**Code**: https://github.com/DC-research/TEMPO

#### Key Innovations

1. **GPT-style Pre-training**: Unified framework for multiple tasks
2. **Prompt-based**: Task specification via prompts
3. **Foundation Model**: Pre-trained on diverse datasets

#### Implementation Guidance

**Effort**: Very High (similar to TIME-LLM)

**Recommendation**: Future work. Similar to TIME-LLM but with prompting interface. Other foundation models are more mature.

**References**:
- Cao et al. (2024): "TEMPO: Prompt-based Generative Pre-trained Transformer for Time Series Forecasting"

---

## Implementation Priority Summary

### Tier 1: Foundation Models (Highest Priority)
**Rationale**: Zero-shot capability, state-of-the-art performance, production-ready

1. **Time-MoE** (ICLR 2025 Spotlight) - Billion-scale, MoE architecture - **Effort**: Very High
2. **Timer** (ICML 2024) - GPT-style, mature implementation - **Effort**: Medium
3. **TimesFM** (Google, ICML 2024) - Industrial-grade, well-supported - **Effort**: Medium
4. **MOMENT** (ICML 2024) - Open-source, research-friendly - **Effort**: Medium

**Recommendation**: Start with **Timer** or **TimesFM** for easiest integration, then explore Time-MoE.

### Tier 2: State Space Models
**Rationale**: Alternative to existing Mamba2/MambaTS implementations

5. **S-Mamba** - Simple Mamba baseline - **Effort**: Low-Medium

### Tier 3: MLP Baselines (Quick Wins)
**Rationale**: Simple, fast, often competitive

6. **N-HiTS** (AAAI 2023) - Successor to N-BEATS - **Effort**: Medium

**Recommendation**: **N-HiTS** for improved N-BEATS baseline.

### Tier 4: Novel Paradigms (Research Interest)
**Rationale**: Different inductive biases, worth exploring

8. **ETSformer** (ICML 2023) - Classical + DL hybrid - **Effort**: Medium-High

**Recommendation**: **ETSformer** for interpretable classical-DL hybrid approach.

### Tier 5: Future Research (Long-term)
**Rationale**: Requires significant infrastructure or immature tooling

10. **TIME-LLM** (ICLR 2024) - LLM-based forecasting
11. **TEMPO** (ICLR 2024) - Prompt-based foundation model

**Recommendation**: Monitor development, wait for community implementations to mature.

---

## Recommended Implementation Roadmap

### Phase 1: Foundation Models (Priority)
**Timeline**: 1-2 months

1. **Timer** or **TimesFM** - Pick one based on preference (THUML vs Google)
   - Integrate pre-trained checkpoints
   - Test zero-shot on synthetic aircraft data
   - Evaluate fine-tuning pipeline
   - **Deliverable**: Foundation model baseline

2. **MOMENT** - Open-source alternative
   - Academic rigor, reproducible
   - **Deliverable**: Research comparison point

### Phase 2: Quick Wins & Baselines
**Timeline**: 1-2 weeks each

3. **N-HiTS** - Successor to N-BEATS

**Deliverable**: Comprehensive MLP baseline suite

### Phase 3: Research Exploration (Optional)
**Timeline**: Variable

4. **S-Mamba** - Simple Mamba variant
5. **ETSformer** - Classical-DL hybrid

**Deliverable**: Research papers, novel approaches

### Phase 4: Future (Monitor)
**Timeline**: TBD

- **Time-MoE** - When compute infrastructure ready (or use smaller checkpoints)
- **TIME-LLM / TEMPO** - When tooling matures

---

## Testing and Validation Strategy

For each new model:

1. **Unit Tests**:
   ```python
   def test_model_forward():
       model = NewModel(input_dim=10, output_dim=5)
       x = torch.randn(32, 128, 10)  # [B, T, D]
       out = model(x)
       assert out["preds"].shape == (32, 1, 5)  # or (32, pred_len, 5)
   ```

2. **Config Test**: Ensure config loads and instantiates

3. **Synthetic Data**: Train on AirTrace synthetic data, verify convergence

4. **Baseline Comparison**: Compare to existing models on same task

5. **Interpretability** (if applicable): Verify attention weights, decompositions

6. **README Update**: Add to Model Registry table (REQUIRED!)

---

## References

### Foundation Models (2024-2025)
- Time-MoE Team (2025): "Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts" (ICLR 2025 Spotlight)
- Liu et al. (2024): "Timer: Generative Pre-trained Transformers Are Large Time Series Models" (ICML 2024)
- Das et al. (2024): "A decoder-only foundation model for time-series forecasting" (ICML 2024, Google)
- Goswami et al. (2024): "MOMENT: A Family of Open Time-series Foundation Models" (ICML 2024)

### State Space Models (2024-2025)
- S-Mamba Authors (2024): "Is Mamba Effective for Time Series Forecasting?" (Neurocomputing 2025)

### MLP and Novel Paradigms (2023)
- Challu et al. (2023): "N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting" (AAAI 2023)
- Woo et al. (2023): "ETSformer: Exponential Smoothing Transformers for Time-series Forecasting" (ICML 2023)

### LLM-Based (2024)
- Jin et al. (2024): "Time-LLM: Time Series Forecasting by Reprogramming Large Language Models" (ICLR 2024)
- Cao et al. (2024): "TEMPO: Prompt-based Generative Pre-trained Transformer for Time Series Forecasting" (ICLR 2024)

### Code Repositories
- Time-MoE: https://github.com/Time-MoE/Time-MoE
- Timer/OpenLTM: https://github.com/thuml/Large-Time-Series-Model
- TimesFM: https://github.com/google-research/timesfm
- MOMENT: https://github.com/moment-timeseries-foundation-model/moment
- S-D-Mamba: https://github.com/wzhwzhwzh0921/S-D-Mamba
- NeuralForecast: https://github.com/Nixtla/neuralforecast
- Time-Series-Library: https://github.com/thuml/Time-Series-Library

---

## Recently Implemented Models

The following models from previous versions of this document have been successfully implemented in AirTrace:

- **FreTS** (NeurIPS 2023) - Frequency-domain MLP with low-frequency focus
- **TSMixer** (KDD 2023) - All-MLP architecture with time and feature mixing
- **TimeXer** (NeurIPS 2024) - Exogenous variable handling
- **SOFTS** (NeurIPS 2024) - Efficient multivariate with STAR module
- **MambaTS** - Improved selective state space model
- **TimesNet** (ICLR 2023) - 2D vision backbone for time series
- **TFT** (Temporal Fusion Transformer)
- **ModernTCN** (ICLR 2024 Spotlight)
- **DLinear / NLinear**
- **Informer** (AAAI 2021)
- **Autoformer** (NeurIPS 2021)
- **FEDformer** (ICML 2022)
- **Non-stationary Transformer** (NeurIPS 2022)
- **Crossformer** (ICLR 2023)
- **N-BEATS** (ICLR 2020)

See [Model Registry](../../README.md#model-registry) for complete list of implemented models.
