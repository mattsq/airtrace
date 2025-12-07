# Model Selection Guide

A comprehensive reference to help you choose the right model architecture for your time series forecasting task based on data characteristics, problem requirements, and computational constraints.

## Table of Contents

1. [Quick Decision Guide](#quick-decision-guide)
2. [Understanding Your Data](#understanding-your-data)
3. [Model Families Overview](#model-families-overview)
4. [Selection by Data Characteristics](#selection-by-data-characteristics)
5. [Selection by Problem Requirements](#selection-by-problem-requirements)
6. [Computational Considerations](#computational-considerations)
7. [Detailed Model Comparisons](#detailed-model-comparisons)
8. [Practical Recommendations](#practical-recommendations)

---

## Quick Decision Guide

**Start here for a rapid recommendation:**

| Your Situation | Recommended Model | Alternative |
|----------------|-------------------|-------------|
| Just starting, need a baseline | `persistence`, `linear_ar` | `moving_average`, `gru_ar` |
| Limited data (<1000 samples) | `dlinear`, `nlinear`, `linear_ar` | `gru_ar`, `tsmixer` |
| Moderate data, good default | `gru_ar`, `moderntcn` | `patchtst`, `tsmixer` |
| Large dataset, best accuracy | `patchtst`, `itransformer` | `softs`, `timemixer` |
| Long sequences (>1024 steps) | `mamba2`, `mambats`, `moderntcn` | `patchtst`, `informer` |
| Strong multivariate dependencies | `itransformer`, `crossformer` | `tft`, `var` |
| Non-stationary, regime changes | `nonstationary_transformer` | `timesnet`, `autoformer` |
| Limited compute/memory | `dlinear`, `nlinear`, `cyclenet` | `tsmixer`, `frets` |
| Need pretrained/transfer learning | `chronos_bolt`, `timer`, `moirai` | `lag_llama`, `mamba2` |
| Need interpretability | `nbeats`, `tft`, `dlinear` | `linear_trend`, `theta` |
| Real-time inference critical | `dlinear`, `nlinear`, `cyclenet` | `gru_ar`, `moderntcn` |

---

## Understanding Your Data

Before selecting a model, characterize your data along these dimensions:

### 1. Sequence Length

**Short sequences (< 256 timesteps):**
- Most models work well
- RNNs (GRU, LSTM) handle these efficiently
- Simple baselines may suffice

**Medium sequences (256-1024 timesteps):**
- Standard Transformers viable with PatchTST patching
- TCNs with dilated convolutions effective
- State-space models (Mamba) excel here

**Long sequences (> 1024 timesteps):**
- Transformers face O(n²) complexity issues
- **Recommended**: Mamba/state-space models (linear complexity)
- **Alternative**: ModernTCN, PatchTST, Informer with sparse attention

### 2. Number of Variables (Multivariate Complexity)

**Univariate (1 variable):**
- Any model works; simpler models often best
- Consider: DLinear, N-BEATS, ARIMA baselines

**Low-dimensional multivariate (2-10 variables):**
- Channel-independent models (PatchTST) work well
- Lightweight MLPs (TSMixer, SOFTS) efficient
- GRU/LSTM handle these easily

**High-dimensional multivariate (>10 variables):**
- **Strong cross-variate dependencies**: iTransformer, Crossformer, TFT
- **Weak cross-variate dependencies**: PatchTST (channel-independent)
- Avoid full attention over all variates (quadratic cost)

### 3. Stationarity

**Stationary data** (constant mean/variance):
- Most models work; simpler is better
- Linear models, persistence baselines strong
- Standard Transformers, GRUs effective

**Non-stationary data** (trends, regime shifts):
- **Decomposition-based**: Autoformer, DLinear, N-BEATS
- **Adaptive**: NonStationary Transformer, TimesNet
- **Preprocessing**: Use `transforms=zscore_diff` to difference

### 4. Seasonality and Periodicity

**No clear seasonality:**
- Standard models: GRU, Transformer, TCN
- Avoid seasonal-specific models

**Regular seasonality (known period):**
- **Explicit modeling**: TimesNet, Holt-Winters, Seasonal Naive
- **Decomposition**: Autoformer, N-BEATS
- **Frequency domain**: FreTS, FEDformer (Fourier/Wavelet)

**Complex multi-scale patterns:**
- TimesNet (intraperiod + interperiod)
- TimeMixer (multiscale mixing)
- Autoformer (auto-correlation)

### 5. Data Availability

**Limited training data (<1000 samples):**
- **Best**: Simple models with strong inductive bias
  - DLinear, NLinear (linear projections)
  - Linear AR, persistence baselines
- **Avoid**: Large Transformers, foundation models (overfitting risk)
- **Consider**: Pretrained foundation models with fine-tuning

**Moderate data (1K-100K samples):**
- Sweet spot for neural models
- GRU, ModernTCN, PatchTST, TSMixer

**Large data (>100K samples):**
- Full Transformers, large foundation models viable
- iTransformer, Crossformer for multivariate
- Consider pretrained models: Chronos-Bolt, Timer

**No task-specific data (zero-shot):**
- Foundation models only: `chronos_bolt`, `timer`, `moirai`

---

## Model Families Overview

### Recurrent Neural Networks (RNNs)

**Models**: `gru_ar`, `lstm_ar`, `gru_seq2seq`, `lstm_seq2seq`

**Theoretical Basis**:
- Sequential state updates with gating mechanisms
- GRU: 2 gates (update, reset) - simpler, faster
- LSTM: 3 gates (input, forget, output) + cell state - more expressive

**Strengths**:
- Naturally handle sequential dependencies
- Work well on short-to-medium sequences (<512 steps)
- Good with limited data (fewer parameters than Transformers)
- GRU training is fast, efficient

**Weaknesses**:
- Gradient issues on very long sequences (though mitigated by gating)
- Sequential computation prevents parallelization
- Can forget distant past despite gating

**Best For**:
- Default choice for moderate-length sequences
- When training speed matters
- Sequential patterns with moderate lookback

**Research Support**: [RNNs, LSTMs, GRUs for time series](https://encord.com/blog/time-series-predictions-with-recurrent-neural-networks/), [Performance comparison](https://pmc.ncbi.nlm.nih.gov/articles/PMC12329085/)

---

### Convolutional Models (TCNs)

**Models**: `tcn`, `moderntcn`, `timesnet`

**Theoretical Basis**:
- Dilated causal convolutions with exponentially growing receptive fields
- ModernTCN: Depthwise separable convolutions, large kernels
- TimesNet: 2D convolutions on period-reshaped series

**Strengths**:
- Highly parallelizable (unlike RNNs)
- Stable gradients via residual connections
- Flexible receptive field via dilation
- Efficient inference (no recurrence)
- ModernTCN: State-of-the-art efficiency

**Weaknesses**:
- Fixed receptive field (determined by depth/dilation)
- May need deep networks for long dependencies
- Less intuitive than attention for interpretability

**Best For**:
- Long sequences where parallelization matters
- Real-time applications (fast inference)
- When stable training is critical
- ModernTCN specifically for efficiency + accuracy

**Research Support**: [TCN overview](https://unit8.com/resources/temporal-convolutional-networks-and-forecasting/), [Dilated convolutions](https://www.activeloop.ai/resources/glossary/temporal-convolutional-networks-tcn/)

---

### Transformer-Based Models

**Models**: `transformer`, `informer`, `autoformer`, `fedformer`, `patchtst`, `itransformer`, `crossformer`, `nonstationary_transformer`, `tft`, `timexer`

**Theoretical Basis**:
- Self-attention mechanism captures pairwise dependencies
- Standard: O(n²) complexity in sequence length
- Variants use sparse attention, decomposition, or patching

**Strengths**:
- Capture long-range dependencies via attention
- Parallelizable (unlike RNNs)
- Interpretable via attention weights (especially TFT)
- Strong performance on complex patterns

**Weaknesses**:
- Quadratic complexity limits sequence length (standard)
- High memory usage
- Require substantial data to train
- Loss of positional information

**Model-Specific Recommendations**:

- **`patchtst`**: Patches reduce complexity, channel-independent = efficient
  - **Use when**: Long sequences, weak cross-variate correlations

- **`itransformer`**: Variates as tokens (inverted), strong multivariate modeling
  - **Use when**: Strong cross-variate dependencies, many variables

- **`crossformer`**: Two-stage (temporal → cross-dimension) attention
  - **Use when**: Explicit cross-sensor dependencies needed

- **`autoformer`**: Series decomposition + auto-correlation
  - **Use when**: Non-stationary, trend/seasonal decomposition helpful

- **`informer`**: ProbSparse attention for long sequences
  - **Use when**: Long-horizon forecasting, efficiency critical

- **`nonstationary_transformer`**: Learnable de-stationarization
  - **Use when**: Distribution shifts, regime changes

- **`tft`**: Variable selection + interpretable attention
  - **Use when**: Interpretability matters, exogenous variables

**Research Support**: [Transformers for time series](https://www.geeksforgeeks.org/deep-learning/transformer-for-time-series-forecasting/), [Local attention](https://arxiv.org/abs/2410.03805), [Long-range dependencies](https://link.springer.com/article/10.1007/s10462-024-11044-2)

---

### State-Space Models (Mamba)

**Models**: `mamba2`, `mambats`, `moirai` (Mamba-based foundation)

**Theoretical Basis**:
- Selective state-space models with hardware-aware implementations
- Linear O(n) complexity via selective scan
- Bidirectional gating, chunked processing

**Strengths**:
- **Linear complexity** - handle very long sequences (100k+ tokens)
- Efficient as convolution in training, recurrent in inference
- Global receptive field without quadratic cost
- Strong performance competitive with Transformers

**Weaknesses**:
- Newer architecture, less mature tooling
- Less interpretable than attention
- Requires careful initialization

**Best For**:
- Very long sequences (>1024 steps)
- When Transformer memory is prohibitive
- Deployment scenarios needing efficiency
- Foundation model use (Moirai)

**Research Support**: [MambaTS](https://arxiv.org/html/2405.16440v1), [Mamba effectiveness](https://arxiv.org/abs/2403.11144), [Linear complexity](https://medium.com/data-science-in-your-pocket/tsmamba-mamba-model-for-time-series-forecasting-c9eeb0d0d23c)

---

### MLP-Based Models

**Models**: `tsmixer`, `timemixer`, `softs`, `nbeats`, `cyclenet`, `dlinear`, `nlinear`, `frets`

**Theoretical Basis**:
- Pure MLP architectures with mixing operations
- DLinear/NLinear: Simple linear projections with decomposition
- TSMixer: Alternating time-mixing and feature-mixing
- SOFTS: STAR module with stochastic aggregate-redistribute
- N-BEATS: Residual stacks with basis expansion

**Strengths**:
- **Simplicity**: Easy to implement, debug, understand
- **Efficiency**: Fewer parameters, fast inference
- **Strong inductive bias**: Linear models work surprisingly well
- **Parameter efficiency**: DLinear/NLinear extremely lightweight
- **Competitive performance**: TSMixer outperforms Transformers on benchmarks

**Weaknesses**:
- Limited expressiveness for highly complex patterns
- Fixed (not data-dependent) temporal relationships (linear models)
- May underfit on very large datasets

**Best For**:
- Limited computational budget
- Real-time inference requirements
- When simpler is better (avoid overfitting)
- DLinear/NLinear: Small data, need speed
- TSMixer/SOFTS: Moderate data, want strong baseline
- N-BEATS: Interpretable trend/seasonality

**Research Support**: [TSMixer overview](https://research.google/blog/tsmixer-an-all-mlp-architecture-for-time-series-forecasting/), [Linear models effectiveness](https://arxiv.org/abs/2303.06053), [Parameter efficiency](https://medium.com/@kdk199604/tsmixer-rethinking-time-series-forecasting-with-all-mlp-design-0db2b169c025)

---

### Foundation Models (Pretrained)

**Models**: `chronos_bolt`, `timer`, `moirai`, `lag_llama`, `mamba2` (with pretraining)

**Theoretical Basis**:
- Pretrained on massive diverse time series datasets
- Transfer learning: leverage patterns from one domain to another
- Zero-shot: predict without task-specific training

**Strengths**:
- **Zero-shot capability**: Work without task-specific data
- **Transfer learning**: Fine-tune on small datasets effectively
- **Generalization**: Leverage broad patterns from pretraining
- **Fast adaptation**: Less data needed for new tasks

**Weaknesses**:
- Large model sizes (memory/compute)
- May not match task-specific models on in-domain data
- Sensitivity to train/test domain alignment
- "Black box" - less interpretable

**Model-Specific**:
- **`chronos_bolt`**: Gated conv-attention, LoRA adapters
- **`timer`**: GPT-style decoder, 260B pretraining points
- **`moirai`**: Multiresolution Mamba, hierarchical patching
- **`lag_llama`**: Retrieval-augmented diffusion forecaster

**Best For**:
- Little/no task-specific training data
- Rapid prototyping across domains
- Transfer learning scenarios
- When pretraining domains align with your task

**Research Support**: [Chronos zero-shot](https://www.manning.com/books/time-series-forecasting-using-foundation-models), [TimesFM decoder](https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/), [Foundation model challenges](https://arxiv.org/html/2510.00742v2)

---

### Baseline Models

**Models**: `persistence`, `moving_average`, `zero`, `mean`, `median`, `linear_trend`, `polynomial_trend`, `drift`, `exponential_smoothing`, `holt_linear_trend`, `holt_winters`, `seasonal_naive`, `theta`, `sarima`, `var`, `linear_ar`, `mlp_ar`

**Theoretical Basis**:
- Simple statistical or linear methods
- Strong assumptions (stationarity, linearity, fixed seasonality)
- Many are parameter-free (non-trainable)

**Strengths**:
- Fast, interpretable, no training needed (most)
- Surprisingly strong on many real-world tasks
- Essential for benchmarking neural models
- Work with tiny datasets

**Weaknesses**:
- Limited capacity for complex patterns
- Fixed assumptions may not hold

**Best For**:
- **Always start here**: Establish baseline performance
- Comparison: Ensure neural models justify complexity
- Simple patterns: Linear trends, basic seasonality
- Interpretability requirements

**Specific Recommendations**:
- **`persistence`**: Naive baseline, predict last value
- **`linear_ar`**, **`mlp_ar`**: Trainable linear/MLP baselines
- **`var`**: Multivariate with cross-sensor dynamics
- **`sarima`**: Classical seasonal ARIMA
- **`theta`**: M3 competition winner, surprisingly effective

---

## Selection by Data Characteristics

### Long Sequences (>1024 timesteps)

**Top Recommendations**:
1. **`mamba2`**, **`mambats`** - Linear complexity, handle 100k+ tokens
2. **`moderntcn`** - Efficient dilated convolutions, large receptive field
3. **`patchtst`** - Patching reduces O(n²) to O((n/p)²)
4. **`informer`** - ProbSparse attention for long horizons

**Avoid**: Standard Transformer (memory limits), RNNs (gradient issues)

### Strong Multivariate Dependencies

**Top Recommendations**:
1. **`itransformer`** - Variates as tokens, explicit correlation modeling
2. **`crossformer`** - Two-stage temporal + cross-dimension attention
3. **`tft`** - Variable selection, handles static/known/unknown variables
4. **`var`** - Classical multivariate autoregression (baseline)

**Avoid**: Channel-independent models (PatchTST), univariate approaches

### Non-Stationary, Distribution Shifts

**Top Recommendations**:
1. **`nonstationary_transformer`** - Learnable de-stationarization
2. **`timesnet`** - Period-based reshaping, adaptive to regime changes
3. **`autoformer`** - Series decomposition for trend/seasonality
4. **Use preprocessing**: `transforms=zscore_diff` (differencing stabilizes)

**Avoid**: Models assuming stationarity without preprocessing

### Strong Seasonality/Periodicity

**Top Recommendations**:
1. **`timesnet`** - Intraperiod + interperiod variation modeling
2. **`autoformer`** - Auto-correlation for seasonal patterns
3. **`fedformer`** - Frequency-domain (Fourier/Wavelet) attention
4. **`frets`** - MLP on Fourier coefficients
5. **Classical**: `holt_winters`, `seasonal_naive` (baselines)

**Avoid**: Models without seasonal awareness (basic GRU, linear models)

---

## Selection by Problem Requirements

### Need Interpretability

**Top Recommendations**:
1. **`nbeats`** - Interpretable trend/seasonality blocks
2. **`tft`** - Variable selection networks, attention weights
3. **`dlinear`** - Transparent trend + seasonal linear projections
4. **Baselines**: `linear_trend`, `holt_winters` (fully interpretable)

**Avoid**: Black-box foundation models, deep RNNs

### Real-Time Inference Critical

**Top Recommendations**:
1. **`dlinear`**, **`nlinear`** - Minimal computation, single forward pass
2. **`cyclenet`** - Extreme efficiency for periodic forecasting
3. **`moderntcn`** - Fast convolutions, no recurrence
4. **`gru_ar`** - Lightweight RNN, fast incremental updates

**Avoid**: Large Transformers, foundation models (high latency)

### Limited Computational Budget

**Training Efficiency**:
1. **`dlinear`**, **`nlinear`** - Fastest to train
2. **`cyclenet`**, **`frets`** - Efficient MLPs
3. **`moderntcn`** - Much faster than Transformers

**Inference Efficiency**:
1. **`dlinear`**, **`nlinear`**, **`cyclenet`** - Smallest memory footprint
2. **`tsmixer`**, **`softs`** - Lightweight MLPs
3. **`gru_ar`** - Small RNN

**Avoid**: Large Transformers, foundation models (high compute)

### Maximum Accuracy (Regardless of Cost)

**Top Recommendations** (based on recent benchmarks):
1. **`patchtst`** - ICLR 2023, strong across benchmarks
2. **`itransformer`** - ICLR 2024 Spotlight, SOTA multivariate
3. **`softs`** - NeurIPS 2024, efficient channel mixing
4. **`timemixer`** - ICLR 2024, multiscale mixing
5. **`moderntcn`** - ICLR 2024 Spotlight, surprising MLP power

**Strategy**: Try multiple, ensemble if possible

---

## Computational Considerations

### Training Complexity

| Model Family | Time Complexity | Memory | Training Speed |
|--------------|-----------------|--------|----------------|
| Linear (DLinear, NLinear) | O(n) | Very Low | Fastest |
| MLP (TSMixer, SOFTS) | O(n) | Low | Very Fast |
| RNN (GRU, LSTM) | O(n) | Low | Fast (but sequential) |
| TCN (ModernTCN) | O(n log n) | Low | Very Fast (parallel) |
| Transformer | O(n²) | High | Slow (quadratic) |
| PatchTST | O((n/p)²) | Medium | Medium (p=patch size) |
| Mamba | O(n) | Low | Fast (linear) |

### Inference Latency

**Fastest** (< 1ms typical):
- `dlinear`, `nlinear`, `cyclenet`
- Baselines: `persistence`, `linear_ar`

**Fast** (1-10ms):
- `moderntcn`, `tsmixer`, `softs`, `frets`
- `gru_ar` (incremental)

**Medium** (10-100ms):
- `patchtst`, `mamba2`
- Small Transformers

**Slow** (>100ms):
- Large Transformers (`itransformer`, `crossformer`, `tft`)
- Foundation models (`chronos_bolt`, `timer`)

### Memory Footprint

**Smallest**:
- Linear models, baselines: <1MB
- `gru_ar`, `lstm_ar`: 1-10MB

**Small**:
- `moderntcn`, `tsmixer`: 10-50MB
- `patchtst` (channel-independent): 10-100MB

**Medium**:
- Small Transformers: 50-200MB
- `mamba2`: 50-200MB

**Large**:
- Full Transformers: 100-500MB
- Foundation models: 200MB-2GB+

---

## Detailed Model Comparisons

### RNNs: GRU vs LSTM

| Aspect | GRU | LSTM |
|--------|-----|------|
| Gates | 2 (update, reset) | 3 (input, forget, output) + cell |
| Parameters | ~33% fewer | More parameters |
| Training Speed | Faster | Slower |
| Long-term Memory | Good | Better (explicit cell state) |
| **Recommendation** | Default choice | Very long dependencies |

**When to use GRU**: Most cases, faster training matters
**When to use LSTM**: Need maximum long-term memory, have compute budget

### Transformers: PatchTST vs iTransformer

| Aspect | PatchTST | iTransformer |
|--------|----------|--------------|
| Token Definition | Time patches | Individual variates |
| Complexity Reduction | Patching (n/p)² | Inverted (depends on #variates) |
| Multivariate Modeling | Channel-independent | Explicit cross-variate attention |
| Best For | Weak cross-variate | Strong cross-variate |
| Lookback Scaling | Excellent (patches) | Good (if few variates) |

**When to use PatchTST**: Long sequences, many independent variables
**When to use iTransformer**: Strong multivariate correlations, fewer variables

### MLPs: DLinear vs TSMixer vs SOFTS

| Aspect | DLinear | TSMixer | SOFTS |
|--------|---------|---------|-------|
| Architecture | Linear projection | Time + feature mixing MLPs | STAR aggregate-redistribute |
| Decomposition | Trend + seasonal | Implicit | Implicit |
| Parameters | Minimal | Low | Low |
| Multivariate | Weak | Good (feature mixing) | Strong (STAR pooling) |
| Interpretability | High | Medium | Medium |
| Speed | Fastest | Very Fast | Very Fast |

**When to use DLinear**: Simplicity, speed, small data
**When to use TSMixer**: Better multivariate, still efficient
**When to use SOFTS**: Strong multivariate channel mixing, SOTA efficiency

### Foundation Models: Chronos-Bolt vs Timer vs Moirai

| Aspect | Chronos-Bolt | Timer | Moirai |
|--------|--------------|-------|--------|
| Architecture | Gated conv-attention | GPT decoder | Mamba SSM |
| Pretraining Data | Diverse (Amazon) | 260B time points (Google) | Multiresolution |
| Parameters | ~100M (with LoRA) | 200M | Varies |
| Adaptation | LoRA fine-tuning | In-context fine-tuning | LoRA adapters |
| Complexity | Medium | Medium | Linear (Mamba) |

**When to use Chronos-Bolt**: Need LoRA adaptation, gated architecture
**When to use Timer**: Google ecosystem, GPT-style decoder
**When to use Moirai**: Long sequences (Mamba), hierarchical patterns

---

## Practical Recommendations

### Workflow for Model Selection

1. **Start with Baselines** (always):
   ```bash
   airtrace train model=persistence
   airtrace train model=linear_ar
   ```
   Establish floor performance. If baselines are strong, simpler is better.

2. **Quick Neural Baseline**:
   ```bash
   airtrace train model=gru_ar  # Solid default
   airtrace train model=dlinear  # Fast MLP baseline
   ```

3. **Characterize Your Data**:
   - Sequence length? → If >1024, try `mamba2`, `moderntcn`
   - Multivariate dependencies? → If strong, try `itransformer`, `crossformer`
   - Non-stationary? → Try `nonstationary_transformer`, use `transforms=zscore_diff`
   - Seasonal? → Try `timesnet`, `autoformer`

4. **Try State-of-the-Art**:
   ```bash
   airtrace train model=patchtst  # ICLR 2023
   airtrace train model=itransformer  # ICLR 2024 Spotlight
   airtrace train model=moderntcn  # ICLR 2024 Spotlight
   airtrace train model=softs  # NeurIPS 2024
   ```

5. **Consider Foundation Models** (if applicable):
   ```bash
   airtrace train model=chronos_bolt  # Zero-shot or fine-tune
   airtrace train model=timer  # GPT-style pretrained
   ```

6. **Optimize Best Candidate**:
   - Hyperparameter tuning
   - Ensemble top 2-3 models

### Red Flags and Warnings

**Warning Signs You Need a Simpler Model**:
- Baselines (persistence, linear) perform within 5-10% of complex models
- Overfitting: validation loss >> training loss
- Limited data (<1000 samples) + large Transformer

**Warning Signs You Need a More Complex Model**:
- Large performance gap between train and validation (underfitting)
- Baselines fail dramatically
- Clear complex patterns (multivariate, long-range) not captured

**Common Mistakes**:
- ❌ Using standard Transformer on >1024 sequences (use PatchTST, Mamba)
- ❌ Using channel-independent models with strong cross-variate dependencies
- ❌ Ignoring non-stationarity (use differencing or adaptive models)
- ❌ Not trying baselines first
- ❌ Using foundation models on in-domain data without comparing to task-specific models

### Domain-Specific Recommendations

**Aircraft Sensor Data** (AirTrace focus):
- **Characteristics**: Multivariate, strong physics-based dependencies, regime changes (takeoff/cruise/landing)
- **Recommended**:
  1. `itransformer` - Capture cross-sensor dependencies (fuel ↔ thrust ↔ altitude)
  2. `nonstationary_transformer` - Handle regime shifts
  3. `gru_ar` - Solid default, efficient
  4. `var` - Baseline with multivariate dynamics
- **Preprocessing**: `transforms=zscore_diff_with_context` (add flight phase context)

**Financial Markets**:
- **Characteristics**: Non-stationary, regime changes, tick-level or high-frequency
- **Recommended**: `nonstationary_transformer`, `timesnet`, foundation models (domain transfer)

**Energy/Demand Forecasting**:
- **Characteristics**: Strong daily/weekly seasonality, weather dependencies
- **Recommended**: `timesnet`, `autoformer`, `tft` (with exogenous weather)

**Medical/Physiological**:
- **Characteristics**: Noisy, irregular sampling, patient-specific
- **Recommended**: `gru_ar` (robust to noise), `moderntcn`, foundation models (transfer across patients)

---

## Summary Table: Model Selection Matrix

| Model | Sequence Length | Multivariate | Non-Stationary | Seasonal | Data Size | Speed | Interpretability |
|-------|-----------------|--------------|----------------|----------|-----------|-------|------------------|
| **persistence** | Any | Weak | No | No | Any | ⚡⚡⚡ | ★★★ |
| **linear_ar** | Short-Med | Weak | No | No | Small | ⚡⚡⚡ | ★★★ |
| **dlinear** | Any | Weak-Med | Yes (decomp) | Yes (decomp) | Small-Med | ⚡⚡⚡ | ★★★ |
| **nlinear** | Any | Weak | Yes (normalize) | No | Small-Med | ⚡⚡⚡ | ★★☆ |
| **gru_ar** | Short-Med | Med | Med | Med | Small-Med | ⚡⚡☆ | ★☆☆ |
| **lstm_ar** | Short-Med | Med | Med | Med | Med | ⚡⚡☆ | ★☆☆ |
| **moderntcn** | Long | Med | Med | Med | Med-Large | ⚡⚡⚡ | ★☆☆ |
| **tsmixer** | Med | Strong | Med | Med | Med | ⚡⚡☆ | ★★☆ |
| **softs** | Med | Strong | Med | Med | Med | ⚡⚡☆ | ★☆☆ |
| **nbeats** | Med | Weak | Yes (decomp) | Yes (decomp) | Med | ⚡⚡☆ | ★★★ |
| **patchtst** | Long | Weak-Med | Med | Med | Med-Large | ⚡⚡☆ | ★☆☆ |
| **itransformer** | Med-Long | Strong | Med | Med | Large | ⚡☆☆ | ★★☆ |
| **crossformer** | Med-Long | Strong | Med | Yes | Large | ⚡☆☆ | ★★☆ |
| **autoformer** | Med-Long | Med | Strong | Strong | Med-Large | ⚡☆☆ | ★★☆ |
| **nonstationary_transformer** | Med | Med | Strong | Med | Large | ⚡☆☆ | ★☆☆ |
| **timesnet** | Med | Strong | Strong | Strong | Large | ⚡☆☆ | ★☆☆ |
| **tft** | Med | Strong | Med | Med | Large | ⚡☆☆ | ★★★ |
| **mamba2** | Very Long | Med | Med | Med | Large | ⚡⚡☆ | ★☆☆ |
| **mambats** | Very Long | Strong | Med | Med | Large | ⚡⚡☆ | ★☆☆ |
| **chronos_bolt** | Any | Med | Strong | Med | Zero-shot/Small | ⚡☆☆ | ★☆☆ |
| **timer** | Any | Med | Strong | Med | Zero-shot/Small | ⚡☆☆ | ★☆☆ |

**Legend**:
- **Speed**: ⚡⚡⚡ = Very Fast, ⚡⚡☆ = Fast, ⚡☆☆ = Slower
- **Interpretability**: ★★★ = High, ★★☆ = Medium, ★☆☆ = Low
- **Capability Levels**: Weak < Med < Strong

---

## References and Further Reading

### Research Papers by Model

- **PatchTST**: [A Time Series is Worth 64 Words (ICLR 2023)](https://github.com/yuqinie98/PatchTST)
- **iTransformer**: [Inverted Transformers (ICLR 2024 Spotlight)](https://github.com/thuml/iTransformer)
- **ModernTCN**: [Pure Convolution Architecture (ICLR 2024 Spotlight)](https://unit8.com/resources/temporal-convolutional-networks-and-forecasting/)
- **TSMixer**: [All-MLP Architecture (KDD 2023)](https://research.google/blog/tsmixer-an-all-mlp-architecture-for-time-series-forecasting/)
- **SOFTS**: [STAR Module (NeurIPS 2024)](https://airtrace.readthedocs.io/)
- **Mamba**: [Selective State Space Models](https://arxiv.org/abs/2403.11144)
- **MambaTS**: [Variable Scan (arXiv 2024)](https://arxiv.org/html/2405.16440v1)
- **Chronos**: [Foundation Model (Amazon)](https://www.manning.com/books/time-series-forecasting-using-foundation-models)
- **Timer**: [GPT-style Pretrained (ICML 2024)](https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/)

### General Resources

- [Transformers for Time Series Forecasting](https://link.springer.com/article/10.1007/s10462-024-11044-2)
- [RNN/LSTM/GRU Performance Analysis](https://pmc.ncbi.nlm.nih.gov/articles/PMC12329085/)
- [TCN Overview](https://www.activeloop.ai/resources/glossary/temporal-convolutional-networks-tcn/)
- [Foundation Models Review](https://otexts.com/fpppy/nbs/15-foundation-models.html)

### AirTrace-Specific Documentation

- [Model Registry (README)](../README.md#model-registry) - Complete list of 49 models
- [Baseline Models](baseline_models.md) - Detailed baseline documentation
- [Architecture Overview](architecture.md) - Framework design principles
- [SOFTS Implementation](models/softs_implementation.md) - SOFTS-specific guide
- [Chronos-Bolt Documentation](models/chronos_bolt.md) - Foundation model guide

---

## Getting Help

If you're still unsure which model to choose:

1. **Post your data characteristics** on the AirTrace GitHub discussions
2. **Run the benchmark suite**: Try top 5 recommended models, compare
3. **Start simple**: Baselines → GRU → PatchTST is a solid progression
4. **Check experiments**: See `docs/experiments.md` for empirical comparisons

**Remember**: The best model is the simplest one that meets your accuracy requirements. Don't over-engineer!
