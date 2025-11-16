# Model Proposals for AirTrace

This document tracks proposals for adding models to AirTrace based on recent literature and identified gaps in the model registry.

**Last Updated**: 2025-11-16
**Status**: Proposed, awaiting implementation decision

---

## Current Status

**Implemented**: 25 models including:
- Modern architectures: PatchTST (ICLR 2023), iTransformer (ICLR 2024), TimeMixer (ICLR 2024), CycleNet (NeurIPS 2024)
- Foundation models: Chronos-Bolt, Moirai, Mamba2, Lag-Llama
- RNNs/Seq2Seq: GRU, LSTM variants
- Baseline models: 16 statistical and simple baselines
- Convolutions: TCN

**Missing**: ~15 significant architectures spanning 2019-2024

---

## Priority Tier 1: Critical Gaps (Implement First)

### 1. Temporal Fusion Transformer (TFT) ⭐⭐⭐ HIGHEST PRIORITY

**Full Title**: Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting
**Venue**: Original paper (2019) + Aviation application (MDPI Aerospace 2024)
**Paper**: https://arxiv.org/abs/1912.09363
**Aviation Paper**: https://www.mdpi.com/2226-4310/11/8/646
**Code**: https://github.com/google-research/google-research/tree/master/tft
**PyTorch Forecasting**: https://pytorch-forecasting.readthedocs.io/en/stable/api/pytorch_forecasting.models.temporal_fusion_transformer.TemporalFusionTransformer.html

#### Key Innovations

1. **Multi-horizon Forecasting**: Explicitly designed for predicting multiple future timesteps simultaneously
2. **Interpretability**: Variable selection networks, temporal attention weights, static covariate encoders
3. **Uncertainty Quantification**: Quantile regression for probabilistic forecasts
4. **Flexible Input Handling**:
   - Time-varying known inputs (flight plan, scheduled altitude)
   - Time-varying unknown inputs (actual sensor readings)
   - Static metadata (aircraft type, engine model, weather)

#### Why Perfect for Aircraft Sensors

**Domain-Specific Success**: The 2024 MDPI Aerospace paper demonstrates explicit success on aircraft sensor data:
- Detects cascading failures in multivariate sensor data
- Identifies precursor events before catastrophic failures
- Handles sensor readout differences and drift
- Provides interpretable attention weights for failure analysis

**Safety-Critical Interpretability**: Aviation requires understanding *why* a model makes predictions:
- Variable importance networks reveal which sensors drive predictions
- Temporal attention shows which historical time periods matter
- Essential for certification and trust in safety applications

**Handles Complex Aircraft Data**:
- Static context: Aircraft type, engine model, configuration
- Known future inputs: Flight plan, scheduled waypoints, weather forecasts
- Unknown future inputs: Actual sensor readings to predict
- Multi-horizon: Predict entire trajectories, not just next timestep

#### Architecture Components

```
Input Layer:
  ├── Static Covariate Encoders (aircraft metadata)
  ├── Variable Selection (learned gating for time-varying inputs)
  └── Temporal Processing
      ├── LSTM Encoder (past context)
      ├── LSTM Decoder (future trajectory)
      └── Multi-head Attention (temporal relationships)

Interpretability Layer:
  ├── Variable Importance Weights
  ├── Temporal Attention Weights
  └── Quantile Outputs (uncertainty bounds)
```

#### Implementation Guidance

**Effort**: High (500-700 lines + tests + config)

**Key Components**:
1. `VariableSelectionNetwork`: Learnable gating mechanism
2. `GatedResidualNetwork`: Building block with GLU activations
3. `InterpretableMultiHeadAttention`: Attention with importance weights
4. `QuantileLoss`: For probabilistic forecasting
5. Static/dynamic input encoders

**Integration with AirTrace**:
- Extend `ARBaseModel` with multi-horizon output `[B, T_out, D_out]`
- Add metadata handling to dataset (static features per flight)
- Create `TFTTask` that handles known future inputs
- Store attention weights in `extras` for visualization

**Config Structure**:
```yaml
# configs/model/tft.yaml
_target_: airtrace.models.tft.TFTModel
input_dim: ${data.input_dim}
output_dim: ${data.output_dim}
hidden_size: 128
lstm_layers: 2
num_heads: 4
dropout: 0.1
quantiles: [0.1, 0.5, 0.9]  # P10, median, P90
static_input_dim: 0  # Aircraft metadata features
known_future_dim: 0  # Flight plan features
```

**Testing Strategy**:
1. Unit tests: Each component (VSN, GRN, attention)
2. Integration tests: Full model forward pass
3. Synthetic data: Verify interpretability outputs
4. Multi-horizon task compatibility

**References**:
- Lim et al. (2021): "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting"
- Ogunfowora & Najjaran (2024): "On the Exploration of Temporal Fusion Transformers for Anomaly Detection with Multivariate Aviation Time-Series Data"

---

### 2. ModernTCN (ICLR 2024 Spotlight) ⭐⭐

**Full Title**: ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis
**Venue**: ICLR 2024 Spotlight
**Paper**: https://openreview.net/forum?id=vpJMJerXHU
**Code**: https://github.com/luodhhh/ModernTCN
**OpenReview**: https://openreview.net/forum?id=vpJMJerXHU

#### Key Innovations

1. **Much Larger Effective Receptive Fields (ERFs)**: Critical improvement over classic TCN
2. **Pure Convolution**: Demonstrates convolutions can match/exceed transformers when properly designed
3. **General Time Series Analysis**: SOTA on 5 tasks (long/short forecasting, imputation, classification, anomaly detection)
4. **Efficiency**: Maintains computational advantages of convolution-based models

#### Why for AirTrace

1. **Natural Evolution**: AirTrace already has TCN; ModernTCN is the SOTA upgrade
2. **Proven Architecture**: ICLR 2024 Spotlight recognition validates quality
3. **Smooth Sensor Data**: Convolutions have excellent inductive bias for continuous sensor readings
4. **Production Deployment**: Lower computational cost than transformers for real-world systems
5. **Multi-Task**: Same architecture works for forecasting, anomaly detection, classification

#### Technical Improvements Over TCN

- Depthwise separable convolutions for efficiency
- Larger receptive fields through architectural innovations
- Better parameter efficiency
- Improved gradient flow for deeper networks
- Modern training techniques (LayerNorm, GELU)

#### Implementation Guidance

**Effort**: Low-Medium (200-300 lines + tests + config)

**Key Components**:
1. `ModernTCNBlock`: Depthwise separable conv + pointwise conv
2. `LargeReceptiveField`: Parallel dilated convolutions
3. `DownsamplingBlock`: Multi-resolution processing
4. Residual connections and layer normalization

**Integration**:
- Similar structure to existing `TCNModel`
- Replace basic dilated convs with modernized blocks
- Add multi-resolution hierarchy

**Config Structure**:
```yaml
# configs/model/moderntcn.yaml
_target_: airtrace.models.moderntcn.ModernTCNModel
input_dim: ${data.input_dim}
output_dim: ${data.output_dim}
num_blocks: 6
kernel_size: 3
hidden_channels: 64
dilation_growth: 2
dropout: 0.1
```

**References**:
- Luo et al. (2024): "ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis"

---

### 3. DLinear / NLinear ⭐⭐

**Full Title**: Are Transformers Effective for Time Series Forecasting?
**Venue**: AAAI 2023
**Paper**: https://arxiv.org/abs/2205.13504
**Code**: https://github.com/cure-lab/LTSF-Linear

#### Key Innovations

1. **Embarrassingly Simple**: One-layer linear models that often beat complex transformers
2. **DLinear**: Decomposition + two linear layers (trend + seasonal)
3. **NLinear**: Instance normalization + one linear layer
4. **Essential Baseline**: Exposes when complexity isn't needed

#### Why for AirTrace

**Critical Missing Baseline**: Current `linear_ar` is generic; DLinear/NLinear are specific SOTA linear methods that:
- Consistently cited in recent papers as strong baselines
- Often beat PatchTST/TimeMixer on some datasets
- Expose model overparameterization
- Extremely fast for ablation studies

**Simple != Weak**: These models leverage:
- Temporal continuity (predictions close to recent values)
- Distribution shift handling (instance normalization)
- Seasonal decomposition (moving average)

#### Implementation Guidance

**Effort**: Very Low (50-100 lines total + tests + config)

**DLinear**:
```python
@register("dlinear")
class DLinearModel(ARBaseModel):
    def __init__(self, input_dim, output_dim, seq_len, pred_len, kernel_size=25):
        super().__init__(input_dim, output_dim)
        self.decomp = SeriesDecomposition(kernel_size)
        self.linear_seasonal = nn.Linear(seq_len, pred_len)
        self.linear_trend = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        # x: [B, T, D]
        seasonal, trend = self.decomp(x)  # [B, T, D] each
        seasonal_out = self.linear_seasonal(seasonal.permute(0,2,1)).permute(0,2,1)
        trend_out = self.linear_trend(trend.permute(0,2,1)).permute(0,2,1)
        return {"preds": seasonal_out + trend_out}
```

**NLinear**:
```python
@register("nlinear")
class NLinearModel(ARBaseModel):
    def __init__(self, input_dim, output_dim, seq_len, pred_len):
        super().__init__(input_dim, output_dim)
        self.linear = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        # Instance normalization
        seq_last = x[:, -1:, :]  # [B, 1, D]
        x = x - seq_last  # [B, T, D]
        # Linear projection per feature
        x_out = self.linear(x.permute(0,2,1)).permute(0,2,1)  # [B, T_out, D]
        # Denormalize
        x_out = x_out + seq_last
        return {"preds": x_out}
```

**Quick Win**: Can implement both in ~2 hours including tests.

**References**:
- Zeng et al. (2023): "Are Transformers Effective for Time Series Forecasting?"

---

## Priority Tier 2: Classic Transformer Baselines

These are foundational transformer models (2020-2022) that established the paradigm and are still widely used as baselines in current papers.

### 4. Informer (AAAI 2021 Best Paper)

**Full Title**: Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting
**Venue**: AAAI 2021 (Best Paper Award)
**Paper**: https://arxiv.org/abs/2012.07436
**Code**: https://github.com/zhouhaoyi/Informer2020
**Citations**: 2000+

#### Key Innovations

1. **ProbSparse Self-Attention**: Reduces complexity from O(L²) to O(L log L)
2. **Self-Attention Distilling**: Halves input at each layer for efficiency
3. **Generative Style Decoder**: Predicts long sequences in one forward pass
4. **Long Sequence**: First to effectively handle 10k+ timesteps

#### Why Important

- **Historical Baseline**: Still the #1 cited baseline in time series papers
- **Long Sequences**: Aircraft flights can be hours of continuous data
- **Proven Architecture**: Widely deployed in production systems

#### Implementation Effort

Medium (300-400 lines) - ProbSparse attention is the complex part.

**References**:
- Zhou et al. (2021): "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting"

---

### 5. Autoformer (NeurIPS 2021)

**Full Title**: Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting
**Venue**: NeurIPS 2021
**Paper**: https://arxiv.org/abs/2106.13008
**Code**: https://github.com/thuml/Autoformer

#### Key Innovations

1. **Auto-Correlation Mechanism**: Replaces self-attention with series-wise connection discovery
2. **Progressive Decomposition**: Extracts trend-cyclical components at each layer
3. **Better for Time Series**: Auto-correlation > self-attention for temporal data

#### Why Important

- Auto-correlation is specifically designed for time series (vs. NLP-borrowed attention)
- Decomposition provides interpretability
- Strong performance on long-term forecasting

#### Implementation Effort

Medium (300-400 lines) - Auto-correlation and decomposition modules.

**References**:
- Wu et al. (2021): "Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting"

---

### 6. FEDformer (ICML 2022)

**Full Title**: FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting
**Venue**: ICML 2022
**Paper**: https://arxiv.org/abs/2201.12740
**Code**: https://github.com/MAZiqing/FEDformer

#### Key Innovations

1. **Frequency Domain**: Operates on Fourier/Wavelet transforms of input
2. **Seasonal-Trend Decomposition**: Architecture-level decomposition
3. **Sparse Attention in Frequency**: Lower frequencies carry most information

#### Why for Aircraft Sensors

- Engine vibrations have strong frequency components
- Periodic patterns (rotation, oscillations)
- More efficient for long sequences in frequency space

#### Implementation Effort

Medium-High (400-500 lines) - FFT/Wavelet transforms + frequency attention.

**References**:
- Zhou et al. (2022): "FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting"

---

### 7. Non-stationary Transformer (NeurIPS 2022)

**Full Title**: Non-stationary Transformers: Exploring the Stationarity in Time Series Forecasting
**Venue**: NeurIPS 2022
**Paper**: https://arxiv.org/abs/2205.14415
**Code**: https://github.com/thuml/Nonstationary_Transformers

#### Key Innovations

1. **De-stationary Attention**: Projects to stationary space before attention
2. **Over-stationarization**: Aggressive normalization to handle distribution shifts
3. **Series Stationarization**: Learnable normalization per series

#### Why for Aircraft Sensors

**Critical for Flight Phases**: Aircraft data is highly non-stationary:
- Takeoff → Climb → Cruise → Descent → Landing
- Each phase has different distributions
- Weight decreases over flight (fuel burn)
- Altitude/temperature/pressure change dramatically

#### Implementation Effort

Medium (300-400 lines) - Stationarization layers + modified attention.

**References**:
- Liu et al. (2022): "Non-stationary Transformers: Exploring the Stationarity in Time Series Forecasting"

---

### 8. Crossformer (ICLR 2023)

**Full Title**: Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting
**Venue**: ICLR 2023
**Paper**: https://openreview.net/forum?id=vSVLM2j9eie
**Code**: https://github.com/Thinklab-SJTU/Crossformer

#### Key Innovations

1. **Two-Stage Attention**: Cross-time dimension, then cross-variable dimension
2. **Dimension-Segment-Wise (DSW) Embedding**: Embeds segments across dimensions
3. **Explicitly Models Cross-Variate Dependencies**: Unlike channel-independent models

#### Why for Aircraft Sensors

**Strong Cross-Sensor Correlations**:
- Fuel flow ↔ Thrust ↔ Speed ↔ Weight
- Temperature ↔ Altitude ↔ Pressure (physics-based)
- Engine sensors are highly coupled

Current models (PatchTST, iTransformer) handle multivariate differently; Crossformer offers another approach.

#### Implementation Effort

Medium-High (400-500 lines) - DSW embedding + two-stage attention.

**References**:
- Zhang & Yan (2023): "Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting"

---

## Priority Tier 3: Pure MLP Models

Simple, efficient, non-sequential architectures that surprisingly compete with transformers.

### 9. N-BEATS (ICLR 2020)

**Full Title**: N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting
**Venue**: ICLR 2020
**Paper**: https://arxiv.org/abs/1905.10437
**Code**: https://github.com/ElementAI/N-BEATS

#### Key Innovations

1. **Doubly Residual Stacking**: Backward (forecast) and forward (backcast) branches
2. **Basis Expansion**: Interpretable trend/seasonality components
3. **Pure MLP**: No convolutions, RNNs, or transformers
4. **Industry Proven**: M4 competition winner, widely deployed

#### Why for AirTrace

- **Interpretable Decomposition**: Explicit trend/seasonal split
- **Strong Baseline**: Often beats complex models
- **Non-Sequential**: Tests if temporal modeling is needed
- **Safety-Critical**: Interpretable components for aviation

#### Implementation Effort

Medium (300-400 lines) - Block structure + basis functions.

**References**:
- Oreshkin et al. (2020): "N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting"

---

### 10. N-HiTS (AAAI 2023)

**Full Title**: N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting
**Venue**: AAAI 2023
**Paper**: https://arxiv.org/abs/2201.12886
**Code**: https://github.com/Nixtla/neuralforecast

#### Key Innovations

1. **Hierarchical Interpolation**: Multi-rate data sampling (like TimeMixer but simpler)
2. **Better than N-BEATS**: Improved long-horizon forecasting
3. **MaxPool Downsampling**: Captures patterns at different scales

#### Why for AirTrace

- Evolution of N-BEATS with better long-term performance
- Multi-scale (like TimeMixer) but pure MLP
- Interpretable hierarchical structure

#### Implementation Effort

Medium (250-350 lines) - Similar to N-BEATS with pooling layers.

**References**:
- Challu et al. (2023): "N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting"

---

### 11. TSMixer (KDD 2023)

**Full Title**: TSMixer: An All-MLP Architecture for Time Series Forecasting
**Venue**: KDD 2023
**Paper**: https://arxiv.org/abs/2303.06053
**Code**: https://github.com/google-research/google-research/tree/master/tsmixer

#### Key Innovations

1. **MLP-Mixer for Time Series**: Adapts vision MLP-Mixer to temporal data
2. **Time-Mixing + Feature-Mixing**: Alternating MLPs for time/channel
3. **All-MLP**: No attention, convolutions, or recurrence

**Note**: Different from TimeMixer (already implemented). TimeMixer = decomposition + multiscale. TSMixer = pure MLP-Mixer adaptation.

#### Implementation Effort

Low-Medium (200-300 lines) - MLP blocks with permutations.

**References**:
- Chen et al. (2023): "TSMixer: An All-MLP Architecture for Time Series Forecasting"

---

## Priority Tier 4: Frequency Domain & Novel Paradigms

### 12. TimesNet (ICLR 2023)

**Full Title**: TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis
**Venue**: ICLR 2023
**Paper**: https://arxiv.org/abs/2210.02186
**Code**: https://github.com/thuml/TimesNet

#### Key Innovations

1. **2D Vision Backbone**: Treats time series as 2D images via period-based reshaping
2. **Intraperiod and Interperiod Variation**: Captures both within-cycle and across-cycle patterns
3. **Parameter Sharing**: Applies same 2D conv across all periods

#### Why for Aircraft

- Engine cycles, rotation periods
- Novel inductive bias

#### Implementation Effort

Medium-High (400-500 lines) - Period detection + 2D convolutions.

**References**:
- Wu et al. (2023): "TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis"

---

### 13. FreTS (NeurIPS 2023)

**Full Title**: FreTS: Frequency-domain MLPs are More Effective Learners in Time Series Forecasting
**Venue**: NeurIPS 2023
**Paper**: https://arxiv.org/abs/2311.06184
**Code**: https://github.com/aikunyi/FreTS

#### Key Innovations

1. **Frequency-domain MLPs**: Apply MLPs to Fourier coefficients
2. **Low-Frequency Focus**: Most signal in low frequencies
3. **Efficient**: FFT is fast, fewer parameters needed

#### Why for Aircraft

- Periodic engine/rotation signals
- Efficient for smooth sensor data

#### Implementation Effort

Low-Medium (200-300 lines) - FFT + MLPs.

**References**:
- Yi et al. (2023): "FreTS: Frequency-domain MLPs are More Effective Learners in Time Series Forecasting"

---

### 14. ETSformer (ICML 2023)

**Full Title**: ETSformer: Exponential Smoothing Transformers for Time-series Forecasting
**Venue**: ICML 2023
**Paper**: https://arxiv.org/abs/2202.01381
**Code**: https://github.com/salesforce/ETSformer

#### Key Innovations

1. **Exponential Smoothing + Transformers**: Combines classical stats with deep learning
2. **Interpretable Latent Components**: Level, growth, seasonality
3. **Frequency Attention**: Attention in Fourier domain

#### Why for Aircraft

- Interpretable components for safety
- Combines physics-based priors (smoothing) with learning

#### Implementation Effort

Medium-High (400-500 lines) - ETS components + frequency attention.

**References**:
- Woo et al. (2023): "ETSformer: Exponential Smoothing Transformers for Time-series Forecasting"

---

## Priority Tier 5: LLM-Based Models (Research Frontier)

### 15. TIME-LLM (ICLR 2024)

**Full Title**: Time-LLM: Time Series Forecasting by Reprogramming Large Language Models
**Venue**: ICLR 2024
**Paper**: https://arxiv.org/abs/2310.01728
**Code**: https://github.com/KimMeen/Time-LLM

#### Key Innovations

1. **LLM Reprogramming**: Uses frozen LLaMA-7B for time series
2. **Patching + Text Prototypes**: Converts time series to "language"
3. **Zero-Shot Transfer**: Leverages LLM's reasoning

#### Why for AirTrace

- Cutting-edge paradigm
- Potential for reasoning about anomalies
- Zero-shot on new aircraft types

#### Challenges

- **Infrastructure**: Requires LLM hosting (7B+ parameters)
- **Integration**: Significant changes to pipeline
- **Compute**: GPU memory intensive

#### Implementation Effort

Very High (1000+ lines + LLM dependencies + infrastructure)

**Recommendation**: Wait for more mature tooling or consider as separate research direction.

**References**:
- Jin et al. (2024): "Time-LLM: Time Series Forecasting by Reprogramming Large Language Models"

---

### 16. TEMPO (ICLR 2024)

**Full Title**: TEMPO: Prompt-based Generative Pre-trained Transformer for Time Series Forecasting
**Venue**: ICLR 2024
**Paper**: https://arxiv.org/abs/2310.04948
**Code**: https://github.com/DC-research/TEMPO

#### Key Innovations

1. **GPT-style Pre-training**: Unified framework for multiple tasks
2. **Prompt-based**: Task specification via prompts
3. **Foundation Model**: Pre-trained on diverse datasets

#### Implementation Effort

Very High (similar to TIME-LLM)

**Recommendation**: Future work, requires significant infrastructure.

**References**:
- Cao et al. (2024): "TEMPO: Prompt-based Generative Pre-trained Transformer for Time Series Forecasting"

---

## Implementation Priority Summary

### Quick Wins (1-2 days each)
1. **DLinear / NLinear** - Essential baseline, 50-100 lines
2. **FreTS** - Frequency MLP, 200-300 lines

### High Impact (1 week each)
3. **TFT** - Domain-specific, interpretable, proven on aircraft
4. **ModernTCN** - Natural TCN upgrade, production-ready
5. **N-BEATS** - Interpretable MLP baseline

### Classic Baselines (1 week each)
6. **Informer** - Most cited baseline
7. **Autoformer** - Auto-correlation paradigm
8. **Non-stationary Transformer** - Critical for flight phases

### Advanced (2 weeks each)
9. **FEDformer** - Frequency domain
10. **Crossformer** - Cross-variate modeling
11. **N-HiTS** - Hierarchical interpolation
12. **TSMixer** - MLP-Mixer adaptation

### Research/Long-term
13. **TimesNet** - 2D vision approach
14. **ETSformer** - Classical-DL hybrid
15. **TIME-LLM / TEMPO** - LLM-based (infrastructure needed)

---

## Recommended Implementation Order

**Phase 1 (Month 1)**: Critical gaps
1. TFT (highest priority for aviation)
2. DLinear/NLinear (quick baseline win)
3. ModernTCN (TCN upgrade)

**Phase 2 (Month 2)**: Classic baselines
4. Informer (most cited)
5. Autoformer (auto-correlation)
6. N-BEATS (interpretable MLP)

**Phase 3 (Month 3)**: Advanced capabilities
7. Non-stationary Transformer (flight phases)
8. FEDformer (frequency domain)
9. N-HiTS (hierarchical)

**Phase 4 (Month 4+)**: Nice-to-have
10. Crossformer, TSMixer, TimesNet, FreTS, ETSformer

**Future Research**: TIME-LLM, TEMPO (requires dedicated infrastructure)

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

6. **README Update**: Add to Model Registry table

---

## References

### Papers
- Lim et al. (2021): Temporal Fusion Transformers
- Luo et al. (2024): ModernTCN
- Zeng et al. (2023): DLinear/NLinear
- Zhou et al. (2021): Informer
- Wu et al. (2021): Autoformer
- Zhou et al. (2022): FEDformer
- Liu et al. (2022): Non-stationary Transformer
- Zhang & Yan (2023): Crossformer
- Oreshkin et al. (2020): N-BEATS
- Challu et al. (2023): N-HiTS
- Chen et al. (2023): TSMixer
- Wu et al. (2023): TimesNet
- Yi et al. (2023): FreTS
- Woo et al. (2023): ETSformer
- Jin et al. (2024): TIME-LLM
- Cao et al. (2024): TEMPO

### Aviation Domain
- Ogunfowora & Najjaran (2024): "On the Exploration of Temporal Fusion Transformers for Anomaly Detection with Multivariate Aviation Time-Series Data", MDPI Aerospace

### Code Repositories
- Time-Series-Library: https://github.com/thuml/Time-Series-Library
- PyTorch Forecasting: https://pytorch-forecasting.readthedocs.io/
- NeuralForecast: https://github.com/Nixtla/neuralforecast

---

## Notes

- Models removed from this document (now implemented): TimeMixer, Mamba2/S-Mamba, Chronos-Bolt, Moirai, Lag-Llama
- Focus on models with public code and reproducible results
- Prioritize interpretability for safety-critical aviation applications
- Balance between cutting-edge research and proven baselines
- Consider computational efficiency for potential onboard deployment
