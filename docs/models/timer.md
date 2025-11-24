# Timer: Generative Pre-trained Transformers for Time Series

**Paper**: [Timer: Generative Pre-trained Transformers Are Large Time Series Models](https://arxiv.org/abs/2402.02368) (ICML 2024)
**Authors**: Yong Liu et al. (THUML, Tsinghua University)
**Code**: https://github.com/thuml/Large-Time-Series-Model
**HuggingFace**: https://huggingface.co/thuml/timer-base-84m

## Overview

Timer is a decoder-only pre-trained Transformer for general time series analysis, trained on 260 billion time points across multiple domains. It represents the first successful application of GPT-style autoregressive generation to time series foundation models, achieving zero-shot forecasting capability without requiring domain-specific training.

### Key Innovations

1. **Single-Series Sequence (S3) Format**: Unified 1D representation that converts heterogeneous time series into a format compatible with language model architectures
2. **Decoder-Only Architecture**: GPT-style autoregressive generation for flexible context and prediction horizons
3. **Zero-Shot Forecasting**: Out-of-the-box predictions on new datasets without training
4. **Scalability**: Pre-trained on massive Unified Time Series Datasets (UTSD) with 260B+ time points

## Architecture

```
Input Time Series [B, T, D]
         ↓
  Per-Variate Z-Score Normalization
         ↓
  Independent Processing (D times)
         ↓ (for each dimension)
  ┌──────────────────────────┐
  │ Timer Backbone (HF)      │
  │  - Decoder-Only Trans.   │
  │  - Token Generation      │
  │  - Max New Tokens = H    │
  └──────────────────────────┘
         ↓
  Stack Predictions [B, H, D]
         ↓
  Denormalization
         ↓
  Final Predictions [B, H, D]
```

## Key Components

### 1. Input Normalization

Timer expects normalized inputs for stable generation. The model applies z-score normalization per series:

```python
normalized = (x - mean) / std
```

Statistics are stored and used for denormalizing predictions.

### 2. Multivariate Handling

**Challenge**: Timer is trained on univariate time series in S3 format.

**Solution**: AirTrace processes each input dimension independently and aggregates predictions:
- Extract each dimension: `x[:, :, d]` → `[B, T]`
- Generate predictions via Timer backbone
- Stack results: `[B, H, D]`

This approach maintains compatibility with Timer's pre-training while supporting multivariate aircraft sensor data.

### 3. Autoregressive Generation

Timer uses HuggingFace's `generate()` method with `max_new_tokens` parameter to control forecast horizon:

```python
predictions = model.generate(input_series, max_new_tokens=pred_len)
```

## Usage

### Zero-Shot Forecasting (No Training)

For immediate predictions without training:

```yaml
# configs/model/timer_zero_shot.yaml
model:
  name: timer
  params:
    pred_len: 96
    checkpoint: thuml/timer-base-84m
    freeze_backbone: true  # Zero-shot mode
    normalize_inputs: true
```

```bash
airtrace train model=timer_zero_shot train.epochs=0 data=my_data
```

### Fine-Tuning for Domain Adaptation

For improved performance on aircraft data:

```yaml
# configs/model/timer_finetune.yaml
model:
  name: timer
  params:
    pred_len: 24
    freeze_backbone: false  # Enable fine-tuning
    normalize_inputs: true
```

```bash
airtrace train model=timer_finetune train.epochs=50 train.learning_rate=1e-4
```

### LoRA Adapter Fine-Tuning (Future)

For parameter-efficient fine-tuning:

```yaml
model:
  name: timer
  params:
    freeze_backbone: true
    lora_rank: 8  # Enable LoRA adapters
    lora_alpha: 8.0
    lora_dropout: 0.05
```

**Note**: Full LoRA implementation is planned for a future release.

## Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pred_len` | int | 24 | Forecast horizon in timesteps |
| `checkpoint` | str | `thuml/timer-base-84m` | HuggingFace model ID or local path |
| `lookback_length` | int | 512 | Context window size (Timer supports variable) |
| `normalize_inputs` | bool | true | Z-score normalization (recommended) |
| `freeze_backbone` | bool | false | Freeze Timer for zero-shot or LoRA |
| `lora_rank` | int | 0 | LoRA adapter rank (0 disables) |
| `lora_alpha` | float | 8.0 | LoRA scaling factor |
| `lora_dropout` | float | 0.05 | LoRA dropout rate |
| `device` | str | `cpu` | Device to load model on (`cpu` or `cuda`) |
| `trust_remote_code` | bool | true | Required for HuggingFace Timer |

## Model Variants

### Timer-Base-84M (Default)

- **Parameters**: 84 million
- **Pre-training**: 260B time points
- **Checkpoint**: `thuml/timer-base-84m`
- **Use Case**: General-purpose forecasting, zero-shot baseline

### Sundial-Base-128M (Future)

- **Parameters**: 128 million
- **Pre-training**: 1 trillion time points
- **Checkpoint**: `thuml/sundial-base-128m`
- **Features**: Probabilistic forecasting with sample generation
- **Status**: Planned for future integration

### Timer-XL (Future)

- **Features**: Extended context lengths (>4096 tokens)
- **Paper**: ICLR 2025
- **Status**: Planned for future integration

## Performance Considerations

### Computational Requirements

**CPU Inference**:
- Timer-base-84M: ~350MB memory
- Inference speed: Moderate (slower than lightweight models)
- Recommended for: Prototyping, small datasets

**GPU Inference**:
- Significant speedup (5-10x)
- Recommended for: Production, large-scale experiments
- Set `device="cuda"` in config

### Memory Usage

- **Model weights**: ~350MB (Timer-base-84M)
- **Activation memory**: Scales with batch size and sequence length
- **Recommended batch size**: 8-32 depending on GPU memory

### First Run

The first run downloads the checkpoint from HuggingFace (~350MB):
- Cached to `~/.cache/huggingface/hub/`
- Subsequent runs are faster
- For offline use, download checkpoint locally and set `checkpoint="/path/to/local/checkpoint"`

## When to Use Timer

### ✅ Use Timer When:

- **Zero-shot baseline needed**: Quick evaluation without training
- **Cross-domain transfer**: Pre-training on diverse domains helps generalization
- **Limited training data**: Foundation model knowledge compensates
- **Long context needed**: Timer supports variable-length contexts
- **Research comparison**: State-of-the-art foundation model baseline

### ❌ Consider Alternatives When:

- **Domain-specific model available**: Aircraft-specific architectures may outperform
- **Strict latency requirements**: Smaller models (GRU, TCN) are faster
- **CPU-only production**: Timer is slower on CPU than lightweight models
- **Interpretability critical**: Simpler baselines (N-BEATS, linear) are more interpretable

## Comparison to Other Foundation Models

| Model | Architecture | Parameters | Pre-training | Multivariate |
|-------|--------------|------------|--------------|--------------|
| **Timer** | Decoder-only Trans. | 84M | 260B points | Independent |
| Chronos-Bolt | Conv + Attention | Custom | Custom | Patching |
| Moirai | SSM | Custom | Custom | Multi-resolution |
| Lag-Llama | Diffusion | Custom | Custom | Retrieval |

**Timer's Advantages**:
- Largest pre-training dataset (260B points)
- True zero-shot capability (no adaptation needed)
- GPT-style architecture (leverages LLM advances)
- Active maintenance by THUML group

## Examples

### Example 1: Zero-Shot Evaluation on Aircraft Data

```bash
# Evaluate Timer zero-shot on Q400 cruise data
airtrace train \
  data=qantas_737 \
  model=timer \
  model.freeze_backbone=true \
  model.pred_len=96 \
  train.epochs=0
```

### Example 2: Fine-Tuning for Multi-Step Forecasting

```bash
# Fine-tune Timer for 24-step ahead forecasting
airtrace train \
  exp=exp_005_timer_zscore \
  model.pred_len=24 \
  train.epochs=50 \
  train.learning_rate=1e-4
```

### Example 3: Compare Zero-Shot vs Fine-Tuned

```bash
# Zero-shot
airtrace train model=timer model.freeze_backbone=true train.epochs=0 exp_name=timer_zero_shot

# Fine-tuned
airtrace train model=timer model.freeze_backbone=false train.epochs=50 exp_name=timer_finetuned
```

## Troubleshooting

### Issue: ImportError - transformers not found

**Solution**: Install transformers library:
```bash
uv pip install "transformers>=4.40.1"
```

### Issue: Slow inference on CPU

**Solution**: Use GPU if available:
```yaml
model:
  params:
    device: cuda
```

Or reduce batch size:
```yaml
train:
  batch_size: 8  # Smaller batches
```

### Issue: Out of memory

**Solution**: Reduce lookback length or batch size:
```yaml
model:
  params:
    lookback_length: 256  # Reduce from default 512
train:
  batch_size: 8
```

### Issue: Checkpoint download fails

**Solution**: Download manually and use local path:
```bash
# Download from HuggingFace
huggingface-cli download thuml/timer-base-84m

# Use local path
model.checkpoint=/path/to/downloaded/checkpoint
```

## Citation

If you use Timer in your research, please cite:

```bibtex
@inproceedings{liu2024timer,
  title={Timer: Generative Pre-trained Transformers Are Large Time Series Models},
  author={Liu, Yong and others},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2024}
}
```

## References

- Paper: https://arxiv.org/abs/2402.02368
- Code: https://github.com/thuml/Large-Time-Series-Model
- HuggingFace: https://huggingface.co/thuml/timer-base-84m
- ICML 2024 Poster: https://icml.cc/virtual/2024/poster/33634

## Related Models in AirTrace

- **chronos_bolt**: Alternative foundation model with conv-attention blocks
- **moirai**: Multiresolution SSM foundation model
- **lag_llama**: Retrieval-augmented diffusion forecaster
- **patchtst**: Efficient patch-based transformer (non-pretrained)
