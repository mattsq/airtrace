# AirTrace Model Parameter Alignment Strategy

## Executive Summary

This document provides a strategic plan to align parameter counts across 28 trainable models in AirTrace, enabling fair comparison and reproducible research.

---

## Current State Analysis

### Parameter Count Ranges

| Category | Model Examples | Param Count | Issue |
|----------|-----------------|------------|-------|
| **Tiny** | DLinear, NLinear | 170 - 600 | Insufficient capacity |
| **Small** | TCN, ModernTCN, TimeMixer | 30K - 150K | Fast training, limited capacity |
| **Medium** | Most Transformers | 150K - 400K | Balanced - ideal for comparison |
| **Large** | iTransformer, CycleNet, NBeats | 2M - 3.6M | High capacity, slower training |
| **Very Large** | Mamba2 | 16M | Memory-intensive |
| **Foundation** | Chronos-Bolt, Moirai, LagLlama | 700M (frozen) | Transfer learning only |

### Key Disparity

**Ratio between smallest and largest (excluding frozen foundation models):**
- 600 params (NLinear) → 16M params (Mamba2) = **26,666:1 ratio**

This heterogeneity makes fair benchmarking impossible without normalization.

---

## Proposed Alignment Strategy

### Tier 1: Standard Capacity Models (200K - 300K params)

**Target Capacity:** 250K parameters (±50K tolerance)

**Baseline Architecture:**
```yaml
# Standard transformer profile
d_model: 128
nhead: 8
num_encoder_layers: 3
dim_feedforward: 256
dropout: 0.1

# Equivalent for other architectures:
hidden_size: 128  # RNNs, NBeats
num_channels: [64, 128, 128]  # TCN
hidden_channels: 64
num_blocks: 6  # ModernTCN
state_dim: 64  # MambaTS
embed_dim: 128  # Most models
```

**Models in this tier:**
- Transformer (~270K) - baseline
- PatchTST (~170K) - reduce dim_feedforward to 200
- Informer (~197K) - reduce ff_dim to 200
- Autoformer (~230K) - reduce d_ff to 200
- FEDformer (~230K) - reduce d_ff to 200
- CrossformerModel (~160K) - increase d_model to 160
- GRUARModel (~100K) - increase hidden_size to 256
- LSTMARModel (~200K) - increase hidden_size to 256
- NonStationaryTransformer (~287K) - reduce num_layers to 2
- TimesNetModel - standardize
- TimeXerModel - standardize
- SOFTSModel - standardize
- TFTModel (~150K) - increase capacity

**Alignment Actions:**
1. Reduce dense dimensions where exceeding (Transformers, TCN)
2. Increase capacity where below (RNNs, linear models)
3. Add capacity layers for 
 models (DLinear, NLinear → ~50K)

**Expected Impact:**
- ✅ Fair comparison across architectures
- ✅ Standardized training time (~1-2 hours/epoch)
- ✅ Unified memory footprint (~2-3GB for 32-batch)
- ✅ Improved reproducibility

---

### Tier 2: High-Capacity Models (500K - 5M params)

**Use Case:** Advanced research, transfer learning

**Current Models:**
- iTransformer (3.6M) → Reduce d_model 512→256 (~900K)
- CycleNet (2M) → Reduce hidden_dim 256→128 (~500K)
- NBeats (400-800K) → Reduce hidden_size 256→200 (~300K)
- Mamba2 (16M) → Reduce embed_dim 512→256 (~2M)
- MambaTSA (145K) → Increase to 500K range

**Strategy:**
- Allow flexibility for advanced practitioners
- Document parameter choices in experiments
- Provide "reference" configurations for Tier 2

---

### Tier 3: Foundation Models (Transfer Learning)

**Models:**
- ChronosBolt (~700M, frozen)
- Moirai (~700M, frozen)
- LagLlama (~700M, frozen)

**Strategy:**
- Keep backbone frozen by default
- Fine-tune via LoRA (parameter-efficient)
- Document number of trainable parameters (LoRA only)

---

## Implementation Plan

### Phase 1: Configuration Updates (Immediate)

For each model in Tier 1, update `/src/airtrace/configs/model/{model}.yaml`:

**Example: PatchTST**
```yaml
# Before: ~170K
model:
  name: patchtst
  params:
    patch_len: 16
    stride: 8
    d_model: 128      # ← Keep
    nhead: 8
    num_layers: 3
    dim_feedforward: 256  # ← Change from 256 to 256 (OK)
    dropout: 0.1
    
# Target: ~200K (already close, minimal change needed)
```

**Example: Informer**
```yaml
# Before: ~197K
model:
  name: informer
  params:
    d_model: 128      # ← Keep
    nhead: 4          # ← Reduce from 4 to 4 (already good)
    e_layers: 2
    d_layers: 1
    ff_dim: 256       # ← Keep
    factor: 5
    dropout: 0.1
    pred_len: 1
    distill: true
    
# Target: ~200K (good, minimal change)
```

**Example: GRUARModel**
```yaml
# Before: ~100K
model:
  name: gru_ar
  params:
    hidden_size: 128   # ← Increase to 256
    num_layers: 2      # ← Or increase to 3
    dropout: 0.1
    bidirectional: false
    use_attention: false
    
# Target: ~250K (doubling hidden_size)
```

**Example: DLinear**
```yaml
# Before: ~170
model:
  name: dlinear
  params:
    seq_len: ${data.window.input_len}
    pred_len: ${data.window.pred_len}
    kernel_size: 25
    # Add hidden layers for capacity?
    # DLinear is inherently simple - may need special handling
    
# Alternative: Keep as lightweight baseline, create "DLinear-Large" variant
```

### Phase 2: Code Updates (Conditional)

Only if config changes require architectural modifications:

1. **GRUARModel**: Already supports arbitrary hidden_size
2. **LSTMARModel**: Already supports arbitrary hidden_size
3. **Transformers**: Most support arbitrary d_model (verify)
4. **DLinear/NLinear**: May need new "expanded" versions

### Phase 3: Validation Testing

1. **Parameter count verification:**
   ```bash
   python -c "
   from airtrace.models import build_model
   from hydra.utils import instantiate
   
   cfg = instantiate(config_path='configs/model/transformer.yaml')
   model = build_model('transformer', input_dim=10, output_dim=5, **cfg.params)
   print(f'Params: {model.get_num_params()}')
   "
   ```

2. **Training benchmarking:**
   - Train each model on standard dataset
   - Measure: wallclock time, GPU memory, convergence speed
   - Target: ±20% variation acceptable

3. **Performance evaluation:**
   - Compare test metrics (MAE, RMSE, etc.)
   - Verify standardization doesn't hurt SOTA models
   - Document any performance regressions

### Phase 4: Documentation

1. Update `README.md` Model Registry with aligned parameters
2. Add `PARAMETER_ALIGNMENT.md` explaining choices
3. Create per-model configuration guide
4. Add validation scripts to CI/CD

---

## Specific Model Adjustments

### RNN Models

#### GRUARModel & LSTMARModel
```python
# Current: hidden_size=128, num_layers=2 → ~100K params
# Target: hidden_size=256, num_layers=2 → ~250K params (approx)

# Config change:
model:
  params:
    hidden_size: 256  # Increased
    num_layers: 2     # Keep
```

#### GRUSeq2Seq & LSTMSeq2Seq
```python
# Current: ~200K (GRU) / ~400K (LSTM)
# Target: Reduce for GRU, keep LSTM baseline OR add "seq2seq_large" variant

# Option A: Reduce both
# hidden_size: 128 → 100
# Target: ~150K

# Option B: Keep as "seq2seq_large" (transfer learning)
```

### Convolutional Models

#### TCN
```python
# Current: num_channels=[64, 128, 128, 256] → ~105K params
# Target: ~250K params → increase channels or layers

# Option A: Deeper TCN
# num_channels: [64, 128, 128, 256, 256]  # Add layer
# Expected: ~150-170K

# Option B: Wider TCN
# num_channels: [128, 256, 256, 512]  # Double channels
# Expected: ~350K (too much)

# Choose: Deeper option
model:
  params:
    num_channels: [64, 128, 128, 256, 256]  # Add 1 layer
```

#### ModernTCN
```python
# Current: num_blocks=6, hidden_channels=64 → ~30-40K
# Target: ~200K (5x increase)

# Options:
# 1. Increase hidden_channels: 64 → 192 (3x)
# 2. Increase num_blocks: 6 → 12 (2x)
# 3. Both: 64 → 128, num_blocks → 8 (total 2x2)

# Choose: 64 → 128, num_blocks → 6 → 10
model:
  params:
    num_blocks: 10
    hidden_channels: 128
    kernel_size: 3
    large_kernel_size: 51
    dilation_growth: 2
    dropout: 0.1
    use_large_kernel: true
```

### Transformer Models

Most Transformers are already in the 150-300K range. Fine-tune:

#### PatchTST (170K → 250K)
```yaml
# Increase d_model: 128 → 160
# Or increase dim_feedforward: 256 → 384
# Or add num_layers: 3 → 4

model:
  params:
    patch_len: 16
    stride: 8
    d_model: 160        # Increased from 128
    nhead: 8
    num_layers: 3
    dim_feedforward: 256
    dropout: 0.1
    activation: "gelu"
```

#### Informer (197K → 250K)
```yaml
# Already good, small tweak:
# Increase ff_dim: 256 → 280
# Or increase nhead: 4 → 8

model:
  params:
    d_model: 128
    nhead: 8            # Increased from 4
    e_layers: 2
    d_layers: 1
    ff_dim: 256
    factor: 5
    dropout: 0.1
    pred_len: 1
    distill: true
```

### MLP-Based Models

#### DLinear (170 → 50K)
```python
# Current: Highly parameter-efficient
# Option A: Keep as lightweight baseline (separate comparison group)
# Option B: Create "DLinear-Plus" with hidden layers

# Recommend: Keep original, create exp variant
# configs/model/dlinear_plus.yaml:
model:
  name: dlinear_plus
  params:
    seq_len: ${data.window.input_len}
    pred_len: ${data.window.pred_len}
    kernel_size: 25
    hidden_dim: 512      # New: Add MLP expansion
    num_hidden_layers: 2 # New
```

#### NLinear (600 → 50K)
```yaml
# Similar strategy: Create "nlinear_plus" variant
model:
  name: nlinear_plus
  params:
    seq_len: ${data.window.input_len}
    pred_len: ${data.window.pred_len}
    center_data: true
    hidden_dim: 512      # New
    num_hidden_layers: 2 # New
```

#### TimeMixer (35K → 150K)
```yaml
# Increase capacity:
# d_model: 64 → 128
# down_sampling_layers: 3 → 4

model:
  params:
    d_model: 128              # Increased from 64
    num_layers: 2
    down_sampling_layers: 4   # Increased from 3
    decomp_kernel: 25
    dropout: 0.1
```

### State-Space Models

#### MambaTSA (145K → 250K)
```yaml
# Increase:
# embed_dim: 128 → 256
# OR state_dim: 16 → 32
# OR num_layers: 4 → 6

model:
  params:
    pred_len: 64
    patch_len: 16
    stride: 8
    embed_dim: 256          # Increased from 128
    state_dim: 16
    num_layers: 4
    expand_factor: 2
    bidirectional_scan: true
    dropout: 0.1
    normalize_input: true
```

#### Mamba2 (16M → 2M)
```yaml
# Reduce for Tier 2 membership:
# embed_dim: 512 → 256
# state_dim: 256 → 128
# num_layers: 8 → 6

model:
  params:
    pred_len: 64
    embed_dim: 256          # Reduced from 512
    state_dim: 128          # Reduced from 256
    num_layers: 6           # Reduced from 8
    conv_kernel_size: 5
    chunk_length: 1024
    bidirectional_scan: true
    decay_init: 0.0
    dropout: 0.1
    ff_expansion: 4
    adapter_rank: 0         # Disable LoRA by default
    freeze_backbone: false
```

### Advanced Models

#### iTransformer (3.6M → 900K)
```yaml
# Reduce d_model: 512 → 256
# Keep d_feedforward: 2048 → 1024

model:
  params:
    d_model: 256            # Reduced from 512
    nhead: 8
    num_layers: 3
    dim_feedforward: 1024   # Reduced from 2048
    dropout: 0.1
    activation: "gelu"
    use_norm: true
    pred_len: 1
```

#### CycleNet (2M → 500K)
```yaml
# Reduce:
# hidden_dim: 256 → 128
# period_len: 32 → 24

model:
  params:
    period_len: 24          # Reduced from 32
    backbone: "mlp"
    hidden_dim: 128         # Reduced from 256
    dropout: 0.1
    activation: "gelu"
```

#### NBeats (400K → 300K)
```yaml
# Reduce hidden_size: 256 → 200
model:
  params:
    stack_types: ["trend", "seasonality"]
    num_blocks_per_stack: 2
    hidden_size: 200        # Reduced from 256
    num_layers: 4
    pred_len: 1
    degree: 2
    harmonics: null
    dropout: 0.0
```

---

## Validation Checklist

- [ ] Update all Tier 1 model configs
- [ ] Verify parameter counts are within ±50K of target
- [ ] Run convergence tests on 3 datasets
- [ ] Document any performance changes
- [ ] Update README.md Model Registry
- [ ] Add alignment validation to CI
- [ ] Create example scripts showing parameter consistency
- [ ] Benchmark training time per model

---

## Expected Benefits

### Immediate
1. **Fair comparison:** All Tier 1 models have comparable capacity
2. **Reproducibility:** Standardized configurations for baselines
3. **Efficiency:** Faster iteration with similar training times
4. **Clarity:** Clear tiers for different use cases

### Long-term
1. **Research integrity:** Parameter normalization is standard in ML
2. **Knowledge sharing:** Results more transferable across organizations
3. **Model zoo consistency:** Sets precedent for future models
4. **Documentation:** Clear guidance for practitioners

---

## Risk Mitigation

### Risk: "Aligned parameters hurt state-of-the-art models"
**Mitigation:** 
- Provide Tier 2 ("large") configurations as alternatives
- Document any performance trade-offs
- Enable quick re-expansion for research purposes

### Risk: "Models need specific parameter counts"
**Mitigation:**
- Verify alignment doesn't significantly impact performance
- Create "expert" configurations for practitioners
- Allow per-experiment overrides in configs

### Risk: "Implementation complexity"
**Mitigation:**
- Most adjustments are config-only (no code changes)
- Incremental rollout (RNNs → Convs → Transformers)
- Comprehensive testing before committing

---

## Timeline

| Phase | Timeline | Deliverables |
|-------|----------|--------------|
| **1. Planning** | Now | This document |
| **2. Config Updates** | Week 1 | Updated YAML configs |
| **3. Implementation** | Week 2 | Code modifications (if needed) |
| **4. Validation** | Week 2-3 | Test results, performance reports |
| **5. Documentation** | Week 3 | Updated README, guides, examples |
| **6. Integration** | Week 4 | CI/CD validation, final review |

---

## Conclusion

This alignment strategy provides a systematic approach to bringing AirTrace's model zoo into comparable parameter ranges, enabling rigorous benchmarking and fair comparison across architectures. The three-tier system accommodates different research needs while maintaining clarity and reproducibility.

