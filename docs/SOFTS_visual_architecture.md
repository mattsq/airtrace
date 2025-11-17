# SOFTS Visual Architecture

A visual guide to understanding the SOFTS model architecture.

## High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         SOFTS MODEL                             │
│                                                                 │
│  Input: [Batch, Time, Channels]                                │
│         [B,    T,    D]                                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   INSTANCE NORMALIZATION                        │
│                     (per channel)                               │
│                                                                 │
│  mean_c = mean(x[:, :, c])    // per channel c                 │
│  std_c  = std(x[:, :, c])                                       │
│  x_norm = (x - mean) / std                                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              CHANNEL-AS-TOKEN EMBEDDING                         │
│         (DataEmbedding_inverted)                                │
│                                                                 │
│  [B, T, D] ──permute──> [B, D, T]                              │
│            ──linear──>  [B, D, d_model]                         │
│            ──dropout──> [B, D, d_model]                         │
│                                                                 │
│  Key: Each channel becomes a "token" with d_model features      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      ENCODER STACK                              │
│              (e_layers EncoderLayers)                           │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │              EncoderLayer 1                               │ │
│  │  ┌──────────────────────────────────────────────────┐    │ │
│  │  │         STAR MODULE                              │    │ │
│  │  │  [B, D, d_model] → [B, D, d_model]              │    │ │
│  │  └──────────────────────────────────────────────────┘    │ │
│  │                      │                                    │ │
│  │                      ▼                                    │ │
│  │                  + Residual                               │ │
│  │                      │                                    │ │
│  │                      ▼                                    │ │
│  │               LayerNorm                                   │ │
│  │                      │                                    │ │
│  │                      ▼                                    │ │
│  │  ┌──────────────────────────────────────────────────┐    │ │
│  │  │         FFN (Conv1D)                             │    │ │
│  │  │  [B, D, d_model] → [B, D, d_ff] → [B, D, d_model]│   │ │
│  │  └──────────────────────────────────────────────────┘    │ │
│  │                      │                                    │ │
│  │                      ▼                                    │ │
│  │                  + Residual                               │ │
│  │                      │                                    │ │
│  │                      ▼                                    │ │
│  │               LayerNorm                                   │ │
│  └───────────────────────────────────────────────────────────┘ │
│                              │                                  │
│                              ▼                                  │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │              EncoderLayer 2                               │ │
│  │                    ...                                    │ │
│  └───────────────────────────────────────────────────────────┘ │
│                              │                                  │
│                              ▼                                  │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │              EncoderLayer N                               │ │
│  │                    ...                                    │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                 │
│  Output: [B, D, d_model]                                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   LINEAR PROJECTION                             │
│                                                                 │
│  [B, D, d_model] ──linear──> [B, D, pred_len]                  │
│                  ──permute─> [B, pred_len, D]                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DE-NORMALIZATION                             │
│                                                                 │
│  x = x * std + mean    // reverse normalization                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        OUTPUT                                   │
│         [Batch, Prediction Length, Channels]                   │
│         [B,     pred_len,          D]                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## STAR Module Deep Dive

```
┌──────────────────────────────────────────────────────────────────┐
│                        STAR MODULE                               │
│         STar Aggregate-Redistribute Module                       │
│                                                                  │
│  Input: [B, C, d_series]   (C = num_channels, d_series = d_model)│
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                    AGGREGATE PHASE                               │
│                                                                  │
│  Step 1: FFN Preprocessing                                      │
│  ┌──────────────────────────────────────────┐                   │
│  │  h = GELU(Linear(x))                     │                   │
│  │  [B, C, d_series] → [B, C, d_series]     │                   │
│  └──────────────────────────────────────────┘                   │
│                              │                                   │
│  Step 2: Compress to Core                                       │
│  ┌──────────────────────────────────────────┐                   │
│  │  z = Linear(h)                           │                   │
│  │  [B, C, d_series] → [B, C, d_core]       │                   │
│  └──────────────────────────────────────────┘                   │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                  STOCHASTIC POOLING                              │
│                                                                  │
│  ┌────────────────────────┬─────────────────────────────────┐   │
│  │   TRAINING MODE        │      EVAL MODE                  │   │
│  │   (Stochastic)         │      (Deterministic)            │   │
│  ├────────────────────────┼─────────────────────────────────┤   │
│  │                        │                                 │   │
│  │  1. Softmax probabilities across channels                │   │
│  │     p = softmax(z, dim=channels)                         │   │
│  │     [B, C, d_core]                                       │   │
│  │                        │                                 │   │
│  │  2. SAMPLE channel     │  2. WEIGHTED AVERAGE            │   │
│  │     idx ~ Multinomial(p)│     z_agg = Σ(p * z)          │   │
│  │     z_agg = z[idx]     │     [B, 1, d_core]             │   │
│  │     [B, 1, d_core]     │                                 │   │
│  │                        │                                 │   │
│  │  3. Broadcast to all channels                            │   │
│  │     z_agg = repeat(z_agg, C)                             │   │
│  │     [B, C, d_core]                                       │   │
│  │                        │                                 │   │
│  └────────────────────────┴─────────────────────────────────┘   │
│                                                                  │
│  Key Benefit: O(C) instead of O(C²) channel interactions        │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                   REDISTRIBUTE PHASE                             │
│                                                                  │
│  Step 1: Concatenate Original + Aggregated                      │
│  ┌──────────────────────────────────────────┐                   │
│  │  fused = concat([x, z_agg], dim=-1)      │                   │
│  │  [B, C, d_series + d_core]               │                   │
│  └──────────────────────────────────────────┘                   │
│                              │                                   │
│  Step 2: Fusion MLP                                             │
│  ┌──────────────────────────────────────────┐                   │
│  │  fused = GELU(Linear(fused))             │                   │
│  │  [B, C, d_series + d_core]               │                   │
│  │           ↓                               │                   │
│  │  [B, C, d_series]                        │                   │
│  └──────────────────────────────────────────┘                   │
│                              │                                   │
│  Step 3: Final Projection                                       │
│  ┌──────────────────────────────────────────┐                   │
│  │  output = Linear(fused)                  │                   │
│  │  [B, C, d_series] → [B, C, d_series]     │                   │
│  └──────────────────────────────────────────┘                   │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                         OUTPUT                                   │
│                   [B, C, d_series]                               │
│                                                                  │
│  Ready for residual connection in EncoderLayer                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## Dimension Transformations Flow

```
Input Timeseries
[B=32, T=96, D=21]          // 32 samples, 96 timesteps, 21 sensors
    │
    │ Instance norm (per channel)
    ▼
[B=32, T=96, D=21]          // normalized
    │
    │ Permute + Linear embedding
    ▼
[B=32, D=21, d_model=512]   // 21 channels, each with 512 features
    │
    │ EncoderLayer 1
    ▼
[B=32, D=21, d_model=512]
    │
    │  STAR Module:
    │  ├─ gen1: [32, 21, 512] → [32, 21, 512]
    │  ├─ gen2: [32, 21, 512] → [32, 21, 128]  (compress to d_core)
    │  │  └─ Stochastic pool: [32, 21, 128] → [32, 21, 128]
    │  ├─ Concat: [32, 21, 512+128] = [32, 21, 640]
    │  ├─ gen3: [32, 21, 640] → [32, 21, 512]
    │  └─ gen4: [32, 21, 512] → [32, 21, 512]
    │
    │  FFN (via Conv1D):
    │  ├─ conv1: [32, 512, 21] → [32, 512, 21]  (d_ff=512, 1x expansion)
    │  └─ conv2: [32, 512, 21] → [32, 512, 21]
    │
    ▼
[B=32, D=21, d_model=512]
    │
    │ EncoderLayer 2, ..., e_layers
    ▼
[B=32, D=21, d_model=512]
    │
    │ Linear projection
    ▼
[B=32, D=21, pred_len=96]
    │
    │ Permute
    ▼
[B=32, pred_len=96, D=21]
    │
    │ De-normalize
    ▼
Output Predictions
[B=32, pred_len=96, D=21]   // 32 samples, 96 future steps, 21 sensors
```

---

## Parameter Count Analysis

For a typical configuration:
```
seq_len = 96
pred_len = 96
input_dim = 21
hidden_dim = 512
d_core = 128
d_ff = 512
e_layers = 3
```

### Embedding Layer
```
Linear(seq_len, hidden_dim) = 96 × 512 = 49,152 params
```

### STAR Module (per layer)
```
gen1: 512 × 512 = 262,144
gen2: 512 × 128 =  65,536
gen3: 640 × 512 = 327,680
gen4: 512 × 512 = 262,144
───────────────────────────
Total per STAR = 917,504 params
```

### EncoderLayer FFN (per layer)
```
conv1: 512 × 512 = 262,144
conv2: 512 × 512 = 262,144
norm1: 512 × 2   =   1,024 (γ, β)
norm2: 512 × 2   =   1,024
───────────────────────────
Total per FFN  = 526,336 params
```

### Total per EncoderLayer
```
STAR + FFN = 917,504 + 526,336 = 1,443,840 params
```

### Encoder Stack (3 layers)
```
3 × 1,443,840 = 4,331,520 params
```

### Projection Layer
```
Linear(hidden_dim, pred_len) = 512 × 96 = 49,152 params
```

### Total Model Parameters
```
Embedding:       49,152
Encoder:     4,331,520
Projection:     49,152
───────────────────────
Total ≈ 4.4M parameters
```

**Scaling**: Parameters scale roughly linearly with:
- Number of layers (e_layers)
- Hidden dimension squared (hidden_dim²)
- Core dimension (d_core)

---

## Channel Interaction Visualization

### Traditional Attention (O(C²))
```
Channel 1 ────┐
              ├──→ Attention weights C × C matrix
Channel 2 ────┤
              │
Channel 3 ────┤
              │
  ...         │
              │
Channel C ────┘

Complexity: O(C²)
Memory: C × C matrix
```

### STAR Stochastic Pooling (O(C))
```
Channel 1 ────┐
              ├──→ Softmax probabilities
Channel 2 ────┤   [sample one]
              │        ↓
Channel 3 ────┤   Selected channel → Broadcast
              │        ↓
  ...         │   Core representation
              │        ↓
Channel C ────┘   Fuse with all channels

Complexity: O(C)
Memory: C × d_core (much smaller)
```

**Benefit**: When C is large (many sensors), STAR is much more efficient.

---

## Normalization Strategy

### Instance Normalization (Input)
```
For each channel c in [1, ..., D]:

  mean_c = (1/T) Σ_{t=1}^T x[t, c]

  std_c = sqrt((1/T) Σ_{t=1}^T (x[t, c] - mean_c)²)

  x_norm[t, c] = (x[t, c] - mean_c) / std_c


Applied BEFORE embedding
Removes per-channel trends and scales
```

### Layer Normalization (Internal)
```
Applied AFTER each residual connection in EncoderLayer

  mean = (1/d_model) Σ_{i=1}^{d_model} x[i]

  std = sqrt((1/d_model) Σ_{i=1}^{d_model} (x[i] - mean)²)

  x_norm[i] = γ * (x[i] - mean) / std + β


Stabilizes training
Normalized across feature dimension
```

### De-Normalization (Output)
```
Reverse the input normalization:

  y[t, c] = y_norm[t, c] * std_c + mean_c


Restores original scale and trend
Uses statistics from input window
```

---

## Computational Complexity

### Forward Pass

| Component | Input Shape | Output Shape | Complexity |
|-----------|-------------|--------------|------------|
| Instance Norm | [B, T, D] | [B, T, D] | O(BTD) |
| Embedding | [B, T, D] | [B, D, d] | O(BTDd) |
| STAR (×L) | [B, D, d] | [B, D, d] | O(LBDd²) |
| FFN (×L) | [B, D, d] | [B, D, d] | O(LBDd·d_ff) |
| Projection | [B, D, d] | [B, P, D] | O(BDdP) |

Where:
- B = batch size
- T = sequence length
- D = number of channels
- d = hidden dimension (d_model)
- L = number of layers
- P = prediction length

**Total**: O(LBD(d² + d·d_ff)) - Linear in channels D!

### Comparison to Transformer

| Model | Channel Mixing | Complexity |
|-------|----------------|------------|
| Transformer | Self-attention | O(D²d) |
| SOFTS | STAR | O(Dd²) |

When D > d, SOFTS is more efficient!

---

## Training vs. Inference Differences

### Stochastic Pooling Behavior

```
┌─────────────────────┬──────────────────────────────────┐
│   Training Mode     │        Inference Mode            │
├─────────────────────┼──────────────────────────────────┤
│                     │                                  │
│  Forward pass 1:    │  Forward pass 1:                 │
│    Different        │    Same output                   │
│    (stochastic      │    (deterministic                │
│     sampling)       │     weighted avg)                │
│         ↓           │         ↓                        │
│  Forward pass 2:    │  Forward pass 2:                 │
│    Different        │    Same output                   │
│                     │                                  │
├─────────────────────┼──────────────────────────────────┤
│  Regularization     │  Stable predictions              │
│  Explores channel   │  Uses all channels               │
│  combinations       │  optimally                       │
│                     │                                  │
│  Faster (sampling)  │  Slightly slower (weighted sum)  │
└─────────────────────┴──────────────────────────────────┘
```

**Important**: Set `model.eval()` for inference to get deterministic outputs!

---

## Summary: Key Architectural Principles

1. **Channel-as-Token**: Treat each variable as a token (invert time and channel dims)
2. **Pure MLP**: No attention mechanisms, only linear layers and MLPs
3. **STAR Module**: Efficient channel mixing via stochastic pooling
4. **Instance Norm**: Per-channel normalization following Non-stationary Transformer
5. **Post-Norm**: Layer normalization after residual connections
6. **Stochastic Regularization**: Multinomial sampling during training
7. **Simple Decoder**: Just a linear projection (no complex decoder)

---

## Quick Reference: Shapes at Each Step

```python
# Input
x = [B, T, D]                    # e.g., [32, 96, 21]

# Normalize
x_norm = instance_norm(x)        # [32, 96, 21]

# Embed (permute first)
x_embed = embed(x_norm)          # [32, 21, 512]

# Encode
for layer in encoder:
    x_embed = layer(x_embed)     # [32, 21, 512]

# Project
x_proj = project(x_embed)        # [32, 21, 96]

# Permute
x_out = x_proj.permute(0, 2, 1)  # [32, 96, 21]

# De-normalize
y = denorm(x_out)                # [32, 96, 21]
```

---

## Common Questions

**Q: Why permute channels and time?**
A: Makes each channel a "token" so the model learns cross-channel patterns. Time is embedded into features.

**Q: Why stochastic pooling in training?**
A: Acts as regularization, prevents overfitting, and is computationally efficient (O(C) vs O(C²)).

**Q: What if I have many channels (D >> d_model)?**
A: SOFTS excels here! Complexity is O(Dd²), not O(D²d) like Transformer attention.

**Q: Can I use different seq_len and pred_len?**
A: Yes! seq_len is embedded into d_model, pred_len is just the projection output size.

**Q: Do I need temporal covariates (x_mark)?**
A: No, they're optional. The model works without them (set to None).

**Q: How does this compare to iTransformer?**
A: Similar channel-as-token idea, but SOFTS uses STAR (MLP) instead of attention for efficiency.
