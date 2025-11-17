# SOFTS Model Architecture - Complete Implementation Guide

**Source**: https://github.com/Secilia-Cxy/SOFTS
**Paper**: NeurIPS 2024 - "SOFTS: Efficient Multivariate Time Series Forecasting with Series-cOre Fused Time Series forecaster"

## Overview

SOFTS (Series-cOre Fused Time Series forecaster) is a pure MLP-based model for multivariate time series forecasting that achieves state-of-the-art performance. The key innovation is the **STAR (STar Aggregate-Redistribute)** module that efficiently handles channel interactions through stochastic pooling.

### Architecture Philosophy

- **Channel-inverted approach**: Treats each variable/channel as a separate token (similar to iTransformer)
- **Pure MLP**: No attention mechanisms, uses only linear layers and MLPs
- **Stochastic aggregation**: Uses multinomial sampling during training for efficient channel mixing
- **Normalization strategy**: Follows Non-stationary Transformer approach

---

## Complete Architecture

```
Input [B, T, D]
    ↓
Normalization (Instance normalization per channel)
    ↓
DataEmbedding_inverted → [B, D, d_model]
    ↓
Encoder (Stack of e_layers EncoderLayers)
    │
    ├─→ EncoderLayer 1
    │   ├─→ STAR Module (Aggregate-Redistribute)
    │   └─→ FFN (Conv1D-based feedforward)
    │
    ├─→ EncoderLayer 2
    │   └─→ ...
    │
    └─→ EncoderLayer N
        └─→ ...
    ↓ [B, D, d_model]
Linear Projection → [B, D, pred_len]
    ↓
Permute → [B, pred_len, D]
    ↓
De-Normalization
    ↓
Output [B, pred_len, D]
```

---

## 1. STAR Module (Core Innovation)

The STAR module is the heart of SOFTS, performing aggregate-redistribute operations across channels.

### Components

```python
class STAR(nn.Module):
    def __init__(self, d_series, d_core):
        super(STAR, self).__init__()
        # d_series: input dimension per channel (typically d_model)
        # d_core: compressed core dimension (bottleneck)

        self.gen1 = nn.Linear(d_series, d_series)  # FFN preprocessing
        self.gen2 = nn.Linear(d_series, d_core)    # Aggregate: compress to core
        self.gen3 = nn.Linear(d_series + d_core, d_series)  # Fuse: combine original + core
        self.gen4 = nn.Linear(d_series, d_series)  # Final projection
```

### Forward Pass Operations

**Input shape**: `[B, C, d_series]` where:
- B = batch size
- C = number of channels/variables
- d_series = feature dimension per channel (= d_model)

#### Step 1: FFN Preprocessing + Aggregation
```python
# Apply GELU activation + linear projection
combined_mean = F.gelu(self.gen1(input))  # [B, C, d_series]
combined_mean = self.gen2(combined_mean)   # [B, C, d_core] - compress to core
```

#### Step 2: Stochastic Pooling (TRAINING MODE)

**Key insight**: Instead of weighted averaging, use multinomial sampling for efficiency.

```python
if self.training:
    # Compute sampling probabilities for each channel
    ratio = F.softmax(combined_mean, dim=1)  # [B, C, d_core] - normalize across channels
    ratio = ratio.permute(0, 2, 1)           # [B, d_core, C]
    ratio = ratio.reshape(-1, C)             # [B*d_core, C]

    # Sample one channel per core dimension
    indices = torch.multinomial(ratio, 1)    # [B*d_core, 1] - sample indices
    indices = indices.view(batch_size, -1, 1).permute(0, 2, 1)  # [B, 1, d_core]

    # Gather sampled values and broadcast back to all channels
    combined_mean = torch.gather(combined_mean, 1, indices)  # [B, 1, d_core]
    combined_mean = combined_mean.repeat(1, channels, 1)     # [B, C, d_core]
```

#### Step 3: Weighted Pooling (EVAL MODE)

```python
else:
    # Soft attention-like weighted average
    weight = F.softmax(combined_mean, dim=1)  # [B, C, d_core]
    combined_mean = torch.sum(combined_mean * weight, dim=1, keepdim=True)  # [B, 1, d_core]
    combined_mean = combined_mean.repeat(1, channels, 1)  # [B, C, d_core]
```

#### Step 4: MLP Fusion (Redistribute)

```python
# Concatenate original input with aggregated core
combined_mean_cat = torch.cat([input, combined_mean], -1)  # [B, C, d_series + d_core]

# Apply fusion MLP
combined_mean_cat = F.gelu(self.gen3(combined_mean_cat))  # [B, C, d_series]
output = self.gen4(combined_mean_cat)                      # [B, C, d_series]
```

**Output shape**: `[B, C, d_series]` - same as input

### Mathematical Formulation

For channel i at position in the sequence:

**Aggregate**:
- h_i = GELU(W₁ · x_i)
- z_i = W₂ · h_i  (compress to core)

**Stochastic Pooling**:
- p_i = softmax(z₁, ..., z_C)  (across channels)
- k ~ Multinomial(p)  (sample channel index)
- z_agg = z_k  (use sampled channel's core)

**Redistribute**:
- y_i = W₄ · GELU(W₃ · [x_i; z_agg])

---

## 2. EncoderLayer (Temporal Mixing)

Each encoder layer combines STAR module with a pointwise feedforward network.

### Structure

```python
class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model  # Default: 4x expansion

        self.attention = attention  # STAR module passed as "attention"

        # Pointwise feedforward: 1x1 convolutions
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)

        # Normalization and regularization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
```

### Forward Pass

```python
def forward(self, x, attn_mask=None, tau=None, delta=None):
    # Input: [B, C, d_model]

    # STAR module + residual connection
    new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
    x = x + self.dropout(new_x)

    # Post-norm 1
    y = x = self.norm1(x)  # [B, C, d_model]

    # Pointwise FFN: expand → activate → compress
    y = y.transpose(-1, 1)  # [B, d_model, C] - prepare for conv1d
    y = self.dropout(self.activation(self.conv1(y)))  # [B, d_ff, C]
    y = self.dropout(self.conv2(y))  # [B, d_model, C]
    y = y.transpose(-1, 1)  # [B, C, d_model] - back to original

    # Post-norm 2 + residual
    return self.norm2(x + y), attn  # [B, C, d_model]
```

**Key points**:
- Uses **post-normalization** (after residual connection)
- Conv1D acts as efficient pointwise MLP when kernel_size=1
- Applies dropout after both STAR and FFN

---

## 3. Encoder Stack

```python
class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)  # List of EncoderLayers
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer  # Optional final normalization

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x: [B, C, d_model]
        attns = []

        if self.conv_layers is not None:
            # With intermediate conv layers (not used in SOFTS)
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            # Simple stacking (SOFTS uses this path)
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns  # [B, C, d_model], list of attention outputs
```

---

## 4. DataEmbedding_inverted (Channel-as-Token)

Transforms time series from `[B, T, D]` to `[B, D, d_model]` - treating each channel as a token.

```python
class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        # c_in: sequence length (T)
        # d_model: embedding dimension

        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        # Input x: [B, T, D]
        x = x.permute(0, 2, 1)  # [B, D, T] - channels become "sequence"

        if x_mark is None:
            x = self.value_embedding(x)  # [B, D, d_model]
        else:
            # Optional: concatenate temporal covariates
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))

        return self.dropout(x)  # [B, D, d_model]
```

**Key insight**: The time dimension T is embedded into d_model, and channels D become the "sequence length" for the encoder.

---

## 5. Complete SOFTS Model

```python
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len      # Input sequence length
        self.pred_len = configs.pred_len    # Prediction horizon
        self.use_norm = configs.use_norm    # Instance normalization flag

        # Embedding: [B, T, D] → [B, D, d_model]
        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len,
            configs.d_model,
            configs.dropout
        )

        # Encoder: Stack of STAR-based EncoderLayers
        self.encoder = Encoder([
            EncoderLayer(
                STAR(configs.d_model, configs.d_core),  # STAR as attention mechanism
                configs.d_model,
                configs.d_ff,
                dropout=configs.dropout,
                activation=configs.activation,
            ) for l in range(configs.e_layers)
        ])

        # Decoder: Simple linear projection
        self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)
```

### Forward Pass

```python
def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
    # Input: x_enc [B, T, D]

    # Step 1: Instance normalization (per channel)
    if self.use_norm:
        means = x_enc.mean(1, keepdim=True).detach()  # [B, 1, D]
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev  # [B, T, D]

    _, _, N = x_enc.shape  # N = D = number of channels

    # Step 2: Embedding [B, T, D] → [B, D, d_model]
    enc_out = self.enc_embedding(x_enc, x_mark_enc)

    # Step 3: Encoder [B, D, d_model] → [B, D, d_model]
    enc_out, attns = self.encoder(enc_out, attn_mask=None)

    # Step 4: Projection [B, D, d_model] → [B, D, pred_len] → [B, pred_len, D]
    dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]

    # Step 5: De-normalization
    if self.use_norm:
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

    return dec_out  # [B, pred_len, D]

def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
    dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
    return dec_out[:, -self.pred_len:, :]  # [B, pred_len, D]
```

---

## 6. Tensor Shape Transformations

### Complete Flow

```
Input: [B, T, D]
    ↓ (permute + linear)
DataEmbedding_inverted: [B, D, d_model]
    ↓
EncoderLayer 1:
    ↓ (STAR module)
    STAR input: [B, D, d_model]
        ↓ (gen1 + gen2)
        Core: [B, D, d_core]
        ↓ (stochastic pooling)
        Pooled: [B, D, d_core]
        ↓ (concat + gen3 + gen4)
    STAR output: [B, D, d_model]
    ↓ (conv1d FFN)
    FFN: [B, D, d_model]
    ↓
EncoderLayer 2, ..., N: [B, D, d_model]
    ↓
Projection: [B, D, pred_len]
    ↓ (permute)
Output: [B, pred_len, D]
```

### Key Dimension Mappings

- **seq_len (T)**: Input time steps (e.g., 96)
- **pred_len**: Output prediction horizon (e.g., 96, 192, 336, 720)
- **D**: Number of channels/variables
- **d_model**: Model hidden dimension (typically 512)
- **d_core**: Core compression dimension (typically 128-512)
- **d_ff**: Feedforward expansion dimension (typically 512-2048)

---

## 7. Key Hyperparameters

### From Weather Script (Typical Configuration)

```yaml
# Model architecture
d_model: 512          # Main model dimension
d_core: 128           # STAR core compression dimension
d_ff: 512             # Feedforward network dimension
e_layers: 3           # Number of encoder layers

# Sequences
seq_len: 96           # Input sequence length
pred_len: 96          # Prediction horizon (variable: 96, 192, 336, 720)

# Regularization
dropout: 0.0          # Dropout probability (often 0.0 in SOFTS)

# Activation
activation: 'gelu'    # Activation function (GELU preferred)

# Normalization
use_norm: True        # Use instance normalization

# Training
batch_size: 16        # Batch size
learning_rate: 0.0003 # Learning rate
train_epochs: 10      # Number of epochs
```

### From run.py Defaults

```yaml
# Alternative configurations
d_model: 512
d_core: 512          # Can be same as d_model (no compression)
d_ff: 2048           # Often 4x d_model
e_layers: 2          # Can use fewer layers
dropout: 0.0         # Very low or zero dropout
batch_size: 32
learning_rate: 0.0001
```

### Typical Ranges

- **d_model**: 128-512 (higher for larger datasets)
- **d_core**: 64-512 (often d_model/2 or d_model/4 for compression)
- **d_ff**: 512-2048 (typically 1x to 4x d_model)
- **e_layers**: 2-4 (diminishing returns beyond 4)
- **dropout**: 0.0-0.1 (often very low or zero)

---

## 8. Activation Functions and Normalization

### Activation Functions

**Primary**: GELU (Gaussian Error Linear Unit)
```python
self.activation = F.gelu
```

Used in:
- STAR module (gen1, gen3)
- EncoderLayer FFN (conv1)

**Alternative**: ReLU (configurable)
```python
self.activation = F.relu if activation == "relu" else F.gelu
```

### Normalization Strategies

#### Instance Normalization (Non-stationary Transformer)

Applied per channel across time dimension:
```python
# Normalize
means = x_enc.mean(1, keepdim=True).detach()  # [B, 1, D]
x_enc = x_enc - means
stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
x_enc /= stdev

# De-normalize
dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
```

**Key points**:
- Normalization is **per channel** (across time)
- Statistics computed on input window, applied to predictions
- Uses `.detach()` to prevent gradient flow through statistics

#### Layer Normalization

Applied within each EncoderLayer:
```python
self.norm1 = nn.LayerNorm(d_model)
self.norm2 = nn.LayerNorm(d_model)
```

**Position**: Post-normalization (after residual connection)

---

## 9. Stochastic Pooling Deep Dive

### Why Stochastic Pooling?

**Problem**: Weighted averaging across all channels is expensive (O(C²) interactions).

**Solution**: Sample one channel per core dimension using learned probabilities.

**Benefits**:
1. **Efficiency**: O(C) instead of O(C²)
2. **Regularization**: Stochasticity prevents overfitting
3. **Scalability**: Works well with many channels

### Training vs. Inference

| Mode | Strategy | Complexity |
|------|----------|------------|
| Training | Stochastic (multinomial sampling) | O(C) |
| Inference | Deterministic (weighted average) | O(C) |

### Mathematical Detail

For each core dimension j:

**Training**:
1. Compute probabilities: p_ij = softmax_i(z_ij)
2. Sample channel: k ~ Multinomial(p_·j)
3. Use sampled value: z_agg,j = z_kj

**Inference**:
1. Compute probabilities: p_ij = softmax_i(z_ij)
2. Weighted average: z_agg,j = Σ_i p_ij · z_ij

---

## 10. Comparison to Other Architectures

### vs. Transformer
- **No attention**: STAR module replaces self-attention
- **No positional encoding**: Temporal info embedded via DataEmbedding_inverted
- **Channel-inverted**: Treats channels as tokens, not time steps

### vs. iTransformer
- **Similarity**: Both use channel-as-token approach
- **Difference**: SOFTS uses STAR instead of attention (more efficient)

### vs. PatchTST
- **No patching**: Works on full sequence
- **No masking**: Standard autoregressive forecasting

---

## 11. Implementation Notes for ARBaseModel

### Required Adaptations

```python
class SOFTS(ARBaseModel):
    def __init__(
        self,
        input_dim: int,      # Number of channels (D)
        hidden_dim: int,     # d_model
        output_dim: int,     # Number of channels to predict
        seq_len: int,        # Input sequence length
        pred_len: int,       # Prediction horizon
        d_core: int = 128,   # STAR core dimension
        d_ff: int = 512,     # FFN dimension
        e_layers: int = 3,   # Number of encoder layers
        dropout: float = 0.0,
        activation: str = "gelu",
        use_norm: bool = True,
        **kwargs
    ):
        super().__init__()
        # Initialize components...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D] input tensor
        Returns:
            [B, pred_len, D] predictions
        """
        # x_mark_enc can be None if not using temporal covariates
        return self.forward(x, None, None, None)
```

### Key Considerations

1. **Input format**: Ensure [B, T, D] format
2. **Normalization**: Handle statistics tracking for inference
3. **Stochastic mode**: Implement training/eval mode switching
4. **Config mapping**: Map ARBaseModel params to SOFTS params

---

## 12. Summary: Key Architectural Components

| Component | Purpose | Input Shape | Output Shape |
|-----------|---------|-------------|--------------|
| DataEmbedding_inverted | Channel-as-token embedding | [B, T, D] | [B, D, d_model] |
| STAR | Aggregate-redistribute channel mixing | [B, D, d_model] | [B, D, d_model] |
| EncoderLayer | STAR + FFN + residual | [B, D, d_model] | [B, D, d_model] |
| Encoder | Stack EncoderLayers | [B, D, d_model] | [B, D, d_model] |
| Projection | Temporal projection | [B, D, d_model] | [B, pred_len, D] |

---

## References

- **Paper**: NeurIPS 2024 - "SOFTS: Efficient Multivariate Time Series Forecasting with Series-cOre Fused Time Series forecaster"
- **Code**: https://github.com/Secilia-Cxy/SOFTS
- **arXiv**: 2404.14197
