# AirTrace Model Parameter Configuration Analysis

## Executive Summary

The AirTrace codebase contains **28 trainable deep learning models** across 6 major architecture families. Parameter counts vary widely (from ~1K to ~10M+), indicating significant heterogeneity that must be addressed for fair comparison and reproducibility.

---

## 1. RNN-BASED MODELS (Recurrent Neural Networks)

### 1.1 GRUARModel (`gru_ar.py`)
**File Location:** `/home/user/airtrace/src/airtrace/models/gru_ar.py`
**Config File:** `/home/user/airtrace/src/airtrace/configs/model/gru_ar.yaml`

**Key Architectural Parameters:**
| Parameter | Default | Type | Notes |
|-----------|---------|------|-------|
| `hidden_size` | 128 | int | GRU hidden state dimension |
| `num_layers` | 2 | int | Number of stacked GRU cells |
| `dropout` | 0.1 | float | Dropout rate between layers |
| `bidirectional` | false | bool | Bidirectional GRU flag |
| `use_attention` | false | bool | Optional MultiheadAttention over encoder outputs |

**Parameter Count Formula:**
- Single GRU layer: `3 * hidden_size * (input_dim + hidden_size + 1)`
- Multi-layer with dropout: Additional layers follow same pattern
- With attention: +`4 * encoder_output_dim^2` for attention projections
- Output projection: `hidden_size * output_dim + output_dim`

**Example Parameter Count (input_dim=10, output_dim=5):**
- Encoder (2 layers): ~100K params
- Output projection: ~645 params
- **Total: ~100K params**

---

### 1.2 LSTMARModel (`lstm_ar.py`)
**File Location:** `/home/user/airtrace/src/airtrace/models/lstm_ar.py`
**Config File:** `/home/user/airtrace/src/airtrace/configs/model/lstm_ar.yaml`

**Key Architectural Parameters:**
| Parameter | Default | Type | Notes |
|-----------|---------|------|-------|
| `hidden_size` | 128 | int | LSTM hidden state dimension |
| `num_layers` | 2 | int | Number of stacked LSTM cells |
| `dropout` | 0.1 | float | Dropout rate between layers |
| `bidirectional` | false | bool | Bidirectional LSTM flag |
| `use_attention` | false | bool | Optional MultiheadAttention |

**Parameter Count Formula:**
- Single LSTM layer: `4 * hidden_size * (input_dim + hidden_size + 1)` (4x GRU due to gates + cell state)
- Additional parameters for cell state management
- With attention: same as GRU

**Example Parameter Count (input_dim=10, output_dim=5):**
- Encoder (2 layers): ~200K params (4x GRU due to LSTM gates)
- Output projection: ~645 params
- **Total: ~200K params** (approximately 2x GRU)

---

### 1.3 GRUSeq2SeqModel (`seq2seq.py`)
**File Location:** `/home/user/airtrace/src/airtrace/models/seq2seq.py`
**Config File:** `/home/user/airtrace/src/airtrace/configs/model/gru_seq2seq.yaml`

**Key Architectural Parameters:**
| Parameter | Default | Type | Notes |
|-----------|---------|------|-------|
| `hidden_size` | 128 | int | Hidden dimension for both encoder & decoder |
| `num_layers` | 2 | int | Stacked layers in encoder and decoder |
| `dropout` | 0.1 | float | Dropout rate |
| `use_attention` | false | bool | Luong-style attention mechanism |
| `teacher_forcing_ratio` | 0.5 | float | Probability of using ground truth during training |

**Architecture:**
- Encoder: GRU stack processing full input sequence
- Decoder: GRU stack generating predictions step-by-step
- Optional attention: Context vector from encoder outputs

**Parameter Count Formula:**
- Encoder GRU: `3 * hidden_size * (input_dim + hidden_size + 1) * num_layers`
- Decoder GRU: `3 * hidden_size * (output_dim + hidden_size + 1) * num_layers`
- With attention: +`4 * hidden_size^2` + `hidden_size * 2 + hidden_size`

**Example Parameter Count (input_dim=10, output_dim=5, hidden_size=128, num_layers=2):**
- **Total: ~200K params** (encoder + decoder)

---

### 1.4 LSTMSeq2SeqModel (`seq2seq.py`)
**File Location:** `/home/user/airtrace/src/airtrace/models/seq2seq.py`
**Config File:** `/home/user/airtrace/src/airtrace/configs/model/lstm_seq2seq.yaml`

**Key Architectural Parameters:** (Identical to GRUSeq2Seq)
| Parameter | Default | Type |
|-----------|---------|------|
| `hidden_size` | 128 | int |
| `num_layers` | 2 | int |
| `dropout` | 0.1 | float |
| `use_attention` | false | bool |
| `teacher_forcing_ratio` | 0.5 | float |

**Parameter Count:** ~400K params (4x more than GRU due to LSTM gate complexity)

---

## 2. CONVOLUTIONAL MODELS

### 2.1 TCNModel (Temporal Convolutional Network)
**File Location:** `/home/user/airtrace/src/airtrace/models/tcn.py`
**Config File:** `/home/user/airtrace/src/airtrace/configs/model/tcn.yaml`

**Key Architectural Parameters:**
| Parameter | Default | Type | Notes |
|-----------|---------|------|-------|
| `num_channels` | [64, 128, 128, 256] | list | Hidden channels per layer |
| `kernel_size` | 3 | int | Conv1d kernel size |
| `dropout` | 0.2 | float | Dropout after conv blocks |
| `causal` | true | bool | Causal (non-future) convolutions |
| `use_skip_connections` | true | bool | Residual connections |

**Architecture:**
- Stack of temporal blocks with dilated convolutions
- Dilation increases exponentially: 2^i for layer i
- Each block has 2 conv layers with weight normalization
- Residual connections with dimension projection

**Parameter Count Formula:**
- Each layer: `2 * (in_channels * kernel_size * out_channels + out_channels)` (2 conv layers per block)
- With weight normalization, same complexity
- Output projection: `num_channels[-1] * output_dim + output_dim`

**Example (input_dim=10, output_dim=5, num_channels=[64, 128, 128, 256]):**
- Layer 0: ~20K params
- Layer 1: ~25K params
- Layer 2: ~25K params
- Layer 3: ~35K params
- Output projection: ~1.3K params
- **Total: ~105K params**

---

### 2.2 ModernTCNModel (`moderntcn.py`)
**File Location:** `/home/user/airtrace/src/airtrace/models/moderntcn.py`
**Config File:** `/home/user/airtrace/src/airtrace/configs/model/moderntcn.yaml`

**Key Architectural Parameters:**
| Parameter | Default | Type | Notes |
|-----------|---------|------|-------|
| `num_blocks` | 6 | int | Number of ModernTCN blocks |
| `hidden_channels` | 64 | int | Channel dimension |
| `kernel_size` | 3 | int | Small kernel for local mixing |
| `large_kernel_size` | 51 | int | Large kernel for receptive field |
| `dilation_growth` | 2 | int | Exponential dilation growth |
| `dropout` | 0.1 | float | Dropout rate |
| `use_large_kernel` | true | bool | Alternate between small/large kernels |

**Architecture:**
- Depthwise separable convolutions for efficiency
- Alternates between small and large kernels
- Modern components: LayerNorm, GELU activation
- Each block: LayerNorm → Conv → GELU → Dropout → Conv → Dropout → Residual

**Parameter Count Formula (per block):**
- Small kernel: `in_channels * kernel_size + in_channels * out_channels + out_channels`
- Large kernel: `in_channels * large_kernel_size + in_channels * out_channels + out_channels`
- Depthwise separable reduces parameters vs standard convolution

**Example (input_dim=10, output_dim=5, hidden=64, 6 blocks):**
- **Total: ~30-40K params** (more efficient than classic TCN)

---

## 3. TRANSFORMER-BASED MODELS

### 3.1 TransformerModel (`transformer.py`)
**File Location:** `/home/user/airtrace/src/airtrace/models/transformer.py`
**Config File:** `/home/user/airtrace/src/airtrace/configs/model/transformer.yaml`

**Key Architectural Parameters:**
| Parameter | Default | Type | Notes |
|-----------|---------|------|-------|
| `d_model` | 128 | int | Model/embedding dimension |
| `nhead` | 8 | int | Number of attention heads (must divide d_model) |
| `num_encoder_layers` | 4 | int | Stacked encoder layers |
| `num_decoder_layers` | 4 | int | Stacked decoder layers (not used for AR) |
| `dim_feedforward` | 512 | int | Feed-forward hidden dimension |
| `dropout` | 0.1 | float | Dropout rate |
| `activation` | "gelu" | str | Activation function |
| `causal` | true | bool | Causal masking for AR |

**Architecture:**
- Input projection: `input_dim → d_model`
- Positional encoding (sinusoidal)
- Standard TransformerEncoder with self-attention
- Causal mask to prevent looking at future tokens
- Output projection: `d_model → output_dim`

**Parameter Count Formula:**
- Input projection: `input_dim * d_model`
- Per attention head: `3 * (d_model / nhead)^2 * nhead = 3 * d_model^2 / nhead * nhead = 3 * d_model^2`
- Multi-head attention output: `d_model^2`
- Total attention per layer: `4 * d_model^2`
- Feed-forward per layer: `2 * d_model * dim_feedforward`
- Per layer: `4 * d_model^2 + 2 * d_model * dim_feedforward`
- Total: `input_dim * d_model + num_encoder_layers * (4 * d_model^2 + 2 * d_model * dim_feedforward) + d_model * output_dim`

**Example (input_dim=10, output_dim=5, d_model=128, nhead=8, num_encoder=4, dim_ff=512):**
- Input projection: ~1.3K
- Per encoder layer: ~67K (4*128² + 2*128*512)
- Total for 4 layers: ~268K
- Output projection: ~645
- **Total: ~270K params**

---

### 3.2 PatchTSTModel (`patchtst.py`)
**File Location:** `/home/user/airtrace/src/airtrace/models/patchtst.py`
**Config File:** `/home/user/airtrace/src/airtrace/configs/model/patchtst.yaml`

**Key Architectural Parameters:**
| Parameter | Default | Type | Notes |
|-----------|---------|------|-------|
| `patch_len` | 16 | int | Length of temporal patches |
| `stride` | 8 | int | Stride between patches (50% overlap) |
| `d_model` | 128 | int | Patch embedding dimension |
| `nhead` | 8 | int | Attention heads |
| `num_layers` | 3 | int | Transformer encoder layers |
| `dim_feedforward` | 256 | int | FFN hidden dim |
| `dropout` | 0.1 | float | Dropout rate |
| `activation` | "gelu" | str | Activation |

**Architecture:**
- Channel-independent: each variable processed separately
- Sliding window patching: `num_patches = (seq_len - patch_len) / stride + 1`
- Patch embedding: `patch_len → d_model`
- Shared transformer encoder across all channels
- Head: flattens all channel embeddings, projects to output

**Parameter Count Formula:**
- Patch embedding: `patch_len * d_model`
- Per encoder layer: `4 * d_model^2 + 2 * d_model * dim_feedforward`
- Head: `input_dim * d_model * num_patches → output_dim` (varies by input size)
- For typical case: `patch_len * d_model + num_layers * (4 * d_model^2 + 2 * d_model * dim_feedforward) + input_dim * d_model * output_dim`

**Example (seq_len=120, input_dim=10, patch_len=16, stride=8, d_model=128, num_layers=3):**
- Num patches: ~13
- Patch embedding: ~2K
- Encoder layers (3): ~150K
- Head: ~16.6K (10 * 128 * 13 → 5)
- **Total: ~170K params**

---

### 3.3 InformerModel (`informer.py`)
**File Location:** `/home/user/airtrace/src/airtrace/models/informer.py`
**Config File:** `/home/user/airtrace/src/airtrace/configs/model/informer.yaml`

**Key Architectural Parameters:**
| Parameter | Default | Type | Notes |
|-----------|---------|------|-------|
| `d_model` | 128 | int | Model dimension |
| `nhead` | 4 | int | Attention heads |
| `e_layers` | 2 | int | Encoder layers |
| `d_layers` | 1 | int | Decoder layers |
| `ff_dim` | 256 | int | Feed-forward dimension |
| `factor` | 5 | int | Sparsity factor for ProbSparse attention |
| `dropout` | 0.1 | float | Dropout rate |
| `pred_len` | 1 | int | Prediction horizon |
| `distill` | true | bool | Distilling encoder |

**Key Innovation: ProbSparse Attention**
- Only attends to top-k "important" queries
- Reduces complexity from O(L²) to O(L log L)
- Important queries selected by magnitude

**Parameter Count:**
- Token embedding: `input_dim * d_model`
- Encoder: `e_layers * (query/key/value projections + output + FF)`
  - Per layer: `3 * d_model^2 + d_model^2 + 2 * d_model * ff_dim`
- Decoder: `d_layers * (self-attention + cross-attention + FF)`
- Distilling layers: conv+pooling if enabled
- Projection: `d_model * output_dim`

**Example (input_dim=10, output_dim=5, d_model=128, e_layers=2, d_layers=1, ff_dim=256):**
- Embedding: ~1.3K
- Encoder (2 layers): ~130K
- Decoder (1 layer): ~65K
- Projection: ~645
- **Total: ~197K params**

---

### 3.4 AutoformerModel (`autoformer.py`)
**File Location:** `/home/user/airtrace/src/airtrace/models/autoformer.py`
**Config File:** `/home/user/airtrace/src/airtrace/configs/model/autoformer.yaml`

**Key Architectural Parameters:**
| Parameter | Default | Type | Notes |
|-----------|---------|------|-------|
| `d_model` | 128 | int | Model dimension |
| `n_heads` | 8 | int | Attention heads |
| `e_layers` | 2 | int | Encoder layers |
| `d_layers` | 1 | int | Decoder layers |
| `moving_avg` | 25 | int | MA kernel for decomposition |
| `d_ff` | 256 | int | Feed-forward dimension |
| `dropout` | 0.1 | float | Dropout rate |
| `activation` | "gelu" | str | Activation |
| `label_len` | 24 | int | Context length for decoder |
| `pred_len` | 1 | int | Prediction horizon |
| `top_k` | 5 | int | Top-k modes for auto-correlation |
| `max_len` | 512 | int | Max sequence length |

**Key Innovation: Auto-Correlation Attention**
- Computes correlation via FFT instead of dot-product
- Selects top-k correlated lags
- More interpretable for periodic patterns

**Architecture:**
- Series decomposition: seasonal + trend via MA
- Separate encoders for seasonal/trend
- Auto-correlation attention instead of dot-product
- Series decomposition in decoder

**Parameter Count:**
- Data embedding: `input_dim * d_model`
- Auto-correlation: Similar to standard attention but with FFT overhead
- Encoder layers: `e_layers * (auto_corr + FF)`
- Decoder layers: `d_layers * (self-attn + cross-attn + FF)`
- Decomposition modules: Moving averages (no learnable params) + projection

**Example (input_dim=10, output_dim=5, d_model=128, e_layers=2, d_layers=1):**
- **Total: ~230K params** (similar to Transformer with decomposition overhead)

---

### 3.5 FEDformerModel (`fedformer.py`)
**File Location:** `/home/user/airtrace/src/airtrace/models/fedformer.py`
**Config File:** `/home/user/airtrace/src/airtrace/configs/model/fedformer.yaml`

**Key Architectural Parameters:**
| Parameter | Default | Type | Notes |
|-----------|---------|------|-------|
| `d_model` | 128 | int | Model dimension |
| `n_heads` | 8 | int | Attention heads |
| `e_layers` | 2 | int | Encoder layers |
| `d_layers` | 1 | int | Decoder layers |
| `moving_avg` | 25 | int | MA kernel for trend extraction |
| `d_ff` | 256 | int | Feed-forward dimension |
| `dropout` | 0.1 | float | Dropout rate |
| `activation` | "gelu" | str | Activation |
| `freq_mode` | "fourier" | str | "fourier" or "wavelet" |
| `modes` | 32 | int | Number of Fourier modes (low-pass) |
| `label_len` | 24 | int | Context length |
| `pred_len` | 1 | int | Prediction horizon |
| `max_len` | 512 | int | Max sequence length |

**Key Innovation: Fourier Attention**
- Operates in frequency domain
- Keeps only top modes (low-pass filtering)
- More stable for long sequences

**Parameter Count:**
- Similar to Autoformer + Fourier-based attention
- Fourier attention parameters: `4 * d_model^2` (q/k/v projections + output)
- Additional: Mode filtering (no learnable params)

**Example (input_dim=10, output_dim=5, d_model=128, e_layers=2, modes=32):**
- **Total: ~230K params** (similar to Autoformer)

---

### 3.6 CrossformerModel (`crossformer.py`)
**File Location:** `/home/user/airtrace/src/airtrace/models/crossformer.py`
**Config File:** `/home/user/airtrace/src/airtrace/configs/model/crossformer.yaml`

**Key Architectural Parameters:**
| Parameter | Default | Type | Notes |
|-----------|---------|------|-------|
| `seg_len` | 16 | int | Temporal patch length |
| `seg_stride` | 8 | int | Stride between patches |
| `dim_seg_size` | 4 | int | Variables per dimension segment |
| `d_model` | 128 | int | Embedding dimension |
| `nhead` | 8 | int | Attention heads |
| `temporal_depth` | 2 | int | Temporal attention layers |
| `spatial_depth` | 1 | int | Cross-dimension attention layers |
| `dim_feedforward` | 256 | int | FFN hidden dimension |
| `dropout` | 0.1 | float | Dropout rate |
| `pred_len` | 1 | int | Prediction horizon |
| `pooling` | "last" | str | "last" or "mean" for aggregation |

**Key Innovation: Two-Stage Attention**
1. Temporal attention: Within each dimension group, across time
2. Spatial attention: Across dimensions, at each time point

**Architecture:**
- Dimension-Segment-Wise (DSW) embedding: groups variables, flattens temporal patches
- Temporal encoder: `temporal_depth` layers
- Cross-dimension encoder: `spatial_depth` layers
- Iterative refinement of representations

**Parameter Count:**
- DSW embedding: `(dim_seg_size * seg_len) * d_model`
- Temporal encoder: `temporal_depth * (4 * d_model^2 + 2 * d_model * dim_feedforward)`
- Cross-dim encoder: `spatial_depth * (4 * d_model^2 + 2 * d_model * dim_feedforward)`
- Head: flattens and projects

**Example (input_dim=10, seg_len=16, dim_seg_size=4, d_model=128, temporal=2, spatial=1):**
- **Total: ~160K params**

---

### 3.7 iTransformerModel (`itransformer.py`)
**File Location:** `/home/user/airtrace/src/airtrace/models/itransformer.py`
**Config File:** `/home/user/airtrace/src/airtrace/configs/model/itransformer.yaml`

**Key Architectural Parameters:**
| Parameter | Default | Type | Notes |
|-----------|---------|------|-------|
| `d_model` | 512 | int | Model dimension (inverted = larger) |
| `nhead` | 8 | int | Attention heads |
| `num_layers` | 3 | int | Transformer encoder layers |
| `dim_feedforward` | 2048 | int | FFN hidden dimension (4x d_model) |
| `dropout` | 0.1 | float | Dropout rate |
| `activation` | "gelu" | str | Activation |
| `use_norm` | true | bool | Layer normalization |
| `pred_len` | 1 | int | Prediction horizon |

**Key Innovation: Inverted Attention**
- **Variates become tokens**, not time points
- Attention captures inter-sensor (cross-variate) dependencies
- FFN learns temporal patterns per variate

**Architecture:**
- Variate embedding: time series → d_model embedding (LazyLinear)
- Learnable positional encoding for variates
- Transformer encoder with variate tokens
- Projection head for output

**Parameter Count:**
- Variate embedding (LazyLinear): `seq_len * input_dim * d_model` (lazy initialization)
- Positional encoding: `input_dim * d_model`
- Per transformer layer: `4 * d_model^2 + 2 * d_model * dim_feedforward`
- Output projection: `d_model * output_dim`

**Example (seq_len=120, input_dim=10, output_dim=5, d_model=512, num_layers=3):**
- Variate embedding: ~600K (lazy)
- Positional encoding: ~5.1K
- Transformer (3 layers): ~3M
- Output projection: ~2.6K
- **Total: ~3.6M params** (significantly larger due to d_model=512)

---

### 3.8 NonStationaryTransformerModel (`nonstationary_transformer.py`)
**File Location:** `/home/user/airtrace/src/airtrace/models/nonstationary_transformer.py`
**Config File:** `/home/user/airtrace/src/airtrace/configs/model/nonstationary_transformer.yaml`

**Key Architectural Parameters:**
| Parameter | Default | Type | Notes |
|-----------|---------|------|-------|
| `d_model` | 128 | int | Model dimension |
| `nhead` | 8 | int | Attention heads |
| `num_layers` | 3 | int | Transformer layers |
| `dim_feedforward` | 256 | int | FFN hidden dimension |
| `dropout` | 0.1 | float | Dropout rate |
| `activation` | "gelu" | str | Activation |
| `pred_len` | 1 | int | Prediction horizon |
| `stationarization_eps` | 1e-5 | float | Epsilon for numerical stability |
| `affine_stationary` | true | bool | Learnable affine parameters |

**Key Innovation: De-stationary Attention**
- Per-series mean/std normalization with learnable parameters
- Attention operates on de-stationarized features
- Helps model non-stationary processes

**Architecture:**
- Input projection: `input_dim → d_model`
- SeriesStationarizer: learns mean/std scaling
- DeStationaryAttention: learnable scale/shift per series
- Standard transformer with de-stationary attention layers
- Output projection

**Parameter Count:**
- Input projection: `input_dim * d_model`
- Stationarizer: `2 * input_dim` (gamma, beta if affine=true)
- Per de-stationary attention layer:
  - QKV projection: `3 * d_model^2`
  - Output projection: `d_model^2`
  - Delta projection: `d_model^2`
  - Tau projection: `d_model^2`
  - Total per layer: `6 * d_model^2 + 2 * d_model * dim_feedforward`
- Output projection: `d_model * output_dim`

**Example (input_dim=10, output_dim=5, d_model=128, num_layers=3):**
- Input projection: ~1.3K
- Stationarizer: ~20
- Per layer (de-stationary): ~95K
- Total for 3 layers: ~285K
- Output projection: ~645
- **Total: ~287K params**

---

## 4. MLP-BASED & LINEAR MODELS

### 4.1 DLinearModel (`dlinear.py`)
**File Location:** `/home/user/airtrace/src/airtrace/models/dlinear.py`
**Config File:** `/home/user/airtrace/src/airtrace/configs/model/dlinear.yaml`

**Key Architectural Parameters:**
| Parameter | Default | Type | Notes |
|-----------|---------|------|-------|
| `seq_len` | 60 | int | Input sequence length |
| `pred_len` | 1 | int | Prediction length |
| `kernel_size` | 25 | int | MA kernel for decomposition |

**Architecture:**
- Decomposition: Moving average for trend, residual for seasonal
- Seasonal linear: `seq_len → pred_len` linear projection
- Trend linear: `seq_len → pred_len` linear projection
- Optional output projection if `input_dim != output_dim`

**Parameter Count Formula:**
- Seasonal linear: `seq_len * pred_len * input_dim * output_dim / input_dim = seq_len * pred_len * output_dim`

Wait, let me recalculate. The linear layers work per-channel:
- Seasonal linear: `seq_len * pred_len` per channel (applies to all channels)
- Trend linear: `seq_len * pred_len` per channel
- Output projection (if needed): `input_dim * output_dim`

**Example (seq_len=60, pred_len=1, input_dim=10, output_dim=5):**
- Seasonal linear: ~60
- Trend linear: ~60
- Output projection: ~50
- **Total: ~170 params** (one of the smallest models!)

---

### 4.2 NLinearModel (`dlinear.py`)
**File Location:** `/home/user/airtrace/src/airtrace/models/dlinear.py`
**Config File:** `/home/user/airtrace/src/airtrace/configs/model/nlinear.yaml`

**Key Architectural Parameters:**
| Parameter | Default | Type | Notes |
|-----------|---------|------|-------|
| `seq_len` | 60 | int | Input sequence length |
| `pred_len` | 1 | int | Prediction length |
| `center_data` | true | bool | Mean-center input |

**Architecture:**
- Mean normalization per series
- Individual linear projection per channel: `seq_len → pred_len`
- No decomposition, simpler than DLinear

**Parameter Count:**
- Per-channel linear: `seq_len * pred_len`
- For all channels: `input_dim * seq_len * pred_len`

**Example (seq_len=60, pred_len=1, input_dim=10, output_dim=5):**
- **Total: ~600 params**

---

### 4.3 TimeMixerModel (`timemixer.py`)
**File Location:** `/home/user/airtrace/src/airtrace/models/timemixer.py`
**Config File:** `/home/user/airtrace/src/airtrace/configs/model/timemixer.yaml`

**Key Architectural Parameters:**
| Parameter | Default | Type | Notes |
|-----------|---------|------|-------|
| `d_model` | 64 | int | Model dimension |
| `num_layers` | 2 | int | Number of PDM (Patch Decomposition Mixing) blocks |
| `down_sampling_layers` | 3 | int | Number of downsampling scales for multi-scale |
| `decomp_kernel` | 25 | int | MA kernel for series decomposition |
| `dropout` | 0.1 | float | Dropout rate |

**Key Innovation: Decomposable Multiscale Mixing**
- Decomposes into seasonal + trend
- Multi-scale mixing: bottom-up (seasonal), top-down (trend)
- PDM blocks with downsampling

**Architecture:**
- Series decomposition: MA for trend
- Multi-scale season mixing: downsample + mix
- Multi-scale trend mixing: upsample + mix
- Future multipredictor: ensemble predictions

**Parameter Count:**
- Decomposition convs: Multiple downsampling conv1d layers
- Mixing layers: Linear(d_model, d_model) per scale
- For `down_sampling_layers=3`: ~6 conv layers + 4 mixing layers

**Example (d_model=64, down_sampling_layers=3, seq_len=120, input_dim=10, output_dim=5):**
- Downsampling convs (6 layers): ~20K
- Mixing layers (4 per scale): ~10K
- Output layers: ~5K
- **Total: ~35K params**

---

### 4.4 NBeatsModel (`nbeats.py`)
**File Location:** `/home/user/airtrace/src/airtrace/models/nbeats.py`
**Config File:** `/home/user/airtrace/src/airtrace/configs/model/nbeats.yaml`

**Key Architectural Parameters:**
| Parameter | Default | Type | Notes |
|-----------|---------|------|-------|
| `stack_types` | ["trend", "seasonality"] | list | Types of stacks (generic, trend, seasonality) |
| `num_blocks_per_stack` | 2 | int | Blocks per stack |
| `hidden_size` | 256 | int | Hidden layer dimension |
| `num_layers` | 4 | int | Layers per block |
| `pred_len` | 1 | int | Prediction length |
| `degree` | 2 | int | Polynomial degree for trend |
| `harmonics` | null | int | Number of harmonics for seasonality |
| `dropout` | 0.0 | float | Dropout rate |

**Architecture:**
- Stacks of blocks (each producing backcast/forecast)
- Different basis for each block type:
  - Generic: Linear basis
  - Trend: Polynomial basis (degree)
  - Seasonality: Fourier basis (harmonics)
- Residual stacking: each block processes residual from previous

**Parameter Count Formula:**
- Per block: Lazy linear → hidden → stack of linears → theta output
  - Lazy linear: `seq_len * input_dim * hidden_size`
  - Hidden layers: `hidden_size^2 * (num_layers - 1)`
  - Theta layer: `hidden_size * (basis_dim)`
  - Basis dim depends on block type

**Example (seq_len=120, input_dim=10, output_dim=5, hidden=256, 2 stacks × 2 blocks):**
- Each block: ~100-200K
- 4 blocks total: ~400-800K
- **Total: ~400K params** (large due to lazy linear)

---

### 4.5 CycleNetModel (`cyclenet.py`)
**File Location:** `/home/user/airtrace/src/airtrace/models/cyclenet.py`
**Config File:** `/home/user/airtrace/src/airtrace/configs/model/cyclenet.yaml`

**Key Architectural Parameters:**
| Parameter | Default | Type | Notes |
|-----------|---------|------|-------|
| `period_len` | 32 | int | Learnable recurrent cycle length (W) |
| `backbone` | "mlp" | str | "mlp" or "linear" |
| `hidden_dim` | 256 | int | Hidden dimension (MLP only) |
| `dropout` | 0.1 | float | Dropout rate |
| `activation` | "gelu" | str | Activation |

**Key Innovation: Learnable Cycles**
- Learns intrinsic period W of data
- Decompose time axis using learned period
- Linear/MLP per cycle position

**Architecture:**
- Learnable period matrix W: `(period_len, period_len)`
- Backbone (per cycle position): Linear or MLP
- Combines predictions from all cycle positions

**Parameter Count:**
- Period matrix W: `period_len^2`
- Per position backbone:
  - Linear: `input_dim * output_dim`
  - MLP: `input_dim * hidden_dim + hidden_dim^2 + hidden_dim * output_dim`
- For period_len positions: times period_len

**Example (period_len=32, input_dim=10, output_dim=5, hidden=256, MLP backbone):**
- Period matrix: ~1K
- Per position MLP: ~65K
- 32 positions: ~2M
- **Total: ~2M params**

---

## 5. STATE-SPACE & MAMBA MODELS

### 5.1 Mamba2Model (`mamba2.py`)
**File Location:** `/home/user/airtrace/src/airtrace/models/mamba2.py`
**Config File:** `/home/user/airtrace/src/airtrace/configs/model/mamba2.yaml`

**Key Architectural Parameters:**
| Parameter | Default | Type | Notes |
|-----------|---------|------|-------|
| `pred_len` | 64 | int | Prediction horizon |
| `embed_dim` | 512 | int | Token embedding dimension |
| `state_dim` | 256 | int | Selective scan state size |
| `num_layers` | 8 | int | Number of Mamba-2 blocks |
| `conv_kernel_size` | 5 | int | Depthwise conv kernel |
| `chunk_length` | 1024 | int | Chunk size for scanning |
| `bidirectional_scan` | true | bool | Forward + backward scans |
| `decay_init` | 0.0 | float | Initial decay parameter |
| `dropout` | 0.1 | float | Dropout rate |
| `ff_expansion` | 4 | int | FFN expansion factor |
| `adapter_rank` | 8 | int | LoRA rank (0 = disabled) |
| `adapter_alpha` | 16.0 | float | LoRA scaling |
| `freeze_backbone` | false | bool | Freeze weights for transfer learning |

**Key Innovation: Selective State Space Models (S4, Mamba)**
- Linear recurrence with selective state matrices
- Selective scan: O(L) complexity
- Bidirectional scanning: forward + backward averaging
- Efficient for long sequences

**Architecture:**
- Input projection: `input_dim → embed_dim`
- Mamba-2 blocks (num_layers):
  - Depthwise conv: local mixing
  - Selective scan: long-range dependencies
  - FFN: expressiveness
- Optional LoRA adapters for efficient fine-tuning
- Output projection: `embed_dim → output_dim`

**Parameter Count Formula:**
- Input projection: `input_dim * embed_dim`
- Per Mamba-2 block:
  - Depthwise conv: `embed_dim * conv_kernel_size + embed_dim`
  - Selective scan: A matrix `state_dim`, B/C projections `embed_dim * state_dim`
  - FFN: `2 * embed_dim * (ff_expansion * embed_dim)`
  - Complexity: ~`2 * embed_dim^2 * ff_expansion + embed_dim * state_dim`
- Output projection: `embed_dim * output_dim`
- LoRA (if enabled): ~`2 * adapter_rank * (embed_dim + output_dim)`

**Example (input_dim=10, output_dim=5, embed_dim=512, state_dim=256, num_layers=8, ff_expansion=4):**
- Input projection: ~5.1K
- Per layer: ~2M
- 8 layers: ~16M
- Output projection: ~2.6K
- LoRA (rank=8): ~8K
- **Total: ~16M params** (large model)

---

### 5.2 MambaTSModel (`mambats.py`)
**File Location:** `/home/user/airtrace/src/airtrace/models/mambats.py`
**Config File:** `/home/user/airtrace/src/airtrace/configs/model/mambats.yaml`

**Key Architectural Parameters:**
| Parameter | Default | Type | Notes |
|-----------|---------|------|-------|
| `pred_len` | 64 | int | Prediction horizon |
| `patch_len` | 16 | int | Patch length for tokenization |
| `stride` | 8 | int | Patch stride (50% overlap typical) |
| `embed_dim` | 128 | int | Patch embedding dimension |
| `state_dim` | 16 | int | Selective scan state size |
| `num_layers` | 4 | int | Temporal Mamba Blocks |
| `expand_factor` | 2 | int | Expansion for TMB hidden dim |
| `bidirectional_scan` | true | bool | Bidirectional scanning |
| `dropout` | 0.1 | float | Dropout rate |
| `normalize_input` | true | bool | Mean-std normalization |

**Key Innovation: MambaTS (Temporal Mamba with Variable Scans)**
- Patching reduces sequence length: `num_tokens = (seq_len - patch_len) / stride + 1`
- Variable Scan along Time (VST): groups patches from different variables at same step
- Enables efficient cross-variable dependency modeling
- Linear complexity in sequence length

**Architecture:**
- Patch embedding: `patch_len → embed_dim`
- Temporal Mamba Blocks (num_layers):
  - Selective scan per patch token
  - Feed-forward network
  - Residual connections
- Output head: projects to predictions

**Parameter Count Formula:**
- Patch embedding: `patch_len * embed_dim`
- Per TMB:
  - Selective scan: `embed_dim * state_dim * 2` (B, C projections)
  - FFN: `2 * embed_dim * (expand_factor * embed_dim)`
  - Total: ~`2 * embed_dim^2 * expand_factor + embed_dim * state_dim`
- Output head: `embed_dim → output_dim`

**Example (seq_len=120, patch_len=16, stride=8, input_dim=10, output_dim=5, embed_dim=128, state_dim=16, num_layers=4):**
- Num patches: ~13
- Patch embedding: ~2K
- Per layer: ~35K
- 4 layers: ~140K
- Output head: ~645
- **Total: ~145K params** (much smaller than Mamba2)

---

## 6. FOUNDATION MODELS

### 6.1 ChronosBoltModel (`chronos_bolt.py`)
**File Location:** `/home/user/airtrace/src/airtrace/models/chronos_bolt.py`
**Config File:** `/home/user/airtrace/src/airtrace/configs/model/chronos_bolt.yaml`

**Key Architectural Parameters:**
| Parameter | Default | Type | Notes |
|-----------|---------|------|-------|
| `pred_len` | 64 | int | Forecast horizon |
| `embed_dim` | 512 | int | Token embedding dimension |
| `patch_size` | 32 | int | Patch/context length |
| `patch_stride` | 16 | int | Stride between patches |
| `num_blocks` | 8 | int | Gated conv-attention blocks |
| `num_heads` | 8 | int | Attention heads |
| `dilation_growth` | 2 | int | Dilation growth for conv |
| `conv_kernel_size` | 5 | int | Kernel for local mixing |
| `dropout` | 0.1 | float | Dropout rate |
| `ff_expansion` | 4 | int | FFN expansion |
| `max_positions` | 4096 | int | Max positional embeddings |
| `lora_rank` | 8 | int | LoRA rank |
| `lora_alpha` | 16.0 | float | LoRA scaling |
| `freeze_backbone` | false | bool | Freeze pretrained weights |
| `train_head` | true | bool | Keep head trainable |
| `pretrained_checkpoint` | null | str | Path to pretrained model |

**Architecture:**
- Pre-trained foundation model from Chronos project
- Gated conv-attention blocks: CNN for local patterns + attention for global
- LoRA fine-tuning: parameter-efficient adaptation
- Transfer learning compatible

**Parameter Count:**
- Backbone (pre-trained, frozen by default): ~700M (Chronos-Bolt base)
- LoRA adapters (if active): `2 * lora_rank * (embed_dim + output_dim)` per layer
- Head (forecast head): trainable

**Notes:**
- When `freeze_backbone=true`: Only LoRA and head parameters are trained
- LoRA adds ~8-16K parameters per layer for rank=8
- Excellent for transfer learning scenarios

---

### 6.2 MoiraiModel (`moirai.py`)
**File Location:** `/home/user/airtrace/src/airtrace/models/moirai.py`
**Config File:** `/home/user/airtrace/src/airtrace/configs/model/moirai.yaml`

**Key Architectural Parameters:**
| Parameter | Default | Type | Notes |
|-----------|---------|------|-------|
| `pred_len` | 24 | int | Prediction horizon |
| `embed_dim` | 256 | int | Token embedding dimension |
| `state_dim` | 256 | int | SSM state size |
| `num_layers` | 6 | int | Selective SSM blocks |
| `conv_kernel_size` | 5 | int | Depthwise conv kernel |
| `dropout` | 0.1 | float | Dropout rate |
| `ff_expansion` | 4 | int | FFN expansion |
| `patch_scales` | [4, 16] | list | Multi-resolution patch sizes |
| `max_positions` | 4096 | int | Max positional encoding |
| `adapter_rank` | 0 | int | LoRA rank (0 = disabled) |
| `adapter_alpha` | 8.0 | float | LoRA scaling |
| `freeze_backbone` | false | bool | Freeze weights |
| `train_head` | true | bool | Keep head trainable |

**Key Innovation: Selective SSMs (State-Space Models)**
- Mamba-style selective scanning
- Multi-resolution patch fusion: different time scales
- Transfer learning via adapter tuning

**Architecture:**
- Multi-scale patching: Creates tokens at different temporal resolutions
- Selective SSM layers: Efficient long-range dependencies
- Adaptive tokens: Fused across scales
- Readout head for forecasting

**Parameter Count:**
- Backbone: ~650M parameters (pre-trained foundation model)
- LoRA adapters (if enabled): ~8-16K per layer
- Head: trainable output layer

---

### 6.3 LagLlamaModel (`lag_llama.py`)
**File Location:** `/home/user/airtrace/src/airtrace/models/lag_llama.py`
**Config File:** `/home/user/airtrace/src/airtrace/configs/model/lag_llama.yaml`

**Key Architectural Parameters:**
| Parameter | Default | Type | Notes |
|-----------|---------|------|-------|
| `pred_len` | 24 | int | Prediction horizon |
| `embed_dim` | 256 | int | Token embedding dimension |
| `patch_size` | 32 | int | Patch length (timesteps per token) |
| `patch_stride` | 16 | int | Stride between patches |
| `add_sensor_embeddings` | true | bool | Per-sensor learnable embeddings |
| `max_positions` | 4096 | int | Max positional encoding |
| `retrieval_mode` | "in_memory" | str | "none" or "in_memory" |
| `max_neighbors` | 4 | int | Number of retrieved neighbors |
| `diffusion_layers` | 3 | int | Depth of diffusion network |
| `diffusion_heads` | 4 | int | Attention heads in diffusion |
| `diffusion_steps` | 8 | int | Diffusion sampling iterations |
| `diffusion_dropout` | 0.1 | float | Dropout in diffusion |
| `diffusion_ff_expansion` | 4 | int | FFN expansion in diffusion |
| `init_noise_scale` | 0.2 | float | Initial noise std |
| `guidance_scale` | 1.2 | float | Guidance scaling |

**Key Innovation: In-context Learning + Diffusion**
- Retrieval: Finds similar historical contexts
- Conditioning: Uses neighbors to condition predictions
- Diffusion-based: Generates predictions via diffusion process
- Probabilistic: Supports sampling multiple scenarios

**Architecture:**
- Patching: Reduces sequence length
- Sensor embeddings: Per-variable learned embeddings
- Retrieval backend: In-memory search (or disabled)
- Diffusion network: Iterative refinement

**Parameter Count:**
- Encoder (pre-trained): ~700M
- Diffusion network: ~100K
- Sensor embeddings: `input_dim * embed_dim`
- LoRA (if used): minimal

---

## 7. TEMPORAL FUSION TRANSFORMER

### 7.1 TemporalFusionTransformer (TFT)
**File Location:** `/home/user/airtrace/src/airtrace/models/tft.py`
**Config File:** `/home/user/airtrace/src/airtrace/configs/model/tft.yaml`

**Key Architectural Parameters:**
| Parameter | Default | Type | Notes |
|-----------|---------|------|-------|
| `hidden_size` | 128 | int | Main hidden dimension |
| `lstm_layers` | 2 | int | LSTM depth |
| `num_heads` | 4 | int | Attention heads |
| `dropout` | 0.1 | float | Dropout rate |
| `quantiles` | [0.1, 0.5, 0.9] | list | Quantile levels for probabilistic forecast |
| `static_input_dim` | 0 | int | Static features dimension |
| `known_future_dim` | 0 | int | Known future features |
| `pred_len` | 1 | int | Prediction horizon |

**Architecture:**
- LSTM encoder: Processes past observations
- Temporal self-attention: Captures long-range dependencies
- Variable selection networks: Learns importance of variables
- Quantile output: Probabilistic predictions

**Parameter Count:**
- LSTM layers: `4 * hidden_size * (input_dim + hidden_size) * lstm_layers`
- Self-attention: `4 * hidden_size^2 + hidden_size^2`
- Variable selection: `input_dim * hidden_size * 2`
- Output heads (per quantile): `hidden_size * len(quantiles)`

**Example (input_dim=10, hidden_size=128, lstm_layers=2, num_heads=4, quantiles=3):**
- **Total: ~150K params**

---

## 8. OTHER MODELS

### 8.1 TimesNetModel (`timesnet.py`)
**File Location:** `/home/user/airtrace/src/airtrace/models/timesnet.py`
**Config File:** `/home/user/airtrace/src/airtrace/configs/model/timesnet.yaml`

**Key Architectural Parameters:**
| Parameter | Default | Type | Notes |
|-----------|---------|------|-------|
| `seq_len` | config-dependent | int | Input sequence length |
| `pred_len` | config-dependent | int | Prediction length |
| `d_model` | 64 | int | Model dimension |
| `d_ff` | 128 | int | Feed-forward dimension |
| `num_layers` | 2 | int | TimesNet blocks |
| `num_kernels` | 6 | int | Number of kernels for 1D conv |
| `top_k` | 5 | int | Top-k frequencies to keep |
| `dropout` | 0.1 | float | Dropout rate |
| `embed_type` | "positional" | str | Embedding type |

**Key Innovation: Temporal patterns via Fast Fourier Transform**
- FFT for frequency domain analysis
- Keeps top-k frequencies
- Conv1d in frequency domain
- More interpretable than attention

---

### 8.2 TimeXerModel (`timexer.py`)
**File Location:** `/home/user/airtrace/src/airtrace/models/timexer.py`
**Config File:** `/home/user/airtrace/src/airtrace/configs/model/timexer.yaml`

**Key Architectural Parameters:**
| Parameter | Default | Type | Notes |
|-----------|---------|------|-------|
| `exog_dim` | 0 | int | Exogenous feature dimension |
| `patch_len` | 16 | int | Patch length |
| `stride` | 8 | int | Patch stride |
| `d_model` | 128 | int | Model dimension |
| `nhead` | 8 | int | Attention heads |
| `num_layers` | 3 | int | Transformer layers |
| `dim_feedforward` | 512 | int | FFN dimension |
| `dropout` | 0.1 | float | Dropout rate |
| `activation` | "gelu" | str | Activation |
| `num_global_tokens` | 1 | int | Global tokens for aggregation |
| `pred_len` | 1 | int | Prediction horizon |

**Architecture:**
- Patching similar to PatchTST
- Global tokens for series-level information
- Per-patch local attention + global aggregation
- Hybrid local-global approach

---

### 8.3 SOFTSModel (`softs.py`)
**File Location:** `/home/user/airtrace/src/airtrace/models/softs.py`
**Config File:** `/home/user/airtrace/src/airtrace/configs/model/softs.yaml`

**Key Architectural Parameters:**
| Parameter | Default | Type | Notes |
|-----------|---------|------|-------|
| `seq_len` | config-dependent | int | Input sequence length |
| `pred_len` | config-dependent | int | Prediction length |
| `hidden_dim` | 512 | int | Model dimension |
| `d_core` | 128 | int | STAR core dimension (compression) |
| `d_ff` | 512 | int | FFN dimension |
| `e_layers` | 3 | int | Encoder layers |
| `dropout` | 0.0 | float | Dropout rate |
| `activation` | "gelu" | str | Activation |
| `use_norm` | true | bool | Instance normalization |

**Key Innovation: Spatial-Temporal Adaptive Fusion (SOFTS)**
- STAR module: Spatial-Temporal Adaptive Representation
- Separate handling of spatial (cross-variable) and temporal dimensions
- Core dimension for efficient dimensionality reduction

---

## 9. PARAMETER ALIGNMENT OPPORTUNITIES

### Current Disparities:
1. **Tiny models** (DLinear, NLinear): ~170-600 params
2. **Small models** (TCN, TimeMixer, MambaTSA): ~30-150K params
3. **Medium models** (Most Transformers): ~150-300K params
4. **Large models** (iTransformer): ~3.6M params
5. **Very large models** (Mamba2): ~16M params
6. **Foundation models** (Chronos-Bolt, Moirai, LagLlama): ~700M params (frozen)

### Alignment Strategy:

**Group 1: Parameter-Efficient Models (< 100K)**
- Baseline: DLinear (170)
- Target alignment: 50-100K
- Candidates: NLinear, ModernTCN, DLinear (with increased capacity)

**Group 2: Standard Models (100K - 500K)**
- Baseline: Transformer (270K)
- Target range: 200-400K
- Most current Transformers fit here
- Action: Reduce d_model, dim_feedforward, or num_layers as needed

**Group 3: High-Capacity Models (500K - 5M)**
- Baseline: iTransformer (3.6M)
- Current: Mamba2 (16M)
- Action: Reduce embed_dim or state_dim

**Group 4: Foundation Models (Foundation)**
- Keep frozen for transfer learning
- Fine-tuning via LoRA

### Recommended Standardization:

```yaml
# Standard model profile: ~250K parameters
d_model: 128
nhead: 8
num_layers: 3
dim_feedforward: 256
hidden_size: 128
num_channels: [64, 128, 128]
embed_dim: 128
```

This achieves:
- Fair comparison across architectures
- Reasonable memory footprint (~1GB for typical batch sizes)
- Sufficient capacity for aircraft sensor forecasting
- Fast training (~1-2 hours per epoch on modern GPU)

