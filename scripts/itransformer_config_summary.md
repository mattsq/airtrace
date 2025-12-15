# iTransformer Configuration to Match Informer Parameters

## Summary

Matched an iTransformer configuration to your Informer model with **99.95%** parameter parity.

### Your Informer Configuration
```yaml
model:
  name: informer
  params:
    d_model: 96
    nhead: 4
    e_layers: 2
    d_layers: 1
    ff_dim: 192
    factor: 5
    dropout: 0.1
    pred_len: 1
    distill: true
```
**Total Parameters: 291,562**

### Matched iTransformer Configuration
```yaml
model:
  name: itransformer
  params:
    d_model: 128
    nhead: 4
    num_layers: 2
    dim_feedforward: 296
    dropout: 0.1
    pred_len: 1
```
**Total Parameters: 291,409** (difference: -153, or 99.95%)

## Parameter Breakdown

### Informer Architecture (Encoder-Decoder)
- **Token Embedding**: Projects input features to d_model=96
- **Positional Encoding**: Sinusoidal, no parameters
- **Encoder**: 2 layers with ProbSparse attention
  - Each layer: Multi-head attention (factor=5 for sparsity) + FFN (192)
  - Distillation convs between layers (when distill=True)
- **Decoder**: 1 layer with self-attention + cross-attention + FFN
- **Projection**: Linear layer to output_dim

### iTransformer Architecture (Encoder-Only)
- **Variate Embedding**: Projects each sensor's time series to d_model=128
- **Positional Encoding**: Learnable, per-variate embeddings
- **Transformer Encoder**: 2 layers with standard multi-head attention
  - Each layer: Multi-head attention + FFN (296)
  - No decoder needed (direct projection)
- **Projection**: Linear layer from embeddings to output horizon

## Key Architectural Differences

### 1. Token Representation
- **Informer**: Time points are tokens → attention across time
- **iTransformer**: Variates (sensors) are tokens → attention across sensors

### 2. Model Complexity
- **Informer**: Encoder-decoder with sparse attention and distillation
- **iTransformer**: Encoder-only with standard attention

### 3. Parameter Distribution
To match parameters with simpler architecture, iTransformer uses:
- Larger d_model (128 vs 96) to compensate for no decoder
- Larger FFN dimension (296 vs 192) to increase capacity
- Learnable positional embeddings (adds params vs sinusoidal)

### 4. Computational Trade-offs
- **Informer**: Sparse attention reduces compute, but adds complexity
- **iTransformer**: Dense attention on fewer tokens (sensors not timesteps)

## Why These Parameters Work

The iTransformer achieves similar parameter count through:

1. **No Decoder**: Saves ~1/3 of parameters (no decoder self-attn, cross-attn, FFN)
2. **Larger d_model**: 128 vs 96 (+33%) compensates for simpler architecture
3. **Wider FFN**: 296 vs 192 (+54%) increases model capacity
4. **Learnable Position**: Adds parameters but improves variate-wise modeling

## Recommendations for Use

### Choose Informer if:
- You need explicit encoder-decoder structure
- Long sequences benefit from sparse attention
- You want interpretable attention patterns on time points

### Choose iTransformer if:
- Your data has strong multivariate correlations (sensor dependencies)
- You want to model cross-sensor relationships explicitly
- You prefer simpler, standard transformer architecture
- Your sequence length is moderate (sparse attention not critical)

## Alternative Configurations

If you want to explore different capacity trade-offs:

### Slightly Lower Parameters (98.5%)
```yaml
model:
  name: itransformer
  params:
    d_model: 128
    nhead: 4
    num_layers: 2
    dim_feedforward: 288
    dropout: 0.1
    pred_len: 1
```
**Parameters: 287,297** (-4,265 from Informer)

### Slightly Higher Parameters (101.4%)
```yaml
model:
  name: itransformer
  params:
    d_model: 128
    nhead: 4
    num_layers: 2
    dim_feedforward: 304
    dropout: 0.1
    pred_len: 1
```
**Parameters: 295,521** (+3,959 from Informer)

### More Layers, Smaller Model
```yaml
model:
  name: itransformer
  params:
    d_model: 112
    nhead: 4
    num_layers: 3
    dim_feedforward: 224
    dropout: 0.1
    pred_len: 1
```
**Parameters: 309,905** (+18,343 from Informer, +6.3%)

## Next Steps

1. Create a config file: `configs/model/itransformer_matched.yaml`
2. Run experiments comparing Informer vs iTransformer with same parameter budget
3. Evaluate on your aircraft sensor data to see which architecture better captures:
   - Temporal patterns (Informer strength)
   - Multivariate correlations (iTransformer strength)

## Notes on Aircraft Sensor Data

For AirTrace's aircraft sensor data, iTransformer may have advantages:
- **Strong physical correlations**: Thrust, fuel flow, Mach number, altitude are highly correlated
- **Moderate sequence length**: Input windows likely don't need sparse attention
- **Multivariate dependencies**: Engine parameters affect each other in complex ways

However, Informer may excel if:
- **Long input sequences**: Sparse attention helps with very long histories
- **Temporal patterns dominate**: If autocorrelation within sensors is more important than cross-sensor correlation
