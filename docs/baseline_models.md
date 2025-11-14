# Baseline Models

Baseline models provide simple, interpretable benchmarks to compare sophisticated models against. A good deep learning model should significantly outperform these baselines.

## Available Baselines

### 1. Persistence Model (`persistence`)

**Description:** Predicts that the next value will be the same as the last observed value.

**Formula:** `ŷ[t+1] = y[t]`

**When to use:**
- Random walk processes
- Highly autocorrelated time series
- As a "sanity check" - most models should beat this

**Config:**
```yaml
model:
  name: persistence
  params: {}
```

**Typical performance:** Often surprisingly strong for smooth, slowly-changing aircraft sensor data.

---

### 2. Moving Average Model (`moving_average`)

**Description:** Predicts the average of the last k values.

**Formula:** `ŷ[t+1] = mean(y[t-k+1:t+1])`

**When to use:**
- Noisy measurements where averaging helps
- Stationary processes
- Comparing against smoothing techniques

**Config:**
```yaml
model:
  name: moving_average
  params:
    window_size: 5  # or null to use all available values
```

**Typical performance:** Better than persistence for noisy data, worse for trending data.

---

### 3. Linear Trend Model (`linear_trend`)

**Description:** Fits a linear trend to the input window and extrapolates one step ahead.

**Formula:**
```
y = a + b*t  (fitted via least squares)
ŷ[t+1] = a + b*(t+1)
```

**When to use:**
- Trending processes (ascent, descent phases)
- Non-stationary data with linear dynamics
- Comparing against autoregressive models

**Config:**
```yaml
model:
  name: linear_trend
  params: {}
```

**Typical performance:** Strong for linear trends (climb/descent), poor for cruise.

---

### 4. Mean Model (`mean`)

**Description:** Always predicts the historical mean of the input sequence.

**Formula:** `ŷ[t+1] = mean(y[0:t+1])`

**When to use:**
- Highly stationary processes
- Zero-centered, normalized data
- Measuring if a model learns temporal patterns at all

**Config:**
```yaml
model:
  name: mean
  params: {}
```

**Typical performance:** Good for cruise (stable values), poor for transient phases.

---

### 5. Zero Model (`zero`)

**Description:** Always predicts zero for all outputs.

**Formula:** `ŷ[t+1] = 0`

**When to use:**
- Differenced or normalized data (where mean ≈ 0)
- Anomaly detection baselines (assume normal = zero deviation)
- Measuring if the model learns anything at all

**Config:**
```yaml
model:
  name: zero
  params: {}
```

**Typical performance:** Baseline floor - any trained model should beat this.

---

## Using Baselines in Experiments

### Quick Comparison

To compare a sophisticated model against baselines:

```bash
# Run baseline
airtrace train exp=my_exp model=persistence

# Run your model
airtrace train exp=my_exp model=gru_ar

# Compare results
airtrace compare exp_001_persistence exp_001_gru_ar
```

### Batch Baseline Evaluation

Create an experiment config that tests all baselines:

```yaml
# configs/exp/baseline_sweep.yaml
defaults:
  - override /data: qantas_737
  - override /transforms: zscore_diff
  - override /task: one_step
  - override /model: persistence  # Override from CLI

# Then run:
# for model in persistence zero mean moving_average linear_trend; do
#   airtrace train exp=baseline_sweep model=$model
# done
```

---

## Interpretation Guidelines

### What makes a good deep learning model?

1. **Must beat Persistence:** If you can't beat naive forecasting, your model is not learning useful patterns.

2. **Should beat Linear Trend:** For trending data, a neural model should capture non-linear dynamics.

3. **Should beat Moving Average:** On noisy data, the model should learn better smoothing than simple averaging.

4. **Large margin on Zero/Mean:** The farther above these baselines, the more your model captures temporal structure.

### Expected Performance Hierarchy (Aircraft Cruise Data)

For stable cruise flight:
```
Persistence ≈ Mean > Moving Average > Linear Trend ≈ Zero
```

For climb/descent:
```
Linear Trend > Persistence > Moving Average > Mean > Zero
```

For your deep learning model:
```
GRU/TCN/Transformer >> Persistence (ideally 30-50% better MSE)
```

---

## Implementation Notes

### Computational Efficiency

- **Zero parameters:** All baselines have ≤ `input_dim × output_dim` parameters (just a projection layer if dimensions differ)
- **No training required:** Baselines compute predictions directly, no gradient descent
- **Fast inference:** Simple numpy/torch operations, no complex forward passes

### Handling Dimension Mismatches

When `input_dim ≠ output_dim`, baselines use a simple linear projection:

```python
# If input_dim=10, output_dim=3
# Baselines compute prediction in input space, then project:
pred_input_space = baseline_logic(x)  # [B, input_dim]
pred = linear_projection(pred_input_space)  # [B, output_dim]
```

This ensures baselines work with any task configuration.

---

## Advanced Usage

### Multi-Step Baselines

Baselines currently predict one step ahead. For multi-step:

```python
# Autoregressive rollout
for step in range(horizon):
    pred = model(context)
    context = torch.cat([context[:, 1:, :], pred], dim=1)
```

### Ensemble Baselines

Combine multiple baselines:

```python
ensemble_pred = (
    0.4 * persistence_pred +
    0.3 * moving_avg_pred +
    0.3 * linear_trend_pred
)
```

Often beats individual baselines!

---

## Testing

All baselines have comprehensive tests in `tests/test_models.py`:

- Forward pass shapes
- Correctness (persistence returns last value, zero returns zeros, etc.)
- Determinism (same input → same output)
- No NaN/Inf values
- Minimal parameter count

Run tests:
```bash
pytest tests/test_models.py -k baseline
```

---

## References

Baseline models are standard practice in time series forecasting:

- **Persistence:** Hyndman & Athanasopoulos, "Forecasting: Principles and Practice"
- **Moving Average:** Classical Box-Jenkins methodology
- **Linear Trend:** Simple linear regression extrapolation

For deep learning models to be useful, they must provide **substantial** improvements over these simple heuristics.
