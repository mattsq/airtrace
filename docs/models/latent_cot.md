# Latent Chain-of-Thought (Latent CoT) Model

## Overview

The Latent Chain-of-Thought (CoT) model implements **adaptive computation time** for timeseries forecasting, allowing the model to "think" by iteratively refining its latent representations before making predictions. Unlike traditional models that use a fixed number of computational steps, Latent CoT learns both **how to refine representations** and **when to stop refining** based on the difficulty of each input.

This approach is inspired by:
- **PonderNet** (Banino et al., 2021): Learning to ponder with probabilistic halting
- **Adaptive Computation Time** (Graves, 2016): Dynamic computation budgets
- **Latent Chain-of-Thought reasoning**: Multi-step reasoning in vector space rather than text tokens

## Architecture

The model consists of three main components:

### 1. Encoder: Input → Initial Latent

Maps the input sequence to an initial latent representation:

```
x [B, T_in, D_in] → Encoder (GRU/LSTM/MLP) → h_0 [B, latent_dim]
```

**Available encoder types:**
- `gru`: GRU encoder (default) - uses final hidden state as initial latent
- `lstm`: LSTM encoder - uses final hidden state as initial latent
- `mlp`: MLP encoder - uses mean-pooled sequence as initial latent

### 2. Pondering: Iterative Latent Refinement

The core innovation: iteratively refines the latent state through learned transformations:

```
for step in 1..max_steps:
    h_t = PonderBlock(h_{t-1})  # Refine latent
    p_halt = HaltingModule(h_t)  # Decide whether to stop
    if should_halt(p_halt):
        break
```

**PonderBlock:**
- Two-layer MLP with LayerNorm and GELU activation
- Optional residual connections (default: enabled)
- Dropout for regularization

**HaltingModule:**
- Small MLP that outputs scalar halting probability p_halt ∈ (0,1)
- During **training**: Geometric sampling from Bernoulli(p_halt)
- During **inference**: Threshold-based halting (stop when cumulative p_halt ≥ threshold)

### 3. Decoder: Final Latent → Predictions

Projects the final refined latent to output predictions:

```
h_final [B, latent_dim] → Decoder (MLP) → y_pred [B, D_out]
```

## Adaptive Computation Time (ACT) Loss

To encourage efficient computation, the model is trained with an auxiliary **ponder cost** loss:

```
Total Loss = Prediction Loss + λ * ACT Loss
```

Where:
- **Prediction Loss**: Standard forecasting loss (MSE, MAE, etc.)
- **ACT Loss**: Expected number of pondering steps (ponder cost)
- **λ** (act_loss_weight): Weight balancing accuracy vs. efficiency (default: 0.01)

The ACT loss penalizes the model for using too many pondering steps, encouraging it to learn efficient reasoning patterns.

## Configuration

### Model Config (`configs/model/latent_cot.yaml`)

```yaml
model:
  name: latent_cot
  params:
    # Latent space configuration
    latent_dim: 256              # Dimension of latent reasoning space

    # Encoder configuration
    encoder_type: gru            # "gru", "lstm", or "mlp"
    encoder_hidden_dim: 512      # Hidden dimension for encoder
    encoder_num_layers: 2        # Number of encoder layers

    # Pondering configuration
    ponder_hidden_dim: 512       # Hidden dim for ponder block MLPs
    max_ponder_steps: 10         # Maximum pondering iterations
    use_residual: true           # Use residual connections in ponder blocks

    # Halting configuration
    halting_threshold: 0.99      # Cumulative halt prob threshold (inference)

    # Loss weighting
    act_loss_weight: 0.01        # Weight for ACT regularization

    # Regularization
    ponder_dropout: 0.1          # Dropout probability
```

### Task Config (`configs/task/cot_one_step.yaml`)

The specialized `cot_one_step` task handles the auxiliary ACT loss:

```yaml
task:
  name: cot_one_step
  loss: mse
  metrics: [rmse, mae, mape]
  horizon: 1

  # ACT loss scheduling
  act_loss_weight: 0.01          # Base ACT loss weight
  act_loss_schedule: constant     # "constant", "linear_decay", "cosine_decay"
  act_warmup_steps: 0            # Warmup steps before applying full ACT loss
```

## Usage Examples

### Basic Training

```bash
# Train with default configuration
airtrace train exp=exp_latent_cot

# Train with custom model parameters
airtrace train exp=exp_latent_cot \
    model.params.latent_dim=512 \
    model.params.max_ponder_steps=15
```

### Hyperparameter Tuning

```bash
# Adjust ACT loss weight (higher = fewer steps)
airtrace train exp=exp_latent_cot \
    model.params.act_loss_weight=0.02 \
    task.act_loss_weight=0.02

# Change encoder architecture
airtrace train exp=exp_latent_cot \
    model.params.encoder_type=lstm \
    model.params.encoder_num_layers=3

# Adjust halting threshold (lower = fewer steps)
airtrace train exp=exp_latent_cot \
    model.params.halting_threshold=0.9
```

### Monitoring Pondering Behavior

The model logs pondering statistics during training:

- `mean_steps`: Average number of pondering steps per batch
- `max_steps`: Maximum number of steps in batch
- `act_loss`: Current ACT regularization loss
- `act_weight`: Current ACT loss weight (if scheduled)

Use TensorBoard to monitor these metrics:

```bash
tensorboard --logdir outputs/latent_cot_zscore_one_step
```

## Design Decisions

### Why Latent Chain-of-Thought?

Traditional neural forecasters use a fixed computational budget (fixed number of layers/steps) regardless of input difficulty. Latent CoT allows the model to:

1. **Adapt to difficulty**: Use more steps for complex/ambiguous inputs, fewer for simple ones
2. **Learn efficient reasoning**: Discover which inputs require deliberation
3. **Improve interpretability**: Inspect intermediate latent states to understand reasoning process
4. **Scale test-time compute**: Trade inference cost for accuracy by adjusting halting threshold

### Stochastic vs. Deterministic Halting

**During Training** (stochastic):
- Samples halt decisions from Bernoulli(p_halt) at each step
- Introduces exploration: model experiences variable-length reasoning paths
- Helps learn robust halting policies

**During Inference** (deterministic):
- Uses threshold-based halting: stops when Σ p_halt ≥ threshold
- Provides reproducible, consistent predictions
- Can be tuned for speed/accuracy tradeoff

### ACT Loss Weight Selection

The `act_loss_weight` parameter balances prediction accuracy vs. computational efficiency:

- **Too low** (e.g., 0.001): Model may always ponder for max_steps, wasting compute
- **Too high** (e.g., 0.1): Model penalized too heavily for thinking, may underperform
- **Recommended range**: 0.005 - 0.05
- **Start with**: 0.01 (default)

### Latent Dimension Guidelines

Larger latent dimensions provide more capacity for complex reasoning patterns:

| Latent Dim | Use Case | Notes |
|------------|----------|-------|
| 64-128 | Simple datasets, few sensors | Faster, less memory |
| 256 | Default, balanced | Good starting point |
| 512-1024 | Complex multivariate, many sensors | More capacity, slower |

## Monitoring and Debugging

### Expected Pondering Behavior

A well-trained Latent CoT model should exhibit:

1. **Variable step counts**: Different inputs use different numbers of steps
2. **Learned efficiency**: Mean steps < max_steps (not always hitting limit)
3. **Stable halting**: Halting probabilities increase over pondering steps
4. **Interpretable patterns**: Similar inputs cluster in latent space

### Common Issues

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| Always max_steps | ACT loss weight too low | Increase act_loss_weight |
| Always 1 step | ACT loss weight too high | Decrease act_loss_weight |
| Unstable training | Exploding gradients | Lower learning rate, add gradient clipping |
| NaN/Inf values | Unstable halting module | Reduce learning rate, check input normalization |
| No improvement over baseline | Insufficient capacity | Increase latent_dim or ponder_hidden_dim |

### Inspecting Latent States

To analyze pondering behavior, use `return_all_steps=True`:

```python
from airtrace.models import LatentCOTModel
import torch

model = LatentCOTModel(input_dim=5, output_dim=3, latent_dim=64)
x = torch.randn(1, 32, 5)

# Get predictions at all pondering steps
output = model(x, return_all_steps=True)

all_preds = output["extras"]["all_preds"]  # List of predictions at each step
latent_states = output["extras"]["latent_states"]  # List of latent states
halt_probs = output["extras"]["halt_probs"]  # List of halting probabilities

# Analyze evolution
for step, (pred, latent, p_halt) in enumerate(zip(all_preds, latent_states, halt_probs)):
    print(f"Step {step}: pred_mean={pred.mean():.3f}, p_halt={p_halt.mean():.3f}")
```

## Advanced Usage

### Custom Pondering Schedules

Adjust ACT loss weight during training:

```yaml
task:
  act_loss_schedule: linear_decay  # Gradually reduce ponder penalty
  act_warmup_steps: 500            # Warmup before applying ACT loss
```

This allows the model to first learn good representations (without ponder penalty), then gradually learn efficiency.

### Integration with Other Components

Latent CoT works with any AirTrace transform and task:

```bash
# With different transforms
airtrace train model=latent_cot transforms=robust_scaler task=cot_one_step

# With multi-step forecasting (requires custom task implementation)
airtrace train model=latent_cot task=multi_step  # Use standard task

# With anomaly detection
airtrace train model=latent_cot task=anomaly  # Use standard task
```

**Note**: For tasks other than `cot_one_step`, the ACT loss is still computed but not added to the total loss. Use `cot_one_step` for full adaptive computation benefits.

## Comparison to Related Approaches

| Approach | Computation | Halting | Reasoning Space |
|----------|-------------|---------|-----------------|
| **Standard Models** | Fixed | N/A | Single forward pass |
| **Ensemble Methods** | Fixed (multiple models) | N/A | Multiple predictions averaged |
| **Iterative Refinement** | Fixed iterations | Fixed | Refine output iteratively |
| **PonderNet (NLP)** | Adaptive | Learned | Text token generation |
| **Latent CoT (Ours)** | Adaptive | Learned | Latent vector space |

**Key advantages over fixed-depth models:**
- Variable compute budget based on input difficulty
- Explicit reasoning traces via intermediate latents
- Tunable speed/accuracy tradeoff at inference time

**Key advantages over textual CoT:**
- No need for text generation or language model
- Direct application to non-linguistic modalities (timeseries, images, etc.)
- Continuous latent space allows smooth interpolation

## References

1. **PonderNet: Learning to Ponder**
   Banino et al., 2021
   https://arxiv.org/abs/2107.05407

2. **Adaptive Computation Time for Recurrent Neural Networks**
   Graves, 2016
   https://arxiv.org/abs/1603.08983

3. **PALBERT: Teaching ALBERT to Ponder**
   Yun et al., 2021
   Early-exit transformers with learned halting

4. **Reasoning Beyond Language: Latent Chain-of-Thought Survey**
   Recent survey on latent CoT across modalities

## Future Directions

Potential extensions to the Latent CoT model:

1. **Hierarchical pondering**: Different pondering depths for different forecast horizons
2. **Multi-resolution reasoning**: Ponder at multiple timescales simultaneously
3. **Uncertainty-aware halting**: Stop pondering when uncertainty is below threshold
4. **Memory-augmented pondering**: External memory for storing intermediate reasoning states
5. **Meta-learned halting**: Transfer halting policies across datasets

## Getting Help

- **Model issues**: Check `tests/models/test_latent_cot.py` for expected behavior
- **Training problems**: Review ACT loss weight and pondering statistics in logs
- **Architecture questions**: See implementation in `src/airtrace/models/latent_cot.py`
- **Task integration**: Examine `src/airtrace/tasks/cot_one_step.py` for ACT loss handling
