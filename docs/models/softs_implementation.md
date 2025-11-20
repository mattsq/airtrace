# SOFTS Implementation Guide for AirTrace

Quick reference for implementing SOFTS following the ARBaseModel interface.

## Implementation Checklist

- [ ] Create `src/airtrace/models/softs.py` with SOFTS model
- [ ] Create `configs/model/softs.yaml` with hyperparameters
- [ ] Add tests in `tests/models/test_softs.py`
- [ ] Update README.md Model Registry section

## File Structure

```
src/airtrace/models/softs.py
    ├── STARModule          # Aggregate-redistribute
    ├── EncoderLayerSOFTS   # STAR + FFN
    ├── EncoderSOFTS        # Stack of EncoderLayers
    └── SOFTS               # Main model (inherits ARBaseModel)

configs/model/softs.yaml    # Hyperparameter config
```

## Code Template

### 1. STAR Module (Core Component)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class STARModule(nn.Module):
    """STar Aggregate-Redistribute Module for channel mixing."""

    def __init__(self, d_series: int, d_core: int):
        super().__init__()
        self.gen1 = nn.Linear(d_series, d_series)        # FFN preprocess
        self.gen2 = nn.Linear(d_series, d_core)          # Aggregate to core
        self.gen3 = nn.Linear(d_series + d_core, d_series)  # Fuse
        self.gen4 = nn.Linear(d_series, d_series)        # Final projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, d_series] - batch, channels, features
        Returns:
            [B, C, d_series] - aggregated and redistributed features
        """
        batch_size, channels, _ = x.shape

        # Aggregate: FFN + compression to core
        h = F.gelu(self.gen1(x))           # [B, C, d_series]
        z = self.gen2(h)                    # [B, C, d_core]

        # Stochastic pooling (training) or weighted average (eval)
        if self.training:
            # Multinomial sampling across channels
            ratio = F.softmax(z, dim=1)                      # [B, C, d_core]
            ratio = ratio.permute(0, 2, 1).reshape(-1, channels)  # [B*d_core, C]
            indices = torch.multinomial(ratio, 1)            # [B*d_core, 1]
            indices = indices.view(batch_size, -1, 1).permute(0, 2, 1)  # [B, 1, d_core]
            z_agg = torch.gather(z, 1, indices)              # [B, 1, d_core]
            z_agg = z_agg.repeat(1, channels, 1)             # [B, C, d_core]
        else:
            # Weighted average
            weight = F.softmax(z, dim=1)                     # [B, C, d_core]
            z_agg = torch.sum(z * weight, dim=1, keepdim=True)  # [B, 1, d_core]
            z_agg = z_agg.repeat(1, channels, 1)             # [B, C, d_core]

        # Redistribute: concatenate + fusion MLP
        fused = torch.cat([x, z_agg], dim=-1)  # [B, C, d_series + d_core]
        fused = F.gelu(self.gen3(fused))       # [B, C, d_series]
        output = self.gen4(fused)              # [B, C, d_series]

        return output
```

### 2. EncoderLayer (STAR + FFN)

```python
class EncoderLayerSOFTS(nn.Module):
    """Encoder layer with STAR module and feedforward network."""

    def __init__(
        self,
        d_model: int,
        d_core: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        super().__init__()
        self.star = STARModule(d_model, d_core)

        # Pointwise feedforward (1x1 convolutions)
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)

        # Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Regularization
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, d_model]
        Returns:
            [B, C, d_model]
        """
        # STAR module + residual
        residual = x
        x = self.star(x)
        x = residual + self.dropout(x)
        x = self.norm1(x)

        # Feedforward network + residual
        residual = x
        y = x.transpose(-1, 1)                          # [B, d_model, C]
        y = self.activation(self.conv1(y))              # [B, d_ff, C]
        y = self.dropout(y)
        y = self.conv2(y)                               # [B, d_model, C]
        y = self.dropout(y).transpose(-1, 1)            # [B, C, d_model]
        x = residual + y
        x = self.norm2(x)

        return x
```

### 3. Main SOFTS Model

```python
from airtrace.models.base import ARBaseModel
from airtrace.registry import register

@register("softs")
class SOFTS(ARBaseModel):
    """
    SOFTS: Series-cOre Fused Time Series forecaster.

    A pure MLP-based model for multivariate time series forecasting using
    the STAR (STar Aggregate-Redistribute) module for efficient channel mixing.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        seq_len: int,
        pred_len: int,
        d_core: int = 128,
        d_ff: int = 512,
        e_layers: int = 3,
        dropout: float = 0.0,
        activation: str = "gelu",
        use_norm: bool = True,
        **kwargs
    ):
        """
        Args:
            input_dim: Number of input channels (D)
            hidden_dim: Model dimension (d_model)
            output_dim: Number of output channels (typically same as input_dim)
            seq_len: Input sequence length (T)
            pred_len: Prediction horizon
            d_core: STAR core compression dimension
            d_ff: Feedforward network dimension
            e_layers: Number of encoder layers
            dropout: Dropout probability
            activation: Activation function ('gelu' or 'relu')
            use_norm: Whether to use instance normalization
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.use_norm = use_norm

        # Channel-as-token embedding: [B, T, D] -> [B, D, d_model]
        self.embedding = nn.Linear(seq_len, hidden_dim)
        self.embed_dropout = nn.Dropout(dropout)

        # Encoder: stack of STAR-based layers
        self.encoder_layers = nn.ModuleList([
            EncoderLayerSOFTS(
                d_model=hidden_dim,
                d_core=d_core,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation
            ) for _ in range(e_layers)
        ])

        # Projection head: [B, D, d_model] -> [B, D, pred_len] -> [B, pred_len, D]
        self.projection = nn.Linear(hidden_dim, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D] input tensor
        Returns:
            [B, pred_len, D] predictions
        """
        batch_size, seq_len, num_channels = x.shape

        # Instance normalization (per channel)
        if self.use_norm:
            means = x.mean(dim=1, keepdim=True)  # [B, 1, D]
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x = x / stdev
        else:
            means, stdev = None, None

        # Embedding: [B, T, D] -> [B, D, d_model]
        x = x.permute(0, 2, 1)              # [B, D, T]
        x = self.embedding(x)               # [B, D, d_model]
        x = self.embed_dropout(x)

        # Encoder: [B, D, d_model] -> [B, D, d_model]
        for layer in self.encoder_layers:
            x = layer(x)

        # Projection: [B, D, d_model] -> [B, D, pred_len]
        x = self.projection(x)              # [B, D, pred_len]

        # Permute to output format: [B, pred_len, D]
        x = x.permute(0, 2, 1)              # [B, pred_len, D]

        # Ensure output matches expected channels
        x = x[:, :, :self.output_dim]

        # De-normalization
        if self.use_norm and means is not None:
            x = x * stdev[:, 0, :].unsqueeze(1)
            x = x + means[:, 0, :].unsqueeze(1)

        return x
```

## Config File: `configs/model/softs.yaml`

```yaml
_target_: airtrace.models.softs.SOFTS

# Model dimensions
hidden_dim: 512      # d_model - main model dimension
d_core: 128          # STAR core compression dimension
d_ff: 512            # Feedforward network dimension

# Architecture
e_layers: 3          # Number of encoder layers

# Regularization
dropout: 0.0         # Dropout probability

# Activation
activation: "gelu"   # 'gelu' or 'relu'

# Normalization
use_norm: true       # Instance normalization

# These will be set by experiment config
input_dim: ???       # From data config
output_dim: ???      # From data config
seq_len: ???         # From data config
pred_len: ???        # From task config
```

## Experiment Config Example: `configs/exp/exp_softs_001.yaml`

```yaml
defaults:
  - override /data: qantas_737
  - override /model: softs
  - override /transforms: zscore
  - override /task: one_step
  - override /train: adam_cosine

experiment_name: "exp_softs_001_onestep"

# Model overrides
model:
  hidden_dim: 512
  d_core: 128
  d_ff: 512
  e_layers: 3
  dropout: 0.0
  use_norm: true

# Training overrides
train:
  batch_size: 16
  learning_rate: 0.0003
  epochs: 10
  patience: 10
```

## Hyperparameter Tuning Guidance

### Small Datasets (< 10 channels, < 1000 samples)
```yaml
hidden_dim: 128
d_core: 64
d_ff: 256
e_layers: 2
dropout: 0.1
```

### Medium Datasets (10-50 channels, 1000-10000 samples)
```yaml
hidden_dim: 256
d_core: 128
d_ff: 512
e_layers: 3
dropout: 0.05
```

### Large Datasets (> 50 channels, > 10000 samples)
```yaml
hidden_dim: 512
d_core: 128
d_ff: 512
e_layers: 3
dropout: 0.0
```

## Key Design Choices

### 1. d_core Selection
- **Bottleneck approach**: `d_core = hidden_dim // 4` (more compression, faster)
- **Balanced approach**: `d_core = hidden_dim // 2` (moderate compression)
- **No compression**: `d_core = hidden_dim` (more expressive, slower)

### 2. d_ff Selection
- **Standard**: `d_ff = hidden_dim` (1x expansion)
- **Transformer-like**: `d_ff = 4 * hidden_dim` (4x expansion)
- **Balanced**: `d_ff = 2 * hidden_dim` (2x expansion)

### 3. Number of Layers
- **Shallow**: 1-2 layers (faster, less overfitting)
- **Medium**: 3-4 layers (good tradeoff)
- **Deep**: 5+ layers (rarely needed, diminishing returns)

### 4. Dropout Strategy
- SOFTS paper often uses **very low or zero dropout**
- If overfitting, try `dropout = 0.05` or `dropout = 0.1`
- Apply consistently across all layers

## Common Issues and Solutions

### Issue 1: Model not learning
**Possible causes**:
- Learning rate too low/high
- d_core too small (over-compression)
- Not enough layers

**Solutions**:
- Try `learning_rate = 0.0003` (SOFTS paper default)
- Increase `d_core` to `hidden_dim // 2`
- Add one more encoder layer

### Issue 2: Overfitting
**Possible causes**:
- Too many parameters
- No regularization

**Solutions**:
- Add dropout: `dropout = 0.1`
- Reduce `e_layers`
- Reduce `d_ff`
- Use more data augmentation in transforms

### Issue 3: Training too slow
**Possible causes**:
- d_core too large (no compression benefit)
- d_ff too large
- Too many layers

**Solutions**:
- Reduce `d_core` to `hidden_dim // 4`
- Use `d_ff = hidden_dim` instead of `4 * hidden_dim`
- Reduce `e_layers`

### Issue 4: NaN/Inf losses
**Possible causes**:
- Learning rate too high
- No normalization
- Gradient explosion

**Solutions**:
- Enable normalization: `use_norm = true`
- Reduce learning rate
- Use gradient clipping in training config
- Check for zero standard deviations in data

## Testing

```python
# tests/models/test_softs.py
import pytest
import torch
from airtrace.models.softs import SOFTS, STARModule, EncoderLayerSOFTS

def test_star_module():
    """Test STAR module forward pass."""
    batch_size, channels, d_series = 4, 10, 128
    d_core = 64

    star = STARModule(d_series, d_core)
    x = torch.randn(batch_size, channels, d_series)

    # Test training mode
    star.train()
    out_train = star(x)
    assert out_train.shape == (batch_size, channels, d_series)

    # Test eval mode
    star.eval()
    out_eval = star(x)
    assert out_eval.shape == (batch_size, channels, d_series)

def test_encoder_layer():
    """Test encoder layer forward pass."""
    batch_size, channels, d_model = 4, 10, 128
    d_core, d_ff = 64, 256

    layer = EncoderLayerSOFTS(d_model, d_core, d_ff)
    x = torch.randn(batch_size, channels, d_model)

    out = layer(x)
    assert out.shape == (batch_size, channels, d_model)

def test_softs_forward():
    """Test SOFTS model forward pass."""
    batch_size, seq_len, num_channels = 4, 96, 10
    pred_len = 24
    hidden_dim = 128

    model = SOFTS(
        input_dim=num_channels,
        hidden_dim=hidden_dim,
        output_dim=num_channels,
        seq_len=seq_len,
        pred_len=pred_len,
        d_core=64,
        d_ff=256,
        e_layers=2
    )

    x = torch.randn(batch_size, seq_len, num_channels)
    out = model(x)

    assert out.shape == (batch_size, pred_len, num_channels)

def test_softs_normalization():
    """Test that normalization is applied correctly."""
    batch_size, seq_len, num_channels = 4, 96, 10
    pred_len = 24

    # With normalization
    model_norm = SOFTS(
        input_dim=num_channels,
        hidden_dim=128,
        output_dim=num_channels,
        seq_len=seq_len,
        pred_len=pred_len,
        use_norm=True
    )

    # Without normalization
    model_no_norm = SOFTS(
        input_dim=num_channels,
        hidden_dim=128,
        output_dim=num_channels,
        seq_len=seq_len,
        pred_len=pred_len,
        use_norm=False
    )

    x = torch.randn(batch_size, seq_len, num_channels) * 100  # Large values

    out_norm = model_norm(x)
    out_no_norm = model_no_norm(x)

    # Both should produce valid outputs
    assert torch.isfinite(out_norm).all()
    assert torch.isfinite(out_no_norm).all()

def test_softs_stochastic_pooling():
    """Test that stochastic pooling differs between train/eval."""
    batch_size, seq_len, num_channels = 4, 96, 10
    pred_len = 24

    model = SOFTS(
        input_dim=num_channels,
        hidden_dim=128,
        output_dim=num_channels,
        seq_len=seq_len,
        pred_len=pred_len
    )

    x = torch.randn(batch_size, seq_len, num_channels)

    # Training mode (stochastic)
    model.train()
    torch.manual_seed(42)
    out_train_1 = model(x)
    torch.manual_seed(43)
    out_train_2 = model(x)

    # Outputs should differ (stochastic)
    assert not torch.allclose(out_train_1, out_train_2)

    # Eval mode (deterministic)
    model.eval()
    out_eval_1 = model(x)
    out_eval_2 = model(x)

    # Outputs should be identical (deterministic)
    assert torch.allclose(out_eval_1, out_eval_2)
```

## README Update

Add to the Model Registry section in `README.md`:

```markdown
### Sequence-to-Sequence Models

| Model | Class | Description | Paper |
|-------|-------|-------------|-------|
| `softs` | `SOFTS` | Pure MLP-based multivariate forecaster using STAR (Aggregate-Redistribute) module with stochastic pooling for efficient channel mixing | NeurIPS 2024 |
```

## Next Steps

1. Implement the model in `src/airtrace/models/softs.py`
2. Create config in `configs/model/softs.yaml`
3. Run tests: `pytest tests/models/test_softs.py -v`
4. Create experiment config
5. Run training: `airtrace train exp=exp_softs_001`
6. Update README.md
7. Update MEMORY.md with any discoveries
