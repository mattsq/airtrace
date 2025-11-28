# Running Experiments

This guide explains how to run and manage experiments in AirTrace.

## Quick Start

Run a pre-configured experiment:

```bash
airtrace train exp=exp_001_gru_zscore
```

This will:
1. Load the experiment config
2. Set up data, model, transforms, and task
3. Train the model
4. Save checkpoints and logs

## Experiment Configuration

Experiments are defined in `configs/exp/`. Example:

```yaml
# configs/exp/exp_001_gru_zscore.yaml
defaults:
  - override /data: qantas_737
  - override /model: gru_ar
  - override /transforms: zscore_diff_with_context  # Use zscore_diff for no context
  - override /task: one_step
  - override /train: default

exp_name: "gru_zscore_one_step"
seed: 123
```

## Command Line Overrides

Override any parameter from the command line:

```bash
# Change batch size and learning rate
airtrace train exp=exp_001 train.batch_size=128 train.optimizer.lr=5e-4

# Use different model
airtrace train exp=exp_001 model=tcn

# Change data
airtrace train data=synthetic
```

## Logging and Checkpoints

Training outputs are organized by experiment:

```
runs/
└── 20240101/              # Date
    └── gru_zscore_one_step/  # Experiment name
        ├── .hydra/        # Hydra config
        ├── checkpoints/   # Model checkpoints
        │   ├── best.ckpt
        │   ├── epoch_10.ckpt
        │   └── ...
        └── events.out.*   # TensorBoard logs
```

## Viewing Results

Use TensorBoard to view training curves:

```bash
tensorboard --logdir runs/
```

## Evaluation

Evaluate a trained model:

```bash
airtrace eval exp=exp_001_gru_zscore checkpoint=runs/.../checkpoints/best.ckpt
```

Or use the evaluation script:

```python
from airtrace.evaluation.eval_runner import EvaluationRunner

evaluator = EvaluationRunner.from_checkpoint(
    checkpoint_path="checkpoints/best.ckpt",
    model_class=GRUARModel,
    task=task,
    test_loader=test_loader
)

results = evaluator.evaluate()
print(results["metrics"])
```

## Comparing Experiments

Run multiple experiments:

```bash
# GRU with z-score
airtrace train exp=exp_001_gru_zscore

# TCN with robust scaling
airtrace train exp=exp_002_tcn_robust

# Transformer with differencing
airtrace train exp=exp_003_transformer_diff
```

Then use notebooks to compare:

```python
# notebooks/01_model_comparison.ipynb
import pandas as pd

experiments = {
    "GRU": "runs/.../exp_001/",
    "TCN": "runs/.../exp_002/",
    "Transformer": "runs/.../exp_003/"
}

# Load results and compare
```

## Reproducibility

Each experiment is fully reproducible via:

1. **Config**: Full config saved in `.hydra/`
2. **Seed**: Random seed set in config
3. **Code Version**: Tag git commit for experiments

To reproduce:

```bash
# Same config + seed = same results
airtrace train exp=exp_001_gru_zscore seed=123
```

## Hyperparameter Search

Use Hydra's multirun feature:

```bash
airtrace train -m \
  exp=exp_001 \
  train.optimizer.lr=1e-3,5e-4,1e-4 \
  model.params.hidden_size=64,128,256
```

This runs 9 experiments (3 × 3 grid).

## Best Practices

1. **Name experiments descriptively:**
   ```yaml
   exp_name: "gru_h128_lr1e3_zscore_one_step"
   ```

2. **Use version control:**
   ```bash
   git commit -m "Add exp_004: test TCN on synthetic data"
   ```

3. **Document experiments:**
   Keep a table in `docs/` or spreadsheet:
   ```
   | Exp | Model | Transform | Task | Val Loss | Notes |
   |-----|-------|-----------|------|----------|-------|
   | 001 | GRU   | z-score   | 1step| 0.0234   | Baseline |
   | 002 | TCN   | robust    | multi| 0.0198   | Best so far |
   ```

4. **Clean up old checkpoints:**
   ```bash
   # Keep only best checkpoints
   find runs/ -name "epoch_*.ckpt" -delete
   ```

## Example Workflow

1. **Start simple:**
   ```bash
   airtrace train exp=baseline
   ```

2. **Iterate on transforms:**
   ```bash
   airtrace train exp=baseline transforms=robust_scaler
   airtrace train exp=baseline transforms=zscore_diff
   ```

3. **Try different models:**
   ```bash
   airtrace train exp=baseline model=tcn
   airtrace train exp=baseline model=transformer
   ```

4. **Tune hyperparameters:**
   ```bash
   airtrace train -m exp=best_model \
     train.optimizer.lr=1e-3,5e-4,1e-4
   ```

5. **Final evaluation:**
   ```bash
   airtrace eval exp=best_model checkpoint=checkpoints/best.ckpt
   ```
