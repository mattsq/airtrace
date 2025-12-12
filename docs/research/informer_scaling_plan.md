# Informer Scaling Laws Experiment Plan

## Objective
Establish scaling laws for the Informer model on the Descent dataset by systematically varying dataset size and model capacity.

## Methodology
We will conduct a grid search over:
1.  **Data Size**: 10%, 20%, 50%, 100% of the training set.
2.  **Model Size**: Small, Medium, Large.

## Configurations

### Data Subsets
Indices have been generated in `data/metadata/`:
- `descent_train_10pct.parquet` (745 samples)
- `descent_train_20pct.parquet` (1491 samples)
- `descent_train_50pct.parquet` (3729 samples)
- `descent_train_100pct.parquet` (7458 samples)

### Model Variants
Base Model: `informer`

| Variant | d_model | nhead | e_layers | d_layers | ff_dim |
|---------|---------|-------|----------|----------|--------|
| **Small** | 64      | 2     | 1        | 1        | 128    |
| **Medium**| 128     | 4     | 2        | 1        | 256    |
| **Large** | 256     | 8     | 3        | 2        | 512    |

## Execution Plan

Run the following commands using `.venv/Scripts/python -m airtrace.cli train`:

### 10% Data
```bash
# Small
python -m airtrace.cli train model=informer data=descent_data data.train_index="metadata/descent_train_10pct.parquet" model.params.d_model=64 model.params.nhead=2 model.params.e_layers=1 model.params.d_layers=1 model.params.ff_dim=128 exp_name="informer_small_10pct"

# Medium
python -m airtrace.cli train model=informer data=descent_data data.train_index="metadata/descent_train_10pct.parquet" model.params.d_model=128 model.params.nhead=4 model.params.e_layers=2 model.params.d_layers=1 model.params.ff_dim=256 exp_name="informer_medium_10pct"

# Large
python -m airtrace.cli train model=informer data=descent_data data.train_index="metadata/descent_train_10pct.parquet" model.params.d_model=256 model.params.nhead=8 model.params.e_layers=3 model.params.d_layers=2 model.params.ff_dim=512 exp_name="informer_large_10pct"
```

### 20% Data
```bash
# Small
python -m airtrace.cli train model=informer data=descent_data data.train_index="metadata/descent_train_20pct.parquet" model.params.d_model=64 model.params.nhead=2 model.params.e_layers=1 model.params.d_layers=1 model.params.ff_dim=128 exp_name="informer_small_20pct"

# Medium
python -m airtrace.cli train model=informer data=descent_data data.train_index="metadata/descent_train_20pct.parquet" model.params.d_model=128 model.params.nhead=4 model.params.e_layers=2 model.params.d_layers=1 model.params.ff_dim=256 exp_name="informer_medium_20pct"

# Large
python -m airtrace.cli train model=informer data=descent_data data.train_index="metadata/descent_train_20pct.parquet" model.params.d_model=256 model.params.nhead=8 model.params.e_layers=3 model.params.d_layers=2 model.params.ff_dim=512 exp_name="informer_large_20pct"
```

### 50% Data
```bash
# Small
python -m airtrace.cli train model=informer data=descent_data data.train_index="metadata/descent_train_50pct.parquet" model.params.d_model=64 model.params.nhead=2 model.params.e_layers=1 model.params.d_layers=1 model.params.ff_dim=128 exp_name="informer_small_50pct"

# Medium
python -m airtrace.cli train model=informer data=descent_data data.train_index="metadata/descent_train_50pct.parquet" model.params.d_model=128 model.params.nhead=4 model.params.e_layers=2 model.params.d_layers=1 model.params.ff_dim=256 exp_name="informer_medium_50pct"

# Large
python -m airtrace.cli train model=informer data=descent_data data.train_index="metadata/descent_train_50pct.parquet" model.params.d_model=256 model.params.nhead=8 model.params.e_layers=3 model.params.d_layers=2 model.params.ff_dim=512 exp_name="informer_large_50pct"
```

### 100% Data
```bash
# Small
python -m airtrace.cli train model=informer data=descent_data data.train_index="metadata/descent_train_100pct.parquet" model.params.d_model=64 model.params.nhead=2 model.params.e_layers=1 model.params.d_layers=1 model.params.ff_dim=128 exp_name="informer_small_100pct"

# Medium
python -m airtrace.cli train model=informer data=descent_data data.train_index="metadata/descent_train_100pct.parquet" model.params.d_model=128 model.params.nhead=4 model.params.e_layers=2 model.params.d_layers=1 model.params.ff_dim=256 exp_name="informer_medium_100pct"

# Large
python -m airtrace.cli train model=informer data=descent_data data.train_index="metadata/descent_train_100pct.parquet" model.params.d_model=256 model.params.nhead=8 model.params.e_layers=3 model.params.d_layers=2 model.params.ff_dim=512 exp_name="informer_large_100pct"
```

## Analysis
Collect validation loss from each run and plot:
1.  Log-Log plot of Data Size vs Loss (for each model size).
2.  Log-Log plot of Parameters vs Loss (for each data size).
