# Scaling Laws Proof-of-Concept Runner
# Smaller models, fewer epochs (10), limited data (max 20%)

$python = ".venv\Scripts\python"

function Run-Experiment {
    param (
        [string]$data_idx,
        [string]$exp_suffix,
        [int]$d_model,
        [int]$nhead,
        [int]$e_layers,
        [int]$d_layers,
        [int]$ff_dim
    )
    
    $exp_name = "informer_poc_$exp_suffix"
    Write-Host "Starting POC Experiment: $exp_name" -ForegroundColor Cyan
    
    $cmdArgs = @(
        "-m", "airtrace.cli", "train",
        "model=informer",
        "data=descent_data",
        "data.train_index=$data_idx",
        "model.params.d_model=$d_model",
        "model.params.nhead=$nhead",
        "model.params.e_layers=$e_layers",
        "model.params.d_layers=$d_layers",
        "model.params.ff_dim=$ff_dim",
        "exp_name=$exp_name",
        "train.epochs=10"
    )
    
    Write-Host "Running: $python $cmdArgs"
    & $python $cmdArgs
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Experiment $exp_name failed with exit code $LASTEXITCODE"
    } else {
        Write-Host "Experiment $exp_name completed successfully." -ForegroundColor Green
    }
}

# 10% Data
Run-Experiment -data_idx "metadata/descent_train_10pct.parquet" -exp_suffix "tiny_10pct" -d_model 16 -nhead 2 -e_layers 1 -d_layers 1 -ff_dim 32
Run-Experiment -data_idx "metadata/descent_train_10pct.parquet" -exp_suffix "small_10pct" -d_model 32 -nhead 2 -e_layers 1 -d_layers 1 -ff_dim 64
Run-Experiment -data_idx "metadata/descent_train_10pct.parquet" -exp_suffix "medium_10pct" -d_model 64 -nhead 4 -e_layers 1 -d_layers 1 -ff_dim 128

# 20% Data
Run-Experiment -data_idx "metadata/descent_train_20pct.parquet" -exp_suffix "tiny_20pct" -d_model 16 -nhead 2 -e_layers 1 -d_layers 1 -ff_dim 32
Run-Experiment -data_idx "metadata/descent_train_20pct.parquet" -exp_suffix "small_20pct" -d_model 32 -nhead 2 -e_layers 1 -d_layers 1 -ff_dim 64
Run-Experiment -data_idx "metadata/descent_train_20pct.parquet" -exp_suffix "medium_20pct" -d_model 64 -nhead 4 -e_layers 1 -d_layers 1 -ff_dim 128
