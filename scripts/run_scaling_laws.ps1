# Revised Scaling Laws Experiment Runner - CPU-Friendly Configuration
# Based on expert critique: smaller models, early stopping, phased execution
# See docs/research/scaling_laws_critique.md for full rationale

$python = ".venv\Scripts\python"

function Run-Experiment {
    param (
        [string]$data_idx,
        [string]$exp_suffix,
        [int]$d_model,
        [int]$nhead,
        [int]$e_layers,
        [int]$d_layers,
        [int]$ff_dim,
        [int]$expected_params
    )

    $exp_name = "scaling_$exp_suffix"
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "Starting: $exp_name" -ForegroundColor Cyan
    Write-Host "Config: d_model=$d_model, nhead=$nhead, layers=$e_layers/$d_layers, ff_dim=$ff_dim" -ForegroundColor Gray
    Write-Host "Expected params: $($expected_params.ToString('N0'))" -ForegroundColor Gray
    Write-Host "========================================" -ForegroundColor Cyan

    $start_time = Get-Date

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
        "train.epochs=30",                    # Increased from POC's 10, decreased from original 50
        "train.early_stopping.patience=10"    # Enable early stopping
    )

    Write-Host "Running: $python $cmdArgs" -ForegroundColor Gray
    & $python $cmdArgs

    if ($LASTEXITCODE -ne 0) {
        Write-Error "Experiment $exp_name failed with exit code $LASTEXITCODE"
        return $false
    }

    $elapsed = ((Get-Date) - $start_time).TotalMinutes
    Write-Host "Completed $exp_name in $([math]::Round($elapsed, 1)) minutes" -ForegroundColor Green
    return $true
}

# =============================================================================
# PHASE 1: VALIDATION (Tiny + Small Models)
# =============================================================================
# Goal: Validate pipeline and check for scaling trends
# Estimated time: 3-4 hours
# Models: Tiny (22K params), Small (85K params)

Write-Host "`n" -ForegroundColor Yellow
Write-Host "================================================================" -ForegroundColor Yellow
Write-Host "  PHASE 1: VALIDATION (Tiny + Small Models)" -ForegroundColor Yellow
Write-Host "  Goal: Validate pipeline and check for scaling trends" -ForegroundColor Yellow
Write-Host "  Estimated time: 3-4 hours" -ForegroundColor Yellow
Write-Host "================================================================" -ForegroundColor Yellow

$phase1_start = Get-Date
$experiments_completed = 0
$experiments_failed = 0

# Tiny model (d_model=32, nhead=2, e_layers=1, d_layers=1, ff_dim=64) -> 22,123 params
Write-Host "`n--- Tiny Model (22K params) ---" -ForegroundColor Magenta
if (Run-Experiment -data_idx "metadata/descent_train_10pct.parquet" -exp_suffix "tiny_10pct" -d_model 32 -nhead 2 -e_layers 1 -d_layers 1 -ff_dim 64 -expected_params 22123) { $experiments_completed++ } else { $experiments_failed++ }
if (Run-Experiment -data_idx "metadata/descent_train_20pct.parquet" -exp_suffix "tiny_20pct" -d_model 32 -nhead 2 -e_layers 1 -d_layers 1 -ff_dim 64 -expected_params 22123) { $experiments_completed++ } else { $experiments_failed++ }
if (Run-Experiment -data_idx "metadata/descent_train_50pct.parquet" -exp_suffix "tiny_50pct" -d_model 32 -nhead 2 -e_layers 1 -d_layers 1 -ff_dim 64 -expected_params 22123) { $experiments_completed++ } else { $experiments_failed++ }
if (Run-Experiment -data_idx "metadata/descent_train_100pct.parquet" -exp_suffix "tiny_100pct" -d_model 32 -nhead 2 -e_layers 1 -d_layers 1 -ff_dim 64 -expected_params 22123) { $experiments_completed++ } else { $experiments_failed++ }

# Small model (d_model=64, nhead=2, e_layers=1, d_layers=1, ff_dim=128) -> 85,195 params
Write-Host "`n--- Small Model (85K params) ---" -ForegroundColor Magenta
if (Run-Experiment -data_idx "metadata/descent_train_10pct.parquet" -exp_suffix "small_10pct" -d_model 64 -nhead 2 -e_layers 1 -d_layers 1 -ff_dim 128 -expected_params 85195) { $experiments_completed++ } else { $experiments_failed++ }
if (Run-Experiment -data_idx "metadata/descent_train_20pct.parquet" -exp_suffix "small_20pct" -d_model 64 -nhead 2 -e_layers 1 -d_layers 1 -ff_dim 128 -expected_params 85195) { $experiments_completed++ } else { $experiments_failed++ }
if (Run-Experiment -data_idx "metadata/descent_train_50pct.parquet" -exp_suffix "small_50pct" -d_model 64 -nhead 2 -e_layers 1 -d_layers 1 -ff_dim 128 -expected_params 85195) { $experiments_completed++ } else { $experiments_failed++ }
if (Run-Experiment -data_idx "metadata/descent_train_100pct.parquet" -exp_suffix "small_100pct" -d_model 64 -nhead 2 -e_layers 1 -d_layers 1 -ff_dim 128 -expected_params 85195) { $experiments_completed++ } else { $experiments_failed++ }

$phase1_elapsed = ((Get-Date) - $phase1_start).TotalMinutes

Write-Host "`n" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green
Write-Host "  PHASE 1 COMPLETE" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green
Write-Host "Experiments completed: $experiments_completed / 8" -ForegroundColor $(if ($experiments_failed -eq 0) { "Green" } else { "Yellow" })
Write-Host "Experiments failed: $experiments_failed" -ForegroundColor $(if ($experiments_failed -eq 0) { "Green" } else { "Red" })
Write-Host "Total time: $([math]::Round($phase1_elapsed, 1)) minutes" -ForegroundColor Cyan

Write-Host "`nNext steps:" -ForegroundColor Yellow
Write-Host "  1. Review results in outputs/ directory" -ForegroundColor White
Write-Host "  2. Check validation loss trends (loss should decrease with more data)" -ForegroundColor White
Write-Host "  3. Check early stopping behavior (did models converge?)" -ForegroundColor White
Write-Host "  4. If results look good, run Phase 2:" -ForegroundColor White
Write-Host "     powershell scripts/run_scaling_laws_phase2.ps1" -ForegroundColor Cyan
