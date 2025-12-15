# Scaling Laws Experiment - Phase 2: Medium + Large Models
# Run this AFTER Phase 1 completes successfully
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
        "train.epochs=30",
        "train.early_stopping.patience=10"
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
# PHASE 2: FULL ANALYSIS (Medium + Large Models)
# =============================================================================
# Goal: Complete the scaling law analysis
# Estimated time: 5-8 hours
# Models: Medium (292K params), Large (516K params)

Write-Host "`n" -ForegroundColor Yellow
Write-Host "================================================================" -ForegroundColor Yellow
Write-Host "  PHASE 2: FULL ANALYSIS (Medium + Large Models)" -ForegroundColor Yellow
Write-Host "  Goal: Complete the scaling law analysis" -ForegroundColor Yellow
Write-Host "  Estimated time: 5-8 hours" -ForegroundColor Yellow
Write-Host "================================================================" -ForegroundColor Yellow

$phase2_start = Get-Date
$experiments_completed = 0
$experiments_failed = 0

# Medium model (d_model=96, nhead=4, e_layers=2, d_layers=1, ff_dim=192) -> 291,755 params
Write-Host "`n--- Medium Model (292K params) ---" -ForegroundColor Magenta
if (Run-Experiment -data_idx "metadata/descent_train_10pct.parquet" -exp_suffix "medium_10pct" -d_model 96 -nhead 4 -e_layers 2 -d_layers 1 -ff_dim 192 -expected_params 291755) { $experiments_completed++ } else { $experiments_failed++ }
if (Run-Experiment -data_idx "metadata/descent_train_20pct.parquet" -exp_suffix "medium_20pct" -d_model 96 -nhead 4 -e_layers 2 -d_layers 1 -ff_dim 192 -expected_params 291755) { $experiments_completed++ } else { $experiments_failed++ }
if (Run-Experiment -data_idx "metadata/descent_train_50pct.parquet" -exp_suffix "medium_50pct" -d_model 96 -nhead 4 -e_layers 2 -d_layers 1 -ff_dim 192 -expected_params 291755) { $experiments_completed++ } else { $experiments_failed++ }
if (Run-Experiment -data_idx "metadata/descent_train_100pct.parquet" -exp_suffix "medium_100pct" -d_model 96 -nhead 4 -e_layers 2 -d_layers 1 -ff_dim 192 -expected_params 291755) { $experiments_completed++ } else { $experiments_failed++ }

# Large model (d_model=128, nhead=4, e_layers=2, d_layers=1, ff_dim=256) -> 515,979 params
Write-Host "`n--- Large Model (516K params) ---" -ForegroundColor Magenta
if (Run-Experiment -data_idx "metadata/descent_train_10pct.parquet" -exp_suffix "large_10pct" -d_model 128 -nhead 4 -e_layers 2 -d_layers 1 -ff_dim 256 -expected_params 515979) { $experiments_completed++ } else { $experiments_failed++ }
if (Run-Experiment -data_idx "metadata/descent_train_20pct.parquet" -exp_suffix "large_20pct" -d_model 128 -nhead 4 -e_layers 2 -d_layers 1 -ff_dim 256 -expected_params 515979) { $experiments_completed++ } else { $experiments_failed++ }
if (Run-Experiment -data_idx "metadata/descent_train_50pct.parquet" -exp_suffix "large_50pct" -d_model 128 -nhead 4 -e_layers 2 -d_layers 1 -ff_dim 256 -expected_params 515979) { $experiments_completed++ } else { $experiments_failed++ }
if (Run-Experiment -data_idx "metadata/descent_train_100pct.parquet" -exp_suffix "large_100pct" -d_model 128 -nhead 4 -e_layers 2 -d_layers 1 -ff_dim 256 -expected_params 515979) { $experiments_completed++ } else { $experiments_failed++ }

$phase2_elapsed = ((Get-Date) - $phase2_start).TotalMinutes

Write-Host "`n" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green
Write-Host "  PHASE 2 COMPLETE - ALL SCALING LAW EXPERIMENTS DONE!" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green
Write-Host "Experiments completed: $experiments_completed / 8" -ForegroundColor $(if ($experiments_failed -eq 0) { "Green" } else { "Yellow" })
Write-Host "Experiments failed: $experiments_failed" -ForegroundColor $(if ($experiments_failed -eq 0) { "Green" } else { "Red" })
Write-Host "Phase 2 time: $([math]::Round($phase2_elapsed, 1)) minutes" -ForegroundColor Cyan

Write-Host "`nNext steps:" -ForegroundColor Yellow
Write-Host "  1. Collect all results from outputs/ directory" -ForegroundColor White
Write-Host "  2. Run analysis script:" -ForegroundColor White
Write-Host "     python scripts/analyze_scaling_laws.py" -ForegroundColor Cyan
Write-Host "  3. Generate scaling law plots (data scaling & model scaling)" -ForegroundColor White
Write-Host "  4. Fit power laws: L = C*D^(-alpha) and L = C*P^(-beta)" -ForegroundColor White
