# Production Model Run (5M Params, 100% Data)

$python = ".venv\Scripts\python"
$exp_name = "informer_production_5m_100pct"

Write-Host "Starting Production Experiment: $exp_name" -ForegroundColor Cyan
Write-Host "Target: ~5M Parameters, 100% Data" -ForegroundColor Gray

$cmdArgs = @(
    "-m", "airtrace.cli", "train",
    "model=informer",
    "data=descent_data",
    "data.train_index=metadata/descent_train_100pct.parquet",
    "model.params.d_model=256",
    "model.params.nhead=8",
    "model.params.e_layers=3",
    "model.params.d_layers=2",
    "model.params.ff_dim=1024",
    "exp_name=$exp_name",
    "train.epochs=30",
    "train.verbose_progress=true"
)

Write-Host "Running: $python $cmdArgs"
& $python $cmdArgs

if ($LASTEXITCODE -ne 0) {
    Write-Error "Experiment $exp_name failed with exit code $LASTEXITCODE"
} else {
    Write-Host "Experiment $exp_name completed successfully." -ForegroundColor Green
}
