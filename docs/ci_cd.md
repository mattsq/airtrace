# Continuous Integration & Continuous Deployment (CI/CD)

This document describes the CI/CD pipeline for the AirTrace project.

## Overview

The AirTrace CI/CD pipeline is implemented using GitHub Actions and consists of two main jobs:

1. **Test Job**: Runs all unit tests with coverage reporting
2. **Model Validation Job**: Validates all registered models against synthetic flight data

## Workflow Configuration

**File**: `.github/workflows/ci.yml`

**Triggers**:
- Push to branches: `main`, `develop`, `claude/**`
- Pull requests to: `main`, `develop`
- Manual workflow dispatch

## Jobs

### 1. Test Job

**Purpose**: Ensure code quality and test coverage

**Matrix**: Python versions 3.9, 3.10, 3.11

**Steps**:
1. Checkout code
2. Set up Python environment
3. Install dependencies (including dev dependencies)
4. Run linting with `ruff`
5. Run type checking with `mypy`
6. Run tests with `pytest` and generate coverage report
7. Upload coverage report as artifact
8. Display coverage summary in GitHub Actions summary

**Quality Gates**:
- All tests must pass
- Linting and type checking run (currently set to `continue-on-error`)

**Artifacts**:
- Coverage reports (XML format) for each Python version
- Retained for 30 days

### 2. Model Validation Job

**Purpose**: Validate that all registered models can train and produce reasonable predictions

**Dependencies**: Runs only if the test job passes

**Steps**:
1. Checkout code
2. Set up Python 3.10
3. Install dependencies
4. Create data directories
5. Run model validation script
6. Upload validation results as artifact
7. Parse and display results in GitHub Actions summary
8. Check validation success (fail job if any model fails)

**Validation Process**:

The model validation script (`src/scripts/validate_models.py`) performs the following:

1. **Generate Synthetic Data**:
   - Creates 20 synthetic cruise flights using `SyntheticCruiseGenerator`
   - Splits into 80% train / 20% validation
   - Each flight is 30 minutes at 1 Hz sampling (1800 samples)

2. **Create Windowed Datasets**:
   - Window size: 60 seconds (60 samples)
   - Prediction horizon: 1 second (1 sample)
   - Stride: 10 seconds
   - Results in ~2000 train windows, ~400 validation windows

3. **Normalize Data**:
   - Z-score normalization using training set statistics
   - Applied to both inputs and targets

4. **For Each Registered Model**:
   - Build model with minimal configuration
   - Train for 10 epochs with Adam optimizer (lr=1e-3)
   - Evaluate on validation set
   - Compute metrics: RMSE, MAE, MAPE, MSE, R²
   - Compute per-sensor metrics

5. **Report Results**:
   - Overall metrics for each model
   - Per-sensor metrics (fuel_flow, mach, altitude, oat, n1, weight)
   - Training time and parameter count
   - Success/failure status

**Quality Gates**:
- All registered models must successfully train
- All models must produce predictions without errors
- Job fails if any model fails validation

**Artifacts**:
- `model_validation_results.json` - Complete validation results
- Retained for 30 days

## Model Validation Metrics

The following metrics are computed for each model:

### Overall Metrics

- **RMSE** (Root Mean Squared Error): Lower is better, measures average prediction error
- **MAE** (Mean Absolute Error): Lower is better, measures average absolute error
- **MAPE** (Mean Absolute Percentage Error): Lower is better, percentage-based error
- **MSE** (Mean Squared Error): Lower is better, squared prediction error
- **R²** (Coefficient of Determination): Higher is better (max 1.0), measures explained variance

### Per-Sensor Metrics

Metrics are computed individually for each sensor:
- `fuel_flow` - Fuel flow rate (kg/hour)
- `mach` - Mach number
- `altitude` - Altitude (feet)
- `oat` - Outside air temperature (°C)
- `n1` - Engine N1 percentage
- `weight` - Aircraft weight (kg)

This helps identify if a model performs poorly on specific sensors.

## Running Locally

### Run Tests Locally

```bash
# Install dependencies
pip install -e ".[dev]"

# Run all tests with coverage
pytest tests/ -v --cov=airtrace --cov-report=term

# Run linting
ruff check src/ tests/

# Run type checking
mypy src/airtrace/
```

### Run Model Validation Locally

```bash
# Basic usage
python src/scripts/validate_models.py

# With custom parameters
python src/scripts/validate_models.py \
  --output my_results.json \
  --seed 42 \
  --n-flights 10 \
  --n-epochs 5 \
  --batch-size 64 \
  --device cpu

# View help
python src/scripts/validate_models.py --help
```

**Parameters**:
- `--output`: Output JSON file path (default: `model_validation_results.json`)
- `--seed`: Random seed for reproducibility (default: 42)
- `--n-flights`: Number of synthetic flights to generate (default: 20)
- `--n-epochs`: Training epochs per model (default: 10)
- `--batch-size`: Batch size for training (default: 32)
- `--device`: Device to use - `cpu` or `cuda` (default: `cpu`)

## Interpreting Results

### GitHub Actions Summary

After a CI run completes, check the **Summary** tab for:

1. **Test Coverage Summary**: Shows which lines are covered by tests
2. **Model Validation Results**: Shows a table with all model results
3. **Per-Sensor Metrics**: Shows detailed metrics for each sensor

### Success Criteria

✅ **Passing CI**:
- All tests pass (test job)
- All models train successfully (model validation job)
- No critical linting or type errors

❌ **Failing CI**:
- Any test fails
- Any model fails to train or evaluate
- Coverage drops significantly (monitored but not enforced)

### Common Failure Modes

1. **Test Failures**:
   - Check test logs for specific failing tests
   - May indicate breaking changes to core functionality

2. **Model Training Failures**:
   - Check model validation logs for error messages
   - Common causes:
     - Incompatible tensor shapes
     - NaN/Inf values during training
     - Missing required parameters
     - Registry issues (model not properly registered)

3. **Linting/Type Errors**:
   - Currently set to `continue-on-error`, so won't fail CI
   - Should still be addressed for code quality

## Adding New Models

When adding a new model to the registry:

1. **Implement the model** in `src/airtrace/models/your_model.py`:
   - Inherit from `ARBaseModel`
   - Implement `__init__` and `forward` methods
   - Add `@register("your_model")` decorator

2. **Create model config** in `configs/model/your_model.yaml`

3. **Add tests** in `tests/test_models.py`

4. **CI will automatically**:
   - Run your unit tests
   - Include your model in validation
   - Report metrics for your model

**No manual CI configuration required** - the model validation script automatically discovers all registered models via `list_models()`.

## Performance Considerations

### CI Runtime

Typical CI run times (on GitHub Actions `ubuntu-latest` runners):

- **Test Job**: ~2-5 minutes per Python version
- **Model Validation Job**: ~5-10 minutes (depends on number of models)

**Total runtime**: ~10-20 minutes (parallel execution across matrix)

### Resource Usage

- **Memory**: Models are small, typically <100MB
- **CPU**: Training on CPU is sufficient for validation
- **Storage**: Artifacts (coverage + validation results) typically <1MB

### Cost Optimization

To reduce CI runtime for faster feedback:

1. Reduce `--n-flights` (fewer flights, faster data generation)
2. Reduce `--n-epochs` (faster training, may reduce metric accuracy)
3. Reduce matrix (test only Python 3.10 instead of 3.9, 3.10, 3.11)

**Example fast configuration** (modify in `.github/workflows/ci.yml`):
```yaml
python src/scripts/validate_models.py \
  --n-flights 10 \
  --n-epochs 5 \
  --batch-size 64
```

## Troubleshooting

### "Model failed validation"

**Check**:
1. Model logs in the validation job output
2. Error message in the GitHub Actions summary
3. Whether the model is properly registered (`list_models()`)
4. Model configuration in `configs/model/`

### "Tests pass locally but fail in CI"

**Common causes**:
1. Different Python version (CI uses 3.9, 3.10, 3.11)
2. Missing dependencies in `pyproject.toml`
3. Platform-specific differences (CI runs on Ubuntu)
4. Random seed issues (not setting seed properly)

**Solutions**:
- Test locally with `tox` or Docker to match CI environment
- Check CI logs for specific error messages
- Ensure all dependencies are in `pyproject.toml`

### "Coverage report shows unexpected gaps"

**Check**:
1. Whether new code has corresponding tests
2. Test file naming (must match `test_*.py`)
3. Whether tests are actually exercising the code paths

### "Model validation is too slow"

**Options**:
1. Reduce `--n-flights` and `--n-epochs` in the workflow
2. Use smaller model configurations for CI
3. Cache synthetic data (not currently implemented)

## Future Enhancements

Potential improvements to the CI pipeline:

1. **Coverage Enforcement**: Fail CI if coverage drops below threshold (e.g., 80%)
2. **Benchmark Tracking**: Track model performance over time
3. **GPU Runners**: Use GPU for faster model training
4. **Model Comparison**: Compare new models against baselines
5. **Deployment**: Auto-deploy passing models to model registry
6. **Slack/Email Notifications**: Alert on failures
7. **PR Comments**: Post validation results as PR comments
8. **Performance Regression Tests**: Fail if model performance degrades

## Related Documentation

- [Architecture](architecture.md) - System design and component structure
- [Synthetic Data Generator](synthetic_data.md) - How synthetic data is generated
- [CLAUDE.md](../CLAUDE.md) - AI agent guide with project conventions
- [README.md](../README.md) - Project overview

## References

- GitHub Actions Documentation: https://docs.github.com/en/actions
- pytest Documentation: https://docs.pytest.org/
- Coverage.py: https://coverage.readthedocs.io/
