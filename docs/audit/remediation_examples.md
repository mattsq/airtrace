# Test Remediation Examples

This document provides concrete "before and after" examples for fixing the test integrity issues identified in the audit.

---

## Pattern 1: Weak Assertions in Visualization Tests

### ❌ BEFORE (Weak)

```python
def test_plot_timeseries_single_sensor(self):
    """Test plotting single sensor timeseries."""
    data = np.random.randn(100, 1)
    fig = plot_timeseries(data)

    assert fig is not None  # Only checks existence
    assert len(fig.axes) == 1
    plt.close(fig)
```

**Problem:** Test passes even if plot is empty or broken.

### ✅ AFTER (Strong)

```python
def test_plot_timeseries_single_sensor(self):
    """Test plotting single sensor timeseries."""
    data = np.random.randn(100, 1)
    fig = plot_timeseries(data)

    # Verify figure exists
    assert fig is not None
    assert len(fig.axes) == 1

    # Verify data was actually plotted
    axis = fig.axes[0]
    lines = axis.get_lines()
    assert len(lines) == 1, "Should plot exactly one line for single sensor"

    # Verify all data points are present
    x_data, y_data = lines[0].get_data()
    assert len(y_data) == 100, "Should plot all 100 datapoints"
    np.testing.assert_array_almost_equal(y_data, data.ravel(), decimal=5)

    # Verify axes are labeled
    assert axis.get_xlabel() != "", "X-axis should be labeled"
    assert axis.get_ylabel() != "", "Y-axis should be labeled"

    plt.close(fig)
```

**Benefits:**
- Catches broken plotting logic
- Ensures data integrity
- Verifies user-facing features (labels)

---

## Pattern 2: Over-Mocked CLI Tests

### ❌ BEFORE (Invalid Test)

```python
def test_evaluate_runs_with_stub_components(monkeypatch, tmp_path, capsys):
    # ... setup code ...

    class _TinyEvaluationRunner:
        def evaluate(self, return_predictions=False):
            return {"metrics": {"mae": 0.0}, "num_samples": 1}  # Hardcoded!

    monkeypatch.setattr("airtrace.evaluation.eval_runner.EvaluationRunner", _TinyEvaluationRunner)
    monkeypatch.setattr("airtrace.data.datamodule.SensorDataModule", _TinyDataModule)
    # ... mock everything ...

    cli.evaluate(cfg)
    output = capsys.readouterr().out
    assert "Evaluation Results" in output
    assert "MAE" in output
```

**Problems:**
- Only tests that `cli.evaluate` can print text
- Doesn't test evaluation logic at all
- Gives false confidence

### ✅ OPTION A: Convert to Narrow Unit Test

```python
def test_evaluate_prints_results_in_correct_format():
    """Test that evaluate() formats and prints evaluation results correctly."""
    # This is what the original test ACTUALLY validated

    fake_results = {
        "metrics": {"mae": 0.123, "rmse": 0.456},
        "num_samples": 100
    }

    output = cli._format_evaluation_results(fake_results)

    assert "Evaluation Results" in output
    assert "MAE" in output
    assert "0.123" in output
    assert "100 samples" in output
```

**Benefits:**
- Honest about what it tests (formatting)
- Fast and focused
- Doesn't give false confidence about evaluation logic

### ✅ OPTION B: Create Real Integration Test

```python
def test_evaluate_end_to_end_with_tiny_model(tmp_path):
    """Integration test: Full evaluation pipeline with real components."""
    # Use a real tiny model and real data
    config = create_minimal_test_config(tmp_path)

    # Create actual test data
    create_test_dataset(tmp_path, num_samples=10)

    # Train a simple model (or load pre-trained)
    model = train_minimal_model(config)
    checkpoint_path = tmp_path / "model.ckpt"
    save_checkpoint(model, checkpoint_path)

    # Run REAL evaluation
    config.checkpoint = str(checkpoint_path)
    results = cli.evaluate(config)

    # Validate real results
    assert "metrics" in results
    assert results["metrics"]["mae"] >= 0.0  # Should be finite
    assert results["num_samples"] == 10
    assert results["metrics"]["mae"] < 100.0  # Should be reasonable
```

**Benefits:**
- Tests actual evaluation pipeline
- Catches integration bugs
- Provides real confidence

### ✅ OPTION C: Split into Multiple Focused Tests

```python
def test_evaluate_loads_checkpoint_correctly(tmp_path):
    """Test that evaluate can load model from checkpoint."""
    checkpoint = create_dummy_checkpoint(tmp_path)
    config = {"checkpoint": str(checkpoint), ...}

    model = cli._load_model_from_checkpoint(config)

    assert isinstance(model, torch.nn.Module)
    assert model.eval()  # Should be in eval mode


def test_evaluate_builds_correct_dataloader(tmp_path):
    """Test that evaluate creates test dataloader with correct settings."""
    config = create_test_config(tmp_path)

    dataloader = cli._build_eval_dataloader(config)

    assert len(dataloader) > 0
    batch = next(iter(dataloader))
    assert "x" in batch and "y" in batch
    assert batch["x"].ndim == 3  # [batch, seq, features]


def test_evaluate_formats_output_correctly(capsys):
    """Test that evaluate prints results in expected format."""
    results = {"metrics": {"mae": 1.23}, "num_samples": 100}

    cli._print_evaluation_results(results)

    output = capsys.readouterr().out
    assert "Evaluation Results" in output
    assert "MAE: 1.23" in output
```

**Benefits:**
- Each test has single responsibility
- Easier to debug failures
- Tests are still fast
- Can use minimal mocking where appropriate

---

## Pattern 3: Model Tests with Weak Validation

### ❌ BEFORE (Weak)

```python
def test_softs_forward():
    """Test SOFTS model forward pass."""
    model = SOFTS(input_dim=10, output_dim=10, seq_len=96, pred_len=24, hidden_dim=128)
    x = torch.randn(4, 96, 10)
    output = model(x)

    assert "preds" in output
    assert output["preds"].shape == (4, 24, 10)
```

**Problems:**
- Doesn't check if model actually learned
- Doesn't verify outputs are reasonable
- Could pass even if model always returns zeros

### ✅ AFTER (Strong)

```python
def test_softs_forward_produces_valid_outputs():
    """Test SOFTS model forward pass produces reasonable predictions."""
    model = SOFTS(input_dim=10, output_dim=10, seq_len=96, pred_len=24, hidden_dim=128)
    x = torch.randn(4, 96, 10)

    output = model(x)

    # Basic shape checks
    assert "preds" in output
    assert output["preds"].shape == (4, 24, 10)

    # Validate outputs are numerically valid
    assert torch.isfinite(output["preds"]).all(), "Predictions should not contain NaN or Inf"

    # Validate model actually does computation
    std = output["preds"].std().item()
    assert std > 1e-6, "Model should produce varied outputs, not all zeros/constants"

    # Validate different inputs produce different outputs
    x2 = torch.randn(4, 96, 10)
    output2 = model(x2)
    assert not torch.allclose(output["preds"], output2["preds"], atol=1e-4), \
        "Different inputs should produce different outputs"


def test_softs_forward_is_deterministic():
    """Test that SOFTS produces consistent outputs with same input."""
    torch.manual_seed(42)
    model = SOFTS(input_dim=10, output_dim=10, seq_len=96, pred_len=24, hidden_dim=128)
    model.eval()

    x = torch.randn(4, 96, 10)

    output1 = model(x)
    output2 = model(x)

    torch.testing.assert_close(output1["preds"], output2["preds"],
                               msg="Model should be deterministic in eval mode")


def test_softs_gradients_flow_correctly():
    """Test that SOFTS model can compute gradients."""
    model = SOFTS(input_dim=10, output_dim=10, seq_len=96, pred_len=24, hidden_dim=128)
    model.train()

    x = torch.randn(4, 96, 10, requires_grad=True)
    target = torch.randn(4, 24, 10)

    output = model(x)
    loss = F.mse_loss(output["preds"], target)
    loss.backward()

    # Check gradients exist and are non-zero
    param_with_grad = False
    for param in model.parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            param_with_grad = True
            break

    assert param_with_grad, "At least some parameters should have non-zero gradients"
```

**Benefits:**
- Catches models that produce garbage
- Validates training capability
- Tests numerical stability

---

## Pattern 4: Stub Classes Used Inappropriately

### ❌ BEFORE (Stub in Integration Test)

```python
class _DummyStore:
    def get_full_window(self, flight_id, start_idx, end_idx, column_names):
        # Always returns same synthetic data
        time = np.arange(start_idx, end_idx, dtype=np.float32)[:, None]
        return np.hstack([time, time + 1]).astype(np.float32)

def test_sensor_data_module_builds_datasets_and_loaders(monkeypatch, tmp_path):
    monkeypatch.setattr(datamodule, "DataStore", _DummyStore)
    # ... test claims to validate data loading but uses stub ...
```

**Problem:** Test doesn't validate real data loading.

### ✅ OPTION A: Use Real Data

```python
def test_sensor_data_module_loads_real_parquet_data(tmp_path):
    """Integration test with real data files."""
    # Create real parquet files
    flight_data = pd.DataFrame({
        "timestamp": np.arange(100),
        "sensor_1": np.random.randn(100),
        "sensor_2": np.random.randn(100),
    })
    data_file = tmp_path / "processed" / "flight_001.parquet"
    data_file.parent.mkdir(parents=True)
    flight_data.to_parquet(data_file)

    # Create real index
    index_df = pd.DataFrame({
        "flight_id": ["flight_001"],
        "start_idx": [0],
        "end_idx": [10]
    })
    index_file = tmp_path / "metadata" / "train.parquet"
    index_file.parent.mkdir(parents=True)
    index_df.to_parquet(index_file)

    # Test with real DataStore (no mocking)
    config = {
        "root": str(tmp_path),
        "sensors": {"use": ["sensor_1", "sensor_2"]},
        "window": {"input_len": 5, "pred_len": 2, "stride": 1, "target_sensors": ["sensor_2"]},
        "train_index": "metadata/train.parquet",
    }

    module = SensorDataModule(config, transforms=None, batch_size=2, num_workers=0)
    module.setup()

    train_loader = module.train_dataloader()
    batch = next(iter(train_loader))

    # Verify real data was loaded
    assert batch["x"].shape[-1] == 2  # Two sensors
    assert not torch.isnan(batch["x"]).any(), "Real data should have no NaNs"
```

### ✅ OPTION B: Keep Stub, but Clarify Purpose

```python
def test_sensor_data_module_integration_with_store(tmp_path):
    """Unit test: Verify SensorDataModule correctly calls DataStore API."""
    # This test validates the integration between SensorDataModule and DataStore,
    # not the DataStore implementation itself (which is tested separately)

    class _SpyStore:
        """Spy object that records calls and returns minimal valid data."""
        def __init__(self, data_root, format="parquet"):
            self.calls = []

        def get_full_window(self, flight_id, start_idx, end_idx, column_names):
            self.calls.append({
                "flight_id": flight_id,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "columns": list(column_names)
            })
            # Return minimal valid data
            length = end_idx - start_idx
            data = np.zeros((length, len(column_names)), dtype=np.float32)
            meta = {"flight_id": flight_id}
            return data, meta

    spy = _SpyStore(tmp_path)
    monkeypatch.setattr(datamodule, "DataStore", lambda *args, **kwargs: spy)

    # ... rest of test ...

    # Verify correct API calls were made
    assert len(spy.calls) > 0, "DataStore.get_full_window should be called"
    assert spy.calls[0]["flight_id"] == "flight_1"
    assert spy.calls[0]["columns"] == ["s1", "s2"]
```

**Benefits:**
- Test purpose is clear from name and docstring
- Separates unit tests (API contracts) from integration tests (end-to-end)

---

## Pattern 5: Missing Negative Tests

### ❌ MISSING: Tests for Error Conditions

Most tests only check the "happy path". We need tests that verify proper error handling.

### ✅ ADD: Error Condition Tests

```python
def test_plot_timeseries_rejects_invalid_shapes():
    """Test that plot_timeseries validates input shape."""
    # 1D input (should be 2D)
    with pytest.raises(ValueError, match="Expected 2D array"):
        plot_timeseries(np.array([1, 2, 3]))

    # 3D input
    with pytest.raises(ValueError, match="Expected 2D array"):
        plot_timeseries(np.zeros((10, 5, 3)))


def test_model_forward_rejects_wrong_sequence_length():
    """Test that SOFTS rejects inputs with wrong sequence length."""
    model = SOFTS(input_dim=10, output_dim=10, seq_len=96, pred_len=24, hidden_dim=128)

    # Wrong sequence length
    x_wrong = torch.randn(4, 50, 10)  # Should be 96

    with pytest.raises(RuntimeError, match="Expected sequence length 96"):
        model(x_wrong)


def test_evaluate_fails_gracefully_with_missing_checkpoint():
    """Test that evaluate shows helpful error when checkpoint is missing."""
    config = OmegaConf.create({
        "checkpoint": "/nonexistent/model.ckpt",
        "mode": "eval",
        # ... other config ...
    })

    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        cli.evaluate(config)


def test_data_module_validates_sensor_names():
    """Test that SensorDataModule validates sensor configuration."""
    config = {
        "sensors": {"use": ["nonexistent_sensor"]},
        # ... other config ...
    }

    module = SensorDataModule(config, transforms=None, batch_size=1, num_workers=0)

    with pytest.raises(ValueError, match="Sensors not found in data"):
        module.setup()
```

**Benefits:**
- Ensures errors are caught early with helpful messages
- Prevents cryptic failures in production
- Documents expected behavior

---

## Pattern 6: Test Organization

### ❌ BEFORE: Monolithic Test File

```python
# test_cli.py (300+ lines mixing unit and integration tests)
def test_prepare_hydra_overrides_defaults():
    ...

def test_train_accepts_all_flag_combinations(monkeypatch, ...):
    # Massive integration test with tons of mocking
    ...

def test_missing_data_assets():
    ...
```

### ✅ AFTER: Organized Test Structure

```
tests/
├── unit/
│   ├── test_cli_parsing.py         # Fast, isolated tests
│   ├── test_cli_formatting.py      # Output formatting logic
│   └── test_cli_validation.py      # Config validation
├── integration/
│   ├── test_cli_train_workflow.py  # End-to-end training
│   ├── test_cli_eval_workflow.py   # End-to-end evaluation
│   └── test_cli_export_workflow.py # End-to-end export
└── test_cli.py                     # Keep existing for compatibility
```

```python
# tests/unit/test_cli_parsing.py
"""Fast unit tests for CLI argument parsing logic."""

def test_prepare_hydra_overrides_defaults():
    """Test default override generation."""
    overrides = cli.prepare_hydra_overrides(["train", "model=tcn"])
    assert overrides[0] == "mode=train"
    assert "model=tcn" in overrides


# tests/integration/test_cli_train_workflow.py
"""Integration tests for training workflow (slower, fewer mocks)."""

@pytest.mark.slow
def test_train_completes_full_epoch_with_tiny_model(tmp_path):
    """Integration test: Train a tiny model for one epoch."""
    config = create_minimal_training_config(tmp_path)
    create_tiny_dataset(tmp_path, num_samples=10)

    # Run real training (minimal mocking)
    cli.train(config)

    # Verify training completed
    checkpoint_dir = tmp_path / "runs" / "tiny" / "checkpoints"
    assert (checkpoint_dir / "epoch_0.ckpt").exists()
    assert (checkpoint_dir / "best.ckpt").exists()
```

**Benefits:**
- Fast unit tests run on every commit
- Slow integration tests run nightly or pre-merge
- Clear separation of concerns
- Easier to maintain

---

## Summary: Test Quality Checklist

When writing or reviewing tests, ask:

### ✅ Good Test Checklist

- [ ] **Single responsibility**: Test does ONE thing
- [ ] **Honest naming**: Name describes what is ACTUALLY tested
- [ ] **Strong assertions**: Checks correctness, not just existence
- [ ] **Minimal mocking**: Only mocks external dependencies
- [ ] **Fast execution**: Runs in <100ms (unit) or <5s (integration)
- [ ] **Deterministic**: Always passes/fails consistently
- [ ] **Tests behavior, not implementation**: Would survive refactoring
- [ ] **Includes negative cases**: Tests error conditions
- [ ] **Self-contained**: Doesn't depend on other tests or external state

### ❌ Bad Test Smells

- [ ] **Weak assertion**: `assert x is not None` without checking value
- [ ] **Over-mocking**: Mocks the component under test
- [ ] **Hardcoded stubs**: Stub returns constant, bypassing logic
- [ ] **No assertions**: Test just calls code without verification
- [ ] **Brittle**: Breaks when implementation details change
- [ ] **Slow**: Takes >5s for a unit test
- [ ] **Flaky**: Sometimes passes, sometimes fails
- [ ] **Unclear purpose**: Can't tell what it's testing from name/docstring

---

## Migration Strategy

To fix existing tests without breaking everything:

### Phase 1: Triage (Week 1)
1. Tag problematic tests with `@pytest.mark.weak` or `@pytest.mark.overmocked`
2. Document what each test ACTUALLY validates
3. Identify which components lack integration tests

### Phase 2: Add Strong Tests (Week 2-3)
1. Write new integration tests with minimal mocking
2. Add strong assertions to existing tests
3. Keep old tests temporarily (mark as deprecated)

### Phase 3: Remove Weak Tests (Week 4)
1. Verify new tests provide adequate coverage
2. Delete or rewrite weak tests
3. Update CI to fail on weak assertion patterns

### Phase 4: Prevention (Ongoing)
1. Add pre-commit hooks to flag weak patterns
2. Require test review in PRs
3. Monitor coverage trends
