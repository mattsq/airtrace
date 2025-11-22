# Test Integrity Audit Report

**Date:** 2025-11-22
**Auditor:** AI Agent (Claude)
**Scope:** Complete test suite (43 test files, ~597 test functions)
**Purpose:** Identify tests that may be "reward hacking" - passing without properly validating functionality

---

## Executive Summary

This audit identified **significant test integrity issues** across multiple categories. While many tests are well-written, there are systematic patterns of weak assertions, excessive mocking, and stub implementations that could allow broken code to pass tests.

### Key Findings

| Category | Count | Severity | Files Affected |
|----------|-------|----------|----------------|
| Weak Assertions (`is not None`) | 49 | ⚠️ MEDIUM-HIGH | 14 files |
| Excessive Mocking (>3 components) | 7 tests | 🔴 CRITICAL | 3 files |
| Stub Classes with Hardcoded Returns | 36+ | 🔴 CRITICAL | 10 files |
| CLI Tests Mocking Core Infrastructure | 2 tests | 🔴 CRITICAL | 2 files |

### Overall Assessment

**Test Coverage:** Likely inflated
**Confidence Level:** 🟡 MODERATE (many tests don't validate behavior)
**Risk Level:** 🔴 HIGH (critical paths may not be properly tested)

---

## Category 1: Weak Assertions (⚠️ MEDIUM-HIGH Risk)

### Description
Tests that only check for existence (`is not None`) without validating correctness.

### Statistics
- **Total instances:** 49
- **Files affected:** 14

### Top Offenders

#### 1. `tests/test_viz_plots.py` - 15 instances
**Lines:** 28, 38, 70, 82, 93, 107, 118, 128, 143, 155, 166, 177, 195, 208, 233

**Example (line 26-30):**
```python
def test_plot_timeseries_single_sensor(self):
    """Test plotting single sensor timeseries."""
    data = np.random.randn(100, 1)
    fig = plot_timeseries(data)

    assert fig is not None  # ❌ WEAK: Only checks existence
    assert len(fig.axes) == 1  # ✅ Better, but incomplete
    plt.close(fig)
```

**Issue:** Tests verify that a figure object is returned but don't validate:
- Data was actually plotted
- Axes labels are correct
- Plot styling matches expectations
- Data points are present in the plot

**Impact:** A broken plotting function that returns an empty figure would pass.

**Recommendation:**
```python
# Better assertions:
assert fig is not None
assert len(fig.axes) == 1
axis = fig.axes[0]
lines = axis.get_lines()
assert len(lines) == 1, "Should plot one line for single sensor"
assert len(lines[0].get_data()[1]) == 100, "Should plot all 100 datapoints"
assert axis.get_ylabel() != "", "Should have y-axis label"
```

#### 2. `tests/test_models.py` - 12 instances
Tests check that model outputs exist but don't validate:
- Output shapes match expected dimensions
- Output values are reasonable (not NaN, not all zeros)
- Forward pass produces different outputs for different inputs

#### 3. Model-specific tests - 13 instances across 8 files
Similar issues in:
- `test_tsmixer.py` (4)
- `test_softs.py` (3)
- `test_timesnet.py` (2)
- `test_mambats.py` (2)
- `test_moderntcn.py`, `test_mamba2.py`, `test_frets.py`, `test_fedformer.py`, `test_crossformer.py` (1 each)

---

## Category 2: Excessive Mocking - Critical Infrastructure (🔴 CRITICAL Risk)

### Description
Tests that mock the exact components they claim to test, providing false confidence.

### Top Offenders

#### 1. `tests/test_cli_additional.py::test_evaluate_runs_with_stub_components` (lines 49-123)

**What it claims to test:** The `cli.evaluate()` function
**What it actually mocks:** EVERYTHING the evaluate function depends on

**Mocked components (lines 91-96):**
```python
monkeypatch.setattr("airtrace.data.datamodule.SensorDataModule", _TinyDataModule)
monkeypatch.setattr("airtrace.models.registry.build_model", lambda config, input_dim, output_dim: torch.nn.Linear(input_dim, output_dim))
monkeypatch.setattr("airtrace.tasks.registry.build_task", lambda cfg: _TinyTask())
monkeypatch.setattr("airtrace.evaluation.eval_runner.EvaluationRunner", _TinyEvaluationRunner)
monkeypatch.setattr("airtrace.training.trainer.set_seed", lambda seed: None)
monkeypatch.setattr("airtrace.transforms.registry.build_transforms", lambda pipeline: pipeline)
```

**Stub implementation (lines 82-89):**
```python
class _TinyEvaluationRunner:
    def __init__(self, model, task, test_loader):
        self.model = model
        self.task = task
        self.test_loader = test_loader

    def evaluate(self, return_predictions=False):
        return {"metrics": {"mae": 0.0}, "num_samples": 1}  # ❌ HARDCODED!
```

**What it validates (lines 120-122):**
```python
assert "Evaluation Results" in output
assert "MAE" in output
```

**Critical Issue:** This test only validates that:
1. `cli.evaluate()` can print strings containing "Evaluation Results" and "MAE"
2. It doesn't crash when calling mocked components

**What it DOESN'T test:**
- ❌ Does `EvaluationRunner` actually run evaluation?
- ❌ Are metrics computed correctly?
- ❌ Does model loading work?
- ❌ Does data loading work?
- ❌ Does the evaluation loop execute?

**Real-world scenario where this fails:**
If `EvaluationRunner.evaluate()` has a bug that crashes on real data, this test would still pass because it never calls the real implementation.

**Classification:** ❌ **INVALID TEST** - This is "reward hacking"

---

#### 2. `tests/test_cli.py::test_train_accepts_all_flag_combinations` (lines 181-291)

**What it claims to test:** Training with different CLI flag combinations
**What it actually mocks:** The entire training infrastructure

**Mocked components (lines 255-258):**
```python
monkeypatch.setattr("airtrace.data.datamodule.SensorDataModule", _TinyDataModule)
monkeypatch.setattr("airtrace.models.registry.build_model", _build_model_stub)
monkeypatch.setattr("airtrace.tasks.registry.build_task", lambda _: _TinyTask())
monkeypatch.setattr("airtrace.training.trainer.Trainer", _TinyTrainer)
```

**Stub Trainer (lines 235-244):**
```python
class _TinyTrainer:
    def __init__(self, model, task, config, train_loader, val_loader):
        trainers.append(self)
        self.model = model
        self.checkpoint_dir = Path(config.get("log_dir", "runs/debug")) / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(config.get("log_dir", "runs/debug"))

    def train(self):
        return None  # ❌ Does nothing!
```

**Critical Issue:** The test validates that `cli.train()` can instantiate mocked components, but doesn't verify:
- ❌ Does training actually run?
- ❌ Are model weights updated?
- ❌ Are metrics logged?
- ❌ Are checkpoints saved correctly?
- ❌ Do CLI flags affect behavior?

**Classification:** ❌ **INVALID TEST** - Provides false confidence

---

#### 3. `tests/test_onnx_exporter_additional.py` - Multiple tests mock torch.onnx.export

**Lines 74, 107:** Mock the core ONNX export functionality

```python
def fake_export(model_arg, args, output_file, *_, **__):
    exported_models.append(model_arg)
    Path(output_file).write_bytes(b"onnx")  # ❌ Writes fake file
```

**Issue:** Tests validate that the exporter *calls* torch.onnx.export with correct arguments, but don't verify:
- ❌ Does the ONNX model actually work?
- ❌ Can it be loaded?
- ❌ Does it produce correct outputs?

**Classification:** ⚠️ **PARTIALLY VALID** - Good for unit testing the wrapper logic, but needs integration tests

---

## Category 3: Stub Implementations with Hardcoded Returns (🔴 CRITICAL Risk)

### Statistics
- **Total stub classes:** 36+
- **Files with stubs:** 10

### Analysis by File

#### 1. `tests/test_cli_additional.py` - 4 stubs
- `_TinyDataModule`: Returns hardcoded `torch.zeros(2, 2)`
- `_TinyTask`: Empty class (passes)
- `_TinyEvaluationRunner`: Returns `{"metrics": {"mae": 0.0}}`
- `_FakeExporter`: Records calls but doesn't export

#### 2. `tests/test_cli.py` - 3 stubs
- `_TinyDataModule`: Returns `torch.zeros(2, 2)`
- `_TinyTask`: Returns `torch.tensor(0.0)` for all losses
- `_TinyTrainer`: train() does nothing

#### 3. `tests/test_data_module.py` - 2 stubs
- `_DummyTransform`: Passthrough, doesn't transform
- `_DummyStore`: Returns synthetic data, never reads real files

**Issue:** While stubs are appropriate for isolated unit tests, these are used in tests that claim to validate end-to-end workflows.

---

## Category 4: Test Classification by File

### 🔴 CRITICAL ISSUES (Require immediate attention)

| File | Issue | Classification |
|------|-------|----------------|
| `test_cli_additional.py` | Mocks entire evaluation/export pipeline | ❌ INVALID |
| `test_cli.py` | Mocks entire training pipeline | ❌ INVALID |

### ⚠️ SIGNIFICANT ISSUES (Require review and improvement)

| File | Issue | Impact |
|------|-------|--------|
| `test_viz_plots.py` | 15 weak assertions | Tests don't validate plot contents |
| `test_models.py` | 12 weak assertions | Models might return garbage |
| `test_data_module.py` | Heavy stubbing | Data loading not tested end-to-end |
| `test_onnx_exporter_additional.py` | Mocks core export | ONNX export not validated |

### ✅ GOOD TESTS (Well-written, minimal issues)

| File | Notes |
|------|-------|
| `test_model_registry.py` | Proper unit test with registry isolation |
| `test_transforms_registry.py` | Tests round-trip transformations |
| `test_core_additional.py` | Tests error handling and edge cases |
| `tests/training/test_trainer.py` | Good mix of unit and integration tests |

---

## Verification Tests

To confirm findings, I recommend "break the code" tests:

### Test 1: Break plot_timeseries
```python
# In airtrace/viz/plots.py, change plot_timeseries to:
def plot_timeseries(data, **kwargs):
    fig, axes = plt.subplots(1, 1, figsize=(10, 6))
    # Don't actually plot anything
    return fig
```

**Expected:** Tests should fail
**Actual:** Most tests in `test_viz_plots.py` would still pass ❌

### Test 2: Break EvaluationRunner
```python
# In airtrace/evaluation/eval_runner.py, change evaluate to:
def evaluate(self, return_predictions=False):
    raise RuntimeError("Evaluation is broken!")
```

**Expected:** `test_evaluate_runs_with_stub_components` should fail
**Actual:** Test would still pass because it mocks EvaluationRunner ❌

---

## Detailed Mocking Cross-Reference

### Legitimate Mocking (✅ Appropriate)

| Test | What's Mocked | Why It's OK |
|------|---------------|-------------|
| `test_compat.py` | `importlib.util.find_spec` | Testing import fallbacks |
| `test_cli.py::test_resolve_version` | `metadata.version` | Testing version resolution logic |
| `test_training/test_trainer.py` | `SummaryWriter` | External logging dependency |
| `test_onnx_export.py` | `torch.onnx.export` | Slow external operation |

### Problematic Mocking (❌ Testing the wrong thing)

| Test | What's Mocked | Why It's Problematic |
|------|---------------|----------------------|
| `test_cli_additional.py::test_evaluate_runs_with_stub_components` | `EvaluationRunner` | That's what evaluate() uses! |
| `test_cli.py::test_train_accepts_all_flag_combinations` | `Trainer` | That's what train() uses! |
| `test_data_module.py::test_sensor_data_module_*` | `DataStore`, `read_parquet` | Should test real data loading |

---

## Recommendations

### Priority 1: CRITICAL - Fix Invalid Tests

**Action Items:**
1. **Rewrite or delete `test_cli_additional.py::test_evaluate_runs_with_stub_components`**
   - Option A: Convert to integration test with real components
   - Option B: Delete and rely on integration tests elsewhere
   - Option C: Rename to `test_evaluate_prints_output_format` (what it actually tests)

2. **Rewrite `test_cli.py::test_train_accepts_all_flag_combinations`**
   - Same options as above
   - Current test only validates config parsing, not training

### Priority 2: HIGH - Strengthen Weak Assertions

**Action Items:**
1. **Update `test_viz_plots.py`** - Add assertions that verify:
   ```python
   # For plot tests, add:
   - Number of data points in plot matches input
   - Axes have labels
   - Legend entries match sensor names
   - Plot data matches input data
   ```

2. **Update model tests** - Add assertions for:
   ```python
   # For model tests, add:
   - Output values are finite (not NaN)
   - Output values are non-zero (model actually runs)
   - Different inputs produce different outputs
   ```

### Priority 3: MEDIUM - Add Integration Tests

**Missing Coverage:**
1. End-to-end evaluation with real EvaluationRunner
2. End-to-end training with real Trainer
3. Real data loading without mocks
4. Real ONNX export verification (load and inference)

**Recommended new test file:** `tests/integration/test_workflows.py`

### Priority 4: LOW - Documentation

**Action Items:**
1. Add comments to stub-heavy tests explaining what they DO and DON'T test
2. Create `tests/README.md` explaining test philosophy
3. Add CI check to flag tests with >3 monkeypatch.setattr calls

---

## Test Quality Metrics

### Current State
- **Lines with assertions:** ~597 test functions
- **Weak assertions (`is not None`):** 49 (8.2%)
- **Tests with excessive mocking:** 7+
- **Stub classes:** 36+

### Target State (Proposed)
- **Weak assertions:** <2% (eliminate most)
- **Excessive mocking in integration tests:** 0
- **Stub classes:** Document and justify each one

---

## Appendix A: Complete File-by-File Breakdown

### Tests with Weak Assertions

```
test_viz_plots.py: 15 instances
test_models.py: 12 instances
models/test_tsmixer.py: 4 instances
models/test_softs.py: 3 instances
test_transforms.py: 3 instances
test_onnx_export.py: 2 instances
models/test_timesnet.py: 2 instances
models/test_mambats.py: 2 instances
test_data_module.py: 1 instance
models/test_moderntcn.py: 1 instance
models/test_mamba2.py: 1 instance
models/test_frets.py: 1 instance
models/test_fedformer.py: 1 instance
models/test_crossformer.py: 1 instance
```

### Tests with Excessive Mocking

```
test_cli_additional.py: 7 monkeypatch.setattr calls
test_cli.py: 6 monkeypatch.setattr calls
test_compat.py: 5 monkeypatch.setattr calls (legitimate)
test_data_module.py: 4 monkeypatch.setattr calls
test_onnx_exporter_additional.py: 2 monkeypatch.setattr calls
data/test_loaders.py: 2 monkeypatch.setattr calls
training/test_trainer.py: 1 monkeypatch.setattr call (legitimate)
```

### Files with Stub Classes

```
test_onnx_export.py: 4 stubs
test_evaluation.py: 4 stubs
test_core_additional.py: 4 stubs
test_cli_additional.py: 4 stubs
training/test_trainer.py: 3 stubs
test_cli.py: 3 stubs
test_data_module.py: 2 stubs
data/test_dataset.py: 2 stubs
training/test_callbacks.py: 1 stub
test_tasks.py: 1 stub
```

---

## Appendix B: Automated Audit Scripts

For future audits, use these commands:

```bash
# Find weak assertions
grep -rn "assert.*is not None" tests/ | wc -l

# Find tests with no assertions (requires Python script)
# (Script not implemented due to complexity)

# Find excessive mocking
grep -rn "monkeypatch\.setattr" tests/ | cut -d: -f1 | sort | uniq -c | sort -rn

# Find stub classes
grep -rn "class _.*\|class Fake\|class Stub\|class Dummy" tests/ | grep -v "MagicMock"
```

---

## Conclusion

This audit identified **systematic test quality issues** that likely inflate test coverage metrics and provide false confidence. The most critical issues are:

1. **Two CLI tests that mock their core dependencies** - These should be rewritten or deleted
2. **49 weak assertions** - Most in visualization tests that don't validate plot contents
3. **36+ stub classes** - Some used inappropriately in integration-style tests

**Estimated effort to remediate:**
- Priority 1 (CRITICAL): 4-6 hours
- Priority 2 (HIGH): 8-12 hours
- Priority 3 (MEDIUM): 4-6 hours
- **Total: 16-24 hours**

**Next Steps:**
1. Review this report with team
2. Create GitHub issues for each priority tier
3. Implement fixes incrementally
4. Add CI checks to prevent regression
