# Test Integrity Audit Report

**Date:** 2025-11-22
**Auditor:** AI Agent (Claude)
**Scope:** Complete test suite (43 test files, ~597 test functions)
**Purpose:** Identify tests that may be "reward hacking" - passing without properly validating functionality

---

## Executive Summary

This audit originally identified **significant test integrity issues** across multiple categories. After targeted remediation, the suite now relies on real components, stronger assertions, and integration coverage; the historical findings below are retained for traceability.

### Key Findings (Current Snapshot)

| Category | Count | Severity | Notes |
|----------|-------|----------|-------|
| Weak Assertions (`is not None`) | **0** (was 49) | 🟢 LOW | Eliminated across the suite; audit script reports no matches. |
| Excessive Mocking (>3 components) | **0** tests over threshold | 🟡 LOW | Remaining mocks are limited to compatibility shims and I/O isolation. |
| Stub Classes with Hardcoded Returns | **Documented only** | 🟡 LOW | Legacy references remain for historical context; active tests use real components. |
| CLI Tests Mocking Core Infrastructure | **0** | 🟢 LOW | CLI coverage now exercises real data-check, dry-run, and ONNX export flows. |

### Overall Assessment

**Test Coverage:** Strengthened by integration workflows and real data-backed fixtures
**Confidence Level:** 🟢 HIGH (core behaviors exercised with real components)
**Risk Level:** 🟡 LOW-MODERATE (documentation/CI guardrails still pending)

### Remediation Progress (2027-02-04)

- ✅ Strengthened visualization assertions in `tests/test_viz_plots.py` to verify plotted data, labels, and guides instead of only checking for figure existence.
- ✅ Replaced over-mocked CLI evaluation/training coverage with focused checks that exercise real CLI formatting and data-check/dry-run flows.
- ✅ Converted `tests/test_cli.py::test_train_accepts_all_flag_combinations` into real data-check and minimal training flows that use the true `SensorDataModule`, `Trainer`, and baseline models instead of stubs.
- ✅ Reinforced transform and model tests to check learned statistics, gradient magnitudes, and wrapper contents rather than only verifying objects exist.
- ✅ Fixed Informer attention reporting, SOFTS normalization statistics, and LoRA adapter initialization to ensure gradient/shape validations in hardened tests reflect true model behavior.
- ✅ Hardened transform coverage for clipping and smoothing by asserting fitted bounds, Gaussian filter outputs, and metadata, eliminating weak "ran without error" assertions in `tests/test_transforms.py`.
- ✅ (2026-05-07) Eliminated remaining `assert ... is not None` checks across visualization and model gradient tests, and updated `scripts/audit_tests.sh` to tolerate zero-match cases so the audit can report a clean slate.
- ✅ (2026-11-23) Replaced the stubbed ONNX export CLI test with a real end-to-end export that loads a persisted checkpoint, runs `onnxruntime` inference, and verifies metadata, and fixed `ONNXExporter.from_checkpoint` to load Hydra configs under PyTorch 2.6's `weights_only=True` default.
- ✅ (2026-11-27) Converted SensorDataModule coverage to use real parquet-backed datasets instead of monkeypatched stores and added `tests/integration/test_workflows.py` to exercise training, checkpointing, and evaluation with actual loaders and models.
- ✅ (2027-02-04) Verified the current suite with `scripts/audit_tests.sh --detailed`: no weak-assertion matches and no tests exceed the mocking threshold; residual monkeypatch usage is limited to compatibility shims.

---

## Category 1: Weak Assertions (Resolved)

### Description
Tests that only check for existence (`is not None`) without validating correctness.

### Statistics
- **Total instances:** **0** (previously 49)
- **Files affected:** 0
- **Current status (2027-02-04):** Automated audit reports no remaining weak assertions. The historical notes below are retained to explain what was fixed and how.

### Historical Offenders (now remediated)

#### 1. `tests/test_viz_plots.py` (previously 15 instances)
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

**Status:** Implemented; the current tests validate plotted content, labels, and data counts.

#### 2. `tests/test_models.py` (previously 12 instances)
Original tests checked that model outputs existed but didn't validate:
- Output shapes match expected dimensions
- Output values are reasonable (not NaN, not all zeros)
- Forward pass produces different outputs for different inputs

#### 3. Model-specific tests (previously 13 instances across 8 files)
Similar issues appeared in:
- `test_tsmixer.py` (4)
- `test_softs.py` (3)
- `test_timesnet.py` (2)
- `test_mambats.py` (2)
- `test_moderntcn.py`, `test_mamba2.py`, `test_frets.py`, `test_fedformer.py`, `test_crossformer.py` (1 each)

---

## Category 2: Excessive Mocking (Stabilized)

### Description
Tests that mock the exact components they claim to test, providing false confidence. The latest audit shows no tests exceeding the monkeypatch threshold; remaining mocks are constrained to compatibility layers.

### Historical Offenders (now remediated)

#### 1. `tests/test_cli_additional.py::test_evaluate_runs_with_stub_components` (lines 49-123, replaced)

**What it claimed to test:** The `cli.evaluate()` function
**What it actually mocked:** EVERYTHING the evaluate function depended on

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

**Resolution:** Replaced with a real ONNX export and evaluation flow that loads an actual checkpoint, runs `onnxruntime`, and validates metadata and outputs.

---

#### 2. `tests/test_cli.py::test_train_accepts_all_flag_combinations` (lines 181-291, replaced)

**What it claimed to test:** Training with different CLI flag combinations
**What it actually mocked:** The entire training infrastructure

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

**Resolution:** Replaced with training flows that perform real data checks, minimal training steps, and checkpoint handling using actual data modules, trainers, and baseline models.

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

### Priority 1: CRITICAL - Fix Invalid Tests (Completed)

**Status:** ✅ Completed via real CLI export, evaluation, and training flows that no longer mock core infrastructure.

**Action Items:**
1. **Rewrite or delete `test_cli_additional.py::test_evaluate_runs_with_stub_components`**
   - **Done:** Replaced with full ONNX export + `onnxruntime` inference validation.

2. **Rewrite `test_cli.py::test_train_accepts_all_flag_combinations`**
   - **Done:** Replaced with data-check and minimal training flows backed by true data modules and trainers.

### Priority 2: HIGH - Strengthen Weak Assertions (Completed)

**Status:** ✅ Completed. Visualization and model tests now validate plotted content, tensor shapes, finiteness, and meaningful gradients/outputs.

**Action Items:**
1. **Update `test_viz_plots.py`**
   - **Done:** Assertions cover plotted lines, labels, and data counts.

2. **Update model tests**
   - **Done:** Assertions check finiteness, non-trivial outputs, and sensitivity to input changes.

### Priority 3: MEDIUM - Add Integration Tests (Completed)

**Status:** ✅ Completed with `tests/integration/test_workflows.py`, CLI training/evaluation coverage, and real ONNX export validation.

**Coverage now present:**
1. End-to-end evaluation with real `EvaluationRunner`
2. End-to-end training with real `Trainer`
3. Real data loading without mocks (parquet-backed)
4. Real ONNX export verification (load and inference)

### Priority 4: LOW - Documentation (In Progress)

**Status:** 🟡 In progress. Added `tests/README.md` summarizing philosophy and mocking guidelines; CI hook for monkeypatch counts remains a future enhancement.

**Action Items:**
1. Add comments to stub-heavy tests explaining what they DO and DON'T test
2. Create `tests/README.md` explaining test philosophy  
   - **Done:** Added guidance on assertions, mocking boundaries, and integration expectations.
3. Add CI check to flag tests with >3 monkeypatch.setattr calls  
   - **Pending:** Keep monitoring via `scripts/audit_tests.sh --detailed` until workflow hook is added.

---

## Test Quality Metrics

### Current State
- **Lines with assertions:** ~597 test functions
- **Weak assertions (`is not None`):** **0**
- **Tests with excessive mocking (>5 monkeypatches):** **0** (13 total monkeypatches concentrated in compatibility coverage)
- **Stub classes bypassing logic:** **0 active** (remaining stubs are documented fixtures only)

### Target State (Proposed)
- **Weak assertions:** <2% (eliminate most)
- **Excessive mocking in integration tests:** 0
- **Stub classes:** Document and justify each one

---

## Appendix A: Complete File-by-File Breakdown

### Tests with Weak Assertions (Historical)

```
The 2025 audit found 49 weak assertions across visualization and model tests. All have since been rewritten to validate real outputs.
```

### Tests with Excessive Mocking (Historical)

```
The 2025 audit flagged over-mocking in CLI and data module tests. These tests now exercise real data modules, trainers, and exporters; remaining monkeypatch usage is limited to compatibility shims.
```

### Files with Stub Classes (Historical)

```
Stub helpers remain documented for fixture setup, but no active tests rely on hardcoded return values to pass.
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

The audit originally identified **systematic test quality issues** that inflated confidence. All critical and high-priority items have been remediated: CLI tests now exercise real data and export flows, weak assertions have been eliminated, and integration coverage is in place. Remaining work is limited to documentation and optional CI enforcement of monkeypatch thresholds.

**Updated posture (2027-02-04):**
- Critical issues: ✅ Resolved
- High-priority issues: ✅ Resolved
- Medium-priority issues: ✅ Resolved
- Low-priority issues: 🟡 Documentation/CI polish

**Next Steps:**
1. Add inline comments to any remaining stub fixtures clarifying scope and limitations.
2. Add a CI check (or extend `audit_tests.sh`) to fail when tests exceed mock thresholds.
3. Continue running `./scripts/audit_tests.sh --detailed` to guard against regressions.
