# Q400 Integration Report & DataStore Bug Fix Implementation Guide

**Date:** November 17, 2025
**Status:** Critical bugs discovered, fixes documented
**Priority:** P0 - Blocking model zoo testing

---

## Executive Summary

During systematic integration of the Q400 turboprop dataset (25.9M timesteps, 6,127 flights), we discovered a **critical bug in the DataStore implementation** that prevents the framework from loading data correctly. This document provides:

1. Complete analysis of the bug and exact fix
2. Comprehensive dataset integration improvement plan
3. Lessons learned from Q400 integration experience
4. Implementation roadmap for the next agent

**Current Status:** Q400 data is fully prepared (1.6M windows generated), but testing is blocked by the DataStore bug.

---

## Part 1: The DataStore Bug

### Location
**File:** `src/airtrace/data/dataset.py`
**Lines:** 137-143
**Severity:** P0 - Completely blocks data loading

### The Bug

```python
# CURRENT BROKEN CODE (Lines 137-143)
# Extract window
window_data = flight_data[start_idx:end_idx]

# Split into input and target based on window spec
# This is a simplified version - real implementation would use WindowSpec
input_len = end_idx - start_idx  # Placeholder
x = window_data[:input_len, [flight_data.columns.get_loc(s) for s in sensor_names]]
y = window_data[input_len:, [flight_data.columns.get_loc(s) for s in target_sensors]]
```

**Problem:** This code attempts NumPy-style 2D indexing `df[rows, cols]` on a pandas DataFrame, which is invalid.

**Error Raised:**
```
pandas.errors.InvalidIndexError: (slice(None, np.int64(144), None), [1, 3, 4, 5])
```

### Root Cause Analysis

1. **Mixed Operations:** `window_data` is a DataFrame (from line 137 slice), but code treats it like a NumPy array
2. **Invalid Pandas Syntax:** Pandas doesn't support `df[row_slice, col_list]` indexing
3. **Missing WindowSpec Integration:** The method doesn't know how to split input/prediction portions correctly
4. **Hardcoded Logic:** `input_len = end_idx - start_idx` is wrong - it assumes entire window is input!

### The Fix (Option 1: Minimal Change)

```python
def get_window(
    self,
    flight_id: str,
    start_idx: int,
    end_idx: int,
    sensor_names: List[str],
    target_sensors: List[str]
) -> tuple:
    """Get a window of data.

    Args:
        flight_id: Flight identifier
        start_idx: Start index of window
        end_idx: End index of window (inclusive of prediction portion)
        sensor_names: Sensor names for inputs
        target_sensors: Sensor names for targets

    Returns:
        Tuple of (x, y, meta) where:
            x: Input array [T_in, D_in]
            y: Target array [T_out, D_out]
            meta: Metadata dict
    """
    # Load flight data (with caching)
    if flight_id not in self._cache:
        self._cache[flight_id] = self._load_flight(flight_id)

    flight_data = self._cache[flight_id]

    # Validate sensors exist
    missing_input = set(sensor_names) - set(flight_data.columns)
    missing_target = set(target_sensors) - set(flight_data.columns)
    if missing_input or missing_target:
        raise ValueError(
            f"Missing sensors in {flight_id}: "
            f"inputs={missing_input}, targets={missing_target}"
        )

    # Extract window using iloc for proper positional indexing
    window_data = flight_data.iloc[start_idx:end_idx]

    # FIXME: This needs proper WindowSpec integration!
    # The DataStore doesn't know input_len vs pred_len
    # For now, use a heuristic: 80% input, 20% prediction
    window_len = end_idx - start_idx
    input_len = int(window_len * 0.8)  # TEMPORARY HACK

    # Select columns and convert to numpy
    x = window_data.iloc[:input_len][sensor_names].values
    y = window_data.iloc[input_len:][target_sensors].values

    meta = {
        "flight_id": flight_id,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "window_len": window_len
    }

    return x, y, meta
```

**Status:** This fixes the immediate bug but uses a hack for window splitting.

### The Fix (Option 2: Proper Architecture - RECOMMENDED)

**Part A: Update DataStore to return full window**

```python
# In DataStore class
def get_full_window(
    self,
    flight_id: str,
    start_idx: int,
    end_idx: int,
    column_names: List[str]
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Get a complete window without splitting.

    Returns the full window [start_idx:end_idx] with specified columns.
    The caller is responsible for splitting into input/target portions.
    """
    # Load flight data (with caching)
    if flight_id not in self._cache:
        self._cache[flight_id] = self._load_flight(flight_id)

    flight_data = self._cache[flight_id]

    # Validate columns
    missing = set(column_names) - set(flight_data.columns)
    if missing:
        raise ValueError(f"Missing columns in {flight_id}: {missing}")

    # Extract window using iloc
    window_data = flight_data.iloc[start_idx:end_idx]

    # Select columns and convert to numpy
    window_array = window_data[column_names].values

    meta = {
        "flight_id": flight_id,
        "start_idx": start_idx,
        "end_idx": end_idx
    }

    return window_array, meta
```

**Part B: Update SensorWindowDataset to handle splitting**

```python
# In SensorWindowDataset.__getitem__ method (around line 62-71)
def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
    """Get a single window.

    Args:
        idx: Index of window

    Returns:
        Dictionary with 'x', 'y', 'meta' keys
    """
    row = self.index_df.iloc[idx]

    # Get all columns we need
    all_columns = list(set(self.sensor_names) | set(self.target_sensors))

    # Get full window from data store
    window_data, meta = self.data_store.get_full_window(
        flight_id=row.flight_id,
        start_idx=row.start_idx,
        end_idx=row.end_idx,
        column_names=all_columns
    )

    # Dataset knows the WindowSpec, so it handles the split
    input_len = self.window_spec.input_len

    # Get column indices
    sensor_indices = [all_columns.index(s) for s in self.sensor_names]
    target_indices = [all_columns.index(s) for s in self.target_sensors]

    # Split window into input and target portions
    x = window_data[:input_len, sensor_indices]
    y = window_data[input_len:, target_indices]

    # Apply transforms
    if self.transforms is not None:
        x, y, meta = self.transforms(x, y, meta)

    # Convert to tensors
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float()

    return {
        "x": x,
        "y": y,
        "meta": meta
    }
```

**Why Option 2 is better:**
- **Cleaner separation of concerns:** DataStore handles storage, Dataset handles windowing logic
- **Correct architecture:** Dataset already has WindowSpec, so it knows the correct input_len
- **More maintainable:** No hacks or hardcoded assumptions
- **Better testability:** Each component has clear responsibilities

---

## Part 2: Q400 Integration Experience

### What We Built

1. **Data Configuration** (`configs/data/q400.yaml`)
   - Sensor mapping: 6 raw columns → 4 selected sensors
   - Window params: input_len=128, pred_len=16, stride=16
   - Data quality notes: documented 50% missing OAT, 6.4% missing weight

2. **Data Preparation Pipeline**
   - **Step 1:** Load and explore (`prepare_q400_data.py`)
   - **Step 2:** Create train/val/test splits (70/15/15, flight-level)
   - **Step 3:** Apply imputation and cleaning
   - **Step 4:** Process into individual flight files (`process_q400_flights.py`)
   - **Step 5:** Generate window indices (`create_q400_window_indices.py`)

3. **Results**
   - 6,127 flights → 6,127 individual parquet files (711 MB)
   - 1,566,641 windows total (Train: 1.1M, Val: 234K, Test: 235K)
   - Complete metadata with statistics

### Pain Points Discovered

#### 1. **Undocumented Multi-Step Process**
**Problem:** Required 5 separate manual scripts with no unified workflow
**Impact:** Took 2+ days of debugging to figure out the process
**Solution:** Create `prepare_dataset_template.py` unified script

#### 2. **Two-Stage Index Creation**
**Problem:** Indices start as flight IDs, then get **overwritten** with window indices
**Impact:** Confusing - not documented anywhere
**Solution:** Document this pattern, add validation

#### 3. **Processed Directory Requirement**
**Problem:** Framework expects `data/processed/{flight_id}.parquet` files but doesn't document this
**Impact:** Silent failures, confusing errors
**Solution:** Add validation script that checks this

#### 4. **No Column Validation**
**Problem:** Sensor names in config aren't validated against actual data until runtime
**Impact:** Late error discovery during training
**Solution:** Validate in prepare step

#### 5. **Missing WindowSpec in DataStore**
**Problem:** DataStore can't properly split windows (the bug we found!)
**Impact:** Complete blocker
**Solution:** Architectural fix (Option 2 above)

### Q400 Dataset Characteristics

```
Dataset: Q400 Turboprop Aircraft
Source: Flight recorder data
Total Records: 25,893,217 timesteps
Unique Flights: 6,127
Flight Length: 360 to 30,936 timesteps (avg: 4,226)

Sensors:
  ✓ fuel_flow: 0.02% missing
  ✓ airspeed_true: 0.02% missing
  ✓ torque: 0.02% missing
  ✓ altitude: 0.02% missing
  ⚠ weight: 6.41% missing (excluded)
  ✗ oat: 50.01% missing (excluded)

Windows Generated:
  Train: 1,096,734 windows
  Val: 234,459 windows
  Test: 235,448 windows
  Total: 1,566,641 windows
```

---

## Part 3: Implementation Plan for Next Agent

### Phase 1: Fix the DataStore Bug (CRITICAL - Do This First!)

**Estimated Time:** 2-3 hours

**Task 1.1: Implement Option 2 Fix**
1. Read `src/airtrace/data/dataset.py`
2. Add new method `DataStore.get_full_window()` (see code above)
3. Update `SensorWindowDataset.__getitem__()` to use new method
4. Update `SensorWindowDataset.__init__()` to store window_spec reference

**Task 1.2: Add Tests**
Create `tests/data/test_datastore_fix.py`:
```python
def test_datastore_get_full_window():
    """Test DataStore returns correct window data."""
    # Create synthetic flight data
    # Call get_full_window
    # Assert correct shape and values

def test_dataset_window_splitting():
    """Test Dataset correctly splits windows."""
    # Create mock DataStore
    # Create Dataset with WindowSpec
    # Assert x has shape (input_len, n_sensors)
    # Assert y has shape (pred_len, n_targets)

def test_missing_sensor_validation():
    """Test validation of missing sensors."""
    # Try to load non-existent sensor
    # Assert ValueError is raised with helpful message
```

**Task 1.3: Verify Backward Compatibility**
Run existing tests to ensure qantas_737 and synthetic datasets still work:
```bash
.venv/Scripts/python -m pytest tests/data/ -v
```

**Task 1.4: Test with Q400**
```bash
.venv/Scripts/python src/scripts/quick_q400_validation.py
```

**Expected Outcome:** All 3 validation models (persistence, linear_ar, gru_ar) should pass!

---

### Phase 2: Create Dataset Integration Tools

**Estimated Time:** 6-8 hours

**Task 2.1: Create Unified Preparation Script**

**File:** `src/scripts/prepare_dataset_template.py`

**Features:**
- CLI interface with argparse
- Loads raw data (CSV, Parquet, or custom loader)
- Analyzes data quality (missing values, distributions)
- Creates train/val/test splits
- Applies imputation strategies
- Processes to individual flight files
- Generates window indices
- Creates config template
- Validates result

**Usage:**
```bash
python src/scripts/prepare_dataset_template.py \
  --name mydata \
  --raw-file data/raw/mydata.parquet \
  --flight-id-col flight_id \
  --time-col timestamp \
  --sensors "sensor1,sensor2,sensor3" \
  --target sensor1 \
  --input-len 128 \
  --pred-len 16 \
  --stride 16 \
  --split 0.7 0.15 0.15
```

**Output:**
```
data/
├── interim/mydata_cleaned.parquet
├── processed/{flight_id}.parquet (many files)
└── metadata/
    ├── mydata_train_index.parquet
    ├── mydata_val_index.parquet
    ├── mydata_test_index.parquet
    └── mydata_stats.csv

configs/data/mydata.yaml (generated template)
```

**Task 2.2: Create Validation Script**

**File:** `src/scripts/validate_dataset.py`

**Checks:**
1. Config file exists and is valid YAML
2. All paths in config exist
3. Index files have correct schema: `(flight_id, start_idx, end_idx)`
4. All flight_ids in indices exist in `processed/`
5. All sensor names in config exist in processed files
6. Window parameters are valid (input_len + pred_len matches indices)
7. No critical missing data in selected sensors

**Usage:**
```bash
python src/scripts/validate_dataset.py --config configs/data/q400.yaml
```

**Output:**
```
Dataset Validation Report: q400
================================================================================

✓ Config file valid
✓ All paths exist
✓ Index files have correct schema
✓ All 6,127 flight files exist
✓ Sensors validated: fuel_flow, airspeed_true, torque, altitude
✓ Window configuration valid
✓ No critical missing data

Total Windows:
  Train: 1,096,734
  Val: 234,459
  Test: 235,448

Dataset is ready for training!
```

**Task 2.3: Create Inspection Tool**

**File:** `src/scripts/inspect_dataset.py`

Quick dataset info and statistics.

---

### Phase 3: Documentation

**Estimated Time:** 4-5 hours

**Task 3.1: Create Comprehensive Integration Guide**

**File:** `docs/dataset_integration_guide.md`

**Outline:**
```markdown
# AirTrace Dataset Integration Guide

## Overview
- What is a "dataset" in AirTrace?
- Required file structure
- Integration workflow

## Quick Start
Using the template script [5 min example]

## Step-by-Step Manual Integration
1. Prepare raw data
2. Create splits
3. Process flights
4. Generate windows
5. Create config
6. Validate

## Index File Specification
Schema, examples, generation

## WindowSpec Explained
How windowing works, parameters

## Common Pitfalls
- Two-stage index creation
- Processed directory requirement
- Sensor name mismatches
- Window parameter conflicts

## Troubleshooting
Error messages and solutions

## Advanced Topics
- Custom imputation strategies
- Multi-target prediction
- Variable-length flights
- Stratified splits

## Reference
- Complete file structure
- Config schema
- API documentation
```

**Task 3.2: Document Q400 Lessons Learned**

**File:** `docs/q400_integration_lessons.md`

- Full Q400 integration story
- Bugs discovered
- Process improvements
- Performance notes

**Task 3.3: Update MEMORY.md**

Add entry:
```markdown
## DataStore Window Splitting Bug (Nov 2025)

**Issue:** DataStore.get_window() used invalid pandas indexing, couldn't load data.

**Root Cause:** Mixed pandas DataFrame operations with NumPy array assumptions.

**Fix:** Moved window splitting logic from DataStore to Dataset. Dataset has WindowSpec
and knows correct input_len/pred_len split.

**Files Changed:**
- src/airtrace/data/dataset.py (DataStore and SensorWindowDataset)

**Lesson:** DataStore should handle storage only. Window-specific logic belongs in Dataset.

**Q400 Integration:** Requires 5-step process. Created prepare_dataset_template.py to automate.

**Index Format:** Window indices must have (flight_id, start_idx, end_idx) schema.
```

---

### Phase 4: Complete Q400 Testing

**Estimated Time:** 4-6 hours

**After bug fixes are complete:**

**Task 4.1: Run Quick Validation**
```bash
.venv/Scripts/python src/scripts/quick_q400_validation.py
```

**Expected:** 3/3 models successful

**Task 4.2: Run Baseline Model Tier**
```bash
.venv/Scripts/python src/scripts/run_q400_model_zoo_tests.py --tier baseline
```

**Expected:** 15 baseline models tested, results in `results/q400_model_zoo/`

**Task 4.3: Run Simple Trainable Tier**
```bash
.venv/Scripts/python src/scripts/run_q400_model_zoo_tests.py --tier simple
```

**Expected:** 6 models with 3-5 epoch training

**Task 4.4: Generate Report**
Create `docs/q400_model_zoo_results.md` with:
- Performance comparison table
- Best models by RMSE/MAE
- Training time analysis
- Framework issues encountered
- Recommendations

---

## Part 4: Testing Checklist

Before considering this work complete, verify:

### DataStore Fix
- [ ] get_full_window() method added to DataStore
- [ ] SensorWindowDataset updated to use new method
- [ ] Window splitting uses WindowSpec.input_len correctly
- [ ] Sensor validation raises helpful errors
- [ ] Unit tests pass
- [ ] Q400 quick validation passes (3/3 models)

### Dataset Integration Tools
- [ ] prepare_dataset_template.py created and tested
- [ ] validate_dataset.py created and tested
- [ ] inspect_dataset.py created
- [ ] Q400 can be recreated using template script
- [ ] Synthetic dataset works with new tools

### Documentation
- [ ] dataset_integration_guide.md complete
- [ ] q400_integration_lessons.md complete
- [ ] MEMORY.md updated
- [ ] README.md updated with dataset integration section
- [ ] All code examples in docs are tested

### Q400 Model Zoo Testing
- [ ] Baseline models (15) tested
- [ ] Simple trainable models (6) tested
- [ ] Results documented
- [ ] Issues logged in MEMORY.md

---

## Part 5: File Reference

### Files to Modify

1. `src/airtrace/data/dataset.py`
   - Add `DataStore.get_full_window()`
   - Update `SensorWindowDataset.__getitem__()`

2. `tests/data/test_dataset.py`
   - Add tests for DataStore fix
   - Add tests for window splitting

3. `docs/MEMORY.md`
   - Add DataStore bug entry
   - Add Q400 integration notes

### Files to Create

1. `src/scripts/prepare_dataset_template.py` - Unified preparation tool
2. `src/scripts/validate_dataset.py` - Dataset validation
3. `src/scripts/inspect_dataset.py` - Quick inspection tool
4. `docs/dataset_integration_guide.md` - Complete integration guide
5. `docs/q400_integration_lessons.md` - Lessons learned report
6. `docs/q400_model_zoo_results.md` - Testing results (after completion)

### Files Already Created (Q400 Integration)

- `configs/data/q400.yaml` ✓
- `data/metadata/q400_{train,val,test}_index.parquet` ✓
- `data/interim/q400_cleaned.parquet` ✓
- `data/processed/{flight_id}.parquet` (6,127 files) ✓
- `src/scripts/prepare_q400_data.py` ✓
- `src/scripts/process_q400_flights.py` ✓
- `src/scripts/create_q400_window_indices.py` ✓
- `notebooks/q400_data_analysis.ipynb` ✓

---

## Part 6: Code Snippets for Quick Copy-Paste

### Complete DataStore.get_full_window() Implementation

```python
def get_full_window(
    self,
    flight_id: str,
    start_idx: int,
    end_idx: int,
    column_names: List[str]
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Get a complete window without splitting.

    Returns the full window [start_idx:end_idx] with specified columns.
    The caller is responsible for splitting into input/target portions.

    Args:
        flight_id: Flight identifier
        start_idx: Start index of window
        end_idx: End index of window
        column_names: List of column names to include

    Returns:
        Tuple of (window_array, meta) where:
            window_array: NumPy array [window_len, n_columns]
            meta: Metadata dictionary

    Raises:
        ValueError: If columns are missing from flight data
    """
    # Load flight data (with caching)
    if flight_id not in self._cache:
        self._cache[flight_id] = self._load_flight(flight_id)

    flight_data = self._cache[flight_id]

    # Validate columns exist
    missing = set(column_names) - set(flight_data.columns)
    if missing:
        available = list(flight_data.columns)
        raise ValueError(
            f"Missing columns in flight {flight_id}: {missing}\n"
            f"Available columns: {available}"
        )

    # Extract window using iloc for proper positional indexing
    window_data = flight_data.iloc[start_idx:end_idx]

    # Select columns and convert to numpy
    window_array = window_data[column_names].values

    meta = {
        "flight_id": flight_id,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "window_len": end_idx - start_idx
    }

    return window_array, meta
```

### Complete SensorWindowDataset.__getitem__() Update

```python
def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
    """Get a single window.

    Args:
        idx: Index of window

    Returns:
        Dictionary with 'x', 'y', 'meta' keys
    """
    row = self.index_df.iloc[idx]

    # Get all unique columns we need (inputs + targets)
    all_columns = list(set(self.sensor_names) | set(self.target_sensors))

    # Get full window from data store (no splitting yet)
    window_data, meta = self.data_store.get_full_window(
        flight_id=row.flight_id,
        start_idx=row.start_idx,
        end_idx=row.end_idx,
        column_names=all_columns
    )

    # Get column indices for splitting into input/target sensors
    sensor_indices = [all_columns.index(s) for s in self.sensor_names]
    target_indices = [all_columns.index(s) for s in self.target_sensors]

    # Split window into input and target portions using WindowSpec
    input_len = self.window_spec.input_len

    x = window_data[:input_len, sensor_indices]
    y = window_data[input_len:, target_indices]

    # Apply transforms
    if self.transforms is not None:
        x, y, meta = self.transforms(x, y, meta)

    # Convert to tensors
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float()

    return {
        "x": x,
        "y": y,
        "meta": meta
    }
```

---

## Part 7: Expected Outcomes

### After Phase 1 (Bug Fix)
- Q400 quick validation: 3/3 models successful
- No more InvalidIndexError
- Proper window splitting with correct input_len/pred_len

### After Phase 2 (Tools)
- New dataset integration time: <2 hours (down from 2+ days)
- Automated validation catches errors early
- Clear error messages guide users to fixes

### After Phase 3 (Documentation)
- New users can integrate datasets independently
- Common pitfalls are documented
- Troubleshooting guide reduces support burden

### After Phase 4 (Q400 Testing)
- Baseline performance benchmarks established
- Framework stress-tested with real data
- Model zoo validation complete
- Performance recommendations available

---

## Part 8: Known Limitations and Future Work

### Current Limitations
1. **Fixed window sizes:** All flights use same input_len/pred_len
2. **Single target prediction:** Multi-task not fully tested
3. **No dynamic window sizing:** Short flights get excluded
4. **Limited imputation strategies:** Only forward/backward fill

### Future Improvements
1. **Adaptive windowing:** Support variable-length sequences with padding/masking
2. **Online data loading:** Stream from database instead of files
3. **Data augmentation:** Time warping, noise injection
4. **Distributed processing:** Parallelize flight processing
5. **Cloud storage:** S3/GCS support for processed files

---

## Conclusion

The Q400 integration was **invaluable** for discovering critical bugs and validating the framework's design. The DataStore bug would have affected any new dataset integration.

**Priority order for next agent:**
1. Fix DataStore bug (2-3 hours) - CRITICAL
2. Test with Q400 (30 min) - Validate fix
3. Create validation script (2-3 hours) - Prevent future issues
4. Document learnings (2-3 hours) - Knowledge transfer
5. Create preparation template (4-5 hours) - Improve DX
6. Complete Q400 testing (4-6 hours) - Original goal

**Total estimated effort:** 15-20 hours over 2-3 days

This systematic testing approach successfully validated the framework's stated interface and discovered real issues that would have impacted all users. The fixes and tools created here will significantly improve the developer experience for anyone integrating new datasets.

---

**End of Document**

**For the next agent:** Start with Phase 1, Task 1.1. The exact code is provided above. Test thoroughly before moving to Phase 2. Good luck! 🚀
