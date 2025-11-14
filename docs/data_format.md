# Data Format

This document describes the data format and pipeline for AirTrace.

## Data Pipeline Overview

AirTrace uses a three-stage data pipeline:

```
Raw → Interim → Processed
```

### 1. Raw Data (`data/raw/`)

Original flight logs as Parquet or CSV files.

**Format:**
- One file per flight: `{flight_id}.parquet` or `{flight_id}.csv`
- Columns:
  - `timestamp`: Datetime or sample index
  - Sensor columns (wide format) OR
  - `sensor_name`, `value` (long format)
- Optional metadata: `aircraft_type`, `route`, `planned_fuel`, etc.

**Example (wide format):**
```
timestamp           | fuel_flow | mach | altitude | oat  | n1
2024-01-01 00:00:00 | 1250.5    | 0.82 | 35000    | -45  | 85.2
2024-01-01 00:00:01 | 1251.2    | 0.82 | 35001    | -45  | 85.3
...
```

### 2. Interim Data (`data/interim/`)

Cleaned and aligned timeseries per flight.

**Processing:**
- Resampled to uniform timesteps (e.g., 1Hz)
- Missing values handled (forward fill with limit)
- Sensors aligned in canonical order
- Saved as Parquet for fast loading

**Format:**
- Same structure as raw (wide format)
- Uniform time index
- Standard sensor ordering

### 3. Processed Data (`data/processed/`)

Sliding windows ready for training.

**Processing:**
- Windows created based on `WindowSpec`
- Index file maps `(flight_id, start_idx)` to windows
- Transforms applied during loading

**Index file format:**
```
flight_id       | start_idx | end_idx
flight_001      | 0         | 288
flight_001      | 32        | 320
flight_002      | 0         | 288
...
```

## Window Specification

Windows are defined by:

```python
@dataclass
class WindowSpec:
    input_len: int      # e.g., 256
    pred_len: int       # e.g., 32
    stride: int         # e.g., 32
    target_sensors: list[str]
```

Example from config:
```yaml
window:
  input_len: 256    # 256 timesteps of history
  pred_len: 32      # Predict 32 timesteps ahead
  stride: 32        # Slide window by 32 steps
  target_sensors: ["fuel_flow", "mach"]
```

## Creating Your Own Dataset

1. **Prepare raw data:**
   ```bash
   # Place flight logs in data/raw/
   data/raw/flight_001.parquet
   data/raw/flight_002.parquet
   ...
   ```

2. **Process to interim:**
   ```python
   from airtrace.data.loaders import RawDataLoader

   loader = RawDataLoader("data")
   loader.process_to_interim("flight_001")
   ```

3. **Create windows:**
   ```python
   from airtrace.data.loaders import InterimDataProcessor
   from airtrace.data.windows import WindowSpec

   processor = InterimDataProcessor("data")
   window_spec = WindowSpec(
       input_len=256,
       pred_len=32,
       stride=32,
       target_sensors=["fuel_flow", "mach"]
   )

   processor.create_windows(
       flight_ids=["flight_001", "flight_002"],
       window_spec=window_spec,
       output_name="train"
   )
   ```

## Metadata Files

Metadata is stored in `data/metadata/`:

- `train_index.parquet`: Training window index
- `val_index.parquet`: Validation window index
- `test_index.parquet`: Test window index
- `sensor_definitions.json`: Sensor metadata (units, ranges, etc.)

## Data Loading

Data is loaded via `SensorDataModule`:

```python
from airtrace.data.datamodule import SensorDataModule

datamodule = SensorDataModule(
    data_config=config,
    transforms=transform_pipeline,
    batch_size=64,
    num_workers=8
)

datamodule.setup()
train_loader = datamodule.train_dataloader()
```
