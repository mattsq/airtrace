# Synthetic Data Generation

AirTrace includes a physics-based synthetic data generator for aircraft cruise sensor readings. This is useful for:

- **Testing**: Validate models and pipelines without real flight data
- **Experimentation**: Generate arbitrary amounts of data for scaling studies
- **Prototyping**: Quickly iterate on features before acquiring real data
- **Teaching**: Demonstrate concepts with known, controlled data

## Quick Start

Generate a synthetic dataset using the command-line script:

```bash
# Generate 20 flights with default parameters
python src/scripts/generate_synthetic_data.py --n-flights 20 --output data/

# Generate longer flights
python src/scripts/generate_synthetic_data.py --n-flights 10 --duration 2.0

# Use Hydra config
python src/scripts/generate_synthetic_data.py --use-config data=synthetic_cruise
```

Or use the Python API:

```python
from pathlib import Path
from airtrace.data.synthetic import create_synthetic_dataset, CruiseProfile

# Generate dataset with default profile
splits = create_synthetic_dataset(
    data_root=Path("data/"),
    n_flights=20,
    seed=42
)

# Returns: {'train': [...], 'val': [...], 'test': [...]}
```

## Physical Model

The synthetic generator models realistic aircraft cruise flight dynamics:

### Sensors Generated

| Sensor | Description | Typical Range | Units |
|--------|-------------|---------------|-------|
| `fuel_flow` | Fuel consumption rate | 1500-3000 | kg/hour |
| `mach` | Aircraft speed | 0.75-0.85 | Mach number |
| `altitude` | Flight level | 30000-40000 | feet |
| `oat` | Outside air temperature | -70 to -40 | Celsius |
| `n1` | Engine fan speed | 75-95 | percent |
| `weight` | Aircraft weight (decreasing) | 50000-80000 | kg |

### Physical Relationships

The generator models these realistic dependencies:

1. **Fuel Flow ↔ N1**: Fuel consumption strongly correlates with engine thrust (N1)
2. **N1 ↔ Weight**: Higher weight requires more thrust to maintain speed
3. **Altitude ↔ OAT**: Outside temperature follows ISA (International Standard Atmosphere) model
4. **Weight → Time**: Aircraft weight decreases linearly as fuel burns
5. **Slow Drift**: All sensors have slow oscillations (pilot corrections, weather)
6. **Measurement Noise**: Realistic sensor noise added to all signals

### Example Timeseries

```
Time (s)  | Altitude (ft) | Mach  | N1 (%)  | Fuel Flow (kg/h)
----------|---------------|-------|---------|------------------
0         | 35012         | 0.820 | 85.2    | 2501
60        | 35034         | 0.821 | 85.5    | 2498
120       | 35021         | 0.819 | 85.1    | 2495
...       | ...           | ...   | ...     | ...
3600      | 34987         | 0.818 | 84.7    | 2442  (less fuel)
```

## Customization

### Cruise Profile

Customize the physical parameters with `CruiseProfile`:

```python
from airtrace.data.synthetic import CruiseProfile, SyntheticCruiseGenerator
from pathlib import Path

# Custom cruise profile
profile = CruiseProfile(
    initial_altitude=40000.0,      # Higher cruise altitude
    initial_mach=0.85,              # Faster speed
    initial_weight=75000.0,         # Heavier aircraft
    cruise_duration=7200,           # 2 hour cruise
    fuel_flow_base=2800.0,          # Higher consumption
    n1_base=88.0,                   # Higher thrust
    sample_rate=1.0,                # 1 Hz sampling

    # Variation parameters
    altitude_variation=200.0,       # More altitude variation
    mach_variation=0.02,            # More speed variation
    turbulence_level=0.3,           # Moderate turbulence (0-1)

    # Noise levels
    fuel_flow_noise=0.015,          # Relative std dev
    mach_noise=0.008,               # Absolute std dev
    altitude_noise=15.0,            # Feet std dev
    n1_noise=0.8,                   # Percent std dev
    oat_noise=1.5                   # Celsius std dev
)

# Generate with custom profile
generator = SyntheticCruiseGenerator(Path("data/"), seed=42)
df = generator.generate_flight("custom_001", profile)
```

### Via Config

Edit `configs/data/synthetic_cruise.yaml`:

```yaml
data:
  generation:
    n_flights: 50
    seed: 42

    cruise_profile:
      initial_altitude: 37000.0
      initial_mach: 0.83
      cruise_duration: 5400  # 1.5 hours
      turbulence_level: 0.2
```

Then generate:

```bash
python src/scripts/generate_synthetic_data.py --use-config data=synthetic_cruise
```

## Integration with AirTrace Pipeline

Generated data follows the standard AirTrace pipeline:

```
Raw (generated)
   ↓
RawDataLoader.process_to_interim()
   ↓
Interim (resampled, aligned)
   ↓
InterimDataProcessor.create_windows()
   ↓
Processed (sliding windows)
   ↓
SensorWindowDataset
```

### Using in Experiments

Once generated, use synthetic data like any other dataset:

```yaml
# configs/exp/my_experiment.yaml
defaults:
  - override /data: synthetic_cruise
  - override /model: gru_ar
  - override /transforms: zscore_diff
  - override /task: one_step
```

Run experiment:

```bash
airtrace train exp=my_experiment
```

## API Reference

### `CruiseProfile`

Dataclass holding physical parameters for cruise flight generation.

**Key Attributes:**
- `initial_altitude` (float): Starting altitude in feet (default: 35000)
- `initial_mach` (float): Starting Mach number (default: 0.82)
- `initial_weight` (float): Starting weight in kg (default: 70000)
- `cruise_duration` (int): Duration in seconds (default: 3600)
- `fuel_flow_base` (float): Base fuel flow in kg/hour (default: 2500)
- `sample_rate` (float): Sampling frequency in Hz (default: 1.0)
- `turbulence_level` (float): Turbulence intensity 0-1 (default: 0.1)

### `SyntheticCruiseGenerator`

Main generator class for creating synthetic flights.

#### `__init__(data_root, seed=None)`

Initialize generator.

**Args:**
- `data_root` (Path): Root directory for data storage
- `seed` (int, optional): Random seed for reproducibility

#### `generate_flight(flight_id, profile, save=True)`

Generate a single synthetic cruise flight.

**Args:**
- `flight_id` (str): Unique identifier for this flight
- `profile` (CruiseProfile): Physical parameters
- `save` (bool): If True, save to raw directory (default: True)

**Returns:**
- `pd.DataFrame`: Timeseries with timestamp and sensor columns

#### `generate_dataset(n_flights, profile, flight_id_prefix="synthetic_cruise")`

Generate multiple synthetic flights with variation.

**Args:**
- `n_flights` (int): Number of flights to generate
- `profile` (CruiseProfile): Base profile (varied per flight)
- `flight_id_prefix` (str): Prefix for flight IDs

**Returns:**
- `List[str]`: List of generated flight IDs

### `create_synthetic_dataset()`

Convenience function for complete dataset creation.

```python
create_synthetic_dataset(
    data_root: Path,
    n_flights: int = 20,
    profile: Optional[CruiseProfile] = None,
    seed: Optional[int] = 42,
    train_val_test_split: Tuple[float, float, float] = (0.7, 0.15, 0.15)
) -> Dict[str, List[str]]
```

**Args:**
- `data_root`: Root directory for data storage
- `n_flights`: Total number of flights to generate
- `profile`: CruiseProfile to use (None for defaults)
- `seed`: Random seed for reproducibility
- `train_val_test_split`: Tuple of (train, val, test) fractions

**Returns:**
- Dictionary with keys `'train'`, `'val'`, `'test'` mapping to flight ID lists

## Reproducibility

The generator is fully deterministic when given a seed:

```python
# Same seed → same data
gen1 = SyntheticCruiseGenerator(Path("data/"), seed=42)
gen2 = SyntheticCruiseGenerator(Path("data/"), seed=42)

df1 = gen1.generate_flight("test", profile)
df2 = gen2.generate_flight("test", profile)

assert df1.equals(df2)  # True
```

This ensures experiments are reproducible.

## Limitations

The synthetic generator is designed for **cruise flight only**:

- ✅ Stable cruise conditions
- ✅ Small variations and drift
- ✅ Realistic sensor correlations
- ❌ **Not modeled**: Takeoff, climb, descent, landing
- ❌ **Not modeled**: Engine failures or anomalies
- ❌ **Not modeled**: Weather events (icing, storms)
- ❌ **Not modeled**: Maneuvers (turns, altitude changes)

For more complex flight phases, consider:
1. Acquiring real flight data
2. Extending the generator with additional physics
3. Using the generator for baseline testing only

## Best Practices

1. **Always set a seed** for reproducibility:
   ```python
   create_synthetic_dataset(seed=42)
   ```

2. **Start with defaults** before customizing:
   ```python
   profile = CruiseProfile()  # Good defaults
   ```

3. **Match your application**: If modeling long-haul flights, increase `cruise_duration`

4. **Validate on real data**: Always test final models on real flight data

5. **Document experiments**: Note that synthetic data was used in experiment logs

## Examples

### Quick Test Dataset

Generate a small dataset for quick testing:

```python
from pathlib import Path
from airtrace.data.synthetic import create_synthetic_dataset, CruiseProfile

# Short flights, small dataset
profile = CruiseProfile(cruise_duration=600, sample_rate=1.0)  # 10 minutes

splits = create_synthetic_dataset(
    data_root=Path("data/"),
    n_flights=5,
    profile=profile,
    seed=42
)

print(f"Generated {sum(len(v) for v in splits.values())} flights")
```

### Large-Scale Study

Generate a large dataset for scaling studies:

```python
# Many long flights
profile = CruiseProfile(cruise_duration=10800)  # 3 hours

splits = create_synthetic_dataset(
    data_root=Path("data/"),
    n_flights=100,
    profile=profile,
    seed=42,
    train_val_test_split=(0.8, 0.1, 0.1)
)
```

### High Turbulence Scenario

Test model robustness with noisy data:

```python
# High turbulence and noise
profile = CruiseProfile(
    turbulence_level=0.6,       # High turbulence
    fuel_flow_noise=0.025,      # More noise
    mach_noise=0.015,
    altitude_noise=30.0
)

generator = SyntheticCruiseGenerator(Path("data/"), seed=42)
generator.generate_dataset(n_flights=20, profile=profile)
```

## See Also

- [Data Format Documentation](data_format.md) - Understanding the data pipeline
- [Architecture](architecture.md) - How synthetic data fits into AirTrace
- [Experiments](experiments.md) - Running experiments with synthetic data
