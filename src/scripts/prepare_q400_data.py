"""
Prepare Q400 dataset for AirTrace framework.

This script:
1. Loads raw Q400 data
2. Creates train/val/test splits at flight level
3. Applies column renaming and imputation
4. Saves cleaned data and split indices
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split


def main():
    print("=" * 70)
    print("Q400 Data Preparation")
    print("=" * 70)

    # Paths
    root_dir = Path(__file__).parent.parent.parent
    data_dir = root_dir / "data"
    raw_path = data_dir / "raw" / "Q400-Ts-Data.parquet"
    interim_dir = data_dir / "interim"
    metadata_dir = data_dir / "metadata"

    # Create directories
    interim_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    # Load raw data
    print(f"\n1. Loading raw data from {raw_path}...")
    df = pd.read_parquet(raw_path)
    print(f"   Shape: {df.shape}")
    print(f"   Flights: {df['flight_id'].nunique()}")

    # Data quality check
    print("\n2. Data quality assessment...")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    print("   Missing values:")
    for col in df.columns:
        if missing[col] > 0:
            print(f"     {col}: {missing[col]:,} ({missing_pct[col]:.2f}%)")

    # Column mapping
    print("\n3. Applying column mapping...")
    column_mapping = {
        'Engine Fuel Flow Total (kg/hr)': 'fuel_flow',
        'Airspeed (true) (knots)': 'airspeed_true',
        '% Torque Average (%)': 'torque',
        'Pressure Altitude (ft)': 'altitude',
        'Air Temperature (outside) (deg C)': 'oat',
        'Gross Weight (Best Uncorrected) (kg)': 'weight'
    }
    df_clean = df.rename(columns=column_mapping)

    # Imputation (forward-fill then backward-fill per flight)
    print("\n4. Applying imputation (ffill then bfill per flight)...")

    # Note: weight and oat columns will still have many missing values
    # This is expected - we'll exclude them from initial modeling
    numeric_cols = ['fuel_flow', 'airspeed_true', 'torque', 'altitude', 'weight', 'oat']

    for col in numeric_cols:
        df_clean[col] = df_clean.groupby('flight_id')[col].ffill().bfill()

    remaining_missing = df_clean[numeric_cols].isnull().sum()
    print(f"   Remaining missing values:")
    for col in numeric_cols:
        if remaining_missing[col] > 0:
            pct = (remaining_missing[col] / len(df_clean)) * 100
            print(f"     {col}: {remaining_missing[col]:,} ({pct:.2f}%)")

    # Train/val/test split
    print("\n5. Creating train/val/test splits...")
    flight_ids = df_clean['flight_id'].unique()
    np.random.seed(42)

    # 70% train, 15% val, 15% test
    train_flights, temp_flights = train_test_split(
        flight_ids, test_size=0.30, random_state=42
    )
    val_flights, test_flights = train_test_split(
        temp_flights, test_size=0.50, random_state=42
    )

    print(f"   Train: {len(train_flights)} flights ({len(train_flights)/len(flight_ids)*100:.1f}%)")
    print(f"   Val:   {len(val_flights)} flights ({len(val_flights)/len(flight_ids)*100:.1f}%)")
    print(f"   Test:  {len(test_flights)} flights ({len(test_flights)/len(flight_ids)*100:.1f}%)")

    # Verify no overlap
    assert len(set(train_flights) & set(val_flights)) == 0
    assert len(set(train_flights) & set(test_flights)) == 0
    assert len(set(val_flights) & set(test_flights)) == 0
    print("   [OK] No overlap verified")

    # Count timesteps in each split
    train_timesteps = df_clean[df_clean['flight_id'].isin(train_flights)].shape[0]
    val_timesteps = df_clean[df_clean['flight_id'].isin(val_flights)].shape[0]
    test_timesteps = df_clean[df_clean['flight_id'].isin(test_flights)].shape[0]

    print(f"\n   Timesteps:")
    print(f"   Train: {train_timesteps:,} ({train_timesteps/len(df_clean)*100:.1f}%)")
    print(f"   Val:   {val_timesteps:,} ({val_timesteps/len(df_clean)*100:.1f}%)")
    print(f"   Test:  {test_timesteps:,} ({test_timesteps/len(df_clean)*100:.1f}%)")

    # Save indices
    print("\n6. Saving split indices...")
    train_index = pd.DataFrame({'flight_id': train_flights})
    val_index = pd.DataFrame({'flight_id': val_flights})
    test_index = pd.DataFrame({'flight_id': test_flights})

    train_index.to_parquet(metadata_dir / 'q400_train_index.parquet', index=False)
    val_index.to_parquet(metadata_dir / 'q400_val_index.parquet', index=False)
    test_index.to_parquet(metadata_dir / 'q400_test_index.parquet', index=False)

    print(f"   [OK] Saved to {metadata_dir}/")
    print(f"     - q400_train_index.parquet")
    print(f"     - q400_val_index.parquet")
    print(f"     - q400_test_index.parquet")

    # Save cleaned data
    print("\n7. Saving cleaned interim data...")
    interim_path = interim_dir / 'q400_cleaned.parquet'
    df_clean.to_parquet(interim_path, index=False)
    file_size_mb = interim_path.stat().st_size / 1e6
    print(f"   [OK] Saved to {interim_path}")
    print(f"     Size: {file_size_mb:.1f} MB")

    # Training statistics
    print("\n8. Computing training statistics...")
    train_data = df_clean[df_clean['flight_id'].isin(train_flights)]
    sensor_cols = ['fuel_flow', 'airspeed_true', 'torque', 'altitude']
    stats = train_data[sensor_cols].describe()

    stats.to_csv(metadata_dir / 'q400_train_stats.csv')
    print(f"   [OK] Saved to {metadata_dir}/q400_train_stats.csv")
    print("\n   Training set statistics:")
    print(stats)

    # Window estimation
    print("\n9. Estimating windowing capacity...")
    input_len = 128
    pred_len = 16
    stride = 16
    min_length = input_len + pred_len

    flight_lengths = df_clean.groupby('flight_id').size()

    def count_windows(length):
        if length < min_length:
            return 0
        return (length - min_length) // stride + 1

    flight_windows = flight_lengths.apply(count_windows)

    train_windows = flight_windows[flight_windows.index.isin(train_flights)].sum()
    val_windows = flight_windows[flight_windows.index.isin(val_flights)].sum()
    test_windows = flight_windows[flight_windows.index.isin(test_flights)].sum()

    print(f"   Window config: input_len={input_len}, pred_len={pred_len}, stride={stride}")
    print(f"   Estimated windows:")
    print(f"     Train: {train_windows:,}")
    print(f"     Val:   {val_windows:,}")
    print(f"     Test:  {test_windows:,}")
    print(f"     Total: {train_windows + val_windows + test_windows:,}")

    print("\n" + "=" * 70)
    print("[OK] Q400 data preparation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
