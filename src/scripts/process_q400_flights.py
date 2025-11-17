"""
Convert Q400 interim cleaned data into per-flight processed files.

This script:
1. Loads the cleaned interim Q400 data
2. Splits it into individual flight files
3. Saves each flight as a separate parquet file in data/processed/
"""

import pandas as pd
from pathlib import Path
from tqdm import tqdm


def main():
    print("=" * 70)
    print("Q400 Flight Processing")
    print("=" * 70)

    # Paths
    root_dir = Path(__file__).parent.parent.parent
    data_dir = root_dir / "data"
    interim_path = data_dir / "interim" / "q400_cleaned.parquet"
    processed_dir = data_dir / "processed"

    # Create processed directory
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Load cleaned data
    print(f"\n1. Loading cleaned data from {interim_path}...")
    df = pd.read_parquet(interim_path)
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")

    # Get unique flights
    flight_ids = df['flight_id'].unique()
    print(f"\n2. Found {len(flight_ids)} unique flights")

    # Process each flight (efficient groupby approach)
    print(f"\n3. Splitting into individual flight files...")
    print(f"   Using efficient groupby method...")

    # Group by flight_id and save each group
    for flight_id, group_data in tqdm(df.groupby('flight_id'), total=len(flight_ids), desc="Processing flights"):
        # Drop flight_id column (no longer needed in individual files)
        flight_data = group_data.drop(columns=['flight_id']).copy()

        # Reset index to be sequential
        flight_data = flight_data.reset_index(drop=True)

        # Save to processed directory
        output_path = processed_dir / f"{flight_id}.parquet"
        flight_data.to_parquet(output_path, index=False)

    print(f"\n4. Verifying processed files...")
    processed_files = list(processed_dir.glob("*.parquet"))
    print(f"   Created {len(processed_files)} flight files")

    # Check a sample file
    if processed_files:
        sample_file = processed_files[0]
        sample_df = pd.read_parquet(sample_file)
        print(f"\n   Sample file: {sample_file.name}")
        print(f"   Shape: {sample_df.shape}")
        print(f"   Columns: {list(sample_df.columns)}")
        print(f"\n   First 3 rows:")
        print(sample_df.head(3))

    # Calculate total size
    total_size_mb = sum(f.stat().st_size for f in processed_files) / 1e6
    print(f"\n5. Storage summary:")
    print(f"   Total files: {len(processed_files)}")
    print(f"   Total size: {total_size_mb:.1f} MB")
    print(f"   Avg file size: {total_size_mb/len(processed_files):.2f} MB")

    print("\n" + "=" * 70)
    print("[OK] Q400 flight processing complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
