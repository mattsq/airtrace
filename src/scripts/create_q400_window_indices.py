"""
Create window indices for Q400 dataset.

Generates indices specifying (flight_id, start_idx, end_idx) for each sliding window.
"""

import pandas as pd
from pathlib import Path
from tqdm import tqdm


def create_window_indices(
    flight_ids,
    processed_dir,
    input_len=128,
    pred_len=16,
    stride=16,
    name="q400"
):
    """Create window indices for given flight IDs.

    Args:
        flight_ids: List of flight IDs
        processed_dir: Directory containing processed flight files
        input_len: Input sequence length
        pred_len: Prediction horizon
        stride: Stride for sliding window
        name: Name for output file

    Returns:
        DataFrame with columns: flight_id, start_idx, end_idx
    """
    window_length = input_len + pred_len
    windows = []

    print(f"Creating window indices for {len(flight_ids)} flights...")
    print(f"  Window config: input_len={input_len}, pred_len={pred_len}, stride={stride}")

    for flight_id in tqdm(flight_ids, desc="Processing flights"):
        # Load flight data to get length
        flight_path = processed_dir / f"{flight_id}.parquet"

        if not flight_path.exists():
            print(f"  Warning: {flight_path} not found, skipping")
            continue

        flight_df = pd.read_parquet(flight_path)
        flight_len = len(flight_df)

        # Generate windows for this flight
        if flight_len < window_length:
            # Flight too short
            continue

        for start_idx in range(0, flight_len - window_length + 1, stride):
            end_idx = start_idx + window_length
            windows.append({
                "flight_id": flight_id,
                "start_idx": start_idx,
                "end_idx": end_idx
            })

    index_df = pd.DataFrame(windows)
    print(f"\nCreated {len(index_df):,} windows from {len(flight_ids)} flights")

    return index_df


def main():
    print("=" * 70)
    print("Q400 Window Index Generation")
    print("=" * 70)

    # Paths
    root_dir = Path(__file__).parent.parent.parent
    data_dir = root_dir / "data"
    processed_dir = data_dir / "processed"
    metadata_dir = data_dir / "metadata"

    # Window config
    input_len = 128
    pred_len = 16
    stride = 16

    # Load flight ID splits
    print("\n1. Loading flight ID splits...")
    train_flights = pd.read_parquet(metadata_dir / "q400_train_index.parquet")
    val_flights = pd.read_parquet(metadata_dir / "q400_val_index.parquet")
    test_flights = pd.read_parquet(metadata_dir / "q400_test_index.parquet")

    print(f"   Train: {len(train_flights)} flights")
    print(f"   Val:   {len(val_flights)} flights")
    print(f"   Test:  {len(test_flights)} flights")

    # Create window indices for each split
    print("\n2. Creating window indices...")

    print("\n  [Train]")
    train_window_index = create_window_indices(
        flight_ids=train_flights["flight_id"].values,
        processed_dir=processed_dir,
        input_len=input_len,
        pred_len=pred_len,
        stride=stride,
        name="train"
    )

    print("\n  [Val]")
    val_window_index = create_window_indices(
        flight_ids=val_flights["flight_id"].values,
        processed_dir=processed_dir,
        input_len=input_len,
        pred_len=pred_len,
        stride=stride,
        name="val"
    )

    print("\n  [Test]")
    test_window_index = create_window_indices(
        flight_ids=test_flights["flight_id"].values,
        processed_dir=processed_dir,
        input_len=input_len,
        pred_len=pred_len,
        stride=stride,
        name="test"
    )

    # Save window indices (overwrite the old flight-only indices)
    print("\n3. Saving window indices...")
    train_window_index.to_parquet(metadata_dir / "q400_train_index.parquet", index=False)
    val_window_index.to_parquet(metadata_dir / "q400_val_index.parquet", index=False)
    test_window_index.to_parquet(metadata_dir / "q400_test_index.parquet", index=False)

    print(f"   [OK] Saved to {metadata_dir}/")
    print(f"     - q400_train_index.parquet ({len(train_window_index):,} windows)")
    print(f"     - q400_val_index.parquet ({len(val_window_index):,} windows)")
    print(f"     - q400_test_index.parquet ({len(test_window_index):,} windows)")

    # Summary
    total_windows = len(train_window_index) + len(val_window_index) + len(test_window_index)
    print(f"\n4. Summary:")
    print(f"   Total windows: {total_windows:,}")
    print(f"   Train: {len(train_window_index):,} ({len(train_window_index)/total_windows*100:.1f}%)")
    print(f"   Val:   {len(val_window_index):,} ({len(val_window_index)/total_windows*100:.1f}%)")
    print(f"   Test:  {len(test_window_index):,} ({len(test_window_index)/total_windows*100:.1f}%)")

    print("\n" + "=" * 70)
    print("[OK] Window index generation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
