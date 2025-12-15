
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

def create_subsets(index_path: str, output_dir: str, fractions: list[float], seed: int = 42):
    """Create subsets of the index dataframe."""
    index_path = Path(index_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading index from {index_path}")
    df = pd.read_parquet(index_path)
    total_rows = len(df)
    print(f"Total rows: {total_rows}")

    # Shuffle
    np.random.seed(seed)
    df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    for frac in fractions:
        subset_size = int(total_rows * frac)
        subset_df = df_shuffled.iloc[:subset_size]
        
        output_filename = f"descent_train_{int(frac*100)}pct.parquet"
        output_path = output_dir / output_filename
        
        print(f"Creating {output_filename} with {len(subset_df)} rows ({frac*100}%)")
        subset_df.to_parquet(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create subset indices for scaling laws.")
    parser.add_argument("--index_path", type=str, default="data/metadata/descent_data_train_index.parquet", help="Path to the original train index.")
    parser.add_argument("--output_dir", type=str, default="data/metadata", help="Output directory for subset indices.")
    
    args = parser.parse_args()
    
    # Fractions to generate
    fractions = [0.1, 0.2, 0.5, 1.0]
    
    create_subsets(args.index_path, args.output_dir, fractions)
