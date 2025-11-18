"""Transform statistics caching for faster training initialization."""

import hashlib
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def compute_cache_key(dataset_name: str, transform_config: Dict[str, Any], index_hash: str) -> str:
    """Compute a unique cache key for transform statistics.

    Args:
        dataset_name: Name of the dataset
        transform_config: Transform pipeline configuration
        index_hash: Hash of the dataset indices (train/val/test splits)

    Returns:
        Cache key string
    """
    # Create a deterministic string representation
    config_str = str(sorted(transform_config.items()))
    combined = f"{dataset_name}_{config_str}_{index_hash}"

    # Hash to get a shorter key
    return hashlib.md5(combined.encode()).hexdigest()


def save_transform_stats(
    stats: Dict[str, Any],
    cache_path: Path,
    dataset_name: str,
    transform_config: Dict[str, Any],
    index_hash: str
) -> None:
    """Save transform statistics to cache.

    Args:
        stats: Dictionary of transform statistics to save
        cache_path: Directory to save cache files
        dataset_name: Name of the dataset
        transform_config: Transform pipeline configuration
        index_hash: Hash of the dataset indices
    """
    cache_key = compute_cache_key(dataset_name, transform_config, index_hash)
    cache_file = cache_path / f"transform_stats_{cache_key}.pkl"

    # Ensure cache directory exists
    cache_path.mkdir(parents=True, exist_ok=True)

    # Save stats
    with open(cache_file, 'wb') as f:
        pickle.dump(stats, f)

    print(f"[INFO] Saved transform stats to cache: {cache_file.name}")


def load_transform_stats(
    cache_path: Path,
    dataset_name: str,
    transform_config: Dict[str, Any],
    index_hash: str
) -> Optional[Dict[str, Any]]:
    """Load transform statistics from cache if available.

    Args:
        cache_path: Directory containing cache files
        dataset_name: Name of the dataset
        transform_config: Transform pipeline configuration
        index_hash: Hash of the dataset indices

    Returns:
        Dictionary of transform statistics, or None if cache miss
    """
    cache_key = compute_cache_key(dataset_name, transform_config, index_hash)
    cache_file = cache_path / f"transform_stats_{cache_key}.pkl"

    if not cache_file.exists():
        return None

    try:
        with open(cache_file, 'rb') as f:
            stats = pickle.load(f)
        print(f"[INFO] Loaded transform stats from cache: {cache_file.name}")
        return stats
    except Exception as e:
        print(f"[WARNING] Failed to load cache: {e}")
        return None


def compute_index_hash(index_df) -> str:
    """Compute a hash of the dataset index for cache invalidation.

    Args:
        index_df: DataFrame containing the dataset index

    Returns:
        Hash string representing the index
    """
    # Use shape and first/last few rows to create hash
    # This catches changes in data splits without reading entire index
    shape_str = str(index_df.shape)

    if len(index_df) > 0:
        # Include first and last few flight IDs
        head_ids = str(index_df['flight_id'].head(5).tolist())
        tail_ids = str(index_df['flight_id'].tail(5).tolist())
        combined = f"{shape_str}_{head_ids}_{tail_ids}"
    else:
        combined = shape_str

    return hashlib.md5(combined.encode()).hexdigest()[:16]
