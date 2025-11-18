import pandas as pd

from airtrace.transforms import cache


def test_compute_cache_key_is_deterministic():
    config = {"pipeline": [{"name": "noop"}, {"name": "normalize", "per_sensor": True}]}
    key_one = cache.compute_cache_key("dataset", config, "index123")
    key_two = cache.compute_cache_key("dataset", config, "index123")
    assert key_one == key_two

    altered = cache.compute_cache_key("dataset", {"pipeline": []}, "index123")
    assert altered != key_one


def test_save_and_load_transform_stats(tmp_path):
    stats = {"transform": {"mean": 0.0, "std": 1.0}}
    config = {"pipeline": [{"name": "noop"}]}

    cache.save_transform_stats(stats, tmp_path, "demo", config, "hash1")

    cached = cache.load_transform_stats(tmp_path, "demo", config, "hash1")
    assert cached == stats

    missing = cache.load_transform_stats(tmp_path, "demo", config, "different-hash")
    assert missing is None


def test_compute_index_hash_changes_with_content():
    empty_df = pd.DataFrame(columns=["flight_id"])
    empty_hash = cache.compute_index_hash(empty_df)

    df = pd.DataFrame({"flight_id": ["A1", "B2", "C3", "D4", "E5", "F6"]})
    populated_hash = cache.compute_index_hash(df)

    assert empty_hash != populated_hash

    df_shuffled = pd.DataFrame({"flight_id": ["F6", "E5", "D4", "C3", "B2", "A1"]})
    assert cache.compute_index_hash(df_shuffled) != populated_hash
