from pathlib import Path

import numpy as np
import pandas as pd

from airtrace.data.datamodule import SensorDataModule
from airtrace.data.windows import WindowSpec
from airtrace.data.dataset import SensorWindowDataset
from airtrace.transforms.context import ContextTransform


class _DummyTransform:
    """Simple transform pipeline with a single context transform."""

    def __init__(self) -> None:
        self.transforms = [ContextTransform(use_static=["aircraft_type"])]
        self.fitted = False
        self.cached_stats = {"static_encoders": {}, "use_static": ["aircraft_type"], "use_plan_deltas": False, "use_env": False}

    def fit(self, dataset: SensorWindowDataset) -> "_DummyTransform":
        self.fitted = True
        return self

    def __call__(self, x, y, meta):
        return x, y, meta

    def get_stats(self):
        return self.cached_stats

    def set_stats(self, stats):
        self.cached_stats = stats


class _DummyStore:
    def __init__(self, data_root, format="parquet") -> None:
        self.data_root = data_root
        self.format = format
        self.calls = []
        self.flight_metadata = pd.DataFrame({"aircraft_type": ["Q400"]}, index=["flight_1"])

    def get_full_window(self, flight_id, start_idx, end_idx, column_names):
        self.calls.append((flight_id, start_idx, end_idx, tuple(column_names)))
        time = np.arange(start_idx, end_idx, dtype=np.float32)[:, None]
        base = np.hstack([time, time + 1])  # two sensors
        meta = {"flight_id": flight_id, "start_idx": start_idx, "end_idx": end_idx, "aircraft_type": "Q400"}
        return base.astype(np.float32), meta


def test_sensor_data_module_builds_datasets_and_loaders(minimal_data_root: Path):
    transforms = _DummyTransform()
    config = {
        "root": minimal_data_root,
        "sensors": {"use": ["s1", "s2"]},
        "window": {"input_len": 4, "pred_len": 1, "stride": 1, "target_sensors": ["s2"]},
        "train_index": "metadata/train.parquet",
        "val_index": "metadata/val.parquet",
        "dataset_name": "tiny",
    }

    module = SensorDataModule(config, transforms=transforms, batch_size=2, num_workers=0)

    module.setup()

    train_loader = module.train_dataloader()
    val_loader = module.val_dataloader()

    batch = next(iter(train_loader))
    assert batch["x"].shape == (1, 4, 2)
    assert batch["y"].shape == (1, 1, 1)
    assert module.in_dim == 3  # two sensors + one context feature
    assert module.out_dim == 1
    assert "SensorDataModule" in repr(module)

    # second loader yields validation batch
    val_batch = next(iter(val_loader))
    assert val_batch["x"].shape[1:] == (4, 2)


def test_sensor_data_module_supports_optional_test_set(minimal_data_root: Path):
    config = {
        "root": minimal_data_root,
        "sensors": {"use": ["s1"]},
        "window": {"input_len": 4, "pred_len": 1, "stride": 1, "target_sensors": ["s1"]},
        "train_index": "metadata/train.parquet",
        "val_index": "metadata/val.parquet",
        "test_index": "metadata/test.parquet",
    }

    module = SensorDataModule(config, transforms=None, batch_size=1, num_workers=0)
    module.setup()

    test_loader = module.test_dataloader()
    assert isinstance(test_loader.dataset, SensorWindowDataset)
    assert len(test_loader) == 1
    test_batch = next(iter(test_loader))
    assert test_batch["x"].shape[-1] == 1


def test_sensor_window_dataset_applies_target_mapping():
    index_df = pd.DataFrame({"flight_id": ["f1"], "start_idx": [0], "end_idx": [4]})
    store = _DummyStore(Path("/tmp"))
    dataset = SensorWindowDataset(
        index_df,
        data_store=store,
        transforms=None,
        sensor_names=["s1", "s2"],
        target_sensors=["s2"],
        window_spec=WindowSpec(3, 1, 1, ["s2"]),
    )

    sample = dataset[0]
    assert sample["x"].shape == (3, 2)
    assert sample["y"].shape == (1, 1)
    assert sample["meta"]["sensor_names"] == ["s1", "s2"]
