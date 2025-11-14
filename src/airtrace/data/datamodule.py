"""PyTorch Lightning-style DataModule for organizing data loading."""

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from torch.utils.data import DataLoader

from .dataset import DataStore, SensorWindowDataset
from .windows import WindowSpec


class SensorDataModule:
    """DataModule for sensor timeseries data.

    Organizes train/val/test datasets and dataloaders.
    Inspired by PyTorch Lightning DataModule but framework-agnostic.
    """

    def __init__(
        self,
        data_config: Dict[str, Any],
        transforms: Optional[Any] = None,
        batch_size: int = 32,
        num_workers: int = 4
    ):
        """Initialize data module.

        Args:
            data_config: Data configuration dictionary
            transforms: Transform pipeline to apply
            batch_size: Batch size for dataloaders
            num_workers: Number of worker processes
        """
        self.data_config = data_config
        self.transforms = transforms
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Extract config
        self.data_root = Path(data_config["root"])
        self.sensor_names = data_config["sensors"]["use"]
        self.target_sensors = data_config["window"]["target_sensors"]

        # Build window spec
        window_cfg = data_config["window"]
        self.window_spec = WindowSpec(
            input_len=window_cfg["input_len"],
            pred_len=window_cfg["pred_len"],
            stride=window_cfg["stride"],
            target_sensors=self.target_sensors
        )

        # Initialize data store
        self.data_store = DataStore(self.data_root)

        # Datasets (to be created in setup)
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self):
        """Set up train/val/test datasets.

        This should be called before creating dataloaders.
        """
        # Load index files
        train_index = pd.read_parquet(self.data_root / self.data_config["train_index"])
        val_index = pd.read_parquet(self.data_root / self.data_config["val_index"])

        # Create datasets
        self.train_dataset = SensorWindowDataset(
            index_df=train_index,
            data_store=self.data_store,
            transforms=self.transforms,
            sensor_names=self.sensor_names,
            target_sensors=self.target_sensors
        )

        self.val_dataset = SensorWindowDataset(
            index_df=val_index,
            data_store=self.data_store,
            transforms=self.transforms,
            sensor_names=self.sensor_names,
            target_sensors=self.target_sensors
        )

        # Test dataset (if available)
        if "test_index" in self.data_config:
            test_index = pd.read_parquet(self.data_root / self.data_config["test_index"])
            self.test_dataset = SensorWindowDataset(
                index_df=test_index,
                data_store=self.data_store,
                transforms=self.transforms,
                sensor_names=self.sensor_names,
                target_sensors=self.target_sensors
            )

    def train_dataloader(self) -> DataLoader:
        """Create training dataloader.

        Returns:
            DataLoader for training
        """
        if self.train_dataset is None:
            raise RuntimeError("Call setup() before creating dataloaders")

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader.

        Returns:
            DataLoader for validation
        """
        if self.val_dataset is None:
            raise RuntimeError("Call setup() before creating dataloaders")

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        """Create test dataloader.

        Returns:
            DataLoader for testing, or None if no test set
        """
        if self.test_dataset is None:
            return None

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    @property
    def in_dim(self) -> int:
        """Input feature dimension."""
        return len(self.sensor_names)

    @property
    def out_dim(self) -> int:
        """Output prediction dimension."""
        return len(self.target_sensors)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  in_dim={self.in_dim},\n"
            f"  out_dim={self.out_dim},\n"
            f"  window_spec={self.window_spec},\n"
            f"  batch_size={self.batch_size}\n"
            f")"
        )
