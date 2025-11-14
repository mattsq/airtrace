"""Tests for transform implementations."""

import numpy as np
import pytest

from airtrace.transforms import (
    ContextTransform,
    DifferenceTransform,
    RobustScalerTransform,
    ZScoreTransform,
)


class MockDataset:
    """Mock dataset for testing transforms."""

    def __init__(self, num_samples=100, seq_len=50, dim=5):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.dim = dim

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = np.random.randn(self.seq_len, self.dim).astype(np.float32)
        y = np.random.randn(10, self.dim).astype(np.float32)
        meta = {"flight_id": f"flight_{idx}"}
        return {"x": x, "y": y, "meta": meta}


def test_zscore_transform():
    """Test z-score normalization transform."""
    dataset = MockDataset()
    transform = ZScoreTransform(per_sensor=True)

    # Fit transform
    transform.fit(dataset)
    assert transform.is_fitted

    # Apply transform
    sample = dataset[0]
    x, y, meta = transform(sample["x"], sample["y"], sample["meta"])

    # Check shapes preserved
    assert x.shape == sample["x"].shape
    assert y.shape == sample["y"].shape

    # Check normalization (approximately zero mean, unit variance)
    assert np.abs(x.mean()) < 0.5  # Approximate due to small sample
    assert np.abs(x.std() - 1.0) < 0.5


def test_robust_scaler_transform():
    """Test robust scaler transform."""
    dataset = MockDataset()
    transform = RobustScalerTransform(per_sensor=True)

    # Fit transform
    transform.fit(dataset)
    assert transform.is_fitted

    # Apply transform
    sample = dataset[0]
    x, y, meta = transform(sample["x"], sample["y"], sample["meta"])

    # Check shapes preserved
    assert x.shape == sample["x"].shape
    assert y.shape == sample["y"].shape


def test_difference_transform():
    """Test differencing transform."""
    transform = DifferenceTransform(order=1)
    transform.fit(MockDataset())

    # Create sample with trend
    x = np.cumsum(np.ones((50, 5)), axis=0).astype(np.float32)
    y = np.cumsum(np.ones((10, 5)), axis=0).astype(np.float32)
    meta = {}

    # Apply transform
    x_diff, y_diff, meta_out = transform(x, y, meta)

    # Check shapes (may be padded)
    assert x_diff.shape[0] == x.shape[0]
    assert x_diff.shape[1] == x.shape[1]


def test_context_transform():
    """Test context transform."""
    transform = ContextTransform(use_static=["aircraft_type"])

    # Create mock dataset with metadata
    class MockDatasetWithMeta(MockDataset):
        def __getitem__(self, idx):
            sample = super().__getitem__(idx)
            sample["meta"]["aircraft_type"] = "737"
            return sample

    dataset = MockDatasetWithMeta()
    transform.fit(dataset)

    # Apply transform
    sample = dataset[0]
    x, y, meta = transform(sample["x"], sample["y"], sample["meta"])

    # Check that context was added
    assert x.shape[1] > sample["x"].shape[1]  # Extra dimension for context
    assert "context_dim" in meta
    assert meta["context_dim"] > 0


def test_transform_invertibility():
    """Test that transforms can be inverted."""
    dataset = MockDataset()
    transform = ZScoreTransform()
    transform.fit(dataset)

    sample = dataset[0]
    x_orig = sample["x"].copy()
    y_orig = sample["y"].copy()

    # Forward transform
    x_trans, y_trans, _ = transform(x_orig, y_orig, {})

    # Inverse transform
    x_inv, y_inv = transform.inverse(x_trans, y_trans)

    # Check reconstruction (within tolerance)
    np.testing.assert_allclose(x_inv, x_orig, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(y_inv, y_orig, rtol=1e-5, atol=1e-5)
