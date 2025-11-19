"""Tests for transform implementations."""

from typing import Any, Dict

import numpy as np
import pytest

from airtrace.transforms import (
    ClipTransform,
    ContextTransform,
    DetrendTransform,
    DifferenceTransform,
    ImputeTransform,
    LogTransform,
    MinMaxTransform,
    NoOpTransform,
    RobustScalerTransform,
    SmoothTransform,
    TemporalFeaturesTransform,
    ZScoreTransform,
)


class MockDataset:
    """Mock dataset for testing transforms with deterministic samples."""

    def __init__(
        self,
        num_samples: int = 100,
        seq_len: int = 50,
        dim: int = 5,
        target_len: int = 10,
        seed: int = 0,
    ) -> None:
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.dim = dim
        self.target_len = target_len

        rng = np.random.default_rng(seed)
        self._x_data = rng.standard_normal(
            size=(num_samples, seq_len, dim), dtype=np.float32
        )
        self._y_data = rng.standard_normal(
            size=(num_samples, target_len, dim), dtype=np.float32
        )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        x = self._x_data[idx].copy()
        y = self._y_data[idx].copy()
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


# ===== Tier 1 Transform Tests =====


def test_minmax_transform():
    """Test MinMax scaling transform."""
    dataset = MockDataset()
    transform = MinMaxTransform(feature_range=(0.0, 1.0))

    # Fit transform
    transform.fit(dataset)
    assert transform.is_fitted

    # Apply transform
    sample = dataset[0]
    x, y, meta = transform(sample["x"], sample["y"], sample["meta"])

    # Check shapes preserved
    assert x.shape == sample["x"].shape
    assert y.shape == sample["y"].shape

    # Check values are in expected range (approximately)
    assert x.min() >= -0.1  # Allow some slack due to sampling
    assert x.max() <= 1.1


def test_minmax_invertibility():
    """Test MinMax transform can be inverted."""
    dataset = MockDataset()
    transform = MinMaxTransform(feature_range=(-1.0, 1.0))
    transform.fit(dataset)

    sample = dataset[0]
    x_orig = sample["x"].copy()
    y_orig = sample["y"].copy()

    # Forward and inverse
    x_trans, y_trans, _ = transform(x_orig, y_orig, {})
    x_inv, y_inv = transform.inverse(x_trans, y_trans)

    # Check reconstruction
    np.testing.assert_allclose(x_inv, x_orig, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(y_inv, y_orig, rtol=1e-5, atol=1e-5)


def test_clip_transform():
    """Test clipping transform."""
    dataset = MockDataset()
    transform = ClipTransform(method="percentile", lower=5.0, upper=95.0)

    # Fit transform
    transform.fit(dataset)
    assert transform.is_fitted
    assert transform.lower_bounds is not None
    assert transform.upper_bounds is not None

    # Apply transform
    sample = dataset[0]
    x, y, meta = transform(sample["x"], sample["y"], sample["meta"])

    # Check shapes preserved
    assert x.shape == sample["x"].shape
    assert y.shape == sample["y"].shape
    assert meta["clipped"] is True


def test_clip_std_method():
    """Test clipping with std method."""
    dataset = MockDataset()
    transform = ClipTransform(method="std", n_std=3.0)

    transform.fit(dataset)
    sample = dataset[0]
    x, y, meta = transform(sample["x"], sample["y"], sample["meta"])

    # Check it ran without error
    assert x.shape == sample["x"].shape


def test_noop_transform():
    """Test no-op transform."""
    dataset = MockDataset()
    transform = NoOpTransform()

    # Fit transform
    transform.fit(dataset)
    assert transform.is_fitted

    # Apply transform
    sample = dataset[0]
    x, y, meta = transform(sample["x"], sample["y"], sample["meta"])

    # Check nothing changed
    np.testing.assert_array_equal(x, sample["x"])
    np.testing.assert_array_equal(y, sample["y"])
    assert meta == sample["meta"]


# ===== Tier 2 Transform Tests =====


def test_log_transform():
    """Test logarithmic transform."""
    dataset = MockDataset()
    transform = LogTransform(base="natural")

    # Fit transform
    transform.fit(dataset)
    assert transform.is_fitted

    # Create positive data
    x = np.abs(np.random.randn(50, 5).astype(np.float32)) + 1.0
    y = np.abs(np.random.randn(10, 5).astype(np.float32)) + 1.0
    meta = {}

    # Apply transform
    x_log, y_log, meta_out = transform(x, y, meta)

    # Check shapes preserved
    assert x_log.shape == x.shape
    assert y_log.shape == y.shape
    assert meta_out["log_transformed"] is True


def test_log_invertibility():
    """Test log transform can be inverted."""
    dataset = MockDataset()
    transform = LogTransform(base="natural")
    transform.fit(dataset)

    # Create positive data
    x_orig = np.abs(np.random.randn(50, 5).astype(np.float32)) + 1.0
    y_orig = np.abs(np.random.randn(10, 5).astype(np.float32)) + 1.0

    # Forward and inverse
    x_trans, y_trans, _ = transform(x_orig, y_orig, {})
    x_inv, y_inv = transform.inverse(x_trans, y_trans)

    # Check reconstruction
    np.testing.assert_allclose(x_inv, x_orig, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(y_inv, y_orig, rtol=1e-3, atol=1e-3)


def test_smooth_transform():
    """Test smoothing transform."""
    dataset = MockDataset()
    transform = SmoothTransform(window_size=5, method="uniform")

    # Fit transform
    transform.fit(dataset)
    assert transform.is_fitted

    # Apply transform
    sample = dataset[0]
    x, y, meta = transform(sample["x"], sample["y"], sample["meta"])

    # Check shapes preserved
    assert x.shape == sample["x"].shape
    assert y.shape == sample["y"].shape
    assert meta["smoothed"] is True


def test_smooth_gaussian():
    """Test Gaussian smoothing."""
    dataset = MockDataset()
    transform = SmoothTransform(window_size=5, method="gaussian")

    transform.fit(dataset)
    sample = dataset[0]
    x, y, meta = transform(sample["x"], sample["y"], sample["meta"])

    # Check it ran without error
    assert x.shape == sample["x"].shape


def test_impute_transform():
    """Test imputation transform."""
    dataset = MockDataset()
    transform = ImputeTransform(method="forward")

    # Fit transform
    transform.fit(dataset)
    assert transform.is_fitted

    # Create data with NaNs
    x = np.random.randn(50, 5).astype(np.float32)
    x[10:15, 2] = np.nan  # Add some missing values
    y = np.random.randn(10, 5).astype(np.float32)
    meta = {}

    # Apply transform
    x_imp, y_imp, meta_out = transform(x, y, meta)

    # Check NaNs are filled
    assert not np.isnan(x_imp).any()
    assert meta_out["imputed"] is True


def test_impute_methods():
    """Test different imputation methods."""
    dataset = MockDataset()

    methods = ["forward", "backward", "linear", "constant"]
    for method in methods:
        transform = ImputeTransform(method=method)
        transform.fit(dataset)

        # Create data with NaNs
        x = np.random.randn(50, 5).astype(np.float32)
        x[10:15, 2] = np.nan
        y = np.random.randn(10, 5).astype(np.float32)

        x_imp, y_imp, _ = transform(x, y, {})

        # Check NaNs are handled
        assert not np.isnan(x_imp).any()


def test_impute_mean():
    """Test mean imputation."""
    dataset = MockDataset()
    transform = ImputeTransform(method="mean")

    transform.fit(dataset)
    assert transform.mean_values is not None

    # Create data with NaNs
    x = np.random.randn(50, 5).astype(np.float32)
    x[10:15, 2] = np.nan
    y = np.random.randn(10, 5).astype(np.float32)

    x_imp, y_imp, _ = transform(x, y, {})
    assert not np.isnan(x_imp).any()


# ===== Tier 3 Transform Tests =====


def test_detrend_transform():
    """Test detrending transform."""
    dataset = MockDataset()
    transform = DetrendTransform(method="linear")

    # Fit transform
    transform.fit(dataset)
    assert transform.is_fitted

    # Create data with trend
    x = np.cumsum(np.ones((50, 5)), axis=0).astype(np.float32)
    y = np.cumsum(np.ones((10, 5)), axis=0).astype(np.float32)
    meta = {}

    # Apply transform
    x_det, y_det, meta_out = transform(x, y, meta)

    # Check shapes preserved
    assert x_det.shape == x.shape
    assert y_det.shape == y.shape
    assert "detrend_x_coeffs" in meta_out


def test_detrend_methods():
    """Test different detrending methods."""
    dataset = MockDataset()

    methods = ["constant", "linear", "polynomial"]
    for method in methods:
        if method == "polynomial":
            transform = DetrendTransform(method=method, degree=2)
        else:
            transform = DetrendTransform(method=method)

        transform.fit(dataset)

        # Create data with trend
        x = np.cumsum(np.random.randn(50, 5), axis=0).astype(np.float32)
        y = np.cumsum(np.random.randn(10, 5), axis=0).astype(np.float32)

        x_det, y_det, meta = transform(x, y, {})

        # Check it ran without error
        assert x_det.shape == x.shape


def test_temporal_features_transform():
    """Test temporal features transform."""
    dataset = MockDataset()
    transform = TemporalFeaturesTransform(
        use_time_idx=True,
        use_cyclic_time=True,
        use_flight_phase=False
    )

    # Fit transform
    transform.fit(dataset)
    assert transform.is_fitted

    # Apply transform
    sample = dataset[0]
    x, y, meta = transform(sample["x"], sample["y"], sample["meta"])

    # Check that features were added
    assert x.shape[0] == sample["x"].shape[0]  # Time dim preserved
    assert x.shape[1] > sample["x"].shape[1]  # Features added
    assert y.shape == sample["y"].shape  # y unchanged
    assert "temporal_features_dim" in meta
    assert meta["temporal_features_dim"] > 0


def test_temporal_features_with_phase():
    """Test temporal features with flight phase."""
    dataset = MockDataset()
    transform = TemporalFeaturesTransform(
        use_time_idx=True,
        use_flight_phase=True
    )

    transform.fit(dataset)

    sample = dataset[0]
    sample["meta"]["flight_phase"] = "cruise"
    x, y, meta = transform(sample["x"], sample["y"], sample["meta"])

    # Check features were added
    assert x.shape[1] > sample["x"].shape[1]


def test_temporal_features_inverse():
    """Test temporal features can be removed."""
    dataset = MockDataset()
    transform = TemporalFeaturesTransform(use_time_idx=True, use_cyclic_time=True)

    transform.fit(dataset)
    sample = dataset[0]
    x_orig = sample["x"].copy()

    # Forward transform
    x_trans, y_trans, meta = transform(x_orig, sample["y"], {})

    # Inverse transform
    x_inv, y_inv = transform.inverse(x_trans, y_trans, meta)

    # Check original shape restored
    assert x_inv.shape == x_orig.shape
