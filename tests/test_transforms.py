"""Tests for transform implementations."""

from typing import Any, Dict

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter1d, uniform_filter1d

from airtrace.transforms import (
    Compose,
    ClipTransform,
    ContextTransform,
    DetrendTransform,
    DifferenceTransform,
    ImputeTransform,
    LogTransform,
    MinMaxTransform,
    NoOpTransform,
    RobustScalerTransform,
    Transform,
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


def test_difference_transform_inverse_and_meta():
    """DifferenceTransform should store initial values and invert with cumulative sum."""
    transform = DifferenceTransform(order=1, sensors=["sensor_a"])
    transform.fit(MockDataset())

    x = np.array([[1.0, 2.0], [3.0, 5.0], [6.0, 9.0]])
    y = np.array([[0.5, 1.5], [1.5, 2.5]])
    meta: Dict[str, Any] = {}

    x_diff, y_diff, meta_out = transform(x, y, meta)

    # Initial values should be stored for both x and y
    assert np.array_equal(meta_out["diff_initial_x"], x[0])
    assert np.array_equal(meta_out["diff_initial_y"], y[0])

    # Inverse should reconstruct the original sequences
    x_inv, y_inv = transform.inverse(x_diff, y_diff)
    assert np.allclose(x_inv, x)
    assert np.allclose(y_inv, y)


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

    # Learned bounds should match dataset percentiles for each feature
    expected_lower = np.percentile(dataset._x_data, 5.0, axis=(0, 1))
    expected_upper = np.percentile(dataset._x_data, 95.0, axis=(0, 1))
    np.testing.assert_allclose(np.asarray(transform.lower_bounds), expected_lower)
    np.testing.assert_allclose(np.asarray(transform.upper_bounds), expected_upper)

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

    # Bounds should reflect dataset statistics
    expected_mean = dataset._x_data.mean(axis=(0, 1))
    expected_std = dataset._x_data.std(axis=(0, 1))
    np.testing.assert_allclose(transform.lower_bounds, expected_mean - 3.0 * expected_std)
    np.testing.assert_allclose(transform.upper_bounds, expected_mean + 3.0 * expected_std)

    # All values should be clipped within the learned bounds
    assert x.shape == sample["x"].shape
    assert x.min() >= transform.lower_bounds.min()
    assert x.max() <= transform.upper_bounds.max()
    assert y.shape == sample["y"].shape
    assert y.min() >= transform.lower_bounds.min()
    assert y.max() <= transform.upper_bounds.max()
    assert meta["clip_method"] == "std"


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


def test_log_transform_global_min_values():
    """LogTransform should use a single global minimum when per_sensor=False."""

    class DeterministicDataset:
        def __init__(self) -> None:
            self._samples = [
                {
                    "x": np.array(
                        [[-5.0, 2.0], [-4.0, -3.0], [1.0, 5.0]], dtype=np.float32
                    ),
                    "y": np.zeros((2, 2), dtype=np.float32),
                    "meta": {},
                }
            ]

        def __len__(self) -> int:
            return len(self._samples)

        def __getitem__(self, idx: int) -> Dict[str, Any]:
            return self._samples[idx]

    transform = LogTransform(base="10", per_sensor=False)
    transform.fit(DeterministicDataset())

    # All sensors should track the same minimum (-5.0)
    np.testing.assert_allclose(transform.min_values, [-5.0, -5.0])

    x = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    y = np.array([[4.0, 5.0]], dtype=np.float32)
    x_trans, y_trans, meta = transform(x, y, {})

    assert meta["log_base"] == "10"
    assert x_trans.shape == x.shape
    assert y_trans.shape == y.shape


def test_log_transform_selected_sensors_and_inverse_handles_missing_targets():
    """Selected sensor paths should only touch available channels and invert cleanly."""

    transform = LogTransform(base="2", sensors=[0, 2])
    transform.fit(MockDataset(dim=3))

    # Use deterministic minima to simplify validation
    transform.min_values = np.array([-1.0, -0.5, -2.0], dtype=np.float32)

    x = np.array(
        [
            [-0.5, 0.1, -1.5],
            [0.0, -0.2, -1.0],
            [0.5, 0.3, -0.75],
            [0.75, 0.0, 0.5],
        ],
        dtype=np.float32,
    )
    # Target has only two sensors, so sensor index 2 should be ignored
    y = np.array(
        [[0.0, -0.25], [0.5, 0.25], [1.0, 0.0], [1.5, 0.5]], dtype=np.float32
    )

    x_trans, y_trans, meta = transform(x, y, {})

    # Sensor 1 untouched, sensor 2 unchanged in y because it doesn't exist
    np.testing.assert_allclose(x_trans[:, 1], x[:, 1])
    np.testing.assert_allclose(y_trans[:, 1], y[:, 1])

    # Inversion should restore original values even when y is omitted
    x_inv, y_inv = transform.inverse(x_trans, None)
    np.testing.assert_allclose(x_inv, x, rtol=1e-6, atol=1e-6)
    assert y_inv is None


def test_log_transform_invalid_base_errors_during_forward():
    """An unknown logarithm base should raise a clear error when applying the transform."""

    dataset = MockDataset()
    transform = LogTransform(base="invalid")
    transform.fit(dataset)

    sample = dataset[0]
    x = np.abs(sample["x"]) + 1.0
    y = np.abs(sample["y"]) + 1.0

    with pytest.raises(ValueError, match="Unknown base"):
        transform(x, y, {})


def test_log_transform_invalid_base_errors_during_inverse():
    """Inverse should also guard against unsupported bases."""

    dataset = MockDataset()
    transform = LogTransform(base="natural")
    transform.fit(dataset)

    sample = dataset[0]
    x = np.abs(sample["x"]) + 1.0
    y = np.abs(sample["y"]) + 1.0
    x_trans, y_trans, _ = transform(x, y, {})

    transform.base = "mystery"

    with pytest.raises(ValueError, match="Unknown base"):
        transform.inverse(x_trans, y_trans)


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

    # Check shapes preserved and smoothing applied
    assert x.shape == sample["x"].shape
    assert y.shape == sample["y"].shape
    expected_x = uniform_filter1d(
        sample["x"], size=transform.window_size, axis=0, mode=transform.mode
    )
    np.testing.assert_allclose(x, expected_x)
    assert meta["smoothed"] is True


def test_smooth_gaussian():
    """Test Gaussian smoothing."""
    dataset = MockDataset(seq_len=8, dim=2, seed=123)
    transform = SmoothTransform(window_size=5, method="gaussian")

    transform.fit(dataset)
    sample = dataset[0]
    x, y, meta = transform(sample["x"], sample["y"], sample["meta"])

    sigma = transform.window_size / 6.0
    expected_x = gaussian_filter1d(sample["x"], sigma=sigma, axis=0, mode=transform.mode)
    expected_y = gaussian_filter1d(sample["y"], sigma=sigma, axis=0, mode=transform.mode)

    assert x.shape == sample["x"].shape
    assert y.shape == sample["y"].shape
    np.testing.assert_allclose(x, expected_x)
    np.testing.assert_allclose(y, expected_y)
    assert meta["smooth_method"] == "gaussian"


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
    expected_mean = dataset._x_data.mean(axis=(0, 1))
    np.testing.assert_allclose(np.asarray(transform.mean_values), expected_mean)

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


def test_detrend_inverse_restores_trend():
    """DetrendTransform inverse should restore the original linear trend."""
    transform = DetrendTransform(method="linear")
    transform.fit(MockDataset())

    time = np.arange(20, dtype=np.float32)[:, None]
    # Simple linear trend: y = 2t + 3
    x = np.hstack([2 * time + 3, 2 * time + 5])
    y = np.vstack([np.array([[1.0, 1.0]], dtype=np.float32), np.array([[2.0, 2.0]], dtype=np.float32)])

    x_det, y_det, meta = transform(x, y, {})
    x_inv, y_inv = transform.inverse(x_det, y_det, meta)

    assert np.allclose(x_inv, x)
    assert np.allclose(y_inv, y)


def test_detrend_requires_fit_and_valid_method():
    """DetrendTransform should enforce fitting and validate methods."""
    transform = DetrendTransform(method="linear")

    with pytest.raises(RuntimeError):
        transform(np.ones((3, 2)), np.ones((1, 2)), {})

    with pytest.raises(ValueError):
        DetrendTransform(method="unknown").fit(MockDataset())._fit_trend(np.ones((3, 1)))


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


def test_clip_transform_variants_and_inverse():
    """ClipTransform should compute bounds and raise for invalid configuration."""
    dataset = MockDataset()

    # Percentile with global bounds
    percentile_transform = ClipTransform(method="percentile", per_sensor=False)
    percentile_transform.fit(dataset)
    x, y, meta = percentile_transform(dataset[0]["x"], dataset[0]["y"], {})
    assert meta["clipped"] is True
    assert meta["clip_method"] == "percentile"

    # Absolute bounds and inverse no-op
    absolute_transform = ClipTransform(method="absolute", lower=-1.0, upper=1.0)
    absolute_transform.fit(dataset)
    clipped_x, clipped_y, _ = absolute_transform(dataset[0]["x"], dataset[0]["y"], {})
    restored_x, restored_y = absolute_transform.inverse(clipped_x, clipped_y)
    assert np.array_equal(clipped_x, restored_x)
    assert np.array_equal(clipped_y, restored_y)

    # Invalid method should raise
    with pytest.raises(ValueError):
        ClipTransform(method="invalid").fit(dataset)

    # Transform must be fitted before use
    unfitted = ClipTransform()
    with pytest.raises(RuntimeError):
        unfitted(dataset[0]["x"], dataset[0]["y"], {})


def test_transform_compose_inverse_order():
    """Compose should apply inverse transforms in reverse order."""

    class AddOneTransform(Transform):
        def fit(self, dataset):
            self.is_fitted = True
            return self

        def __call__(self, x, y, meta):
            return x + 1, y + 1, meta

        def inverse(self, x, y=None):
            return x - 1, y - 1 if y is not None else None

    class MultiplyTransform(Transform):
        def fit(self, dataset):
            self.is_fitted = True
            return self

        def __call__(self, x, y, meta):
            return x * 2, y * 2, meta

        def inverse(self, x, y=None):
            return x / 2, y / 2 if y is not None else None

    compose = Compose([AddOneTransform(), MultiplyTransform()])
    compose.fit(MockDataset())

    x, y = np.array([[1.0]]), np.array([[2.0]])
    x_out, y_out, _ = compose(x, y, {})
    x_inv, y_inv = compose.inverse(x_out, y_out)

    assert np.allclose(x_out, (x + 1) * 2)
    assert np.allclose(y_out, (y + 1) * 2)
    assert np.allclose(x_inv, x)
    assert np.allclose(y_inv, y)

    class NoInverseTransform(Transform):
        def fit(self, dataset):
            self.is_fitted = True
            return self

        def __call__(self, x, y, meta):
            return x, y, meta

    base_transform = NoInverseTransform().fit(MockDataset())
    base_transform(np.array([[1.0]]), np.array([[2.0]]), {})
    with pytest.raises(NotImplementedError):
        base_transform.inverse(np.array([[1.0]]))


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


# ===== Transform Statistics Persistence Tests =====


def test_difference_transform_get_set_stats():
    """Test that DifferenceTransform can save and restore stats."""
    dataset = MockDataset()

    # Create and fit transform
    transform = DifferenceTransform(sensors=['sensor_a', 'sensor_b'], order=2)
    transform.fit(dataset)

    # Get stats
    stats = transform.get_stats()
    assert 'sensors' in stats
    assert 'order' in stats
    assert 'sensor_indices' in stats
    assert stats['order'] == 2
    assert stats['sensors'] == ['sensor_a', 'sensor_b']

    # Create new transform and restore stats
    transform2 = DifferenceTransform()
    transform2.set_stats(stats)

    assert transform2.sensors == transform.sensors
    assert transform2.order == transform.order
    assert transform2.sensor_indices == transform.sensor_indices
    assert transform2.is_fitted


def test_difference_transform_get_stats_requires_fit():
    """Test that DifferenceTransform.get_stats() requires fitting first."""
    transform = DifferenceTransform(order=1)

    with pytest.raises(RuntimeError, match="Transform not fitted"):
        transform.get_stats()


def test_robust_scaler_get_set_stats():
    """Test that RobustScalerTransform can save and restore stats."""
    dataset = MockDataset()

    # Create and fit transform
    transform = RobustScalerTransform(
        per_sensor=True,
        quantile_range=(10.0, 90.0),
        with_centering=True,
        with_scaling=True
    )
    transform.fit(dataset)

    # Get stats
    stats = transform.get_stats()
    assert 'scaler_x_center' in stats
    assert 'scaler_x_scale' in stats
    assert 'scaler_x_n_features_in' in stats
    assert 'scaler_y_center' in stats
    assert 'scaler_y_scale' in stats
    assert 'scaler_y_n_features_in' in stats
    assert 'per_sensor' in stats
    assert 'quantile_range' in stats
    assert 'with_centering' in stats
    assert 'with_scaling' in stats

    # Create new transform and restore stats
    transform2 = RobustScalerTransform()
    transform2.set_stats(stats)

    # Verify attributes restored
    assert transform2.per_sensor == transform.per_sensor
    assert transform2.quantile_range == transform.quantile_range
    assert transform2.with_centering == transform.with_centering
    assert transform2.with_scaling == transform.with_scaling
    assert transform2.is_fitted

    # Verify scalers restored (including sklearn metadata)
    np.testing.assert_array_equal(transform2.scaler_x.center_, transform.scaler_x.center_)
    np.testing.assert_array_equal(transform2.scaler_x.scale_, transform.scaler_x.scale_)
    assert transform2.scaler_x.n_features_in_ == transform.scaler_x.n_features_in_
    np.testing.assert_array_equal(transform2.scaler_y.center_, transform.scaler_y.center_)
    np.testing.assert_array_equal(transform2.scaler_y.scale_, transform.scaler_y.scale_)
    assert transform2.scaler_y.n_features_in_ == transform.scaler_y.n_features_in_

    # Verify both transforms produce same output
    sample = dataset[0]
    x1, y1, _ = transform(sample["x"], sample["y"], sample["meta"])
    x2, y2, _ = transform2(sample["x"], sample["y"], sample["meta"])

    np.testing.assert_allclose(x1, x2, rtol=1e-6)
    np.testing.assert_allclose(y1, y2, rtol=1e-6)


def test_robust_scaler_get_stats_requires_fit():
    """Test that RobustScalerTransform.get_stats() requires fitting first."""
    transform = RobustScalerTransform()

    with pytest.raises(RuntimeError, match="Transform not fitted"):
        transform.get_stats()


def test_zscore_transform_get_set_stats():
    """Test that ZScoreTransform can save and restore stats (regression test)."""
    dataset = MockDataset()

    # Create and fit transform
    transform = ZScoreTransform(per_sensor=True, center=True, scale=True)
    transform.fit(dataset)

    # Get stats
    stats = transform.get_stats()
    assert 'scaler_x_mean' in stats
    assert 'scaler_x_scale' in stats
    assert 'scaler_y_mean' in stats
    assert 'scaler_y_scale' in stats

    # Create new transform and restore stats
    transform2 = ZScoreTransform()
    transform2.set_stats(stats)

    assert transform2.is_fitted

    # Verify both transforms produce same output
    sample = dataset[0]
    x1, y1, _ = transform(sample["x"], sample["y"], sample["meta"])
    x2, y2, _ = transform2(sample["x"], sample["y"], sample["meta"])

    np.testing.assert_allclose(x1, x2, rtol=1e-6)
    np.testing.assert_allclose(y1, y2, rtol=1e-6)
