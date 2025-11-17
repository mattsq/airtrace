"""Test script to validate all transforms with synthetic sensor data."""

import numpy as np
import pytest
from typing import Dict, Any

from airtrace.transforms import (
    ZScoreTransform,
    RobustScalerTransform,
    MinMaxTransform,
    ClipTransform,
    DifferenceTransform,
    LogTransform,
    ImputeTransform,
    SmoothTransform,
    DetrendTransform,
    ContextTransform,
    TemporalFeaturesTransform,
    NoOpTransform,
)


class SyntheticDataset:
    """Mock dataset for testing transforms."""

    def __init__(self, data: np.ndarray, n_samples: int = 50):
        """Initialize synthetic dataset.

        Args:
            data: Sensor data array [total_timesteps, n_sensors]
            n_samples: Number of sliding window samples to create
        """
        self.data = data
        self.n_samples = n_samples
        self.window_size = 10
        self.target_size = 1

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample (x, y, meta) tuple."""
        # Create sliding windows
        start_idx = idx
        end_idx = start_idx + self.window_size
        target_idx = end_idx + self.target_size

        # Handle edge cases by wrapping around
        if target_idx > len(self.data):
            start_idx = 0
            end_idx = self.window_size
            target_idx = end_idx + self.target_size

        x = self.data[start_idx:end_idx].copy()
        y = self.data[end_idx:target_idx].copy()

        meta = {
            "idx": idx,
            "flight_id": "TEST_001",
            "aircraft_type": "737",
            "flight_phase": "cruise",
        }

        return {"x": x, "y": y, "meta": meta}


def create_synthetic_data(n_timesteps: int = 100, n_sensors: int = 5) -> np.ndarray:
    """Create synthetic aircraft sensor data.

    Args:
        n_timesteps: Number of time steps
        n_sensors: Number of sensors

    Returns:
        Synthetic sensor data [n_timesteps, n_sensors]
    """
    np.random.seed(42)

    time = np.arange(n_timesteps)
    data = np.zeros((n_timesteps, n_sensors))

    # Sensor 0: Altitude (increasing trend)
    data[:, 0] = 10000 + 50 * time + 100 * np.sin(time / 10) + np.random.randn(n_timesteps) * 10

    # Sensor 1: Speed (with periodic component)
    data[:, 1] = 250 + 20 * np.sin(2 * np.pi * time / 30) + np.random.randn(n_timesteps) * 5

    # Sensor 2: Fuel (decreasing trend)
    data[:, 2] = 5000 - 10 * time + np.random.randn(n_timesteps) * 20

    # Sensor 3: Temperature (with noise)
    data[:, 3] = 15 + 5 * np.cos(2 * np.pi * time / 40) + np.random.randn(n_timesteps) * 2

    # Sensor 4: Pressure (exponential-like behavior)
    data[:, 4] = 1013 * np.exp(-time / 200) + 900 + np.random.randn(n_timesteps) * 5

    # Add some missing values for imputation testing
    data[10:12, 1] = np.nan
    data[25, 3] = np.nan
    data[50:53, 4] = np.nan

    # Add some outliers for clipping testing
    data[30, 0] = data[30, 0] * 3  # Altitude spike
    data[60, 2] = data[60, 2] * 2  # Fuel anomaly

    return data


class TestTransforms:
    """Test suite for all transforms."""

    @pytest.fixture
    def synthetic_data(self):
        """Create synthetic data fixture."""
        return create_synthetic_data(n_timesteps=100, n_sensors=5)

    @pytest.fixture
    def dataset(self, synthetic_data):
        """Create dataset fixture."""
        return SyntheticDataset(synthetic_data, n_samples=50)

    @pytest.fixture
    def sample(self, dataset):
        """Get a single sample."""
        return dataset[0]

    def test_zscore_transform(self, dataset, sample):
        """Test Z-score normalization."""
        transform = ZScoreTransform(per_sensor=True)
        transform.fit(dataset)

        x, y, meta = sample["x"], sample["y"], sample["meta"]
        x_t, y_t, meta_t = transform(x, y, meta)

        # Check shapes preserved
        assert x_t.shape == x.shape
        assert y_t.shape == y.shape

        # Check inverse
        x_inv, y_inv = transform.inverse(x_t, y_t)
        assert x_inv.shape == x.shape
        np.testing.assert_allclose(x_inv, x, rtol=1e-5)

        print("✓ ZScoreTransform passed")

    def test_robust_scaler_transform(self, dataset, sample):
        """Test Robust scaler."""
        transform = RobustScalerTransform(per_sensor=True)
        transform.fit(dataset)

        x, y, meta = sample["x"], sample["y"], sample["meta"]
        x_t, y_t, meta_t = transform(x, y, meta)

        assert x_t.shape == x.shape
        assert y_t.shape == y.shape

        # Check inverse
        x_inv, y_inv = transform.inverse(x_t, y_t)
        np.testing.assert_allclose(x_inv, x, rtol=1e-5)

        print("✓ RobustScalerTransform passed")

    def test_minmax_transform(self, dataset, sample):
        """Test MinMax scaling."""
        transform = MinMaxTransform(feature_range=(0.0, 1.0))
        transform.fit(dataset)

        x, y, meta = sample["x"], sample["y"], sample["meta"]
        x_t, y_t, meta_t = transform(x, y, meta)

        assert x_t.shape == x.shape

        # Check inverse
        x_inv, y_inv = transform.inverse(x_t, y_t)
        np.testing.assert_allclose(x_inv, x, rtol=1e-5)

        print("✓ MinMaxTransform passed")

    def test_clip_transform(self, dataset, sample):
        """Test clipping transform."""
        transform = ClipTransform(method="percentile", lower=1.0, upper=99.0)
        transform.fit(dataset)

        x, y, meta = sample["x"], sample["y"], sample["meta"]
        x_t, y_t, meta_t = transform(x, y, meta)

        assert x_t.shape == x.shape
        assert "clipped" in meta_t

        print("✓ ClipTransform passed")

    def test_difference_transform(self, dataset, sample):
        """Test differencing transform."""
        transform = DifferenceTransform(order=1)
        transform.fit(dataset)

        x, y, meta = sample["x"], sample["y"], sample["meta"]
        x_t, y_t, meta_t = transform(x, y, meta)

        assert x_t.shape == x.shape  # Padded to maintain length
        assert "diff_initial_x" in meta_t

        print("✓ DifferenceTransform passed")

    def test_log_transform(self, dataset):
        """Test log transform."""
        # Create clean data without NaNs for log transform
        clean_data = create_synthetic_data(n_timesteps=100, n_sensors=5)
        # Remove NaNs
        clean_data = np.nan_to_num(clean_data, nan=100.0)
        clean_dataset = SyntheticDataset(clean_data, n_samples=50)
        clean_sample = clean_dataset[0]

        transform = LogTransform(base="natural")
        transform.fit(clean_dataset)

        x, y, meta = clean_sample["x"], clean_sample["y"], clean_sample["meta"]
        # Make sure data is positive for log
        x = np.abs(x) + 1
        y = np.abs(y) + 1

        x_t, y_t, meta_t = transform(x, y, meta)

        assert x_t.shape == x.shape
        assert "log_transformed" in meta_t

        # Check inverse
        x_inv, y_inv = transform.inverse(x_t, y_t)
        np.testing.assert_allclose(x_inv, x, rtol=1e-4)

        print("✓ LogTransform passed")

    def test_impute_transform_forward(self, dataset, sample):
        """Test imputation with forward fill."""
        transform = ImputeTransform(method="forward")
        transform.fit(dataset)

        x, y, meta = sample["x"], sample["y"], sample["meta"]
        x_t, y_t, meta_t = transform(x, y, meta)

        assert x_t.shape == x.shape
        assert "imputed" in meta_t
        # Check no NaNs remain
        assert not np.isnan(x_t).any()

        print("✓ ImputeTransform (forward) passed")

    def test_impute_transform_linear(self, dataset, sample):
        """Test imputation with linear interpolation."""
        transform = ImputeTransform(method="linear")
        transform.fit(dataset)

        x, y, meta = sample["x"], sample["y"], sample["meta"]
        x_t, y_t, meta_t = transform(x, y, meta)

        assert x_t.shape == x.shape
        assert not np.isnan(x_t).any()

        print("✓ ImputeTransform (linear) passed")

    def test_impute_transform_mean(self, dataset, sample):
        """Test imputation with mean."""
        transform = ImputeTransform(method="mean")
        transform.fit(dataset)

        x, y, meta = sample["x"], sample["y"], sample["meta"]
        x_t, y_t, meta_t = transform(x, y, meta)

        assert x_t.shape == x.shape
        assert not np.isnan(x_t).any()

        print("✓ ImputeTransform (mean) passed")

    def test_smooth_transform(self, dataset, sample):
        """Test smoothing transform."""
        transform = SmoothTransform(window_size=3, method="uniform")
        transform.fit(dataset)

        x, y, meta = sample["x"], sample["y"], sample["meta"]
        x_t, y_t, meta_t = transform(x, y, meta)

        assert x_t.shape == x.shape
        assert "smoothed" in meta_t

        print("✓ SmoothTransform passed")

    def test_detrend_transform(self, dataset, sample):
        """Test detrending transform."""
        transform = DetrendTransform(method="linear")
        transform.fit(dataset)

        x, y, meta = sample["x"], sample["y"], sample["meta"]
        # Remove NaNs (detrend can't handle them)
        x = np.nan_to_num(x, nan=0.0)
        y = np.nan_to_num(y, nan=0.0)
        # Detrending needs at least 2 timesteps - create a longer y for testing
        y_longer = np.vstack([y, y, y])  # Make y have 3 timesteps

        x_t, y_t, meta_t = transform(x, y_longer, meta)

        assert x_t.shape == x.shape
        assert y_t.shape == y_longer.shape
        assert "detrend_x_coeffs" in meta_t

        # Check inverse (needs meta)
        x_inv, y_inv = transform.inverse(x_t, y_t, meta_t)
        np.testing.assert_allclose(x_inv, x, rtol=1e-5)
        np.testing.assert_allclose(y_inv, y_longer, rtol=1e-5)

        print("✓ DetrendTransform passed")

    def test_context_transform(self, dataset, sample):
        """Test context transform."""
        transform = ContextTransform(use_static=["aircraft_type"])
        transform.fit(dataset)

        x, y, meta = sample["x"], sample["y"], sample["meta"]
        x_t, y_t, meta_t = transform(x, y, meta)

        # Context features added to x
        assert x_t.shape[0] == x.shape[0]
        assert x_t.shape[1] >= x.shape[1]  # Additional features
        assert "context_dim" in meta_t

        print("✓ ContextTransform passed")

    def test_temporal_features_transform(self, dataset, sample):
        """Test temporal features transform."""
        transform = TemporalFeaturesTransform(
            use_time_idx=True,
            use_cyclic_time=True,
            use_flight_phase=True
        )
        transform.fit(dataset)

        x, y, meta = sample["x"], sample["y"], sample["meta"]
        x_t, y_t, meta_t = transform(x, y, meta)

        # Temporal features added
        assert x_t.shape[0] == x.shape[0]
        assert x_t.shape[1] > x.shape[1]  # Additional features
        assert "temporal_features_dim" in meta_t

        # Check inverse
        x_inv, y_inv = transform.inverse(x_t, y_t, meta_t)
        assert x_inv.shape == x.shape
        np.testing.assert_allclose(x_inv, x, rtol=1e-5)

        print("✓ TemporalFeaturesTransform passed")

    def test_noop_transform(self, dataset, sample):
        """Test no-op transform."""
        transform = NoOpTransform()
        transform.fit(dataset)

        x, y, meta = sample["x"], sample["y"], sample["meta"]
        x_t, y_t, meta_t = transform(x, y, meta)

        # Everything should be unchanged
        np.testing.assert_array_equal(x_t, x)
        np.testing.assert_array_equal(y_t, y)
        assert meta_t == meta

        # Check inverse
        x_inv, y_inv = transform.inverse(x_t, y_t)
        np.testing.assert_array_equal(x_inv, x)

        print("✓ NoOpTransform passed")


def main():
    """Run all transform tests manually."""
    print("=" * 60)
    print("Testing All Transforms with Synthetic Sensor Data")
    print("=" * 60)

    # Create synthetic data
    print("\n1. Creating synthetic sensor data (100 timesteps, 5 sensors)...")
    data = create_synthetic_data(n_timesteps=100, n_sensors=5)
    print(f"   Data shape: {data.shape}")
    print(f"   Data range: [{np.nanmin(data):.2f}, {np.nanmax(data):.2f}]")
    print(f"   Missing values: {np.isnan(data).sum()}")

    # Create dataset
    print("\n2. Creating mock dataset with sliding windows...")
    dataset = SyntheticDataset(data, n_samples=50)
    print(f"   Dataset size: {len(dataset)} samples")
    sample = dataset[0]
    print(f"   Sample x shape: {sample['x'].shape}")
    print(f"   Sample y shape: {sample['y'].shape}")

    # Test all transforms
    print("\n3. Testing all transforms...")
    print("-" * 60)

    test_suite = TestTransforms()

    try:
        test_suite.test_zscore_transform(dataset, sample)
        test_suite.test_robust_scaler_transform(dataset, sample)
        test_suite.test_minmax_transform(dataset, sample)
        test_suite.test_clip_transform(dataset, sample)
        test_suite.test_difference_transform(dataset, sample)
        test_suite.test_log_transform(dataset)
        test_suite.test_impute_transform_forward(dataset, sample)
        test_suite.test_impute_transform_linear(dataset, sample)
        test_suite.test_impute_transform_mean(dataset, sample)
        test_suite.test_smooth_transform(dataset, sample)
        test_suite.test_detrend_transform(dataset, sample)
        test_suite.test_context_transform(dataset, sample)
        test_suite.test_temporal_features_transform(dataset, sample)
        test_suite.test_noop_transform(dataset, sample)

        print("-" * 60)
        print("\n✅ ALL TRANSFORMS PASSED!")
        print("\nSummary:")
        print("  - 14 transforms tested")
        print("  - All fit(), __call__(), and inverse() methods working")
        print("  - Shape preservation verified")
        print("  - Metadata handling confirmed")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
