"""Tests for synthetic data generation."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from airtrace.data.synthetic import (
    CruiseProfile,
    SyntheticCruiseGenerator,
    create_synthetic_dataset,
)


class TestCruiseProfile:
    """Tests for CruiseProfile dataclass."""

    def test_default_profile_creation(self):
        """Test creating a profile with default values."""
        profile = CruiseProfile()

        assert profile.initial_altitude == 35000.0
        assert profile.initial_mach == 0.82
        assert profile.sample_rate == 1.0

    def test_custom_profile_creation(self):
        """Test creating a profile with custom values."""
        profile = CruiseProfile(
            initial_altitude=40000.0,
            initial_mach=0.85,
            cruise_duration=7200
        )

        assert profile.initial_altitude == 40000.0
        assert profile.initial_mach == 0.85
        assert profile.cruise_duration == 7200

    def test_profile_validation(self):
        """Test that profile parameters are reasonable."""
        profile = CruiseProfile()

        assert profile.initial_altitude > 0
        assert 0 < profile.initial_mach < 1.0
        assert profile.fuel_flow_base > 0
        assert 0 <= profile.n1_base <= 100


class TestSyntheticCruiseGenerator:
    """Tests for SyntheticCruiseGenerator."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create a temporary directory for test data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def generator(self, temp_data_dir):
        """Create a generator with fixed seed."""
        return SyntheticCruiseGenerator(temp_data_dir, seed=42)

    @pytest.fixture
    def short_profile(self):
        """Create a short profile for fast testing."""
        return CruiseProfile(
            cruise_duration=300,  # 5 minutes
            sample_rate=1.0
        )

    def test_generator_initialization(self, temp_data_dir):
        """Test generator initialization creates directories."""
        generator = SyntheticCruiseGenerator(temp_data_dir, seed=42)

        assert generator.data_root == temp_data_dir
        assert (temp_data_dir / "raw").exists()

    def test_generate_single_flight(self, generator, short_profile):
        """Test generating a single flight."""
        df = generator.generate_flight(
            flight_id="test_001",
            profile=short_profile,
            save=False
        )

        # Check structure
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 300  # 5 minutes at 1Hz

        # Check columns
        expected_columns = ["timestamp", "fuel_flow", "mach", "altitude", "oat", "n1", "weight"]
        assert all(col in df.columns for col in expected_columns)

        # Check timestamp
        assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])

    def test_sensor_values_plausible(self, generator, short_profile):
        """Test that generated sensor values are physically plausible."""
        df = generator.generate_flight(
            flight_id="test_002",
            profile=short_profile,
            save=False
        )

        # Altitude: typical cruise range
        assert df["altitude"].min() > 20000
        assert df["altitude"].max() < 45000

        # Mach: subsonic cruise
        assert df["mach"].min() > 0.7
        assert df["mach"].max() < 0.9

        # Fuel flow: positive and reasonable
        assert (df["fuel_flow"] > 0).all()
        assert df["fuel_flow"].mean() > 1000  # kg/hour

        # N1: realistic range
        assert df["n1"].min() > 70
        assert df["n1"].max() < 100

        # OAT: cold at altitude
        assert df["oat"].max() < 0  # Should be negative Celsius at cruise altitude

        # Weight: decreases over time
        assert df["weight"].iloc[-1] < df["weight"].iloc[0]

    def test_reproducibility(self, temp_data_dir, short_profile):
        """Test that same seed produces same data."""
        gen1 = SyntheticCruiseGenerator(temp_data_dir, seed=123)
        gen2 = SyntheticCruiseGenerator(temp_data_dir, seed=123)

        df1 = gen1.generate_flight("test_003", short_profile, save=False)
        df2 = gen2.generate_flight("test_003", short_profile, save=False)

        # Should be identical
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_produce_different_data(self, temp_data_dir, short_profile):
        """Test that different seeds produce different data."""
        gen1 = SyntheticCruiseGenerator(temp_data_dir, seed=123)
        gen2 = SyntheticCruiseGenerator(temp_data_dir, seed=456)

        df1 = gen1.generate_flight("test_004", short_profile, save=False)
        df2 = gen2.generate_flight("test_004", short_profile, save=False)

        # Should be different
        assert not np.allclose(df1["fuel_flow"].values, df2["fuel_flow"].values)

    def test_save_flight_to_disk(self, generator, short_profile):
        """Test saving flight to parquet file."""
        flight_id = "test_save"
        df = generator.generate_flight(
            flight_id=flight_id,
            profile=short_profile,
            save=True
        )

        # Check file was created
        file_path = generator.raw_dir / f"{flight_id}.parquet"
        assert file_path.exists()

        # Check we can load it
        loaded_df = pd.read_parquet(file_path)
        pd.testing.assert_frame_equal(df, loaded_df)

    def test_generate_multiple_flights(self, generator, short_profile):
        """Test generating multiple flights."""
        flight_ids = generator.generate_dataset(
            n_flights=5,
            profile=short_profile,
            flight_id_prefix="test_multi"
        )

        assert len(flight_ids) == 5

        # Check all files exist
        for flight_id in flight_ids:
            file_path = generator.raw_dir / f"{flight_id}.parquet"
            assert file_path.exists()

    def test_flight_variation(self, generator, short_profile):
        """Test that flights in a dataset have variation."""
        flight_ids = generator.generate_dataset(
            n_flights=3,
            profile=short_profile,
            flight_id_prefix="test_var"
        )

        # Load all flights
        dfs = []
        for flight_id in flight_ids:
            file_path = generator.raw_dir / f"{flight_id}.parquet"
            dfs.append(pd.read_parquet(file_path))

        # Check that initial conditions vary
        initial_altitudes = [df["altitude"].iloc[0] for df in dfs]
        assert len(set(initial_altitudes)) > 1  # Not all identical

    def test_altitude_generation(self, generator, short_profile):
        """Test altitude timeseries characteristics."""
        df = generator.generate_flight("test_alt", short_profile, save=False)

        altitude = df["altitude"].values

        # Should have slow variations
        altitude_diff = np.diff(altitude)
        assert np.abs(altitude_diff).max() < 500  # No huge jumps

        # Should stay around initial value (no long-term drift)
        assert np.abs(altitude.mean() - short_profile.initial_altitude) < 1000

    def test_mach_generation(self, generator, short_profile):
        """Test Mach number timeseries characteristics."""
        df = generator.generate_flight("test_mach", short_profile, save=False)

        mach = df["mach"].values

        # Should have small variations
        assert mach.std() < 0.05

        # Should stay near initial value
        assert np.abs(mach.mean() - short_profile.initial_mach) < 0.02

    def test_weight_decreases(self, generator, short_profile):
        """Test that weight decreases monotonically (fuel burn)."""
        df = generator.generate_flight("test_weight", short_profile, save=False)

        weight = df["weight"].values

        # Should decrease overall (allowing for noise)
        # Check that moving average decreases
        window = 30
        moving_avg = np.convolve(weight, np.ones(window)/window, mode='valid')
        assert moving_avg[-1] < moving_avg[0]

    def test_oat_altitude_relationship(self, generator, short_profile):
        """Test that OAT follows ISA relationship with altitude."""
        df = generator.generate_flight("test_oat", short_profile, save=False)

        # At cruise altitude (~35000 ft), OAT should be around -50°C
        mean_oat = df["oat"].mean()
        assert -70 < mean_oat < -30  # Reasonable range

    def test_fuel_flow_n1_correlation(self, generator, short_profile):
        """Test that fuel flow correlates with N1."""
        df = generator.generate_flight("test_corr", short_profile, save=False)

        # Compute correlation
        corr = np.corrcoef(df["fuel_flow"].values, df["n1"].values)[0, 1]

        # Should be strongly positively correlated
        assert corr > 0.7


class TestCreateSyntheticDataset:
    """Tests for create_synthetic_dataset convenience function."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create a temporary directory for test data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_create_complete_dataset(self, temp_data_dir):
        """Test creating a complete dataset with splits."""
        short_profile = CruiseProfile(cruise_duration=60, sample_rate=1.0)

        splits = create_synthetic_dataset(
            data_root=temp_data_dir,
            n_flights=10,
            profile=short_profile,
            seed=42,
            train_val_test_split=(0.6, 0.2, 0.2)
        )

        # Check splits
        assert "train" in splits
        assert "val" in splits
        assert "test" in splits

        # Check counts
        assert len(splits["train"]) == 6
        assert len(splits["val"]) == 2
        assert len(splits["test"]) == 2

        # Check all flights exist
        all_flights = splits["train"] + splits["val"] + splits["test"]
        raw_dir = temp_data_dir / "raw"
        for flight_id in all_flights:
            assert (raw_dir / f"{flight_id}.parquet").exists()

    def test_split_no_overlap(self, temp_data_dir):
        """Test that train/val/test splits don't overlap."""
        short_profile = CruiseProfile(cruise_duration=60, sample_rate=1.0)

        splits = create_synthetic_dataset(
            data_root=temp_data_dir,
            n_flights=9,
            profile=short_profile,
            seed=42
        )

        # Check no overlap
        train_set = set(splits["train"])
        val_set = set(splits["val"])
        test_set = set(splits["test"])

        assert len(train_set & val_set) == 0
        assert len(train_set & test_set) == 0
        assert len(val_set & test_set) == 0

    def test_default_profile(self, temp_data_dir):
        """Test using default profile (None)."""
        splits = create_synthetic_dataset(
            data_root=temp_data_dir,
            n_flights=3,
            profile=None,  # Should use defaults
            seed=42
        )

        assert len(splits["train"]) + len(splits["val"]) + len(splits["test"]) == 3
