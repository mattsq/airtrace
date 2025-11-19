"""Synthetic aircraft sensor data generator for cruise flight.

This module generates physically plausible aircraft cruise sensor readings
for testing and experimentation. The generator models realistic relationships
between sensors and includes configurable noise.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class CruiseProfile:
    """Physical parameters for a cruise flight segment.

    Attributes:
        initial_altitude: Starting altitude in feet
        initial_mach: Starting Mach number
        initial_weight: Starting aircraft weight in kg
        cruise_duration: Duration of cruise in seconds
        fuel_flow_base: Base fuel flow in kg/hour
        n1_base: Base N1 percentage
        sample_rate: Sampling rate in Hz
    """

    initial_altitude: float = 35000.0  # feet
    initial_mach: float = 0.82
    initial_weight: float = 70000.0  # kg
    cruise_duration: int = 3600  # seconds (1 hour)
    fuel_flow_base: float = 2500.0  # kg/hour
    n1_base: float = 85.0  # percent
    sample_rate: float = 1.0  # Hz

    # Variation parameters
    altitude_variation: float = 100.0  # feet
    mach_variation: float = 0.01
    turbulence_level: float = 0.1  # 0-1 scale

    # Noise levels (standard deviation as fraction of mean)
    fuel_flow_noise: float = 0.01
    mach_noise: float = 0.005
    altitude_noise: float = 10.0  # feet
    n1_noise: float = 0.5  # percent
    oat_noise: float = 1.0  # Celsius


class SyntheticCruiseGenerator:
    """Generate synthetic aircraft cruise sensor data.

    This generator creates physically plausible timeseries data for aircraft
    sensors during cruise flight. It models realistic relationships between:
    - Fuel flow, engine thrust (N1), and aircraft weight
    - Altitude and outside air temperature (ISA model)
    - Mach number stability with small perturbations
    - Correlated noise and slow drift

    The generated data follows the AirTrace raw data format and can be
    processed through the standard interim → processed pipeline.
    """

    def __init__(
        self,
        data_root: Path,
        seed: Optional[int] = None
    ):
        """Initialize synthetic generator.

        Args:
            data_root: Root directory for data storage
            seed: Random seed for reproducibility
        """
        self.data_root = Path(data_root)
        self.raw_dir = self.data_root / "raw"
        self.raw_dir.mkdir(exist_ok=True, parents=True)

        self.rng = np.random.default_rng(seed)
        self._seed = seed

    def generate_flight(
        self,
        flight_id: str,
        profile: CruiseProfile,
        save: bool = True
    ) -> pd.DataFrame:
        """Generate a single synthetic cruise flight.

        Args:
            flight_id: Unique identifier for this flight
            profile: CruiseProfile with physical parameters
            save: If True, save to raw directory

        Returns:
            DataFrame with timestamp and sensor columns
        """
        # Time vector
        n_samples = int(profile.cruise_duration * profile.sample_rate)
        time_seconds = np.arange(n_samples) / profile.sample_rate

        # Generate base trajectories with slow drift
        altitude = self._generate_altitude(n_samples, profile)
        mach = self._generate_mach(n_samples, profile)
        weight = self._generate_weight(n_samples, profile)

        # Compute derived quantities
        oat = self._compute_oat(altitude, profile)
        n1 = self._compute_n1(weight, altitude, mach, profile)
        fuel_flow = self._compute_fuel_flow(n1, altitude, profile)

        # Create DataFrame
        df = pd.DataFrame({
            "timestamp": pd.Timestamp("2024-01-01") + pd.to_timedelta(time_seconds, unit="s"),
            "fuel_flow": fuel_flow,
            "mach": mach,
            "altitude": altitude,
            "oat": oat,
            "n1": n1,
            "weight": weight
        })

        # Save if requested
        if save:
            output_path = self.raw_dir / f"{flight_id}.parquet"
            df.to_parquet(output_path, index=False)

        return df

    def generate_dataset(
        self,
        n_flights: int,
        profile: CruiseProfile,
        flight_id_prefix: str = "synthetic_cruise"
    ) -> List[str]:
        """Generate multiple synthetic flights.

        Args:
            n_flights: Number of flights to generate
            profile: Base CruiseProfile (will be varied per flight)
            flight_id_prefix: Prefix for flight IDs

        Returns:
            List of generated flight IDs
        """
        flight_ids = []

        for i in range(n_flights):
            flight_id = f"{flight_id_prefix}_{i:04d}"

            # Vary profile slightly per flight
            varied_profile = self._vary_profile(profile)

            # Generate and save
            self.generate_flight(flight_id, varied_profile, save=True)
            flight_ids.append(flight_id)

        return flight_ids

    def _generate_altitude(
        self,
        n_samples: int,
        profile: CruiseProfile
    ) -> np.ndarray:
        """Generate altitude timeseries.

        Models stable cruise with small variations and occasional step climbs.
        """
        altitude = np.full(n_samples, profile.initial_altitude, dtype=float)

        # Add slow drift (pressure changes, pilot corrections)
        drift_freq = 2 * np.pi / (300 * profile.sample_rate)  # ~5 min period
        t = np.arange(n_samples)
        drift = profile.altitude_variation * np.sin(drift_freq * t)
        altitude += drift

        # Add small random walk
        walk = np.cumsum(self.rng.normal(0, profile.altitude_noise, n_samples))
        walk -= walk.mean()  # Keep centered
        altitude += walk * 0.5

        # Occasional step climbs (1-2 during long flights)
        if n_samples > 1800 * profile.sample_rate:  # If > 30 min
            n_steps = self.rng.poisson(1.5)
            for _ in range(n_steps):
                step_idx = self.rng.integers(n_samples // 4, 3 * n_samples // 4)
                step_size = self.rng.choice([2000, 4000])  # FL200 or FL400
                altitude[step_idx:] += step_size

        # Add measurement noise
        altitude += self.rng.normal(0, profile.altitude_noise, n_samples)

        return altitude

    def _generate_mach(
        self,
        n_samples: int,
        profile: CruiseProfile
    ) -> np.ndarray:
        """Generate Mach number timeseries.

        Models stable cruise speed with small variations from turbulence/wind.
        """
        mach = np.full(n_samples, profile.initial_mach, dtype=float)

        # Add slow oscillation (speed adjustments)
        drift_freq = 2 * np.pi / (600 * profile.sample_rate)  # ~10 min period
        t = np.arange(n_samples)
        drift = profile.mach_variation * np.sin(drift_freq * t)
        mach += drift

        # Add turbulence (higher frequency)
        if profile.turbulence_level > 0:
            turb_freq = 2 * np.pi / (30 * profile.sample_rate)  # ~30 sec period
            turbulence = (
                profile.turbulence_level *
                profile.mach_variation *
                np.sin(turb_freq * t + self.rng.uniform(0, 2*np.pi))
            )
            mach += turbulence

        # Add measurement noise
        mach += self.rng.normal(0, profile.mach_noise, n_samples)

        # Ensure positive and reasonable
        mach = np.clip(mach, 0.7, 0.9)

        return mach

    def _generate_weight(
        self,
        n_samples: int,
        profile: CruiseProfile
    ) -> np.ndarray:
        """Generate aircraft weight timeseries.

        Models monotonic decrease as fuel is burned.
        """
        # Total fuel burn over cruise
        total_fuel_kg = (profile.fuel_flow_base / 3600) * profile.cruise_duration

        # Linear decrease (simplified, real burn rate varies slightly)
        time_fraction = np.linspace(0, 1, n_samples)
        weight = profile.initial_weight - total_fuel_kg * time_fraction

        # Add tiny variations (measurement noise only)
        weight += self.rng.normal(0, 10, n_samples)

        return weight

    def _compute_oat(
        self,
        altitude: np.ndarray,
        profile: CruiseProfile
    ) -> np.ndarray:
        """Compute outside air temperature using ISA model.

        ISA (International Standard Atmosphere):
        - Sea level: 15°C
        - Lapse rate: -6.5°C per 1000m up to 11km
        - Stratosphere: -56.5°C above 11km
        """
        # Convert altitude from feet to meters
        altitude_m = altitude * 0.3048

        # ISA temperature
        oat = np.where(
            altitude_m <= 11000,
            15.0 - 6.5 * (altitude_m / 1000),  # Troposphere
            -56.5  # Stratosphere
        )

        # Add weather variation (non-ISA days)
        oat += self.rng.normal(0, 5, size=altitude.shape)  # ±5°C variation

        # Add measurement noise
        oat += self.rng.normal(0, profile.oat_noise, size=altitude.shape)

        return oat

    def _compute_n1(
        self,
        weight: np.ndarray,
        altitude: np.ndarray,
        mach: np.ndarray,
        profile: CruiseProfile
    ) -> np.ndarray:
        """Compute engine N1 based on thrust requirements.

        N1 correlates with thrust required to maintain speed.
        Thrust requirements depend on weight, altitude, and speed.
        """
        # Simplified thrust model
        # Higher weight → more thrust
        # Higher altitude → more thrust (less dense air)
        # Higher speed → more thrust

        weight_factor = (weight / profile.initial_weight)
        altitude_factor = (altitude / profile.initial_altitude)
        speed_factor = (mach / profile.initial_mach)

        n1 = profile.n1_base * weight_factor * (0.5 + 0.5 * altitude_factor) * speed_factor

        # Add slow variations (engine adjustments)
        drift_freq = 2 * np.pi / (180 * profile.sample_rate)  # ~3 min period
        t = np.arange(len(weight))
        drift = 2.0 * np.sin(drift_freq * t)
        n1 += drift

        # Add measurement noise
        n1 += self.rng.normal(0, profile.n1_noise, size=weight.shape)

        # Ensure reasonable range
        n1 = np.clip(n1, 75, 95)

        return n1

    def _compute_fuel_flow(
        self,
        n1: np.ndarray,
        altitude: np.ndarray,
        profile: CruiseProfile
    ) -> np.ndarray:
        """Compute fuel flow based on thrust and altitude.

        Fuel flow strongly correlates with N1 (engine thrust).
        Also affected by altitude (air density).
        """
        # Base relationship: fuel flow ∝ N1
        n1_factor = (n1 / profile.n1_base)

        # Altitude effect (less dense air → more efficient)
        altitude_factor = 1.0 - 0.1 * ((altitude - 30000) / 10000)
        altitude_factor = np.clip(altitude_factor, 0.8, 1.2)

        fuel_flow = profile.fuel_flow_base * n1_factor * altitude_factor

        # Add measurement noise
        fuel_flow += self.rng.normal(0, profile.fuel_flow_base * profile.fuel_flow_noise, size=n1.shape)

        # Ensure positive
        fuel_flow = np.maximum(fuel_flow, 1000)

        return fuel_flow

    def _vary_profile(self, base_profile: CruiseProfile) -> CruiseProfile:
        """Create a varied version of a cruise profile.

        Adds realistic flight-to-flight variation in parameters.
        """
        return CruiseProfile(
            initial_altitude=base_profile.initial_altitude + self.rng.normal(0, 2000),
            initial_mach=base_profile.initial_mach + self.rng.normal(0, 0.02),
            initial_weight=base_profile.initial_weight + self.rng.normal(0, 5000),
            cruise_duration=base_profile.cruise_duration,
            fuel_flow_base=base_profile.fuel_flow_base + self.rng.normal(0, 100),
            n1_base=base_profile.n1_base + self.rng.normal(0, 2),
            sample_rate=base_profile.sample_rate,
            altitude_variation=base_profile.altitude_variation,
            mach_variation=base_profile.mach_variation,
            turbulence_level=np.clip(base_profile.turbulence_level + self.rng.normal(0, 0.1), 0, 1),
            fuel_flow_noise=base_profile.fuel_flow_noise,
            mach_noise=base_profile.mach_noise,
            altitude_noise=base_profile.altitude_noise,
            n1_noise=base_profile.n1_noise,
            oat_noise=base_profile.oat_noise
        )


def create_synthetic_dataset(
    data_root: Path,
    n_flights: int = 20,
    profile: Optional[CruiseProfile] = None,
    seed: Optional[int] = 42,
    train_val_test_split: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    flight_id_prefix: str = "synthetic_cruise"
) -> Dict[str, List[str]]:
    """Create a complete synthetic dataset with train/val/test splits.

    This is a convenience function that:
    1. Generates multiple synthetic flights
    2. Splits them into train/val/test sets
    3. Returns the flight IDs for each split

    Args:
        data_root: Root directory for data storage
        n_flights: Total number of flights to generate
        profile: CruiseProfile to use (None for defaults)
        seed: Random seed for reproducibility
        train_val_test_split: Tuple of (train, val, test) fractions
        flight_id_prefix: Prefix for flight IDs (default: "synthetic_cruise")

    Returns:
        Dictionary with keys 'train', 'val', 'test' mapping to flight ID lists
    """
    if profile is None:
        profile = CruiseProfile()

    # Generate flights
    generator = SyntheticCruiseGenerator(data_root, seed=seed)
    flight_ids = generator.generate_dataset(
        n_flights=n_flights,
        profile=profile,
        flight_id_prefix=flight_id_prefix
    )

    # Split into train/val/test
    n_train = int(n_flights * train_val_test_split[0])
    n_val = int(n_flights * train_val_test_split[1])

    splits = {
        "train": flight_ids[:n_train],
        "val": flight_ids[n_train:n_train + n_val],
        "test": flight_ids[n_train + n_val:]
    }

    return splits
