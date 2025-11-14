"""Generate toy synthetic data for testing."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate_synthetic_flight(
    flight_id: str,
    duration_minutes: int = 60,
    sample_rate_hz: float = 1.0,
    num_sensors: int = 5,
    noise_level: float = 0.1
) -> pd.DataFrame:
    """Generate synthetic flight data.

    Args:
        flight_id: Flight identifier
        duration_minutes: Duration of flight in minutes
        sample_rate_hz: Sampling rate in Hz
        num_sensors: Number of sensors to simulate
        noise_level: Noise level (std dev as fraction of signal)

    Returns:
        DataFrame with synthetic sensor timeseries
    """
    num_samples = int(duration_minutes * 60 * sample_rate_hz)
    timestamps = pd.date_range(
        start="2024-01-01 00:00:00",
        periods=num_samples,
        freq=f"{1/sample_rate_hz}S"
    )

    data = {}
    data["timestamp"] = timestamps

    # Generate synthetic sensor data with different patterns
    t = np.arange(num_samples)

    for i in range(num_sensors):
        # Different patterns for different sensors
        if i == 0:
            # Slowly increasing trend (fuel consumption)
            signal = 100 + 0.01 * t + 5 * np.sin(2 * np.pi * t / (10 * sample_rate_hz))
        elif i == 1:
            # Oscillating (mach number)
            signal = 0.8 + 0.05 * np.sin(2 * np.pi * t / (30 * sample_rate_hz))
        elif i == 2:
            # Step changes (altitude)
            signal = 10000 + 5000 * (t > num_samples // 3) + 5000 * (t > 2 * num_samples // 3)
        elif i == 3:
            # Random walk (temperature)
            signal = 20 + np.cumsum(np.random.randn(num_samples) * 0.1)
        else:
            # Mixed pattern
            signal = 50 + 10 * np.sin(2 * np.pi * t / (20 * sample_rate_hz)) + \
                    0.005 * t

        # Add noise
        noise = np.random.randn(num_samples) * noise_level * np.abs(signal).mean()
        data[f"sensor_{i}"] = signal + noise

    df = pd.DataFrame(data)
    df.set_index("timestamp", inplace=True)

    return df


def main():
    """Generate toy dataset."""
    parser = argparse.ArgumentParser(description="Generate synthetic flight data")
    parser.add_argument("--output-dir", type=str, default="data/raw",
                       help="Output directory")
    parser.add_argument("--num-flights", type=int, default=10,
                       help="Number of flights to generate")
    parser.add_argument("--duration", type=int, default=60,
                       help="Flight duration in minutes")
    parser.add_argument("--num-sensors", type=int, default=5,
                       help="Number of sensors")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.num_flights} synthetic flights...")

    for i in range(args.num_flights):
        flight_id = f"synthetic_flight_{i:03d}"

        df = generate_synthetic_flight(
            flight_id=flight_id,
            duration_minutes=args.duration,
            num_sensors=args.num_sensors
        )

        # Save as parquet
        output_path = output_dir / f"{flight_id}.parquet"
        df.to_parquet(output_path)

        print(f"  Saved {flight_id} to {output_path}")

    print(f"\nDone! Generated {args.num_flights} flights in {output_dir}")


if __name__ == "__main__":
    main()
