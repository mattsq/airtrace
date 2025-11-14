"""Sliding window specification and utilities."""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class WindowSpec:
    """Specification for sliding window creation.

    Attributes:
        input_len: Length of input sequence
        pred_len: Length of prediction/target sequence
        stride: Stride for sliding window (in timesteps)
        target_sensors: List of sensor names to use as targets
    """

    input_len: int
    pred_len: int
    stride: int
    target_sensors: List[str]

    def __post_init__(self):
        """Validate window specification."""
        if self.input_len <= 0:
            raise ValueError(f"input_len must be > 0, got {self.input_len}")
        if self.pred_len <= 0:
            raise ValueError(f"pred_len must be > 0, got {self.pred_len}")
        if self.stride <= 0:
            raise ValueError(f"stride must be > 0, got {self.stride}")

    @property
    def total_len(self) -> int:
        """Total length required for one window.

        Returns:
            input_len + pred_len
        """
        return self.input_len + self.pred_len

    def create_windows(
        self,
        data: np.ndarray,
        flight_id: Optional[str] = None
    ) -> pd.DataFrame:
        """Create sliding windows from a timeseries array.

        Args:
            data: Timeseries data [T, D]
            flight_id: Optional flight identifier

        Returns:
            DataFrame with columns: flight_id, start_idx, end_idx
        """
        T = len(data)
        if T < self.total_len:
            # Not enough data for even one window
            return pd.DataFrame(columns=["flight_id", "start_idx", "end_idx"])

        windows = []
        start_idx = 0
        while start_idx + self.total_len <= T:
            windows.append({
                "flight_id": flight_id,
                "start_idx": start_idx,
                "end_idx": start_idx + self.total_len
            })
            start_idx += self.stride

        return pd.DataFrame(windows)

    def __repr__(self):
        return (
            f"WindowSpec(\n"
            f"  input_len={self.input_len},\n"
            f"  pred_len={self.pred_len},\n"
            f"  stride={self.stride},\n"
            f"  total_len={self.total_len},\n"
            f"  target_sensors={self.target_sensors}\n"
            f")"
        )
