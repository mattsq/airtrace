import numpy as np
import pandas as pd
import pytest

from airtrace.data.windows import WindowSpec


def test_window_spec_validation_and_repr():
    spec = WindowSpec(input_len=4, pred_len=2, stride=1, target_sensors=["altitude"])
    assert "input_len=4" in repr(spec)
    assert spec.total_len == 6

    with pytest.raises(ValueError):
        WindowSpec(input_len=0, pred_len=1, stride=1, target_sensors=[])
    with pytest.raises(ValueError):
        WindowSpec(input_len=1, pred_len=0, stride=1, target_sensors=[])
    with pytest.raises(ValueError):
        WindowSpec(input_len=1, pred_len=1, stride=0, target_sensors=[])


def test_create_windows_generates_indices():
    data = np.arange(10).reshape(10, 1)
    spec = WindowSpec(input_len=3, pred_len=2, stride=2, target_sensors=["speed"])

    windows = spec.create_windows(data, flight_id="AB123")

    expected = pd.DataFrame(
        [
            {"flight_id": "AB123", "start_idx": 0, "end_idx": 5},
            {"flight_id": "AB123", "start_idx": 2, "end_idx": 7},
            {"flight_id": "AB123", "start_idx": 4, "end_idx": 9},
        ]
    )
    pd.testing.assert_frame_equal(windows, expected)


def test_create_windows_returns_empty_when_too_short():
    data = np.zeros((4, 1))
    spec = WindowSpec(input_len=3, pred_len=2, stride=1, target_sensors=["speed"])

    windows = spec.create_windows(data)
    assert windows.empty
    assert list(windows.columns) == ["flight_id", "start_idx", "end_idx"]
