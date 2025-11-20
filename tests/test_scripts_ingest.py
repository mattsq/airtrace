"""Tests for ingest CLI script."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from airtrace.scripts.ingest import (
    ingest_dataset,
    main,
    parse_args,
    parse_split_ratios,
    print_summary,
    split_flights,
)


class TestParseSplitRatios:
    """Tests for parse_split_ratios function."""

    def test_valid_split_ratios(self):
        """Test parsing valid split ratios."""
        ratios = parse_split_ratios("0.7,0.15,0.15")
        assert ratios == [0.7, 0.15, 0.15]
        assert np.isclose(sum(ratios), 1.0)

    def test_valid_split_ratios_different(self):
        """Test parsing different valid split ratios."""
        ratios = parse_split_ratios("0.8,0.1,0.1")
        assert ratios == [0.8, 0.1, 0.1]

    def test_invalid_count(self):
        """Test error when not exactly 3 ratios."""
        with pytest.raises(SystemExit):
            parse_split_ratios("0.5,0.5")

    def test_invalid_sum(self):
        """Test error when ratios don't sum to 1.0."""
        with pytest.raises(SystemExit):
            parse_split_ratios("0.5,0.3,0.3")

    def test_invalid_format(self):
        """Test error with invalid format."""
        with pytest.raises(SystemExit):
            parse_split_ratios("abc,def,ghi")


class TestSplitFlights:
    """Tests for split_flights function."""

    def test_split_flights_basic(self):
        """Test basic flight splitting."""
        flight_ids = [f"flight_{i:03d}" for i in range(10)]
        train, val, test = split_flights(flight_ids, [0.7, 0.15, 0.15], seed=42)

        assert len(train) == 7
        assert len(val) == 1
        assert len(test) == 2
        assert len(set(train + val + test)) == 10  # No duplicates

    def test_split_flights_reproducible(self):
        """Test that same seed produces same split."""
        flight_ids = [f"flight_{i:03d}" for i in range(20)]
        train1, val1, test1 = split_flights(flight_ids, [0.6, 0.2, 0.2], seed=123)
        train2, val2, test2 = split_flights(flight_ids, [0.6, 0.2, 0.2], seed=123)

        assert train1 == train2
        assert val1 == val2
        assert test1 == test2

    def test_split_flights_different_seeds(self):
        """Test that different seeds produce different splits."""
        flight_ids = [f"flight_{i:03d}" for i in range(20)]
        train1, val1, test1 = split_flights(flight_ids, [0.6, 0.2, 0.2], seed=42)
        train2, val2, test2 = split_flights(flight_ids, [0.6, 0.2, 0.2], seed=999)

        # At least one split should differ
        assert train1 != train2 or val1 != val2 or test1 != test2

    def test_split_flights_small_dataset(self, caplog):
        """Test warning with small dataset."""
        flight_ids = ["flight_001", "flight_002"]
        train, val, test = split_flights(flight_ids, [0.5, 0.25, 0.25], seed=42)

        # Should still work but may have empty splits
        assert len(train) + len(val) + len(test) == 2


class TestPrintSummary:
    """Tests for print_summary function."""

    def test_print_summary(self, capsys):
        """Test summary printing."""
        print_summary(
            dataset_name="test_dataset",
            train_ids=["f1", "f2", "f3"],
            val_ids=["f4"],
            test_ids=["f5", "f6"],
            sensors=["fuel_flow", "mach", "altitude"],
            target_sensors=["fuel_flow", "mach"],
            input_len=256,
            pred_len=32,
            stride=32,
            train_windows=100,
            val_windows=20,
            test_windows=40
        )

        captured = capsys.readouterr()
        assert "test_dataset" in captured.out
        assert "Train flights: 3 (100 windows)" in captured.out
        assert "Val flights: 1 (20 windows)" in captured.out
        assert "Test flights: 2 (40 windows)" in captured.out
        assert "fuel_flow, mach, altitude" in captured.out
        assert "Input length: 256" in captured.out


class TestParseArgs:
    """Tests for argument parsing."""

    def test_parse_args_minimal(self):
        """Test parsing minimal required arguments."""
        test_args = [
            "ingest",
            "data/flights.parquet",
            "--dataset-name", "test_dataset"
        ]
        with patch('sys.argv', test_args):
            args = parse_args()
            assert args.input_path == "data/flights.parquet"
            assert args.dataset_name == "test_dataset"
            assert args.input_len == 256  # Default
            assert args.pred_len == 32  # Default
            assert args.stride == 32  # Default

    def test_parse_args_with_window_params(self):
        """Test parsing with custom window parameters."""
        test_args = [
            "ingest",
            "data/flights.parquet",
            "--dataset-name", "test",
            "--input-len", "512",
            "--pred-len", "64",
            "--stride", "16"
        ]
        with patch('sys.argv', test_args):
            args = parse_args()
            assert args.input_len == 512
            assert args.pred_len == 64
            assert args.stride == 16

    def test_parse_args_with_sensors(self):
        """Test parsing with target sensors."""
        test_args = [
            "ingest",
            "data/flights.parquet",
            "--dataset-name", "test",
            "--target-sensors", "fuel_flow,mach,n1"
        ]
        with patch('sys.argv', test_args):
            args = parse_args()
            assert args.target_sensors == "fuel_flow,mach,n1"

    def test_parse_args_with_resample(self):
        """Test parsing with resample rate."""
        test_args = [
            "ingest",
            "data/flights.parquet",
            "--dataset-name", "test",
            "--resample-rate", "1S"
        ]
        with patch('sys.argv', test_args):
            args = parse_args()
            assert args.resample_rate == "1S"

    def test_parse_args_with_flags(self):
        """Test parsing with boolean flags."""
        test_args = [
            "ingest",
            "data/flights.parquet",
            "--dataset-name", "test",
            "--dry-run",
            "--force",
            "--skip-config"
        ]
        with patch('sys.argv', test_args):
            args = parse_args()
            assert args.dry_run is True
            assert args.force is True
            assert args.skip_config is True


class TestIngestDataset:
    """Tests for ingest_dataset function."""

    def test_ingest_dataset_dry_run(self, tmp_path, caplog):
        """Test dry run mode."""
        import logging
        caplog.set_level(logging.INFO)
        
        # Create a simple test parquet file
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=1000, freq='1S'),
            'fuel_flow': np.random.randn(1000) + 1250,
            'mach': np.random.randn(1000) * 0.01 + 0.82,
            'altitude': np.random.randn(1000) * 100 + 35000,
        })
        df = df.set_index('timestamp')
        
        input_file = tmp_path / "test_flight.parquet"
        df.to_parquet(input_file)

        # Create mock args
        args = MagicMock()
        args.input_path = str(input_file)
        args.dataset_name = "test_dataset"
        args.split = "0.7,0.15,0.15"
        args.seed = 42
        args.input_len = 256
        args.pred_len = 32
        args.stride = 32
        args.target_sensors = None
        args.resample_rate = None
        args.timestamp_column = None
        args.flight_id_column = None
        args.skip_processed = False
        args.skip_config = False
        args.force = False
        args.dry_run = True

        ingest_dataset(args)

        # Check logging output
        log_output = caplog.text
        assert "[DRY RUN]" in log_output
        assert "Would create:" in log_output

    def test_main_keyboard_interrupt(self, caplog):
        """Test handling of keyboard interrupt."""
        import logging
        caplog.set_level(logging.INFO)
        
        with patch('airtrace.scripts.ingest.parse_args') as mock_parse:
            with patch('airtrace.scripts.ingest.ingest_dataset', side_effect=KeyboardInterrupt):
                with pytest.raises(SystemExit) as exc_info:
                    main()
                assert exc_info.value.code == 1

        log_output = caplog.text
        assert "cancelled by user" in log_output

    def test_main_exception(self, caplog):
        """Test handling of general exceptions."""
        with patch('airtrace.scripts.ingest.parse_args') as mock_parse:
            with patch('airtrace.scripts.ingest.ingest_dataset', side_effect=ValueError("Test error")):
                with pytest.raises(SystemExit) as exc_info:
                    main()
                assert exc_info.value.code == 1

        log_output = caplog.text
        assert "Ingestion failed" in log_output
