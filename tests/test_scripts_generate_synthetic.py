"""Tests for CLI scripts."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import DictConfig, OmegaConf

from airtrace.scripts.generate_synthetic_data import cli, generate_from_config


class TestGenerateSyntheticData:
    """Tests for generate_synthetic_data.py script."""

    def test_generate_from_config_missing_generation_section(self, tmp_path, capsys):
        """Test error when config lacks generation section."""
        config = OmegaConf.create({
            "data": {
                "root": str(tmp_path),
                "dataset_name": "test_dataset",
                # Missing 'generation' section
            }
        })

        with pytest.raises(SystemExit) as exc_info:
            generate_from_config(config)

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "missing a 'generation' section" in captured.out

    def test_generate_from_config_success(self, tmp_path, capsys):
        """Test successful synthetic data generation."""
        config = OmegaConf.create({
            "data": {
                "root": str(tmp_path),
                "dataset_name": "test_synthetic",
                "generation": {
                    "n_flights": 3,
                    "seed": 42,
                    "train_val_test_split": [0.6, 0.2, 0.2],
                    "cruise_profile": {
                        "initial_altitude": 35000,
                        "initial_mach": 0.82,
                        "initial_weight": 70000,
                        "cruise_duration": 1800,
                        "fuel_flow_base": 1250,
                        "n1_base": 85,
                        "sample_rate": 1.0,
                        "altitude_variation": 500,
                        "mach_variation": 0.02,
                        "turbulence_level": 0.3,
                        "fuel_flow_noise": 10,
                        "mach_noise": 0.001,
                        "altitude_noise": 50,
                        "n1_noise": 0.5,
                        "oat_noise": 1.0,
                    }
                },
                "sensors": {
                    "use": ["fuel_flow", "mach", "altitude", "oat", "n1", "weight"]
                },
                "window": {
                    "input_len": 64,
                    "pred_len": 16,
                    "stride": 16,
                    "target_sensors": ["fuel_flow", "mach"]
                }
            }
        })

        generate_from_config(config)

        # Verify directories created
        assert (tmp_path / "raw").exists()
        assert (tmp_path / "interim").exists()
        assert (tmp_path / "processed").exists()
        assert (tmp_path / "metadata").exists()

        # Verify files created
        raw_files = list((tmp_path / "raw").glob("*.parquet"))
        assert len(raw_files) == 3

        interim_files = list((tmp_path / "interim").glob("*.parquet"))
        assert len(interim_files) == 3

        processed_files = list((tmp_path / "processed").glob("*.parquet"))
        assert len(processed_files) == 3

        # Verify window indices created
        metadata_files = list((tmp_path / "metadata").glob("*_index.parquet"))
        assert len(metadata_files) == 3  # train, val, test

        # Check output messages
        captured = capsys.readouterr()
        assert "Generating 3 synthetic flights" in captured.out
        assert "Train:" in captured.out
        assert "Val:" in captured.out
        assert "Test:" in captured.out

    def test_cli_adds_default_config(self):
        """Test that CLI adds default data=synthetic_cruise if not specified."""
        with patch('sys.argv', ['generate_synthetic_data']):
            with patch('airtrace.scripts.generate_synthetic_data.main') as mock_main:
                cli()
                assert 'data=synthetic_cruise' in sys.argv

    def test_cli_preserves_user_config(self):
        """Test that CLI preserves user-specified data config."""
        original_argv = ['generate_synthetic_data', 'data=custom']
        with patch('sys.argv', original_argv.copy()):
            with patch('airtrace.scripts.generate_synthetic_data.main') as mock_main:
                # Save argv before cli() modifies it
                import sys
                argv_before = sys.argv.copy()
                cli()
                # Check that data=custom is still present
                assert any('data=custom' in arg for arg in argv_before)
                # Should not have added another data= argument
                data_args = [arg for arg in sys.argv if arg.startswith('data=')]
                assert len(data_args) == 1
