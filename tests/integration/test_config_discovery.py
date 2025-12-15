"""
Integration tests for config discovery with Hydra.

These tests validate the end-to-end config discovery behavior including:
- User directory initialization
- Config file writing
- Hydra config discovery via SearchPath plugin
- Priority ordering (project > user > package)
"""

import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch

from airtrace.data.ingest import ConfigGenerator
from airtrace.utils.config_paths import initialize_user_config_dir


class TestConfigGeneratorIntegration:
    """Integration tests for ConfigGenerator with auto-detection."""

    def test_writes_to_user_dir_by_default(self, tmp_path):
        """Test that ConfigGenerator writes to user directory by default."""
        with patch("pathlib.Path.home", return_value=tmp_path):
            # Initialize config directory
            initialize_user_config_dir()

            # Create config generator
            config_gen = ConfigGenerator(
                dataset_name="test_dataset",
                sensors=["fuel_flow", "mach", "altitude"],
                input_len=256,
                pred_len=32,
                stride=32,
                target_sensors=["fuel_flow", "mach"]
            )

            # Generate config without specifying path
            config_gen.generate(
                output_path=None,  # Auto-detect
                n_flights=10,
                input_path="/data/raw/test.parquet",
                force=True
            )

            # Verify config was written to user directory
            expected_path = tmp_path / ".airtrace" / "configs" / "data" / "test_dataset.yaml"
            assert expected_path.exists()

            # Verify content
            content = expected_path.read_text()
            assert "test_dataset" in content
            assert "fuel_flow" in content
            assert "mach" in content
            assert "altitude" in content

    def test_respects_explicit_config_dir(self, tmp_path):
        """Test that explicit config_dir takes precedence."""
        custom_dir = tmp_path / "custom_configs"
        custom_dir.mkdir(parents=True)

        config_gen = ConfigGenerator(
            dataset_name="custom_dataset",
            sensors=["sensor1", "sensor2"],
            input_len=128,
            pred_len=16,
            stride=16,
            target_sensors=["sensor1"]
        )

        # Generate with explicit path
        explicit_path = custom_dir / "custom_dataset.yaml"
        config_gen.generate(
            output_path=explicit_path,
            n_flights=5,
            input_path="/data/custom.parquet",
            force=True
        )

        # Verify config was written to explicit path
        assert explicit_path.exists()
        content = explicit_path.read_text()
        assert "custom_dataset" in content

    def test_respects_env_var(self, tmp_path):
        """Test that AIRTRACE_CONFIG_DIR environment variable is respected."""
        env_config_dir = tmp_path / "env_configs"
        env_config_dir.mkdir(parents=True)

        with patch.dict(os.environ, {"AIRTRACE_CONFIG_DIR": str(env_config_dir)}):
            config_gen = ConfigGenerator(
                dataset_name="env_dataset",
                sensors=["s1", "s2", "s3"],
                input_len=256,
                pred_len=32,
                stride=32,
                target_sensors=["s1"]
            )

            # Generate without explicit path
            config_gen.generate(
                output_path=None,
                n_flights=7,
                input_path="/data/env.parquet",
                force=True
            )

            # Verify config was written to env directory
            expected_path = env_config_dir / "data" / "env_dataset.yaml"
            assert expected_path.exists()

    def test_creates_parent_directories(self, tmp_path):
        """Test that parent directories are created automatically."""
        with patch("pathlib.Path.home", return_value=tmp_path):
            # Don't initialize directory manually

            config_gen = ConfigGenerator(
                dataset_name="auto_dir_test",
                sensors=["x", "y", "z"],
                input_len=100,
                pred_len=10,
                stride=10,
                target_sensors=["x"]
            )

            # Generate config
            config_gen.generate(
                output_path=None,
                n_flights=3,
                input_path="/data/test.parquet",
                force=True
            )

            # Verify directory structure was created
            config_dir = tmp_path / ".airtrace" / "configs" / "data"
            assert config_dir.exists()
            assert config_dir.is_dir()

            # Verify config file exists
            config_file = config_dir / "auto_dir_test.yaml"
            assert config_file.exists()


class TestUserDirectoryInitialization:
    """Integration tests for user directory initialization."""

    def test_initialize_creates_full_structure(self, tmp_path):
        """Test that initialization creates complete directory structure."""
        with patch("pathlib.Path.home", return_value=tmp_path):
            config_dir = initialize_user_config_dir()

            # Verify main directory
            assert config_dir.exists()
            assert config_dir == tmp_path / ".airtrace" / "configs"

            # Verify all subdirectories
            for subdir in ["data", "model", "transforms", "exp"]:
                subdir_path = config_dir / subdir
                assert subdir_path.exists()
                assert subdir_path.is_dir()

            # Verify README
            readme = tmp_path / ".airtrace" / "README.txt"
            assert readme.exists()
            content = readme.read_text()
            assert "AirTrace User Configuration Directory" in content
            assert "Config Discovery Search Priority" in content
            assert "project-local" in content.lower()
            assert "user-level" in content.lower()

    def test_can_write_and_read_config(self, tmp_path):
        """End-to-end test: initialize dir, write config, read it back."""
        with patch("pathlib.Path.home", return_value=tmp_path):
            # Initialize
            config_dir = initialize_user_config_dir()

            # Write a config manually
            test_config_path = config_dir / "data" / "test.yaml"
            test_config_content = """# @package _global_
data:
  dataset_name: "test"
  sensors:
    use:
      - "sensor1"
      - "sensor2"
"""
            test_config_path.write_text(test_config_content)

            # Read it back
            assert test_config_path.exists()
            content = test_config_path.read_text()
            assert "test" in content
            assert "sensor1" in content


class TestConfigPriorityOrdering:
    """Integration tests for config priority ordering."""

    def test_project_overrides_user_config(self, tmp_path):
        """Test that project-local configs take precedence over user configs."""
        from airtrace.utils.config_paths import get_config_search_paths

        # Create user config
        user_home = tmp_path / "home"
        user_config_dir = user_home / ".airtrace" / "configs"
        user_config_dir.mkdir(parents=True)
        user_config_file = user_config_dir / "data" / "test.yaml"
        user_config_file.parent.mkdir(parents=True)
        user_config_file.write_text("# User config\ndata:\n  source: user")

        # Create project config
        project_dir = tmp_path / "project"
        project_config_dir = project_dir / ".airtrace" / "configs"
        project_config_dir.mkdir(parents=True)
        project_config_file = project_config_dir / "data" / "test.yaml"
        project_config_file.parent.mkdir(parents=True)
        project_config_file.write_text("# Project config\ndata:\n  source: project")

        # Get search paths
        with patch("pathlib.Path.home", return_value=user_home):
            with patch("pathlib.Path.cwd", return_value=project_dir):
                paths = get_config_search_paths()

                # Project should come first
                assert len(paths) >= 2
                assert paths[0] == project_config_dir

                # User should come second
                assert paths[1] == user_config_dir

    def test_user_config_available_when_no_project_config(self, tmp_path):
        """Test that user configs work when project config doesn't exist."""
        from airtrace.utils.config_paths import get_config_search_paths

        # Create only user config
        user_home = tmp_path / "home"
        user_config_dir = user_home / ".airtrace" / "configs"
        user_config_dir.mkdir(parents=True)

        # Don't create project config
        project_dir = tmp_path / "project"
        project_dir.mkdir(parents=True)

        # Get search paths
        with patch("pathlib.Path.home", return_value=user_home):
            with patch("pathlib.Path.cwd", return_value=project_dir):
                paths = get_config_search_paths()

                # Should have user config
                assert user_config_dir in paths


class TestEnvVarOverride:
    """Integration tests for AIRTRACE_CONFIG_DIR environment variable."""

    def test_env_var_overrides_default(self, tmp_path):
        """Test that env var takes precedence over default user dir."""
        from airtrace.utils.config_paths import get_default_config_write_path

        user_home = tmp_path / "home"
        env_dir = tmp_path / "custom_env"

        with patch("pathlib.Path.home", return_value=user_home):
            # Without env var
            path_without_env = get_default_config_write_path("test")
            assert user_home / ".airtrace" in path_without_env.parents

            # With env var
            with patch.dict(os.environ, {"AIRTRACE_CONFIG_DIR": str(env_dir)}):
                path_with_env = get_default_config_write_path("test")
                assert env_dir in path_with_env.parents
                assert path_with_env == env_dir / "data" / "test.yaml"

    def test_env_var_persists_across_calls(self, tmp_path):
        """Test that env var setting persists for multiple config writes."""
        from airtrace.utils.config_paths import get_default_config_write_path

        env_dir = tmp_path / "persistent_env"
        env_dir.mkdir(parents=True)

        with patch.dict(os.environ, {"AIRTRACE_CONFIG_DIR": str(env_dir)}):
            # First call
            path1 = get_default_config_write_path("dataset1")
            assert env_dir in path1.parents

            # Second call
            path2 = get_default_config_write_path("dataset2")
            assert env_dir in path2.parents

            # Both should use the same env dir
            assert path1.parent.parent == path2.parent.parent == env_dir
