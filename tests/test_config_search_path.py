"""
Unit tests for config path utilities and SearchPath plugin.
"""

import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from airtrace.utils.config_paths import (
    get_default_config_write_path,
    initialize_user_config_dir,
    get_config_search_paths,
)

from hydra_plugins.airtrace_searchpath_plugin import AirTraceSearchPathPlugin


class TestConfigWritePath:
    """Tests for get_default_config_write_path()."""

    def test_explicit_path_as_directory(self):
        """Test with explicit directory path."""
        explicit = Path("/tmp/explicit")
        result = get_default_config_write_path("test_dataset", config_dir=explicit)
        assert result == explicit / "test_dataset.yaml"

    def test_explicit_path_as_yaml_file(self):
        """Test with explicit YAML file path."""
        explicit = Path("/tmp/explicit/config.yaml")
        result = get_default_config_write_path("test_dataset", config_dir=explicit)
        assert result == explicit
        assert result.suffix == ".yaml"

    def test_environment_variable(self, tmp_path):
        """Test AIRTRACE_CONFIG_DIR environment variable."""
        env_dir = str(tmp_path / "env_config")
        with patch.dict(os.environ, {"AIRTRACE_CONFIG_DIR": env_dir}):
            result = get_default_config_write_path("test_dataset")
            expected = Path(env_dir) / "data" / "test_dataset.yaml"
            assert result == expected

    def test_user_directory_default(self, tmp_path):
        """Test default user directory (~/.airtrace/configs/data/)."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove AIRTRACE_CONFIG_DIR if present
            with patch("pathlib.Path.home", return_value=tmp_path):
                result = get_default_config_write_path("test_dataset")
                expected = tmp_path / ".airtrace" / "configs" / "data" / "test_dataset.yaml"
                assert result == expected

    def test_package_directory_fallback(self, tmp_path):
        """Test package directory fallback (prefer_user_dir=False)."""
        with patch.dict(os.environ, {}, clear=True):
            result = get_default_config_write_path(
                "test_dataset",
                prefer_user_dir=False
            )
            # Should contain configs/data/ from package
            assert "configs" in str(result)
            assert "data" in str(result)
            assert result.name == "test_dataset.yaml"

    def test_priority_order(self, tmp_path):
        """Test that priority order is: explicit > env var > user dir."""
        explicit = Path("/tmp/explicit")
        env_dir = str(tmp_path / "env_config")

        # Explicit takes priority
        with patch.dict(os.environ, {"AIRTRACE_CONFIG_DIR": env_dir}):
            result = get_default_config_write_path("test", config_dir=explicit)
            assert result == explicit / "test.yaml"

        # Env var takes priority over default
        with patch.dict(os.environ, {"AIRTRACE_CONFIG_DIR": env_dir}):
            with patch("pathlib.Path.home", return_value=tmp_path):
                result = get_default_config_write_path("test")
                assert env_dir in str(result)


class TestInitializeUserConfigDir:
    """Tests for initialize_user_config_dir()."""

    def test_creates_directory_structure(self, tmp_path):
        """Test that directory structure is created correctly."""
        with patch("pathlib.Path.home", return_value=tmp_path):
            result = initialize_user_config_dir()

            assert result == tmp_path / ".airtrace" / "configs"
            assert result.exists()

            # Check subdirectories
            for subdir in ["data", "model", "transforms", "exp"]:
                subdir_path = result / subdir
                assert subdir_path.exists()
                assert subdir_path.is_dir()

    def test_creates_readme(self, tmp_path):
        """Test that README.txt is created."""
        with patch("pathlib.Path.home", return_value=tmp_path):
            initialize_user_config_dir()

            readme = tmp_path / ".airtrace" / "README.txt"
            assert readme.exists()
            content = readme.read_text()
            assert "AirTrace User Configuration Directory" in content
            assert "Config Discovery Search Priority" in content

    def test_idempotent(self, tmp_path):
        """Test that function is idempotent (safe to call multiple times)."""
        with patch("pathlib.Path.home", return_value=tmp_path):
            # Call twice
            result1 = initialize_user_config_dir()
            result2 = initialize_user_config_dir()

            assert result1 == result2
            assert result1.exists()

            # README should still exist
            readme = tmp_path / ".airtrace" / "README.txt"
            assert readme.exists()

    def test_creates_missing_subdirs(self, tmp_path):
        """Test that missing subdirectories are created on subsequent calls."""
        with patch("pathlib.Path.home", return_value=tmp_path):
            # Initialize once
            result = initialize_user_config_dir()

            # Remove one subdirectory
            (result / "model").rmdir()
            assert not (result / "model").exists()

            # Call again
            initialize_user_config_dir()

            # Should be recreated
            assert (result / "model").exists()


class TestGetConfigSearchPaths:
    """Tests for get_config_search_paths()."""

    def test_returns_existing_paths_only(self, tmp_path):
        """Test that only existing paths are returned."""
        # Create user config dir
        user_config = tmp_path / "home" / ".airtrace" / "configs"
        user_config.mkdir(parents=True)

        # Create project config dir
        project_config = tmp_path / "project" / ".airtrace" / "configs"
        project_config.mkdir(parents=True)

        with patch("pathlib.Path.home", return_value=tmp_path / "home"):
            with patch("pathlib.Path.cwd", return_value=tmp_path / "project"):
                paths = get_config_search_paths()

                # Should include project and user dirs
                assert len(paths) >= 2
                assert project_config in paths
                assert user_config in paths

    def test_priority_order(self, tmp_path):
        """Test that paths are returned in priority order."""
        # Create both directories
        user_config = tmp_path / "home" / ".airtrace" / "configs"
        user_config.mkdir(parents=True)

        project_config = tmp_path / "project" / ".airtrace" / "configs"
        project_config.mkdir(parents=True)

        with patch("pathlib.Path.home", return_value=tmp_path / "home"):
            with patch("pathlib.Path.cwd", return_value=tmp_path / "project"):
                paths = get_config_search_paths()

                # Project should come before user
                project_idx = None
                user_idx = None
                for i, path in enumerate(paths):
                    if path == project_config:
                        project_idx = i
                    if path == user_config:
                        user_idx = i

                assert project_idx is not None
                assert user_idx is not None
                assert project_idx < user_idx


class TestSearchPathPlugin:
    """Tests for AirTraceSearchPathPlugin."""

    def test_plugin_registration(self):
        """Test that plugin can be instantiated."""
        plugin = AirTraceSearchPathPlugin()
        assert plugin is not None

    def test_manipulate_search_path_adds_user_dir(self, tmp_path):
        """Test that user directory is added to search path if it exists."""
        user_config = tmp_path / ".airtrace" / "configs"
        user_config.mkdir(parents=True)

        with patch("pathlib.Path.home", return_value=tmp_path):
            plugin = AirTraceSearchPathPlugin()

            # Create mock search path
            mock_search_path = MagicMock()
            mock_search_path.config_search_path = []

            plugin.manipulate_search_path(mock_search_path)

            # Should have called append with user directory
            mock_search_path.append.assert_called_once()
            call_args = mock_search_path.append.call_args
            assert call_args[1]["provider"] == "airtrace-user"
            assert user_config.as_posix() in call_args[1]["path"]

    def test_manipulate_search_path_adds_project_dir(self, tmp_path):
        """Test that project directory is prepended to search path if it exists."""
        project_config = tmp_path / "project" / ".airtrace" / "configs"
        project_config.mkdir(parents=True)

        with patch("pathlib.Path.cwd", return_value=tmp_path / "project"):
            plugin = AirTraceSearchPathPlugin()

            # Create mock search path
            mock_search_path = MagicMock()
            mock_search_path.config_search_path = []

            plugin.manipulate_search_path(mock_search_path)

            # Should have called prepend with project directory
            mock_search_path.prepend.assert_called_once()
            call_args = mock_search_path.prepend.call_args
            assert call_args[1]["provider"] == "airtrace-project"
            assert project_config.as_posix() in call_args[1]["path"]

    def test_does_not_add_nonexistent_directories(self, tmp_path):
        """Test that non-existent directories are not added."""
        # Don't create the directories
        with patch("pathlib.Path.home", return_value=tmp_path / "nonexistent"):
            with patch("pathlib.Path.cwd", return_value=tmp_path / "also_nonexistent"):
                plugin = AirTraceSearchPathPlugin()

                # Create mock search path
                mock_search_path = MagicMock()
                mock_search_path.config_search_path = []

                plugin.manipulate_search_path(mock_search_path)

                # Should not have called append or prepend
                mock_search_path.append.assert_not_called()
                mock_search_path.prepend.assert_not_called()

    def test_project_dir_prepended_before_user_dir(self, tmp_path):
        """Test that project directory is prepended (higher priority) before user directory."""
        user_config = tmp_path / "home" / ".airtrace" / "configs"
        user_config.mkdir(parents=True)

        project_config = tmp_path / "project" / ".airtrace" / "configs"
        project_config.mkdir(parents=True)

        with patch("pathlib.Path.home", return_value=tmp_path / "home"):
            with patch("pathlib.Path.cwd", return_value=tmp_path / "project"):
                plugin = AirTraceSearchPathPlugin()

                # Create mock search path
                mock_search_path = MagicMock()
                mock_search_path.config_search_path = []

                plugin.manipulate_search_path(mock_search_path)

                # Should have called both append (user) and prepend (project)
                mock_search_path.append.assert_called_once()
                mock_search_path.prepend.assert_called_once()

                # Verify user was appended
                append_call_args = mock_search_path.append.call_args
                assert append_call_args[1]["provider"] == "airtrace-user"

                # Verify project was prepended (higher priority)
                prepend_call_args = mock_search_path.prepend.call_args
                assert prepend_call_args[1]["provider"] == "airtrace-project"
