"""Utilities for managing AirTrace config file locations.

This module provides functions for determining where to write config files
and for initializing user-level config directories, supporting the multi-path
config discovery system.
"""

import os
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def get_default_config_write_path(
    dataset_name: str,
    config_dir: Optional[Path] = None,
    prefer_user_dir: bool = True
) -> Path:
    """
    Determine the best location to write a dataset config file.

    Priority order:
      1. Explicit config_dir argument (if provided)
      2. AIRTRACE_CONFIG_DIR environment variable
      3. User directory: ~/.airtrace/configs/data/ (default)
      4. Package directory (fallback, not recommended)

    Args:
        dataset_name: Name of the dataset (used for filename)
        config_dir: Optional explicit directory path
        prefer_user_dir: If True, prefer user dir over package dir (default: True)

    Returns:
        Path to the config file to write

    Examples:
        >>> # Default behavior - writes to user directory
        >>> path = get_default_config_write_path("my_dataset")
        >>> # ~/.airtrace/configs/data/my_dataset.yaml

        >>> # Explicit directory
        >>> path = get_default_config_write_path("my_dataset", Path("/custom/path"))
        >>> # /custom/path/my_dataset.yaml

        >>> # Using environment variable
        >>> os.environ["AIRTRACE_CONFIG_DIR"] = "/shared/configs"
        >>> path = get_default_config_write_path("my_dataset")
        >>> # /shared/configs/data/my_dataset.yaml
    """
    # 1. Explicit argument takes highest priority
    if config_dir is not None:
        config_path = Path(config_dir)
        # If config_dir points to a .yaml file, use it directly
        if config_path.suffix == ".yaml":
            return config_path
        # Otherwise treat it as a directory
        return config_path / f"{dataset_name}.yaml"

    # 2. Environment variable
    env_config_dir = os.getenv("AIRTRACE_CONFIG_DIR")
    if env_config_dir:
        env_path = Path(env_config_dir) / "data"
        return env_path / f"{dataset_name}.yaml"

    # 3. User directory (default and recommended)
    if prefer_user_dir:
        user_dir = Path.home() / ".airtrace" / "configs" / "data"
        return user_dir / f"{dataset_name}.yaml"

    # 4. Package directory (fallback - only works with editable install)
    import airtrace
    pkg_path = Path(airtrace.__file__).parent
    return pkg_path / "configs" / "data" / f"{dataset_name}.yaml"


def initialize_user_config_dir() -> Path:
    """
    Initialize the user-level AirTrace config directory with subdirectories and README.

    Creates the following structure:
      ~/.airtrace/
      ├── configs/
      │   ├── data/        # Dataset configs (written by airtrace-ingest)
      │   ├── model/       # Custom model configs
      │   ├── transforms/  # Custom transform configs
      │   └── exp/         # Experiment configs
      └── README.txt       # Documentation about the directory

    This directory is used as the default write location for airtrace-ingest
    and is included in Hydra's config search path via the SearchPath plugin.

    Returns:
        Path to the user config directory (~/.airtrace/configs/)

    Note:
        This function is idempotent - safe to call multiple times.
        If the directory already exists, it will only create missing subdirectories
        and update the README if needed.
    """
    user_root = Path.home() / ".airtrace"
    user_config = user_root / "configs"

    # Create main config directory
    user_config.mkdir(parents=True, exist_ok=True)

    # Create subdirectories for different config types
    for subdir in ["data", "model", "transforms", "exp"]:
        (user_config / subdir).mkdir(exist_ok=True)

    # Create/update README
    readme_path = user_root / "README.txt"
    readme_content = """AirTrace User Configuration Directory
======================================

This directory stores your personal AirTrace configurations.

Structure:
  configs/data/        - Dataset configs (created by airtrace-ingest)
  configs/model/       - Custom model configs
  configs/transforms/  - Custom transform configs
  configs/exp/         - Experiment configs

Config Discovery Search Priority:
  1. <project>/.airtrace/configs/  (project-local, highest priority)
  2. ~/.airtrace/configs/          (user-level, THIS DIRECTORY)
  3. Package configs               (built-in defaults, lowest priority)

Usage:
  - Configs created by 'airtrace-ingest' are automatically written here
  - You can manually add custom configs to any subdirectory
  - To override a package config, copy it here and modify
  - For team sharing, use project-local configs instead (see below)

Project-Local Configs (Optional):
  Create <project>/.airtrace/configs/ in your project directory for configs
  that should be shared with your team via git. These take priority over
  user-level configs.

Environment Variables:
  AIRTRACE_CONFIG_DIR - Override default config write location
                        Example: export AIRTRACE_CONFIG_DIR=/shared/configs

Generated by AirTrace. Safe to delete - will be recreated as needed.

For more information, see: https://github.com/your-org/airtrace
"""

    if not readme_path.exists():
        readme_path.write_text(readme_content.strip())
        logger.info(f"Initialized AirTrace user directory at: {user_root}")
    else:
        # Update README if content has changed (for future updates)
        current_content = readme_path.read_text()
        if current_content.strip() != readme_content.strip():
            readme_path.write_text(readme_content.strip())
            logger.debug(f"Updated README at: {readme_path}")

    return user_config


def get_config_search_paths() -> list[Path]:
    """
    Get all config search paths in priority order.

    This is useful for debugging and introspection.

    Returns:
        List of paths that Hydra searches for configs (project, user, package)

    Example:
        >>> paths = get_config_search_paths()
        >>> for path in paths:
        ...     print(f"Searching: {path}")
    """
    import airtrace

    paths = []

    # 1. Project-local (highest priority)
    project_config = Path.cwd() / ".airtrace" / "configs"
    if project_config.exists():
        paths.append(project_config)

    # 2. User-level
    user_config = Path.home() / ".airtrace" / "configs"
    if user_config.exists():
        paths.append(user_config)

    # 3. Package (lowest priority)
    pkg_config = Path(airtrace.__file__).parent / "configs"
    if pkg_config.exists():
        paths.append(pkg_config)

    return paths
