"""AirTrace SearchPath plugin for Hydra config discovery.

This plugin extends Hydra's config search path to include user and project directories,
making config discovery more portable and persistent across package reinstalls.

Search order (first match wins):
  1. <cwd>/.airtrace/configs/  (project-local, highest priority)
  2. ~/.airtrace/configs/       (user-level, medium priority)
  3. pkg://airtrace.configs     (package defaults, lowest priority)
"""

from pathlib import Path
from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin


class AirTraceSearchPathPlugin(SearchPathPlugin):
    """
    Extends Hydra's config search path to include user and project directories.

    This plugin enables configs created by airtrace-ingest to be discovered
    regardless of where the command is run, solving the portability issue where
    configs written to the package directory were lost on reinstall or inaccessible
    from different directories.
    """

    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        """
        Add user and project config directories to Hydra's search path.

        The search path is modified to include:
        - User directory: ~/.airtrace/configs/ (added if exists)
        - Project directory: <cwd>/.airtrace/configs/ (prepended if exists)

        Project configs override user configs, which override package configs.

        Args:
            search_path: Hydra's ConfigSearchPath instance to modify
        """
        # Add user-level config directory (persistent across reinstalls)
        user_config = Path.home() / ".airtrace" / "configs"
        if user_config.exists() and user_config.is_dir():
            search_path.append(
                provider="airtrace-user",
                path=f"file://{user_config.as_posix()}"
            )

        # Add project-local config directory (highest priority, for team sharing)
        project_config = Path.cwd() / ".airtrace" / "configs"
        if project_config.exists() and project_config.is_dir():
            search_path.prepend(
                provider="airtrace-project",
                path=f"file://{project_config.as_posix()}"
            )
