"""Pytest configuration for path setup.

Ensures the project's ``src`` directory is importable when tests are executed
without installing the package.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Resolve the repository root (two levels above this file) and ensure the ``src``
# directory is on ``sys.path`` so ``import airtrace`` works during test runs.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"

if SRC_PATH.exists():
    sys.path.insert(0, str(SRC_PATH))
