"""Pytest configuration ensuring the `airtrace` package is importable during local runs."""
from __future__ import annotations

import sys
from pathlib import Path

# Add the repository's ``src`` directory to ``sys.path`` so that ``import airtrace``
# works without requiring an editable installation before running pytest.
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
