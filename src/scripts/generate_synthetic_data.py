#!/usr/bin/env python3
"""Wrapper for backwards-compatible synthetic data generation.

This wrapper delegates to the packaged ``airtrace.scripts.generate_synthetic_data``
module so it works both from a git checkout and an installed wheel.
"""

from airtrace.scripts.generate_synthetic_data import main_cli


if __name__ == "__main__":
    main_cli()
