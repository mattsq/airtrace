"""AirTrace package initialization."""

from airtrace.utils import ensure_parquet_support

# Ensure pandas parquet helpers are available even when optional parquet
# dependencies (``pyarrow``/``fastparquet``) are missing in the runtime
# environment. This allows the synthetic data generator tests to persist data
# without requiring heavy optional dependencies.
ensure_parquet_support()

__all__ = ["ensure_parquet_support"]
