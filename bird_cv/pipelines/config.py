"""Shared config loading utilities for all pipelines."""

from datetime import datetime
from pathlib import Path

import msgspec.toml


def load_config(config_path: Path, config_type: type):
    """Load and decode a TOML config file into a typed msgspec struct.

    Args:
        config_path: Path to the TOML file.
        config_type: A msgspec.Struct subclass to decode into.

    Returns:
        Decoded config struct.
    """
    return msgspec.toml.decode(config_path.read_bytes(), type=config_type)


def resolve_run_dir(base_path: Path, run_id: str | None) -> Path:
    """Resolve the output directory for a pipeline run.

    If ``run_id`` is None or empty, generates a timestamped ID of the form
    ``YYYYMMDD_HHMMSS``. Otherwise uses the provided value.

    Args:
        base_path: Root directory under which the run directory is created.
        run_id: Optional fixed run identifier.

    Returns:
        Path to the run directory (not yet created).
    """
    if not run_id:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_path / run_id
    print(f"Run directory: {run_dir}")
    return run_dir
