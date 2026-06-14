"""Backward-compatible entry point.

Prefer running `crossmedia-pid` after installation or
`python -m crossmedia_pid.cli` from the repository root.
"""

from __future__ import annotations

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crossmedia_pid.app import CrossMediaPID
from crossmedia_pid.cli import cli, setup_logging
from crossmedia_pid.config import load_config

__all__ = ["CrossMediaPID", "cli", "load_config", "setup_logging"]


if __name__ == "__main__":
    cli()
