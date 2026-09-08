"""Command-line entry point."""

from __future__ import annotations

import sys


def main() -> int:
    """Print the package version and exit."""
    from lidar_bedlam import __version__

    sys.stdout.write(f"lidar-bedlam {__version__}\n")
    return 0
