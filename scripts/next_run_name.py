"""Print the next free ``<experiment>-NNN`` run name for a config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lidar_bedlam.train.config import load_config
from lidar_bedlam.train.loop import new_run_name


def main() -> int:
    """Entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--checkpoint-root", type=Path, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    root = args.checkpoint_root or Path(cfg.checkpoint_root)
    root.mkdir(parents=True, exist_ok=True)
    name = new_run_name(root, cfg.experiment)
    (root / name).mkdir(exist_ok=True)  # claim it
    sys.stdout.write(name + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
