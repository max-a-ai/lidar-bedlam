"""Real-data shards: Waymo (train / val) and SLOPER4D (train / test).

Usage::

    uv run python lidar_bedlam/scripts/generate_real.py \
        --out resources/data/generated/real/v1 \
        --datasets sloper4d waymo

Waymo uses the SAM 3 masks from ``resources/data/generated/waymo_masks``
when present.
SLOPER4D split: train = seq002, seq003, seq005, seq007; test = seq008, seq009.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from lidar_bedlam.body.smpl import SmplModel
from lidar_bedlam.data.sloper4d import Sloper4dSource
from lidar_bedlam.data.waymo import WaymoSource
from lidar_bedlam.generate.real import RealRecordBuilder, build_real_shards

DATA = Path("resources/data")
BODY = DATA / "generated" / "body_models"
MASKS = DATA / "generated" / "waymo_masks"
SLOPER_TRAIN = [
    "seq002_football_001",
    "seq003_street_002",
    "seq005_library_002",
    "seq007_garden_001",
]
SLOPER_TEST = ["seq008_running_001", "seq009_running_002"]


def main(argv: list[str] | None = None) -> int:
    """CLI."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--datasets", nargs="+", default=["sloper4d", "waymo"])
    ap.add_argument("--sloper-stride", type=int, default=1)
    args = ap.parse_args(argv)
    smpl = SmplModel(BODY)
    results = {}
    if "sloper4d" in args.datasets:
        for split, seqs in (("train", SLOPER_TRAIN), ("test", SLOPER_TEST)):
            src = Sloper4dSource(DATA / "sloper4d", smpl, seqs)
            ids = list(range(0, len(src), args.sloper_stride))
            builder = RealRecordBuilder(smpl)
            results[f"sloper4d_{split}"] = build_real_shards(
                src, builder, args.out, f"sloper4d_{split}", indices=ids
            )
            sys.stdout.write(
                f"sloper4d {split}: {results[f'sloper4d_{split}']}\n"
            )
    if "waymo" in args.datasets:
        root = DATA / "waymo_perception" / "waymo_pose_complete_4"
        for split in ("train", "val"):
            wsrc = WaymoSource(root, split)
            builder = RealRecordBuilder(smpl, mask_dir=MASKS)
            results[f"waymo_{split}"] = build_real_shards(
                wsrc, builder, args.out, f"waymo_{split}"
            )
            sys.stdout.write(f"waymo {split}: {results[f'waymo_{split}']}\n")
    (args.out / "stats.json").write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
