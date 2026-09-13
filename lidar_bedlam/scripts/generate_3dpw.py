"""Camera + simulated-LiDAR shards from 3DPW (real image, real SMPL, LiDAR
rendered on the clothing-offset mesh).

    uv run python lidar_bedlam/scripts/generate_3dpw.py \\
        --out resources/data/generated/meshlidar/v1 --split train \\
        --workers 12

Writes ``<out>/threedpw_<split>_wNN_XXXXX.npz`` shards (512 records each)
plus a stats json per worker. Frames are taken every ``--frame-stride``
frames of every sequence of the split.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path
from typing import Any

from lidar_bedlam.body.smpl import SmplModel
from lidar_bedlam.data.threedpw import ThreeDPWSource
from lidar_bedlam.generate.mesh_synth import (
    MeshLidarGenerator,
    MeshSynthConfig,
    generate_mesh,
)
from lidar_bedlam.generate.synth import SynthConfig

DATA = Path("resources/data")
ROOT = DATA / "generated" / "threedpw"
BODY = DATA / "generated" / "body_models"


def _models() -> dict[str, SmplModel]:
    return {
        g: SmplModel(BODY, gender=g) for g in ("neutral", "male", "female")
    }


def _worker(
    args: tuple[
        int, str, list[int], SynthConfig, MeshSynthConfig, Path, int, Path
    ],
) -> dict[str, Any]:
    wid, split, indices, cfg, mesh_cfg, out, stride, image_root = args
    models = _models()
    src = ThreeDPWSource(
        ROOT, split, models, frame_stride=stride, image_root=image_root
    )
    gen = MeshLidarGenerator(src, models, cfg, mesh_cfg, DATA, seed_offset=wid)
    return generate_mesh(gen, indices, out, f"threedpw_{split}_w{wid:02d}")


def main(argv: list[str] | None = None) -> int:
    """CLI."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--frame-stride", type=int, default=5)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--offset-min", type=float, default=0.01)
    ap.add_argument("--offset-max", type=float, default=0.04)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--image-root",
        type=Path,
        default=ROOT,
        help="folder holding imageFiles/ (may be the NAS)",
    )
    args = ap.parse_args(argv)
    cfg = replace(SynthConfig(), seed=args.seed, distance_aug_fraction=0.0)
    mesh_cfg = MeshSynthConfig(args.offset_min, args.offset_max)
    # index only (no SMPL models): torch must not be initialised before the
    # pool forks, or the workers deadlock in OpenMP
    src = ThreeDPWSource(ROOT, args.split, {}, args.frame_stride)
    indices = list(range(len(src)))
    if args.max_frames:
        indices = indices[: args.max_frames]
    sys.stdout.write(f"{len(indices)} frames in split {args.split}\n")
    n = max(1, min(args.workers, len(indices)))
    jobs = [
        (
            w,
            args.split,
            indices[w::n],
            cfg,
            mesh_cfg,
            args.out,
            args.frame_stride,
            args.image_root,
        )
        for w in range(n)
    ]
    with Pool(n) as pool:
        stats = pool.map(_worker, jobs)
    total = {
        "frames": sum(s["frames"] for s in stats),
        "records": sum(s["records"] for s in stats),
        "shards": sum(s["shards"] for s in stats),
        "rejected_box": sum(s["rejected_box"] for s in stats),
        "rejected_points": sum(s["rejected_points"] for s in stats),
    }
    sys.stdout.write(json.dumps(total, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
