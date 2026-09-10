"""Generate synthetic shards from extracted BEDLAM groups (multiprocess).

Usage::

    uv run python scripts/generate_synthetic.py --out data/generated/synth/v1 \
        --groups 20221024_3-10_100_batch01handhair_static_highSchoolGym \
        --workers 8 [--max-frames 20] [--speed-mean 20 --speed-std 20]

Each worker owns a slice of the frames and writes ``w<id>_<shard>.npz``
plus ``w<id>_stats.json``; ``stats.json`` merges the workers.
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
from lidar_bedlam.data.bedlam import BedlamFramesSource
from lidar_bedlam.generate.synth import (
    SynthConfig,
    SynthGenerator,
    frame_indices,
    generate,
)

DATA = Path("data")
RAW = DATA / "generated" / "bedlam_raw"
LABELS = DATA / "generated" / "bedlam_labels" / "smpl" / "bedlam-labels"
BODY = DATA / "generated" / "body_models"


def _worker(
    args: tuple[int, list[str], list[int], SynthConfig, Path],
) -> dict[str, Any]:
    wid, groups, indices, cfg, out = args
    smpl = SmplModel(BODY)
    src = BedlamFramesSource(
        RAW, groups=groups, frame_stride=5, labels_dir=LABELS, smpl_model=smpl
    )
    gen = SynthGenerator(src, smpl, cfg, DATA, seed_offset=wid)
    return generate(gen, indices, out, f"w{wid:02d}")


def main(argv: list[str] | None = None) -> int:
    """CLI."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--groups", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--speed-mean", type=float, default=0.0)
    ap.add_argument("--speed-std", type=float, default=0.0)
    ap.add_argument("--distance-aug", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    cfg = replace(
        SynthConfig(),
        speed_mean_kmh=a.speed_mean,
        speed_std_kmh=a.speed_std,
        distance_aug_fraction=a.distance_aug,
        seed=a.seed,
    )
    src = BedlamFramesSource(RAW, groups=a.groups, frame_stride=5)
    frames = frame_indices(src)
    if a.max_frames:
        frames = frames[: a.max_frames]
    sys.stdout.write(f"{len(frames)} frames in {len(a.groups)} groups\n")
    slices = [frames[w :: a.workers] for w in range(a.workers)]
    jobs = [(w, a.groups, s, cfg, a.out) for w, s in enumerate(slices) if s]
    with Pool(len(jobs)) as pool:
        results = pool.map(_worker, jobs)
    merged: dict[str, Any] = {
        "config": results[0]["config"],
        "workers": len(results),
    }
    for key in (
        "frames",
        "frames_distance_aug",
        "persons_total",
        "persons_unlabelled",
        "rejected_box",
        "rejected_points",
        "records",
        "shards",
    ):
        merged[key] = sum(int(r[key]) for r in results)
    merged["per_group"] = {}
    for r in results:
        for g, n in r["per_group"].items():
            merged["per_group"][g] = merged["per_group"].get(g, 0) + n
    (a.out / "stats.json").write_text(json.dumps(merged, indent=2))
    sys.stdout.write(
        json.dumps(
            {k: v for k, v in merged.items() if k != "config"}, indent=1
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
