"""Waymo shards whose LiDAR is simulated on the pseudo-GT mesh.

Domain test for the Waymo hand problem: is it the real LiDAR? Every Waymo
record with an accepted pseudo-GT v2 mesh gets its real returns replaced by
a scan simulated on that mesh exactly like the BEDLAM and 3DPW records
(one spinning sensor with fixed rows, a random OS1 spec from the same
channel and step choices, placed in a 1 m ball around the camera, a 1 to
4 cm clothing offset), so the point structure is identical to the
synthetic sources. Records without a mesh keep no points (image only).

    uv run python lidar_bedlam/scripts/make_waymo_simlidar_shards.py \\
        --src resources/data/generated/real/v1_pseudo2 \\
        --out resources/data/generated/real/v1_simlidar \\
        --pattern "waymo_train_*.npz"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from lidar_bedlam.body.smpl import SmplModel, SmplParams
from lidar_bedlam.data.shards import ShardDataset
from lidar_bedlam.generate.records import MAX_POINTS, rewrite_shard
from lidar_bedlam.generate.synth import SynthConfig
from lidar_bedlam.geometry.camera import PinholeCamera
from lidar_bedlam.lidar.mesh_depth import offset_mesh, render_depth
from lidar_bedlam.lidar.placement import BallPlacement
from lidar_bedlam.lidar.simulate import ouster, simulate

BODY_MODELS = Path("resources/data/generated/body_models")
RENDER_SCALE = 2  # depth rendered at 512 px for the 256 px crop


def main(argv: list[str] | None = None) -> int:
    """CLI."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pattern", default="waymo_train_*.npz")
    ap.add_argument("--offset-min", type=float, default=0.01)
    ap.add_argument("--offset-max", type=float, default=0.04)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    cfg = SynthConfig()
    ball = BallPlacement(cfg.ball_r_max_m, cfg.tilt_max_deg)
    smpl = SmplModel(BODY_MODELS)
    rng = np.random.default_rng(args.seed)
    shards = sorted(
        p for p in args.src.glob(args.pattern) if ".fit." not in p.name
    )
    stats = {"records": 0, "simulated": 0, "no_mesh": 0}
    counts: list[int] = []
    t0 = time.time()
    for shard in shards:
        with np.load(shard, allow_pickle=False) as z:
            n = len(z["key"])
            has = z["has_smpl"].astype(bool)
            go, bp = z["global_orient"], z["body_pose"]
            betas, tr = z["betas"], z["transl"]
            intr = z["intrinsics"].astype(np.float64)
            crop = int(z["mask"].shape[1])
        pts = np.zeros((n, MAX_POINTS, 3), np.float16)
        ch = np.full((n, MAX_POINTS), -1, np.int16)
        col = np.full((n, MAX_POINTS), -1, np.int16)
        cnt = np.zeros(n, np.int32)
        chans = np.zeros(n, np.int16)
        steps = np.zeros(n, np.int16)
        poses = np.tile(np.eye(4, dtype=np.float32), (n, 1, 1))
        for i in range(n):
            stats["records"] += 1
            if not has[i]:
                stats["no_mesh"] += 1
                continue
            verts, _ = smpl.forward(
                SmplParams(
                    go[i].astype(np.float64),
                    bp[i].astype(np.float64),
                    betas[i].astype(np.float64),
                    tr[i].astype(np.float64),
                )
            )
            offset = float(rng.uniform(args.offset_min, args.offset_max))
            mesh = offset_mesh(verts, smpl.faces, offset)
            k = intr[i] * RENDER_SCALE
            cam = PinholeCamera(
                k[0, 0], k[1, 1], k[0, 2], k[1, 2],
                crop * RENDER_SCALE, crop * RENDER_SCALE,
            )  # fmt: skip
            depth, _ = render_depth([(mesh, smpl.faces)], cam)
            spec = ouster(
                cfg.family,
                int(rng.choice(cfg.channels)),
                int(rng.choice(cfg.steps)),
            )
            pose = ball.sample(rng)
            scan = simulate(depth.astype(np.float64), cam, spec, pose, rng)
            m = min(len(scan.points), MAX_POINTS)
            pts[i, :m] = scan.points[:m]
            ch[i, :m] = scan.channel[:m]
            col[i, :m] = scan.column[:m]
            cnt[i] = m
            chans[i] = spec.channels
            steps[i] = spec.horizontal_steps
            poses[i] = pose
            stats["simulated"] += 1
            counts.append(m)
        rewrite_shard(
            shard,
            args.out / shard.name,
            {
                "scan/real/points": pts,
                "scan/real/channel": ch,
                "scan/real/column": col,
                "scan/real/count": cnt,
                "scan/real/channels": chans,
                "scan/real/steps": steps,
                "scan/real/sensor_pose": poses,
            },
        )
        tok = ShardDataset.tokens_path(shard)
        dst_tok = ShardDataset.tokens_path(args.out / shard.name)
        if tok.exists() and not dst_tok.exists():
            try:
                os.link(tok, dst_tok)
            except OSError:
                dst_tok.symlink_to(tok.resolve())
        sys.stdout.write(
            f"{shard.name}: {int(has.sum())}/{n} simulated, "
            f"{time.time() - t0:.0f} s\n"
        )
        sys.stdout.flush()
    pts_arr = np.array(counts)
    summary = {
        **stats,
        "points_median": float(np.median(pts_arr)) if len(pts_arr) else 0,
        "points_p10": float(np.percentile(pts_arr, 10)) if len(pts_arr) else 0,
        "seconds": time.time() - t0,
    }
    prefix = args.pattern.split("*")[0].strip("_") or "all"
    (args.out / f"simlidar_stats_{prefix}.json").write_text(
        json.dumps(summary, indent=1)
    )
    sys.stdout.write(json.dumps(summary, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
