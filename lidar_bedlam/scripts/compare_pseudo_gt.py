"""Fit quality of LiDAR-HMR's published Waymo pseudo-GT against ours.

Both label sets are SMPL fits to the same Waymo keypoints. The same two
measures are computed for each: the error of the COCO limb joints
regressed from the fitted mesh against the labelled 3D keypoints, and the
median nearest-vertex distance of the person's LiDAR returns to the mesh.

    uv run python lidar_bedlam/scripts/compare_pseudo_gt.py \\
        --lidar-hmr resources/data/external/lidar_hmr_waymov2 \\
        --ours resources/data/generated/real/v1_pseudo
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from lidar_bedlam.body.smpl import SmplModel, SmplParams
from lidar_bedlam.data.schema import WAYMO15_TO_COCO17
from lidar_bedlam.generate.records import Shard

FloatArray = NDArray[np.float64]

# COCO-17 limb joints: shoulders, elbows, wrists, hips, knees, ankles
LIMBS = frozenset({5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16})
# LiDAR-HMR's 14-keypoint order -> COCO index (nose and head left out)
THEIRS_TO_COCO = {
    1: 5,
    7: 6,
    2: 7,
    8: 8,
    3: 9,
    9: 10,
    4: 11,
    10: 12,
    5: 13,
    11: 14,
    6: 15,
    12: 16,
}
PERSON_RADIUS_M = 1.5


def report(
    name: str, kp: FloatArray, p2m: FloatArray, dist: FloatArray
) -> None:
    """Print the two measures overall and per distance band."""
    sys.stdout.write(
        f"{name}: n={len(kp)}  keypoint err mean {kp.mean() * 1000:.1f} mm, "
        f"median {np.median(kp) * 1000:.1f} mm | point-to-mesh median "
        f"{np.median(p2m) * 100:.2f} cm, "
        f"p90 {np.percentile(p2m, 90) * 100:.2f} cm"
        f" | distance median {np.median(dist):.1f} m\n"
    )
    for lo, hi in ((0, 10), (10, 20), (20, 40)):
        m = (dist >= lo) & (dist < hi)
        if m.any():
            sys.stdout.write(
                f"   {lo}-{hi} m: n={int(m.sum())} "
                f"kp {kp[m].mean() * 1000:.1f} mm,"
                f" p2m {np.median(p2m[m]) * 100:.2f} cm\n"
            )


def lidar_hmr(path: Path, smpl: SmplModel, limit: int) -> None:
    """Measures on one of LiDAR-HMR's ``save_data/waymov2`` pickles."""
    with open(path, "rb") as fh:
        records: list[dict[str, Any]] = pickle.load(fh)
    kp_err, p2m, dist = [], [], []
    for r in records[:limit]:
        verts = np.asarray(r["smpl_verts"], np.float64)
        coco = smpl.coco_joints(verts)
        kp = np.asarray(r["keypoints"], np.float64)
        flag = np.asarray(r["flag"], bool)
        errs = [
            float(np.linalg.norm(coco[c] - kp[i]))
            for i, c in THEIRS_TO_COCO.items()
            if flag[i]
        ]
        if not errs:
            continue
        kp_err.append(float(np.mean(errs)))
        pts = np.asarray(r["human_points"][0], np.float64)
        p2m.append(float(np.median(cKDTree(verts).query(pts)[0])))
        dist.append(float(np.linalg.norm(kp[flag].mean(0))))
    report(
        f"LiDAR-HMR {path.name}",
        np.array(kp_err),
        np.array(p2m),
        np.array(dist),
    )


def ours(root: Path, smpl: SmplModel, n_shards: int) -> None:
    """Measures on our ``real/v1_pseudo`` Waymo training shards."""
    shards = sorted(
        p for p in root.glob("waymo_train_*.npz") if ".fit." not in p.name
    )[:n_shards]
    kp_err, p2m, dist = [], [], []
    for sp in shards:
        s = Shard(sp)
        for i in range(len(s)):
            if not bool(s.row("has_smpl", i)):
                continue
            params = SmplParams(
                global_orient=s.row("global_orient", i).astype(np.float64),
                body_pose=s.row("body_pose", i).astype(np.float64),
                betas=s.row("betas", i).astype(np.float64),
                transl=s.row("transl", i).astype(np.float64),
            )
            verts, _ = smpl.forward(params)
            coco = smpl.coco_joints(verts)
            j = s.row("joints3d", i)[:15].astype(np.float64)
            v = s.row("joints3d_valid", i)[:15].astype(bool)
            errs = [
                float(np.linalg.norm(coco[c] - j[w]))
                for w, c in enumerate(WAYMO15_TO_COCO17[:15])
                if c in LIMBS and v[w]
            ]
            pts = np.asarray(s.scan(i, "real").points, np.float64)
            centre = s.row("box3d", i).astype(np.float64)[:3]
            near = pts[np.linalg.norm(pts - centre, axis=1) < PERSON_RADIUS_M]
            if not errs or not len(near):
                continue
            kp_err.append(float(np.mean(errs)))
            p2m.append(float(np.median(cKDTree(verts).query(near)[0])))
            dist.append(float(np.linalg.norm(j[v].mean(0))))
    report(
        f"ours {root.name} ({len(shards)} shards)",
        np.array(kp_err),
        np.array(p2m),
        np.array(dist),
    )


def main(argv: list[str] | None = None) -> int:
    """CLI."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lidar-hmr", type=Path, required=True)
    ap.add_argument("--ours", type=Path, required=True)
    ap.add_argument(
        "--body-models",
        type=Path,
        default=Path("resources/data/generated/body_models"),
    )
    ap.add_argument(
        "--limit", type=int, default=2000, help="records per pickle"
    )
    ap.add_argument("--shards", type=int, default=4, help="of our shards")
    args = ap.parse_args(argv)
    smpl = SmplModel(args.body_models, gender="neutral")
    for name in ("test.pkl", "train.pkl"):
        lidar_hmr(args.lidar_hmr / name, smpl, args.limit)
    ours(args.ours, smpl, args.shards)
    return 0


if __name__ == "__main__":
    sys.exit(main())
