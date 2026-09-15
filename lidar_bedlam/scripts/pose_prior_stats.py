"""Per-joint rotation statistics of the BEDLAM poses: the pose prior.

For every SMPL body joint the chordal mean rotation over the synthetic
records and the standard deviation of the geodesic angle to that mean.
Written as ``pose_prior_bedlam.npz`` (``mean_rot`` (23, 3, 3), ``sigma``
(23,) in radians, ``count``) next to the body models; ``losses.pose_prior``
turns it into a loss for joints that no label supervises.

    uv run python lidar_bedlam/scripts/pose_prior_stats.py \\
        resources/data/generated/synth/v1 \\
        --out resources/data/generated/body_models/pose_prior_bedlam.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

FloatArray = NDArray[np.float64]
NUM_BODY = 23


def chordal_mean(mats: FloatArray) -> FloatArray:
    """Rotation closest to the arithmetic mean of ``mats`` (N, 3, 3)."""
    m = mats.mean(0)
    u, _, vt = np.linalg.svd(m)
    d = np.sign(np.linalg.det(u @ vt))
    return np.asarray(u @ np.diag([1.0, 1.0, d]) @ vt, dtype=np.float64)


def geodesic(mats: FloatArray, ref: FloatArray) -> FloatArray:
    """Angle in radians between every rotation in ``mats`` and ``ref``."""
    rel = np.einsum("nij,kj->nik", mats, ref)  # R @ ref^T
    tr = np.trace(rel, axis1=1, axis2=2)
    return np.asarray(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0)))


def main(argv: list[str] | None = None) -> int:
    """CLI."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("shard_dirs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-records", type=int, default=200000)
    args = ap.parse_args(argv)
    poses = []
    n = 0
    for d in args.shard_dirs:
        for p in sorted(d.glob("*.npz")):
            if p.name.endswith("stats.npz") or ".fit." in p.name:
                continue
            with np.load(p) as z:
                if "body_pose" not in z.files:
                    continue
                has = z["has_smpl"].astype(bool)
                bp = z["body_pose"][has].astype(np.float64)
            poses.append(bp)
            n += len(bp)
            if n >= args.max_records:
                break
        if n >= args.max_records:
            break
    aa = np.concatenate(poses)[: args.max_records].reshape(-1, NUM_BODY, 3)
    mean_rot = np.zeros((NUM_BODY, 3, 3))
    sigma = np.zeros(NUM_BODY)
    for j in range(NUM_BODY):
        mats = Rotation.from_rotvec(aa[:, j]).as_matrix()
        mean_rot[j] = chordal_mean(mats)
        sigma[j] = float(np.sqrt(np.mean(geodesic(mats, mean_rot[j]) ** 2)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, mean_rot=mean_rot, sigma=sigma, count=np.array(len(aa)))
    sys.stdout.write(
        f"{len(aa)} poses; sigma per joint (deg): "
        + " ".join(f"{np.degrees(s):.0f}" for s in sigma)
        + f"\nwrote {args.out}\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
