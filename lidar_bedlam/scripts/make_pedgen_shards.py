"""Waymo training shards with the pedestrian-generation SMPL fits as labels.

The colleague's fits (``ped_gen/waymo_smpl_matched.npz``, cached by
``cache_ped_gen_smpl.py``) come as body pose, betas, a root rotation and
the mesh centroid in the record's camera frame, ego-relative in height:
the notebook ``waymo_pseudo_gt`` lifts each mesh along the world up axis
until its joints sit on the Waymo keypoints. This script does the same and
writes the result as SMPL parameters into a copy of the ``real/v1`` Waymo
training shards (``has_smpl`` set on the matched records, the rest keep
their keypoint-only label), so a config can point its Waymo source at
``real/v1_pedgen``.

    uv run python lidar_bedlam/scripts/make_pedgen_shards.py \\
        --src resources/data/generated/real/v1 \\
        --out resources/data/generated/real/v1_pedgen
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from lidar_bedlam.body.smpl import SmplModel, SmplParams
from lidar_bedlam.data.schema import WAYMO15_TO_COCO17
from lidar_bedlam.scripts.make_lidarhmr_shards import (
    BODY_MODELS,
    relabel_shards,
)

PED_GEN = Path("resources/data/generated/ped_gen")
EGO_TOL = 0.05  # m; tracks whose ego-frame check misses by more are skipped


def main(argv: list[str] | None = None) -> int:
    """CLI."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--matched", type=Path, default=PED_GEN / "waymo_smpl_matched.npz"
    )
    ap.add_argument("--match", type=Path, default=PED_GEN / "waymo_match.npz")
    ap.add_argument("--pattern", default="waymo_train_*.npz")
    ap.add_argument("--check", type=int, default=8, help="records per shard")
    args = ap.parse_args(argv)
    pg = np.load(args.matched)
    keep = pg["ego_residual_m"] < EGO_TOL
    at = {str(k): i for i, k in enumerate(pg["key"]) if keep[i]}
    match = np.load(args.match)
    up = {
        str(k): np.asarray(mat, np.float64)[:3, :3] @ np.array([0.0, 0.0, 1.0])
        for k, mat in zip(match["key"], match["world_to_camera"], strict=True)
    }
    smpl = SmplModel(BODY_MODELS)
    sel = np.nonzero(WAYMO15_TO_COCO17 >= 0)[0]
    csel = WAYMO15_TO_COCO17[sel]
    keypoints: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for shard in sorted(args.src.glob(args.pattern)):
        if ".fit." in shard.name:
            continue
        with np.load(shard, allow_pickle=False) as z:
            for k, j, v in zip(
                z["key"], z["joints3d"][:, :15], z["joints3d_valid"][:, :15],
                strict=True,
            ):  # fmt: skip
                keypoints[str(k)] = (j.astype(np.float64), v.astype(bool))
    rebuilt: list[float] = []

    def label(key: str) -> SmplParams | None:
        i = at.get(key)
        if i is None or key not in up:
            return None
        bp = pg["body_pose"][i].astype(np.float64)
        betas = pg["betas"][i].astype(np.float64)
        rot = pg["orient"][i].astype(np.float64)
        centroid = pg["centroid"][i].astype(np.float64)
        posed, _ = smpl.forward(
            SmplParams(np.zeros(3), bp, betas, np.zeros(3))
        )
        delivered = (posed - posed.mean(0)) @ rot.T + centroid
        kp, valid = keypoints[key]
        ok = valid[sel]
        shift = np.zeros(3)
        if ok.any():
            coco = smpl.coco_joints(delivered)
            shift = (kp[sel][ok] - coco[csel][ok]).mean(0)
        lifted = delivered + float(shift @ up[key]) * up[key]
        # SMPL rotates about the pelvis: v = R (v0 - p) + p + t
        pelvis = smpl.rest_pelvis(betas)
        transl = (
            rot @ (pelvis - posed.mean(0)) - pelvis + centroid
            + float(shift @ up[key]) * up[key]
        )  # fmt: skip
        params = SmplParams(
            Rotation.from_matrix(rot).as_rotvec(), bp, betas, transl
        )
        if len(rebuilt) < 200:
            verts, _ = smpl.forward(params)
            rebuilt.append(float(np.abs(verts - lifted).max()))
        return params

    summary = relabel_shards(
        args.src, args.out, args.pattern, label, args.check
    )
    sys.stdout.write(
        f"parameter rebuild vs lifted mesh, max over {len(rebuilt)} records: "
        f"{max(rebuilt) * 1000:.2f} mm\n" + summary
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
