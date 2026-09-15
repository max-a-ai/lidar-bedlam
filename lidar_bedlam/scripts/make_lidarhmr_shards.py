"""Waymo training shards with LiDAR-HMR's published SMPL fits as mesh labels.

``match_lidar_hmr_records.py`` pairs our Waymo training records with the
fit LiDAR-HMR ships for the same frame (``lidar_hmr/waymo_match.npz``:
SMPL parameters in the Waymo vehicle frame plus the vehicle-to-camera
transform of our record). This script copies the ``real/v1`` Waymo
training shards, writes those parameters, expressed in the record's
camera frame, into the SMPL fields, sets ``has_smpl`` for the matched
records and links the token sidecars, so a config can point its Waymo
source at ``real/v1_lidarhmr`` and the full-mesh losses apply.

    uv run python lidar_bedlam/scripts/make_lidarhmr_shards.py \\
        --src resources/data/generated/real/v1 \\
        --match resources/data/generated/lidar_hmr/waymo_match.npz \\
        --out resources/data/generated/real/v1_lidarhmr
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

from lidar_bedlam.body.smpl import SmplModel, SmplParams, transform_smpl_params
from lidar_bedlam.data.schema import WAYMO15_TO_COCO17
from lidar_bedlam.data.shards import ShardDataset
from lidar_bedlam.generate.records import rewrite_shard

BODY_MODELS = Path("resources/data/generated/body_models")


def relabel_shards(
    src: Path,
    out: Path,
    pattern: str,
    label: Callable[[str], SmplParams | None],
    check: int,
) -> str:
    """Copy the shards, replacing the SMPL fields of every record ``label``
    knows; returns a one-line summary with the keypoint check."""
    smpl = SmplModel(BODY_MODELS)
    sel = np.nonzero(WAYMO15_TO_COCO17 >= 0)[0]
    csel = WAYMO15_TO_COCO17[sel]
    total = matched = 0
    errs: list[float] = []
    for shard in sorted(src.glob(pattern)):
        if ".fit." in shard.name:
            continue
        with np.load(shard, allow_pickle=False) as z:
            keys = [str(k) for k in z["key"]]
            go = z["global_orient"].astype(np.float64).copy()
            bp = z["body_pose"].astype(np.float64).copy()
            betas = z["betas"].astype(np.float64).copy()
            tr = z["transl"].astype(np.float64).copy()
            has = np.zeros(len(keys), dtype=bool)
            joints = z["joints3d"][:, :15].astype(np.float64)
            valid = z["joints3d_valid"][:, :15].astype(bool)
        checked = 0
        for i, key in enumerate(keys):
            ours = label(key)
            if ours is None:
                continue
            go[i], bp[i], betas[i], tr[i] = (
                ours.global_orient,
                ours.body_pose,
                ours.betas,
                ours.transl,
            )
            has[i] = True
            if checked < check:  # the label must sit on the keypoints
                verts, _ = smpl.forward(ours)
                coco = smpl.coco_joints(verts)
                ok = valid[i][sel]
                if ok.any():
                    err = np.linalg.norm(
                        coco[csel][ok] - joints[i][sel][ok], axis=1
                    ).mean()
                    errs.append(float(err))
                checked += 1
        rewrite_shard(
            shard,
            out / shard.name,
            {
                "global_orient": go,
                "body_pose": bp,
                "betas": betas,
                "transl": tr,
                "has_smpl": has,
            },
        )
        tok = ShardDataset.tokens_path(shard)
        dst_tok = ShardDataset.tokens_path(out / shard.name)
        if tok.exists() and not dst_tok.exists():
            try:
                os.link(tok, dst_tok)
            except OSError:
                dst_tok.symlink_to(tok.resolve())
        total += len(keys)
        matched += int(has.sum())
        sys.stdout.write(
            f"{shard.name}: {int(has.sum())}/{len(keys)} labelled\n"
        )
        sys.stdout.flush()
    return (
        f"{matched}/{total} records carry the new labels; keypoint error of "
        f"the camera-frame label on {len(errs)} checked records: "
        f"mean {np.mean(errs) * 1000:.1f} mm, "
        f"max {np.max(errs) * 1000:.1f} mm\n"
    )


def main(argv: list[str] | None = None) -> int:
    """CLI."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--match", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pattern", default="waymo_train_*.npz")
    ap.add_argument("--check", type=int, default=8, help="records per shard")
    args = ap.parse_args(argv)
    m = np.load(args.match)
    at = {str(k): i for i, k in enumerate(m["key"])}
    smpl = SmplModel(BODY_MODELS)

    def label(key: str) -> SmplParams | None:
        j = at.get(key)
        if j is None:
            return None
        theirs = SmplParams(
            m["global_orient"][j].astype(np.float64),
            m["body_pose"][j].astype(np.float64),
            m["betas"][j].astype(np.float64),
            m["transl"][j].astype(np.float64),
        )
        return transform_smpl_params(
            theirs, m["vehicle_to_camera"][j], smpl.rest_pelvis(theirs.betas)
        )

    sys.stdout.write(
        relabel_shards(args.src, args.out, args.pattern, label, args.check)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
