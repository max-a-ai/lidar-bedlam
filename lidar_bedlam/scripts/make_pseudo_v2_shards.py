"""Waymo training shards with pseudo-GT v2 mesh labels.

Runs ``generate.pseudo_fit_v2`` over every Waymo training record: the
records are grouped by track (one Waymo object seen by one camera),
ordered by frame time, fitted in batches, and written as a copy of the
``real/v1`` shards with the fitted SMPL parameters, ``has_smpl`` set on the
accepted
records, and four new per-record arrays: ``label_conf`` (the confidence
the training loss weights the mesh terms with), ``pseudo_kp_error_m``,
``pseudo_chamfer_m`` and ``pseudo_prior_energy``. Token sidecars are
linked.

The fit starts from LiDAR-HMR's published fit of the same frame
(``real/v1_lidarhmr``, 4,586 of 4,591 records); records without one start
from TokenHMR's image prediction where the earlier TokenHMR repository
produced one, and are skipped otherwise.

    uv run python lidar_bedlam/scripts/make_pseudo_v2_shards.py \\
        --src resources/data/generated/real/v1 \\
        --init resources/data/generated/real/v1_lidarhmr \\
        --out resources/data/generated/real/v1_pseudo2
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from lidar_bedlam.data.shards import ShardDataset
from lidar_bedlam.generate.pseudo_fit_v2 import (
    FitBatch,
    FitV2Config,
    FitV2Result,
    PseudoFitterV2,
    summarize,
)
from lidar_bedlam.generate.records import rewrite_shard

BODY_MODELS = Path("resources/data/generated/body_models")
TOKENIZER = Path("resources/pretrained-checkpoints/tokenhmr_tokenizer.pt")
TOKENHMR_PKL = Path("/home/max/Documents/tokenHMR/debug_output/pkl-pseudoGT")


@dataclass
class Record:
    """One record's evidence and initialisation, numpy."""

    shard: int
    row: int
    key: str
    track: str
    time_s: float
    joints3d: Any
    joints3d_valid: Any
    kp2d: Any
    intrinsics: Any
    points: Any
    points_valid: Any
    box3d: Any
    init: dict[str, Any] | None


def tokenhmr_init(key: str) -> dict[str, Any] | None:
    """TokenHMR's image prediction of the record, if the old repository
    produced one (``<frame>_<object>.pkl``)."""
    _, _, frame, obj = key.split("/")
    path = TOKENHMR_PKL / f"{frame}_{obj}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as fh:
        d = pickle.load(fh)  # noqa: S301
    go = Rotation.from_matrix(np.asarray(d["global_orient"]).reshape(3, 3))
    bp = Rotation.from_matrix(np.asarray(d["body_pose"]).reshape(-1, 3, 3))
    return {
        "global_orient": go.as_rotvec(),
        "body_pose": bp.as_rotvec().reshape(-1)[:69],
        "betas": np.asarray(d["betas"]).reshape(-1)[:10].astype(np.float64),
        "transl": None,  # image depth is not metric: the centroid is used
    }


def load_records(
    src: Path, init_dir: Path, pattern: str, n_points: int
) -> tuple[list[Path], list[Record]]:
    """Every record of the shards with its LiDAR-HMR or TokenHMR init."""
    shards = sorted(p for p in src.glob(pattern) if ".fit." not in p.name)
    records: list[Record] = []
    for si, shard in enumerate(shards):
        init_path = init_dir / shard.name
        with (
            np.load(shard, allow_pickle=False) as z,
            np.load(init_path, allow_pickle=False) as zi,
        ):
            keys = [str(k) for k in z["key"]]
            assert keys == [str(k) for k in zi["key"]], shard.name
            has_init = zi["has_smpl"].astype(bool)
            pts_all = z["scan/real/points"].astype(np.float32)
            counts = z["scan/real/count"].astype(int)
            for i, key in enumerate(keys):
                parts = key.split("/")
                init: dict[str, Any] | None
                if has_init[i]:
                    init = {
                        "global_orient": zi["global_orient"][i].astype(
                            np.float64
                        ),
                        "body_pose": zi["body_pose"][i].astype(np.float64),
                        "betas": zi["betas"][i].astype(np.float64),
                        "transl": zi["transl"][i].astype(np.float64),
                    }
                else:
                    init = tokenhmr_init(key)
                pts = pts_all[i, : counts[i]]
                valid = np.zeros(n_points, dtype=bool)
                out = np.zeros((n_points, 3), dtype=np.float32)
                m = min(len(pts), n_points)
                if m:
                    sel = (
                        np.random.default_rng(i).choice(
                            len(pts), n_points, replace=False
                        )
                        if len(pts) > n_points
                        else np.arange(len(pts))
                    )
                    out[: len(sel)] = pts[sel]
                    valid[: len(sel)] = True
                if init is not None and init["transl"] is None:
                    # image init: place the pelvis on the point centroid
                    init["transl"] = (
                        out[valid].mean(0).astype(np.float64)
                        if valid.any()
                        else z["box3d"][i][:3].astype(np.float64)
                    )
                records.append(
                    Record(
                        si,
                        i,
                        key,
                        parts[3],
                        int(parts[2].split("_")[0]) / 1e6,
                        z["joints3d"][i][:15].astype(np.float64),
                        z["joints3d_valid"][i][:15].astype(bool),
                        z["kp2d"][i][:15].astype(np.float64),
                        z["intrinsics"][i].astype(np.float64),
                        out,
                        valid,
                        z["box3d"][i].astype(np.float64),
                        init,
                    )
                )
    return shards, records


def batches(records: list[Record], size: int) -> list[list[Record]]:
    """Tracks kept together, ordered by time, packed to about ``size``."""
    by_track: dict[str, list[Record]] = {}
    for r in records:
        if r.init is not None:
            by_track.setdefault(r.track, []).append(r)
    out: list[list[Record]] = []
    cur: list[Record] = []
    for track in sorted(by_track):
        recs = sorted(by_track[track], key=lambda r: r.time_s)
        if cur and len(cur) + len(recs) > size:
            out.append(cur)
            cur = []
        cur.extend(recs)
    if cur:
        out.append(cur)
    return out


def to_fit_batch(recs: list[Record]) -> FitBatch:
    """Stack the records; track indices are dense in order of appearance."""
    tracks: dict[str, int] = {}
    idx = np.array([tracks.setdefault(r.track, len(tracks)) for r in recs])

    def stack(name: str) -> Any:
        return np.stack([getattr(r, name) for r in recs])

    def init(name: str) -> Any:
        return np.stack([r.init[name] for r in recs if r.init is not None])

    return FitBatch(
        keys=[r.key for r in recs],
        track=idx.astype(np.int64),
        time_s=np.array([r.time_s for r in recs], dtype=np.float64),
        joints3d=stack("joints3d"),
        joints3d_valid=stack("joints3d_valid"),
        kp2d=stack("kp2d"),
        intrinsics=stack("intrinsics"),
        points=stack("points").astype(np.float64),
        points_valid=stack("points_valid"),
        box3d=stack("box3d"),
        init_global_orient=init("global_orient"),
        init_body_pose=init("body_pose"),
        init_betas=init("betas"),
        init_transl=init("transl"),
    )


def main(argv: list[str] | None = None) -> int:
    """CLI."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--init", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pattern", default="waymo_train_*.npz")
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--n-points", type=int, default=1024)
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--limit", type=int, default=0, help="records, for tests")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    shards, records = load_records(
        args.src, args.init, args.pattern, args.n_points
    )
    if args.limit:
        records = records[: args.limit]
    n_init = sum(r.init is not None for r in records)
    sys.stdout.write(
        f"{len(records)} records in {len(shards)} shards, {n_init} with an "
        f"initialisation\n"
    )
    fitter = PseudoFitterV2(
        BODY_MODELS,
        TOKENIZER,
        torch.device(args.device),
        FitV2Config(iters=args.iters),
    )
    results: dict[str, tuple[FitV2Result, int]] = {}
    all_results: list[FitV2Result] = []
    t0 = time.time()
    packs = batches(records, args.batch_size)
    for bi, recs in enumerate(packs):
        res = fitter.fit(to_fit_batch(recs))
        all_results.append(res)
        for j, r in enumerate(recs):
            results[r.key] = (res, j)
        done = sum(len(p) for p in packs[: bi + 1])
        sys.stdout.write(
            f"batch {bi + 1}/{len(packs)}: {done} records, "
            f"{done / (time.time() - t0):.1f} records/s, accepted so far "
            f"{sum(int(x.accepted.sum()) for x in all_results)}\n"
        )
        sys.stdout.flush()
    # write the shards
    for shard in shards:
        with np.load(shard, allow_pickle=False) as z:
            keys = [str(k) for k in z["key"]]
            go = z["global_orient"].astype(np.float64).copy()
            bp = z["body_pose"].astype(np.float64).copy()
            betas = z["betas"].astype(np.float64).copy()
            tr = z["transl"].astype(np.float64).copy()
        n = len(keys)
        has = np.zeros(n, dtype=bool)
        conf = np.zeros(n, dtype=np.float32)
        kp_err = np.full(n, np.nan, dtype=np.float32)
        cham = np.full(n, np.nan, dtype=np.float32)
        energy = np.full(n, np.nan, dtype=np.float32)
        for i, key in enumerate(keys):
            if key not in results:
                continue
            res, j = results[key]
            kp_err[i], cham[i], energy[i] = (
                res.kp_error_m[j],
                res.chamfer_m[j],
                res.prior_energy[j],
            )
            if not res.accepted[j]:
                continue
            go[i], bp[i], betas[i], tr[i] = (
                res.global_orient[j],
                res.body_pose[j],
                res.betas[j],
                res.transl[j],
            )
            has[i] = True
            conf[i] = res.confidence[j]
        rewrite_shard(
            shard,
            args.out / shard.name,
            {
                "global_orient": go,
                "body_pose": bp,
                "betas": betas,
                "transl": tr,
                "has_smpl": has,
                "label_conf": conf,
                "pseudo_kp_error_m": kp_err,
                "pseudo_chamfer_m": cham,
                "pseudo_prior_energy": energy,
            },
        )
        tok = ShardDataset.tokens_path(shard)
        dst_tok = ShardDataset.tokens_path(args.out / shard.name)
        if tok.exists() and not dst_tok.exists():
            try:
                os.link(tok, dst_tok)
            except OSError:
                dst_tok.symlink_to(tok.resolve())
        sys.stdout.write(f"{shard.name}: {int(has.sum())}/{n} labelled\n")
    stats = summarize(all_results)
    stats["seconds"] = time.time() - t0
    (args.out / "pseudo_v2_stats.json").write_text(json.dumps(stats, indent=1))
    sys.stdout.write(json.dumps(stats, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
