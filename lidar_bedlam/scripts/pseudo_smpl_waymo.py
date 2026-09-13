"""Fit pseudo ground-truth SMPL to the Waymo training records.

    uv run python lidar_bedlam/scripts/pseudo_smpl_waymo.py \\
        --checkpoint outputs/helma/main-mixed-000/best.pt \\
        --config configs/main_mixed.yaml \\
        --shards resources/data/generated/real/v1 \\
        --out resources/data/generated/real/v1_pseudo

Writes ``waymo_train_*.npz`` copies with the fitted SMPL parameters and
``has_smpl=True`` for accepted fits (real keypoints are kept as the joint
labels), hard-links the token files, and writes ``<shard>.fit.npz``
(per-record errors) plus ``pseudo_stats.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from lidar_bedlam.data.shards import ShardDataset, ShardDatasetConfig
from lidar_bedlam.generate.pseudo_smpl import (
    FitResult,
    PseudoFitConfig,
    PseudoSmplFitter,
    summarize,
)
from lidar_bedlam.generate.records import rewrite_shard
from lidar_bedlam.train.config import load_config
from lidar_bedlam.train.loop import build_model, tensors_only, to_device


def _concat(results: list[FitResult]) -> FitResult:
    return FitResult(
        *[
            np.concatenate([getattr(r, f) for r in results])
            for f in FitResult.__dataclass_fields__
        ]
    )


def main(argv: list[str] | None = None) -> int:
    """CLI."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--shards", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pattern", default="waymo_train_*.npz")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--max-error-m", type=float, default=0.08)
    args = ap.parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(args.config)
    model = build_model(cfg)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    fit_cfg = PseudoFitConfig(iters=args.iters, max_error_m=args.max_error_m)
    fitter = PseudoSmplFitter(model, fit_cfg, device)
    ds_cfg = ShardDatasetConfig(
        variant="real", n_points=cfg.data.n_points, use_augmented_image=0.0,
        require_tokens=True,
    )  # fmt: skip
    all_results: list[FitResult] = []
    for shard in sorted(args.shards.glob(args.pattern)):
        ds = ShardDataset([shard], ds_cfg)
        loader = DataLoader(ds, batch_size=args.batch, num_workers=4)
        results = []
        for batch in loader:
            results.append(fitter.fit(tensors_only(to_device(batch, device))))
        r = _concat(results)
        all_results.append(r)
        has_smpl = r.accepted.copy()
        rewrite_shard(
            shard,
            args.out / shard.name,
            {
                "global_orient": r.global_orient,
                "body_pose": r.body_pose,
                "betas": r.betas,
                "transl": r.transl,
                "has_smpl": has_smpl,
            },
        )
        np.savez(
            args.out / (shard.stem + ".fit.npz"),
            error_m=r.error_m,
            error_init_m=r.error_init_m,
            accepted=r.accepted,
        )
        tok = ShardDataset.tokens_path(shard)
        dst_tok = ShardDataset.tokens_path(args.out / shard.name)
        if tok.exists() and not dst_tok.exists():
            try:
                os.link(tok, dst_tok)
            except OSError:
                dst_tok.symlink_to(tok.resolve())
        sys.stdout.write(
            f"{shard.name}: {int(r.accepted.sum())}/{len(r.accepted)} "
            f"accepted, error {r.error_init_m.mean() * 1000:.1f} -> "
            f"{r.error_m.mean() * 1000:.1f} mm\n"
        )
        sys.stdout.flush()
    stats = summarize(all_results)
    stats["config"] = fit_cfg.__dict__
    (args.out / "pseudo_stats.json").write_text(json.dumps(stats, indent=1))
    sys.stdout.write(json.dumps(stats, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
