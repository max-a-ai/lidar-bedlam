"""Score baseline predictions with the paper protocol.

    uv run python lidar_bedlam/scripts/score_baselines.py \\
        --config configs/main_mixed.yaml \\
        --pred outputs/baselines/*/waymo_val.npz \\
               outputs/baselines/*/sloper4d_test.npz \\
        --out outputs/baselines/results.json

Each prediction file (written by ``lidar_bedlam/scripts/baselines``) holds
camera-frame vertices, the SMPL translation and the global orientation per
record key. The 24 SMPL joints, the COCO joints for Waymo and the 3D box
are derived here exactly as for our model, then ``metrics.protocol`` runs
unchanged. Records without a prediction are skipped and counted.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from lidar_bedlam.body.smpl import SmplModel
from lidar_bedlam.data.base import SMPL_FORWARD
from lidar_bedlam.geometry.boxes import heading_yaw, oriented_box_from_points
from lidar_bedlam.geometry.camera import CAMERA_UP_AXIS
from lidar_bedlam.geometry.rotations import axis_angle_to_matrix
from lidar_bedlam.metrics.protocol import (
    MetricSummary,
    SampleMetrics,
    sample_metrics,
    summarize,
)
from lidar_bedlam.train.config import load_config
from lidar_bedlam.train.loop import build_dataset

FloatArray = np.ndarray[Any, np.dtype[np.float64]]


class PredictionTable:
    """Predictions of one method on one split, addressable by record key."""

    def __init__(self, path: Path) -> None:
        with np.load(path) as z:
            self.index = {str(k): i for i, k in enumerate(z["key"])}
            self.vertices = z["vertices"]
            self.transl = z["transl"].astype(np.float64)
            self.global_orient = z["global_orient"].astype(np.float64)
            self.meta = {
                k[5:]: z[k].item() for k in z.files if k.startswith("meta/")
            }
        self.name = path.parent.name
        self.split = path.stem

    def batch(
        self, keys: list[str], smpl: SmplModel
    ) -> tuple[dict[str, torch.Tensor], list[int]]:
        """Model-style prediction dict for the keys found (and their rows)."""
        rows = [i for i, k in enumerate(keys) if k in self.index]
        idx = [self.index[keys[i]] for i in rows]
        verts = self.vertices[idx].astype(np.float64)
        regressor = np.asarray(
            smpl._model.J_regressor.cpu().numpy(), dtype=np.float64
        )
        joints = np.einsum("jv,nvk->njk", regressor, verts)
        boxes = []
        for v, aa in zip(verts, self.global_orient[idx], strict=True):
            forward = axis_angle_to_matrix(aa) @ SMPL_FORWARD
            yaw = heading_yaw(forward, CAMERA_UP_AXIS)
            boxes.append(oriented_box_from_points(v, yaw, CAMERA_UP_AXIS))
        pred = {
            "vertices": torch.from_numpy(verts),
            "joints3d": torch.from_numpy(joints),
            "transl": torch.from_numpy(self.transl[idx]),
            "box3d": torch.from_numpy(np.stack(boxes)),
            "box_conf": torch.ones(len(idx), dtype=torch.float64),
        }
        return pred, rows


def _select(batch: dict[str, Any], rows: list[int]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v[rows]
        else:
            out[k] = [v[i] for i in rows]
    return out


def score(
    table: PredictionTable, loader: DataLoader[Any], smpl: SmplModel
) -> tuple[MetricSummary, int]:
    """Summary over the records that have a prediction, and the miss count."""
    samples: list[SampleMetrics] = []
    missing = 0
    for batch in loader:
        keys = [str(k) for k in batch["key"]]
        pred, rows = table.batch(keys, smpl)
        missing += len(keys) - len(rows)
        if rows:
            samples.extend(sample_metrics(pred, _select(batch, rows), smpl))
    return summarize(samples), missing


def main(argv: list[str] | None = None) -> int:
    """CLI."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--pred", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="score only the first n records per source (0 = all)",
    )
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    smpl = SmplModel(Path(cfg.body_models))
    loaders: dict[str, DataLoader[Any]] = {}
    for s in cfg.data.val:
        ds: Any = build_dataset(s, cfg, train=False)
        if args.max_samples:  # the trainer's evaluation subset: the first n
            n = min(len(ds), args.max_samples)
            ds = torch.utils.data.Subset(ds, list(range(n)))
        loaders[s.name] = DataLoader(
            ds, batch_size=64, num_workers=args.num_workers
        )
    results: dict[str, dict[str, Any]] = {}
    for path in args.pred:
        table = PredictionTable(path)
        if table.split not in loaders:
            sys.stdout.write(f"skip {path}: no val source {table.split}\n")
            continue
        summary, missing = score(table, loaders[table.split], smpl)
        results.setdefault(table.name, {})[table.split] = {
            **asdict(summary),
            "missing": missing,
            "meta": table.meta,
        }
        sys.stdout.write(
            f"{table.name:>18} {table.split:<14} n={summary.n:<5} "
            f"MPJPE {summary.mpjpe:6.1f}  PA {summary.pa_mpjpe:6.1f}  "
            f"transl {summary.transl_err_m:6.3f} m  mAP {summary.map:.3f}  "
            f"IoU {summary.mean_iou:.3f}  missing {missing}\n"
        )
        sys.stdout.flush()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
