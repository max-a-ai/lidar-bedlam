"""Evaluation protocol: joint conventions and per-sample metrics.

Predictions are SMPL (24 joints + mesh). Ground truth is either SMPL
(``smpl24``, SLOPER4D and synthetic data) or Waymo's 15 keypoints
(``waymo15``). For Waymo, COCO-17 joints are regressed from the predicted
mesh and compared on the 13 shared body joints; forehead and head centre
have no COCO counterpart.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from numpy.typing import NDArray

from lidar_bedlam.body.smpl import SmplModel, SmplParams
from lidar_bedlam.data.schema import WAYMO15_TO_COCO17
from lidar_bedlam.geometry.boxes import iou3d
from lidar_bedlam.geometry.camera import CAMERA_UP_AXIS
from lidar_bedlam.metrics.pose import mpjpe, pa_mpjpe, pve

FloatArray = NDArray[np.float64]

PELVIS = 0


@dataclass
class SampleMetrics:
    """Metrics of one evaluated sample (mm unless stated)."""

    key: str
    dataset: str
    mpjpe: float
    pa_mpjpe: float
    abs_mpjpe: float  # joints as predicted, no centring: pose + placement
    transl_err_m: float
    gt_depth_m: float
    iou: float
    box_conf: float
    pve: float = float("nan")
    n_joints: int = 0


@dataclass
class MetricSummary:
    """Aggregates over a set of samples."""

    n: int
    mpjpe: float
    pa_mpjpe: float
    abs_mpjpe: float
    pve: float
    transl_err_m: float
    ap: dict[float, float]
    map: float
    mean_iou: float
    by_distance: dict[str, float] = field(default_factory=dict)


def _to_np(t: torch.Tensor) -> FloatArray:
    return np.asarray(t.detach().cpu().double().numpy(), dtype=np.float64)


def sample_metrics(
    pred: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor | str],
    smpl: SmplModel,
) -> list[SampleMetrics]:
    """Per-sample metrics for a batch of predictions."""
    verts = _to_np(pred["vertices"])
    joints = _to_np(pred["joints3d"])
    transl = _to_np(pred["transl"])
    boxes = _to_np(pred["box3d"])
    # Waymo rows are scored on the padded-convention box when the model
    # predicts one (the labels are padded, the mesh extent is not)
    boxes_padded = (
        _to_np(pred["box3d_padded"]) if "box3d_padded" in pred else None
    )
    conf = (
        _to_np(pred["box_conf"]) if "box_conf" in pred else np.ones(len(verts))
    )
    gt_joints = _to_np(_t(batch["joints3d"]))
    gt_valid = _t(batch["joints3d_valid"]).cpu().numpy().astype(bool)
    gt_box = _to_np(_t(batch["box3d"]))
    gt_transl = _to_np(_t(batch["transl"]))
    has_smpl = _t(batch["has_smpl"]).cpu().numpy().astype(bool)
    conventions = batch["joint_convention"]
    keys, datasets = batch["key"], batch["dataset"]
    out = []
    for i in range(len(verts)):
        conv = str(conventions[i])
        box = boxes[i]
        if conv == "waymo15":
            if boxes_padded is not None:
                box = boxes_padded[i]
            coco = smpl.coco_joints(verts[i])
            sel = WAYMO15_TO_COCO17[:15] >= 0
            p = coco[WAYMO15_TO_COCO17[:15][sel]]
            g = gt_joints[i][:15][sel]
            valid = gt_valid[i][:15][sel]
            # placement error on Waymo: pelvis approximated by the hip centre;
            # unlabelled joints carry garbage coordinates, so records without
            # both hips get no placement error (nan, skipped by the mean)
            p_root = (coco[11] + coco[12]) / 2.0
            g_root = (gt_joints[i][4] + gt_joints[i][10]) / 2.0
            if not (gt_valid[i][4] and gt_valid[i][10]):
                g_root = np.full(3, np.nan)
        else:
            p, g, valid = joints[i], gt_joints[i][:24], gt_valid[i][:24]
            p_root, g_root = transl[i], gt_transl[i]
        rel_p = p - p[valid].mean(0) if valid.any() else p
        rel_g = g - g[valid].mean(0) if valid.any() else g
        m = SampleMetrics(
            key=str(keys[i]),
            dataset=str(datasets[i]),
            mpjpe=mpjpe(rel_p, rel_g, valid),
            pa_mpjpe=pa_mpjpe(p, g, valid),
            abs_mpjpe=mpjpe(p, g, valid),
            transl_err_m=float(np.linalg.norm(p_root - g_root)),
            gt_depth_m=float(g_root[2]),
            iou=iou3d(box, gt_box[i], CAMERA_UP_AXIS),
            box_conf=float(conf[i]),
            n_joints=int(valid.sum()),
        )
        if has_smpl[i]:
            # per-vertex error, root-relative like MPJPE: the GT mesh comes
            # from the record's SMPL parameters (neutral model)
            if "gt_vertices" in batch:
                gt_verts = _to_np(_t(batch["gt_vertices"]))[i]
            else:
                gt_verts, _ = smpl.forward(
                    SmplParams(
                        _to_np(_t(batch["global_orient"]))[i],
                        _to_np(_t(batch["body_pose"]))[i],
                        _to_np(_t(batch["betas"]))[i],
                        gt_transl[i],
                    )
                )
            m.pve = pve(verts[i] - joints[i][0], gt_verts - gt_joints[i][0])
        out.append(m)
    return out


def _t(x: torch.Tensor | str) -> torch.Tensor:
    assert isinstance(x, torch.Tensor)
    return x


def summarize(
    samples: list[SampleMetrics],
    thresholds: tuple[float, ...] = (0.25, 0.5, 0.7),
    bins: tuple[float, ...] = (0.0, 10.0, 20.0, 30.0, np.inf),
) -> MetricSummary:
    """Aggregate per-sample metrics into the paper's numbers."""
    from lidar_bedlam.metrics.detection import (
        average_precision,
        error_by_distance,
    )

    if not samples:
        return MetricSummary(
            0, *(float("nan"),) * 5, {}, float("nan"), float("nan")
        )
    ious = np.array([s.iou for s in samples])
    confs = np.array([s.box_conf for s in samples])
    ap = {t: average_precision(ious >= t, confs) for t in thresholds}
    err = np.array([s.transl_err_m for s in samples])
    depth = np.array([s.gt_depth_m for s in samples])
    if not np.isfinite(err).any():
        err = np.array([float("nan")])
        depth = np.array([float("nan")])
    pves = np.array([s.pve for s in samples])
    return MetricSummary(
        n=len(samples),
        mpjpe=float(np.nanmean([s.mpjpe for s in samples])),
        pa_mpjpe=float(np.nanmean([s.pa_mpjpe for s in samples])),
        abs_mpjpe=float(np.nanmean([s.abs_mpjpe for s in samples])),
        pve=float(np.nanmean(pves))
        if np.isfinite(pves).any()
        else float("nan"),
        transl_err_m=float(np.nanmean(err)),  # nan: GT root unlabelled
        ap=ap,
        map=float(np.mean(list(ap.values()))),
        mean_iou=float(ious.mean()),
        by_distance=error_by_distance(err, depth, bins),
    )
