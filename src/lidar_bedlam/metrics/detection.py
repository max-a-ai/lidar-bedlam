"""3D placement metrics: translation error and box AP / mAP.

Each evaluated sample is one person crop with exactly one ground-truth
3D box and one predicted box (with an optional confidence score). AP at an
IoU threshold is the area under the precision-recall curve obtained by
ranking predictions by score; with uniform scores it reduces to the
fraction of boxes above the threshold.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from lidar_bedlam.geometry.boxes import iou3d
from lidar_bedlam.geometry.camera import CAMERA_UP_AXIS

FloatArray = NDArray[np.float64]
DEFAULT_IOU_THRESHOLDS = (0.25, 0.5, 0.7)
DEFAULT_DISTANCE_BINS_M = (0.0, 10.0, 20.0, 30.0, np.inf)


def translation_error(
    pred_t: NDArray[np.floating], gt_t: NDArray[np.floating]
) -> FloatArray:
    """Euclidean 3D placement error per sample, in metres (N,)."""
    d = np.asarray(pred_t, np.float64) - np.asarray(gt_t, np.float64)
    return np.asarray(np.linalg.norm(d, axis=-1), dtype=np.float64)


def error_by_distance(
    errors: NDArray[np.floating],
    gt_depth: NDArray[np.floating],
    bins: tuple[float, ...] = DEFAULT_DISTANCE_BINS_M,
) -> dict[str, float]:
    """Mean error per ground-truth distance bin (keys like ``10-20m``)."""
    out: dict[str, float] = {}
    e, z = np.asarray(errors, np.float64), np.asarray(gt_depth, np.float64)
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        sel = (z >= lo) & (z < hi)
        label = f"{lo:g}-{hi:g}m" if np.isfinite(hi) else f">{lo:g}m"
        out[label] = float(e[sel].mean()) if sel.any() else float("nan")
    return out


def average_precision(
    matched: NDArray[np.bool_], scores: NDArray[np.floating]
) -> float:
    """AP (all-point interpolation) for one prediction per ground truth."""
    order = np.argsort(-np.asarray(scores, np.float64), kind="stable")
    tp = np.asarray(matched, bool)[order].astype(np.float64)
    n_gt = len(tp)
    if n_gt == 0:
        return float("nan")
    recall = np.cumsum(tp) / n_gt
    precision = np.cumsum(tp) / np.arange(1, n_gt + 1)
    # precision envelope
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    r_prev = np.concatenate([[0.0], recall[:-1]])
    return float(((recall - r_prev) * precision).sum())


@dataclass
class DetectionResult:
    """AP per IoU threshold, their mean, and IoU statistics."""

    ap: dict[float, float]
    map: float
    mean_iou: float
    ious: FloatArray


def box_detection_metrics(
    pred_boxes: NDArray[np.floating],
    gt_boxes: NDArray[np.floating],
    scores: NDArray[np.floating] | None = None,
    thresholds: tuple[float, ...] = DEFAULT_IOU_THRESHOLDS,
    up_axis: int = CAMERA_UP_AXIS,
) -> DetectionResult:
    """IoU-based AP at each threshold and the mAP over thresholds."""
    p = np.asarray(pred_boxes, np.float64).reshape(-1, 7)
    g = np.asarray(gt_boxes, np.float64).reshape(-1, 7)
    ious = np.array([iou3d(a, b, up_axis) for a, b in zip(p, g, strict=True)])
    s = np.ones(len(p)) if scores is None else np.asarray(scores, np.float64)
    ap = {t: average_precision(ious >= t, s) for t in thresholds}
    return DetectionResult(
        ap=ap,
        map=float(np.mean(list(ap.values()))) if ap else float("nan"),
        mean_iou=float(ious.mean()) if len(ious) else float("nan"),
        ious=ious,
    )
