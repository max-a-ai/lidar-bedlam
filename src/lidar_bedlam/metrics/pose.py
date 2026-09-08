"""Pose and shape metrics (numpy, millimetres).

Inputs are per-sample arrays; the caller aggregates over the dataset.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
MM = 1000.0


def mpjpe(
    pred: NDArray[np.floating],
    gt: NDArray[np.floating],
    valid: NDArray[np.bool_] | None = None,
) -> float:
    """Mean per-joint position error in mm over the valid joints."""
    p, g = np.asarray(pred, np.float64), np.asarray(gt, np.float64)
    v = np.ones(len(g), bool) if valid is None else valid
    if not v.any():
        return float("nan")
    return float(np.linalg.norm(p[v] - g[v], axis=-1).mean() * MM)


def root_relative(joints: NDArray[np.floating], root_index: int) -> FloatArray:
    """Subtract the root joint."""
    j = np.asarray(joints, np.float64)
    return np.asarray(j - j[root_index : root_index + 1], dtype=np.float64)


def procrustes_align(
    pred: NDArray[np.floating],
    gt: NDArray[np.floating],
    valid: NDArray[np.bool_] | None = None,
) -> FloatArray:
    """Similarity transform (scale, rotation, translation) of pred onto gt.

    The transform is fitted on the valid joints and applied to all joints.
    """
    p, g = np.asarray(pred, np.float64), np.asarray(gt, np.float64)
    v = np.ones(len(g), bool) if valid is None else valid
    mu_p, mu_g = p[v].mean(0), g[v].mean(0)
    p0, g0 = p[v] - mu_p, g[v] - mu_g
    var_p = (p0**2).sum()
    u, s, vt = np.linalg.svd(p0.T @ g0)
    d = np.sign(np.linalg.det(u @ vt))
    dm = np.diag([1.0, 1.0, d])
    rot = u @ dm @ vt
    scale = (s @ np.diag([1.0, 1.0, d])).sum() / var_p
    return np.asarray(scale * (p - mu_p) @ rot + mu_g, dtype=np.float64)


def pa_mpjpe(
    pred: NDArray[np.floating],
    gt: NDArray[np.floating],
    valid: NDArray[np.bool_] | None = None,
) -> float:
    """MPJPE after Procrustes alignment (mm)."""
    return mpjpe(procrustes_align(pred, gt, valid), gt, valid)


def pve(
    pred_verts: NDArray[np.floating], gt_verts: NDArray[np.floating]
) -> float:
    """Per-vertex error in mm (no alignment)."""
    return mpjpe(pred_verts, gt_verts)
