"""Virtual distance: move the camera back along its optical axis.

Scaling a scene about the camera centre changes nothing a sensor can see
(angles are preserved), so "far" persons must be made by translating the
camera. Moving it back by ``d`` metres turns a planar depth ``z`` into
``z + d`` and moves pixel ``(u, v)`` towards the principal point by the
factor ``z / (z + d)``. The depth map and the person masks are re-rendered
by forward splatting with a z-buffer; LiDAR simulated on the result has
the density of the larger distance, and labels shift by ``(0, 0, d)``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from lidar_bedlam.geometry.camera import PinholeCamera

FloatArray = NDArray[np.float64]
SKY_M = 1e6


def move_camera_back(
    depth_m: NDArray[np.floating],
    masks: dict[str, NDArray[np.bool_]],
    camera: PinholeCamera,
    d: float,
    sky_threshold_m: float = 1e4,
) -> tuple[FloatArray, dict[str, NDArray[np.bool_]]]:
    """Re-render depth (planar z, metres) and masks for a camera moved back.

    Returns the new depth map (unseen regions set to ``SKY_M``) and the
    re-rendered masks. Pixels that map onto the same target keep the
    nearest surface.
    """
    depth = np.asarray(depth_m, dtype=np.float64)
    h, w = depth.shape
    valid = depth < sky_threshold_m
    v, u = np.nonzero(valid)
    z = depth[v, u]
    z_new = z + d
    scale = z / z_new
    u_new = np.rint(camera.cx + (u - camera.cx) * scale).astype(int)
    v_new = np.rint(camera.cy + (v - camera.cy) * scale).astype(int)
    inside = (u_new >= 0) & (u_new < w) & (v_new >= 0) & (v_new < h)
    flat = v_new[inside] * w + u_new[inside]
    zbuf = np.full(h * w, SKY_M)
    np.minimum.at(zbuf, flat, z_new[inside])
    out_depth = zbuf.reshape(h, w)
    out_masks: dict[str, NDArray[np.bool_]] = {}
    src_flat = v * w + u
    for pid, m in masks.items():
        sel = m.reshape(-1)[src_flat] & inside
        visible = np.isclose(zbuf[flat[sel[inside]]], z_new[sel], atol=1e-6)
        target = np.zeros(h * w, dtype=bool)
        target[flat[sel[inside]][visible]] = True
        out_masks[pid] = target.reshape(h, w)
    return out_depth, out_masks


def shrink_factor(z: float, d: float) -> float:
    """Image scale of an object at depth ``z`` after moving back by ``d``."""
    return float(z / (z + d))
