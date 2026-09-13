"""Depth and id images rendered from SMPL meshes (no scene).

Real datasets with SMPL labels but no depth (3DPW) get a simulated LiDAR
by rasterising the labelled mesh into a planar depth map that the ray
marcher in :mod:`lidar_bedlam.lidar.simulate` consumes like BEDLAM depth.
Because a real LiDAR hits the clothing, not the skin, every vertex is
pushed outwards along its normal by a few centimetres first.

Pixels without a mesh get :data:`SKY_M` (no return), like BEDLAM's sky.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from lidar_bedlam.geometry.camera import PinholeCamera

FloatArray = NDArray[np.float64]
SKY_M = 1e6


def vertex_normals(verts: FloatArray, faces: NDArray[np.int64]) -> FloatArray:
    """Area-weighted vertex normals (N, 3), unit length."""
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    normals = np.zeros_like(verts)
    for k in range(3):
        np.add.at(normals, faces[:, k], fn)
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.maximum(length, 1e-12)


def offset_mesh(
    verts: FloatArray, faces: NDArray[np.int64], offset_m: float
) -> FloatArray:
    """Vertices moved outwards along their normals (the clothing layer)."""
    return verts + offset_m * vertex_normals(verts, faces)


def render_depth(
    meshes: list[tuple[FloatArray, NDArray[np.int64]]],
    camera: PinholeCamera,
) -> tuple[NDArray[np.float32], NDArray[np.int16]]:
    """Z-buffer rasterisation of camera-frame meshes.

    Returns the planar depth (H, W) in metres with ``SKY_M`` where nothing
    was hit, and the id image (H, W) holding the mesh index or -1.
    """
    h, w = camera.height, camera.width
    depth = np.full((h, w), SKY_M, dtype=np.float32)
    ids = np.full((h, w), -1, dtype=np.int16)
    for mesh_id, (verts, faces) in enumerate(meshes):
        z = verts[:, 2]
        uv = camera.project(verts)
        _rasterise(depth, ids, uv, z, faces, mesh_id)
    return depth, ids


def _rasterise(
    depth: NDArray[np.float32],
    ids: NDArray[np.int16],
    uv: FloatArray,
    z: FloatArray,
    faces: NDArray[np.int64],
    mesh_id: int,
) -> None:
    h, w = depth.shape
    tri = uv[faces]  # (F, 3, 2)
    tz = z[faces]  # (F, 3)
    keep = (tz > 1e-3).all(1)
    x0 = np.clip(np.floor(tri[:, :, 0].min(1)), 0, w - 1).astype(int)
    x1 = np.clip(np.ceil(tri[:, :, 0].max(1)), 0, w - 1).astype(int)
    y0 = np.clip(np.floor(tri[:, :, 1].min(1)), 0, h - 1).astype(int)
    y1 = np.clip(np.ceil(tri[:, :, 1].max(1)), 0, h - 1).astype(int)
    keep &= (x1 >= x0) & (y1 >= y0)
    for f in np.nonzero(keep)[0]:
        (ax, ay), (bx, by), (cx, cy) = tri[f]
        det = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay)
        if abs(det) < 1e-9:
            continue
        xs = np.arange(x0[f], x1[f] + 1)
        ys = np.arange(y0[f], y1[f] + 1)
        px, py = np.meshgrid(xs + 0.5, ys + 0.5)
        l1 = ((bx - px) * (cy - py) - (cx - px) * (by - py)) / det
        l2 = ((cx - px) * (ay - py) - (ax - px) * (cy - py)) / det
        l3 = 1.0 - l1 - l2
        inside = (l1 >= 0) & (l2 >= 0) & (l3 >= 0)
        if not inside.any():
            continue
        # perspective-correct depth: interpolate 1/z
        inv_z = l1 / tz[f, 0] + l2 / tz[f, 1] + l3 / tz[f, 2]
        zf = 1.0 / np.maximum(inv_z, 1e-9)
        win_d = depth[y0[f] : y1[f] + 1, x0[f] : x1[f] + 1]
        win_i = ids[y0[f] : y1[f] + 1, x0[f] : x1[f] + 1]
        closer = inside & (zf < win_d)
        win_d[closer] = zf[closer]
        win_i[closer] = mesh_id
