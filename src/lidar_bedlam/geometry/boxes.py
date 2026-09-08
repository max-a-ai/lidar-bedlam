"""3D bounding boxes for the human "detection" evaluation.

A box is a 7-vector ``[cx, cy, cz, dx, dy, dz, yaw]``: centre, full extents
along the box axes, and a heading angle (rad) around the ``up`` axis of the
frame it lives in. In the OpenCV camera frame the up axis is -y, so boxes
rotate in the x-z plane; ``up_axis`` selects that behaviour.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def aabb_from_points(points: NDArray[np.floating]) -> FloatArray:
    """Axis-aligned box (yaw = 0) enclosing points (N, 3)."""
    p = np.asarray(points, dtype=np.float64)
    lo, hi = p.min(axis=0), p.max(axis=0)
    return np.concatenate([(lo + hi) / 2.0, hi - lo, [0.0]])


def _plane_axes(up_axis: int) -> tuple[int, int]:
    a, b = (i for i in range(3) if i != up_axis)
    return a, b


def oriented_box_from_points(
    points: NDArray[np.floating], yaw: float, up_axis: int
) -> FloatArray:
    """Box with the given heading enclosing points (N, 3) tightly."""
    p = np.asarray(points, dtype=np.float64)
    a, b = _plane_axes(up_axis)
    c, s = np.cos(yaw), np.sin(yaw)
    # rotate the in-plane coordinates into the box frame
    pa = c * p[:, a] + s * p[:, b]
    pb = -s * p[:, a] + c * p[:, b]
    lo_a, hi_a = pa.min(), pa.max()
    lo_b, hi_b = pb.min(), pb.max()
    lo_u, hi_u = p[:, up_axis].min(), p[:, up_axis].max()
    ca, cb = (lo_a + hi_a) / 2.0, (lo_b + hi_b) / 2.0
    centre = np.zeros(3)
    centre[a] = c * ca - s * cb
    centre[b] = s * ca + c * cb
    centre[up_axis] = (lo_u + hi_u) / 2.0
    size = np.zeros(3)
    size[a], size[b], size[up_axis] = hi_a - lo_a, hi_b - lo_b, hi_u - lo_u
    return np.concatenate([centre, size, [yaw]])


def heading_yaw(forward: NDArray[np.floating], up_axis: int) -> float:
    """Yaw (rad) of a heading vector in the plane orthogonal to ``up``.

    Matches the box convention: yaw 0 points along the first in-plane axis.
    """
    a, b = _plane_axes(up_axis)
    f = np.asarray(forward, dtype=np.float64)
    return float(np.arctan2(f[b], f[a]))


def box_corners_bev(box: NDArray[np.floating], up_axis: int) -> FloatArray:
    """The 4 in-plane corners (4, 2) of a box, counter-clockwise."""
    a, b = _plane_axes(up_axis)
    cx, cy = box[a], box[b]
    ha, hb = box[3 + a] / 2.0, box[3 + b] / 2.0
    c, s = np.cos(box[6]), np.sin(box[6])
    local = np.array([[-ha, -hb], [ha, -hb], [ha, hb], [-ha, hb]])
    rot = np.array([[c, -s], [s, c]])
    return np.asarray(local @ rot.T + np.array([cx, cy]), dtype=np.float64)


def _cross2(u: FloatArray, v: FloatArray) -> float:
    return float(u[0] * v[1] - u[1] * v[0])


def _inside(p: FloatArray, ea: FloatArray, eb: FloatArray) -> bool:
    return _cross2(eb - ea, p - ea) >= 0


def _intersect(
    p: FloatArray, q: FloatArray, ea: FloatArray, eb: FloatArray
) -> FloatArray:
    d1, d2 = q - p, eb - ea
    t = _cross2(ea - p, d2) / _cross2(d1, d2)
    return np.asarray(p + t * d1, dtype=np.float64)


def _clip_edge(poly: FloatArray, ea: FloatArray, eb: FloatArray) -> FloatArray:
    out: list[FloatArray] = []
    for j in range(len(poly)):
        cur, prev = poly[j], poly[j - 1]
        if _inside(cur, ea, eb):
            if not _inside(prev, ea, eb):
                out.append(_intersect(prev, cur, ea, eb))
            out.append(cur)
        elif _inside(prev, ea, eb):
            out.append(_intersect(prev, cur, ea, eb))
    return np.asarray(out, dtype=np.float64).reshape(-1, 2)


def _clip_polygon(subject: FloatArray, clip: FloatArray) -> FloatArray:
    """Sutherland-Hodgman clipping of convex polygons (both CCW)."""
    output = subject
    for i in range(len(clip)):
        if len(output) == 0:
            break
        output = _clip_edge(output, clip[i], clip[(i + 1) % len(clip)])
    return output


def _signed_area(poly: FloatArray) -> float:
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return float(0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _ensure_ccw(poly: FloatArray) -> FloatArray:
    return poly[::-1] if _signed_area(poly) < 0 else poly


def iou3d(
    box_a: NDArray[np.floating], box_b: NDArray[np.floating], up_axis: int
) -> float:
    """Exact 3D IoU of two oriented boxes sharing the same up axis."""
    pa = _ensure_ccw(box_corners_bev(box_a, up_axis))
    pb = _ensure_ccw(box_corners_bev(box_b, up_axis))
    inter_area = abs(_signed_area(_clip_polygon(pa, pb)))
    ha, hb = box_a[3 + up_axis], box_b[3 + up_axis]
    lo = max(box_a[up_axis] - ha / 2.0, box_b[up_axis] - hb / 2.0)
    hi = min(box_a[up_axis] + ha / 2.0, box_b[up_axis] + hb / 2.0)
    inter = inter_area * max(hi - lo, 0.0)
    vol_a = float(np.prod(box_a[3:6]))
    vol_b = float(np.prod(box_b[3:6]))
    union = vol_a + vol_b - inter
    return float(inter / union) if union > 0 else 0.0


def iou3d_matrix(
    boxes_a: NDArray[np.floating], boxes_b: NDArray[np.floating], up_axis: int
) -> FloatArray:
    """Pairwise IoU (len(a), len(b))."""
    out = np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float64)
    for i, a in enumerate(boxes_a):
        for j, b in enumerate(boxes_b):
            out[i, j] = iou3d(a, b, up_axis)
    return out
