"""Point-cloud augmentations for robustness to occlusion and sparsity.

All functions take camera-frame person points (N, 3) and return a subset.
They never invent points; the model must learn that a partial cloud still
belongs to a full body.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class OcclusionConfig:
    """Probabilities and ranges of the random occlusions."""

    p_halfspace: float = 0.3  # cut everything on one side of a random plane
    p_box: float = 0.3  # remove a random axis-aligned box
    p_height_cut: float = 0.3  # remove the lower or upper part of the body
    height_cut_frac: tuple[float, float] = (0.2, 0.6)
    box_frac: tuple[float, float] = (
        0.2,
        0.5,
    )  # box side as fraction of extent
    p_subsample: float = 0.5
    subsample_keep: tuple[float, float] = (0.2, 0.8)
    min_points: int = 16


def halfspace_cut(
    points: FloatArray, rng: np.random.Generator
) -> NDArray[np.bool_]:
    """Keep the points on one side of a plane through a random point."""
    normal = rng.normal(size=3)
    normal[1] *= 0.3  # mostly vertical planes (occluders stand on the ground)
    normal /= np.linalg.norm(normal)
    anchor = points[rng.integers(len(points))]
    side = np.asarray((points - anchor) @ normal, dtype=np.float64)
    keep = side >= 0 if rng.random() < 0.5 else side <= 0
    return np.asarray(keep, dtype=bool)


def box_cut(
    points: FloatArray, rng: np.random.Generator, frac: tuple[float, float]
) -> NDArray[np.bool_]:
    """Remove the points inside a random axis-aligned box."""
    lo, hi = points.min(0), points.max(0)
    extent = hi - lo
    size = extent * rng.uniform(frac[0], frac[1], size=3)
    centre = rng.uniform(lo, hi)
    inside = np.all(np.abs(points - centre) <= size / 2.0, axis=1)
    return ~inside


def height_cut(
    points: FloatArray, rng: np.random.Generator, frac: tuple[float, float]
) -> NDArray[np.bool_]:
    """Remove the lowest or highest fraction of the body (up = -y)."""
    y = points[:, 1]
    lo, hi = y.min(), y.max()
    f = rng.uniform(frac[0], frac[1])
    if rng.random() < 0.7:  # legs occluded is the common case
        return y <= hi - f * (hi - lo)
    return y >= lo + f * (hi - lo)


def subsample(
    n: int, rng: np.random.Generator, keep: tuple[float, float]
) -> NDArray[np.bool_]:
    """Keep a random fraction of the points."""
    k = max(1, int(round(n * rng.uniform(keep[0], keep[1]))))
    mask = np.zeros(n, dtype=bool)
    mask[rng.choice(n, k, replace=False)] = True
    return mask


def occlude(
    points: NDArray[np.floating],
    cfg: OcclusionConfig,
    rng: np.random.Generator,
) -> NDArray[np.bool_]:
    """Random combination of occlusions; returns the keep mask (N,).

    Falls back to keeping everything if an augmentation would leave fewer
    than ``cfg.min_points`` points.
    """
    p = np.asarray(points, dtype=np.float64)
    keep = np.ones(len(p), dtype=bool)
    if len(p) < cfg.min_points:
        return keep
    ops: list[tuple[float, Callable[[FloatArray], NDArray[np.bool_]]]] = [
        (cfg.p_halfspace, lambda q: halfspace_cut(q, rng)),
        (cfg.p_box, lambda q: box_cut(q, rng, cfg.box_frac)),
        (cfg.p_height_cut, lambda q: height_cut(q, rng, cfg.height_cut_frac)),
    ]
    for prob, op in ops:
        if rng.random() >= prob:
            continue
        idx = np.flatnonzero(keep)
        sub = op(p[idx])
        if sub.sum() >= cfg.min_points:
            keep[idx[~sub]] = False
    if rng.random() < cfg.p_subsample:
        idx = np.flatnonzero(keep)
        sub = subsample(len(idx), rng, cfg.subsample_keep)
        if sub.sum() >= cfg.min_points:
            keep[idx[~sub]] = False
    return keep
