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


# --- deterministic (parametrised) variants for interactive inspection -------


def height_cut_fraction(
    points: NDArray[np.floating], frac: float, from_bottom: bool = True
) -> NDArray[np.bool_]:
    """Remove ``frac`` of the body height from the bottom (or top)."""
    p = np.asarray(points, dtype=np.float64)
    y = p[:, 1]  # up is -y
    lo, hi = y.min(), y.max()
    if from_bottom:
        return np.asarray(y <= hi - frac * (hi - lo), dtype=bool)
    return np.asarray(y >= lo + frac * (hi - lo), dtype=bool)


def axis_cut_fraction(
    points: NDArray[np.floating], axis: int, frac: float, from_min: bool
) -> NDArray[np.bool_]:
    """Remove ``frac`` of the extent along ``axis`` from its min (or max)."""
    p = np.asarray(points, dtype=np.float64)
    v = p[:, axis]
    lo, hi = v.min(), v.max()
    if from_min:
        return np.asarray(v >= lo + frac * (hi - lo), dtype=bool)
    return np.asarray(v <= hi - frac * (hi - lo), dtype=bool)


# --- sensor-level effects ----------------------------------------------------


def jitter(
    points: NDArray[np.floating], std_m: float, rng: np.random.Generator
) -> NDArray[np.float32]:
    """Add isotropic Gaussian noise to every point."""
    p = np.asarray(points, dtype=np.float64)
    if std_m <= 0:
        return p.astype(np.float32)
    return (p + rng.normal(0.0, std_m, size=p.shape)).astype(np.float32)


def range_jitter(
    points: NDArray[np.floating],
    origin: NDArray[np.floating],
    std_m: float,
    rng: np.random.Generator,
) -> NDArray[np.float32]:
    """Noise along the ray from ``origin`` (how a range sensor errs)."""
    p = np.asarray(points, dtype=np.float64)
    d = p - np.asarray(origin, dtype=np.float64)
    r = np.linalg.norm(d, axis=1, keepdims=True).clip(1e-6)
    noise = rng.normal(0.0, std_m, size=(len(p), 1))
    return np.asarray(p + d / r * noise, dtype=np.float32)


def add_outliers(
    points: NDArray[np.floating],
    fraction: float,
    spread_m: float,
    rng: np.random.Generator,
) -> NDArray[np.float32]:
    """Append spurious returns around the cloud (dust, edge effects).

    ``fraction`` of the point count is added, uniformly in the cloud's
    bounding box dilated by ``spread_m``.
    """
    p = np.asarray(points, dtype=np.float64)
    n = int(round(len(p) * fraction))
    if n == 0:
        return p.astype(np.float32)
    lo, hi = p.min(0) - spread_m, p.max(0) + spread_m
    extra = rng.uniform(lo, hi, size=(n, 3))
    return np.concatenate([p, extra]).astype(np.float32)


def miscalibrated_pose(
    camera_from_sensor: NDArray[np.floating],
    rot_std_deg: float,
    trans_std_m: float,
    rng: np.random.Generator,
) -> FloatArray:
    """Perturb a sensor pose (calibration error between LiDAR and camera)."""
    from lidar_bedlam.geometry.rotations import euler_to_matrix

    pose = np.asarray(camera_from_sensor, dtype=np.float64).copy()
    d_rot = euler_to_matrix("xyz", rng.normal(0.0, rot_std_deg, size=3))
    pose[:3, :3] = d_rot @ pose[:3, :3]
    pose[:3, 3] += rng.normal(0.0, trans_std_m, size=3)
    return pose


@dataclass(frozen=True)
class SensorCover:
    """A physical obstruction in front of the LiDAR, in scan coordinates.

    Fractions of the camera's azimuth window covered from the left / right
    and fractions of the channels covered from the top / bottom.
    """

    left: float = 0.0
    right: float = 0.0
    top: float = 0.0
    bottom: float = 0.0


def cover_mask(
    channel: NDArray[np.integer],
    azimuth_deg: NDArray[np.floating],
    n_channels: int,
    az_min_deg: float,
    az_max_deg: float,
    cover: SensorCover,
) -> NDArray[np.bool_]:
    """Keep mask for returns not hidden by ``cover``."""
    az = np.asarray(azimuth_deg, dtype=np.float64)
    ch = np.asarray(channel)
    span = az_max_deg - az_min_deg
    keep = np.ones(len(az), dtype=bool)
    keep &= az >= az_min_deg + cover.left * span
    keep &= az <= az_max_deg - cover.right * span
    keep &= ch >= cover.top * n_channels
    keep &= ch < n_channels - cover.bottom * n_channels
    return keep


def drop_channels(
    channel: NDArray[np.integer],
    n_channels: int,
    fraction: float,
    rng: np.random.Generator,
) -> NDArray[np.bool_]:
    """Keep mask that removes a random ``fraction`` of the channels."""
    n_drop = int(round(n_channels * fraction))
    dropped = rng.choice(n_channels, n_drop, replace=False)
    return np.asarray(~np.isin(np.asarray(channel), dropped), dtype=bool)


@dataclass(frozen=True)
class PointAugmentConfig:
    """Training-time point-cloud augmentation, applied in this order."""

    occlusion: OcclusionConfig = OcclusionConfig()
    jitter_std_m: tuple[float, float] = (0.0, 0.03)
    p_outliers: float = 0.3
    outlier_fraction: tuple[float, float] = (0.02, 0.1)
    outlier_spread_m: float = 0.3
    p_channel_dropout: float = 0.2
    channel_dropout: tuple[float, float] = (0.1, 0.5)


def augment_points(
    points: NDArray[np.floating],
    channel: NDArray[np.integer] | None,
    n_channels: int,
    cfg: PointAugmentConfig,
    rng: np.random.Generator,
) -> NDArray[np.float32]:
    """Occlusion, channel dropout, jitter and outliers for one cloud."""
    p = np.asarray(points, dtype=np.float64)
    keep = occlude(p, cfg.occlusion, rng)
    if channel is not None and rng.random() < cfg.p_channel_dropout:
        frac = rng.uniform(*cfg.channel_dropout)
        drop_keep = drop_channels(channel, n_channels, frac, rng)
        if (keep & drop_keep).sum() >= cfg.occlusion.min_points:
            keep &= drop_keep
    p = p[keep]
    p = jitter(p, rng.uniform(*cfg.jitter_std_m), rng)
    if rng.random() < cfg.p_outliers:
        p = add_outliers(
            p, rng.uniform(*cfg.outlier_fraction), cfg.outlier_spread_m, rng
        )
    return np.asarray(p, dtype=np.float32)
