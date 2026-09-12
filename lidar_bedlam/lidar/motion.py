"""Ego motion during a scan: vehicle speed and LiDAR rolling shutter.

A spinning LiDAR at ``spin_hz`` sweeps the azimuth columns one after the
other, so a scan taken from a moving platform is distorted: every column
is measured from a slightly different sensor position. Uncompensated
datasets (nuScenes, and Waymo range images before per-column pose
correction) hand such distorted clouds to the model, so training data
should show the same effect.

The distortion is applied to a static-scene scan: the true point of a
column captured at time ``t`` is expressed in the sensor frame of that
time and then naively assembled as if the sensor had not moved. For pure
translation ``v`` that is ``p_reported = p_true - v * (t - t_ref)``; a yaw
rate adds a rotation about the up axis (-y in the camera frame).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from lidar_bedlam.lidar.simulate import AzimuthWindow, LidarScan, LidarSpec

FloatArray = NDArray[np.float64]
KMH_TO_MPS = 1.0 / 3.6
SPEED_STEP_KMH = 10.0


@dataclass(frozen=True)
class SpeedSetting:
    """Ego speed as expected value and standard deviation in km/h.

    Both are multiples of 10 km/h (the available settings); samples are
    clipped to ``[0, max_kmh]``.
    """

    mean_kmh: float = 0.0
    std_kmh: float = 0.0
    max_kmh: float = 130.0

    def __post_init__(self) -> None:
        for v in (self.mean_kmh, self.std_kmh):
            if (
                v < 0
                or abs(v / SPEED_STEP_KMH - round(v / SPEED_STEP_KMH)) > 1e-9
            ):
                msg = f"speed settings are multiples of {SPEED_STEP_KMH} km/h"
                raise ValueError(msg)

    def sample_mps(self, rng: np.random.Generator) -> float:
        """One speed in m/s."""
        if self.std_kmh == 0:
            return self.mean_kmh * KMH_TO_MPS
        v = rng.normal(self.mean_kmh, self.std_kmh)
        return float(np.clip(v, 0.0, self.max_kmh)) * KMH_TO_MPS


# Measured on Waymo Open validation ego poses (see docs): urban driving
# with frequent stops. Sampled in 10 km/h steps around that distribution.
STATIONARY = SpeedSetting(0.0, 0.0)
WAYMO_URBAN = SpeedSetting(20.0, 20.0, max_kmh=70.0)


@dataclass(frozen=True)
class EgoMotion:
    """Platform motion during one scan, in the camera frame."""

    speed_mps: float = 0.0
    direction: tuple[float, float, float] = (0.0, 0.0, 1.0)  # camera z
    yaw_rate_deg_s: float = 0.0
    spin_hz: float = 10.0
    scan_start_deg: float | None = None  # azimuth captured at t=0
    reference: str = "center"  # "center" | "start" | "end" of the window

    @property
    def velocity(self) -> FloatArray:
        """Velocity vector (m/s) in the camera frame."""
        d = np.asarray(self.direction, dtype=np.float64)
        n = float(np.linalg.norm(d))
        return d / n * self.speed_mps if n > 0 else np.zeros(3)


def column_times(
    azimuth_deg: NDArray[np.floating], motion: EgoMotion, window: AzimuthWindow
) -> FloatArray:
    """Capture time (s) of each return relative to the reference time."""
    period = 1.0 / motion.spin_hz
    start = (
        window.az_min_deg
        if motion.scan_start_deg is None
        else motion.scan_start_deg
    )
    az = np.asarray(azimuth_deg, dtype=np.float64)
    t = ((az - start) % 360.0) / 360.0 * period
    ref_az = {
        "start": window.az_min_deg,
        "end": window.az_max_deg,
        "center": (window.az_min_deg + window.az_max_deg) / 2.0,
    }[motion.reference]
    t_ref = ((ref_az - start) % 360.0) / 360.0 * period
    return np.asarray(t - t_ref, dtype=np.float64)


def _yaw_matrix(angle_rad: NDArray[np.floating]) -> FloatArray:
    """Rotation about the camera up axis (-y) for each angle, (N, 3, 3)."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    z, o = np.zeros_like(c), np.ones_like(c)
    # rotation about +y by -angle == rotation about -y by +angle
    return np.stack(
        [
            np.stack([c, z, -s], -1),
            np.stack([z, o, z], -1),
            np.stack([s, z, c], -1),
        ],
        -2,
    )


def apply_rolling_shutter(
    scan: LidarScan, motion: EgoMotion, window: AzimuthWindow
) -> LidarScan:
    """Distort a static-scene scan by the ego motion during the sweep."""
    if motion.speed_mps == 0 and motion.yaw_rate_deg_s == 0:
        return scan
    t = column_times(scan.azimuth_deg, motion, window)
    p = scan.points.astype(np.float64)
    v = motion.velocity
    shifted = p - v[None, :] * t[:, None]
    if motion.yaw_rate_deg_s != 0:
        ang = np.deg2rad(motion.yaw_rate_deg_s) * t
        rot = _yaw_matrix(
            -ang
        )  # undo the platform yaw, as a naive assembly does
        shifted = np.einsum("nij,nj->ni", rot, shifted)
    return replace(scan, points=shifted.astype(np.float32))


def sweep_duration_s(
    window: AzimuthWindow, spec: LidarSpec, spin_hz: float
) -> float:
    """Time the sweep needs to cross the camera window."""
    return (window.az_max_deg - window.az_min_deg) / 360.0 / spin_hz
