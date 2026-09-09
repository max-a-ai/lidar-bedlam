"""Spinning-LiDAR simulation against a rendered depth map.

The depth map (planar z, metres) of a pinhole camera is the only scene
geometry we have. Every LiDAR beam is a ray from the sensor origin; we
march along it, look up the rendered depth at the projected pixel, and
take the first crossing of the ray with the depth surface. This is exact
for a sensor at the camera origin and a good approximation for a sensor
offset by up to a few decimetres (the depth surface only knows what the
camera saw, so surfaces hidden from the camera stay unknown).

Everything is expressed in the OpenCV camera frame (x right, y down,
z forward); the sensor's own frame has the same axes rotated/translated
by ``sensor_from_camera``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from lidar_bedlam.geometry.camera import PinholeCamera, invert_se3, se3
from lidar_bedlam.geometry.rotations import euler_to_matrix

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class LidarSpec:
    """A spinning LiDAR: ``beams`` elevations over a vertical FOV."""

    name: str
    beams: int
    elevation_min_deg: float
    elevation_max_deg: float
    azimuth_res_deg: float
    max_range_m: float = 80.0
    min_range_m: float = 0.5
    range_noise_std_m: float = 0.02
    dropout: float = 0.0

    @property
    def elevations_deg(self) -> FloatArray:
        """Beam elevations, top to bottom, evenly spaced."""
        return np.linspace(
            self.elevation_max_deg, self.elevation_min_deg, self.beams
        )


# Presets. Ouster-like sensors share a 45 deg vertical FOV; the beam count
# sets the angular resolution. "waymo64" mimics the Waymo top LiDAR.
PRESETS: dict[str, LidarSpec] = {
    "os32": LidarSpec("os32", 32, -22.5, 22.5, 0.35),
    "os64": LidarSpec("os64", 64, -22.5, 22.5, 0.35),
    "os128": LidarSpec("os128", 128, -22.5, 22.5, 0.35),
    "os256": LidarSpec("os256", 256, -22.5, 22.5, 0.175),
    "waymo64": LidarSpec("waymo64", 64, -17.6, 2.4, 0.14, 75.0),
}


def sensor_pose(
    translation_m: NDArray[np.floating] | None = None,
    pitch_deg: float = 0.0,
    yaw_deg: float = 0.0,
    roll_deg: float = 0.0,
) -> FloatArray:
    """4x4 pose of the sensor in the camera frame (camera_from_sensor).

    ``translation_m`` is the sensor origin in camera coordinates (x right,
    y down, z forward), e.g. ``[0, -0.3, 0]`` mounts it 30 cm above the
    camera. Angles tilt the sensor's optical axis relative to the camera.
    """
    rot = euler_to_matrix("xyz", np.array([pitch_deg, yaw_deg, roll_deg]))
    t = np.zeros(3) if translation_m is None else np.asarray(translation_m)
    return se3(rot, t)


@dataclass
class LidarScan:
    """Simulated returns in the camera frame."""

    points: NDArray[np.float32]  # (N, 3)
    beam: NDArray[np.int32]  # (N,) beam index, 0 = top
    azimuth_deg: NDArray[np.float32]  # (N,)
    range_m: NDArray[np.float32]  # (N,) measured range (with noise)
    pixel: NDArray[np.int32]  # (N, 2) hit pixel (u, v) in the depth map


def _beam_directions(
    spec: LidarSpec, az_min: float, az_max: float
) -> tuple[FloatArray, NDArray[np.int32], FloatArray]:
    """Unit directions (R, 3) in the sensor frame for the given azimuths."""
    az = np.arange(az_min, az_max + 1e-9, spec.azimuth_res_deg)
    el = spec.elevations_deg
    az_g, el_g = np.meshgrid(np.deg2rad(az), np.deg2rad(el))
    beam_g = np.broadcast_to(np.arange(spec.beams)[:, None], az_g.shape)
    # azimuth about the -y (up) axis, elevation towards -y
    x = np.cos(el_g) * np.sin(az_g)
    y = -np.sin(el_g)
    z = np.cos(el_g) * np.cos(az_g)
    dirs = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    return (
        dirs,
        beam_g.reshape(-1).astype(np.int32),
        np.rad2deg(az_g).reshape(-1),
    )


def camera_azimuth_span(
    camera: PinholeCamera, camera_from_sensor: NDArray[np.floating]
) -> tuple[float, float]:
    """Azimuth range (deg, sensor frame) that can hit the camera image."""
    half = np.rad2deg(np.arctan((camera.width / 2.0) / camera.fx))
    yaw = float(
        np.rad2deg(
            np.arctan2(camera_from_sensor[0, 2], camera_from_sensor[2, 2])
        )
    )
    margin = 2.0
    return -half - yaw - margin, half - yaw + margin


def _march(
    origin: FloatArray,
    dirs: FloatArray,
    depth: FloatArray,
    camera: PinholeCamera,
    spec: LidarSpec,
    steps: int,
) -> tuple[FloatArray, NDArray[np.bool_], NDArray[np.int64]]:
    """First crossing of each ray with the depth surface.

    Returns the ray parameter ``t`` (metres along the unit direction), a hit
    mask, and the hit pixel index (row-major) for each ray.
    """
    h, w = depth.shape
    ts = np.geomspace(spec.min_range_m, spec.max_range_m, steps)
    n = len(dirs)
    t_hit = np.full(n, np.nan)
    hit = np.zeros(n, dtype=bool)
    pix = np.zeros(n, dtype=np.int64)
    prev_gap = np.full(n, np.nan)
    prev_t = np.full(n, np.nan)
    prev_pix = np.zeros(n, dtype=np.int64)
    for t in ts:
        p = origin + t * dirs
        z = p[:, 2]
        u = np.rint(
            camera.fx * p[:, 0] / np.where(z > 0, z, np.nan) + camera.cx
        )
        v = np.rint(
            camera.fy * p[:, 1] / np.where(z > 0, z, np.nan) + camera.cy
        )
        inside = (z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        idx = np.where(inside, (v * w + u), 0).astype(np.int64)
        surf = np.where(inside, depth.reshape(-1)[idx], np.nan)
        gap = z - surf  # > 0 once the ray passed behind the surface
        # a hit needs the previous sample inside the image and in front of
        # the surface; rays entering the image from outside stay unknown
        crossing = (
            ~hit & inside & (gap > 0) & np.isfinite(prev_gap) & (prev_gap <= 0)
        )
        frac = -prev_gap / np.maximum(gap - prev_gap, 1e-9)
        t_est = prev_t + frac * (t - prev_t)
        t_hit[crossing] = t_est[crossing]
        pix[crossing] = prev_pix[crossing]
        hit |= crossing
        prev_gap = np.where(inside, gap, np.nan)
        prev_t = np.full(n, t)
        prev_pix = idx
    return t_hit, hit, pix


def simulate(
    depth_m: NDArray[np.floating],
    camera: PinholeCamera,
    spec: LidarSpec,
    camera_from_sensor: NDArray[np.floating] | None = None,
    rng: np.random.Generator | None = None,
    steps: int = 192,
) -> LidarScan:
    """Cast one full scan and return the returns inside the camera image."""
    rng = rng or np.random.default_rng()
    pose = (
        np.eye(4)
        if camera_from_sensor is None
        else np.asarray(camera_from_sensor)
    )
    az_min, az_max = camera_azimuth_span(camera, pose)
    dirs_s, beam, az = _beam_directions(spec, az_min, az_max)
    dirs = dirs_s @ pose[:3, :3].T
    origin = pose[:3, 3]
    depth = np.asarray(depth_m, dtype=np.float64)
    t_hit, hit, pix = _march(origin, dirs, depth, camera, spec, steps)
    if spec.dropout > 0:
        hit &= rng.random(len(hit)) >= spec.dropout
    t = t_hit[hit]
    if spec.range_noise_std_m > 0:
        t = t + rng.normal(0.0, spec.range_noise_std_m, size=len(t))
    pts = origin + t[:, None] * dirs[hit]
    w = depth.shape[1]
    pixel = np.stack([pix[hit] % w, pix[hit] // w], axis=-1)
    return LidarScan(
        points=pts.astype(np.float32),
        beam=beam[hit],
        azimuth_deg=az[hit].astype(np.float32),
        range_m=t.astype(np.float32),
        pixel=pixel.astype(np.int32),
    )


def select_mask(scan: LidarScan, mask: NDArray[np.bool_]) -> LidarScan:
    """Keep only the returns whose hit pixel lies inside ``mask``."""
    keep = mask[scan.pixel[:, 1], scan.pixel[:, 0]]
    return LidarScan(
        points=scan.points[keep],
        beam=scan.beam[keep],
        azimuth_deg=scan.azimuth_deg[keep],
        range_m=scan.range_m[keep],
        pixel=scan.pixel[keep],
    )


def with_beams(spec: LidarSpec, beams: int) -> LidarSpec:
    """Same sensor with a different beam count (resolution ablation)."""
    return replace(spec, name=f"{spec.name}-{beams}", beams=beams)


def sensor_from_camera(camera_from_sensor: NDArray[np.floating]) -> FloatArray:
    """Inverse pose, for expressing camera-frame points in the sensor."""
    return invert_se3(camera_from_sensor)
