"""Spinning-LiDAR simulation against a rendered depth map.

The depth map (planar z, metres) of a pinhole camera is the only scene
geometry we have. Every LiDAR channel is a ray from the sensor origin; we
march along it, look up the rendered depth at the projected pixel, and
take the first crossing of the ray with a part of the depth surface that
faces the sensor (depth-map normals, back-facing patches never return).
This is exact for a sensor at the camera origin and an approximation for
an offset sensor: the depth surface only knows what the camera saw, so
surfaces hidden from the camera (a person's side) stay unknown and return
nothing rather than something wrong.

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
    """A spinning LiDAR: ``channels`` elevations over a vertical FOV.

    Azimuth sampling follows the sensor's ``horizontal_steps`` per full
    revolution (Ouster modes: 512 / 1024 / 2048), so a camera with a
    horizontal FOV of ``h`` degrees sees ``h / 360 * horizontal_steps``
    columns of the scan (90 deg at 1024 steps = 256 columns).
    """

    name: str
    channels: int
    elevation_min_deg: float
    elevation_max_deg: float
    horizontal_steps: int = 1024
    max_range_m: float = 80.0
    min_range_m: float = 0.5
    range_noise_std_m: float = 0.02
    dropout: float = 0.0

    @property
    def azimuth_res_deg(self) -> float:
        """Azimuth step in degrees."""
        return 360.0 / self.horizontal_steps

    @property
    def elevations_deg(self) -> FloatArray:
        """Channel elevations, top to bottom, evenly spaced."""
        return np.linspace(
            self.elevation_max_deg, self.elevation_min_deg, self.channels
        )


# Ouster families: vertical FOV and range differ, channels are 32/64/128.
OUSTER_FAMILIES: dict[str, tuple[float, float]] = {
    "OS0": (45.0, 50.0),  # half vertical FOV (deg), max range (m)
    "OS1": (22.5, 120.0),
    "OS2": (11.25, 240.0),
}


def ouster(family: str, channels: int, steps: int = 1024) -> LidarSpec:
    """Ouster-style spec, e.g. ``ouster("OS1", 64)`` -> ``OS1-64``.

    256 channels do not exist as a product; they are a resolution ablation.
    """
    half_fov, max_range = OUSTER_FAMILIES[family]
    return LidarSpec(
        f"{family}-{channels}", channels, -half_fov, half_fov, steps, max_range
    )


PRESETS: dict[str, LidarSpec] = {
    "OS1-32": ouster("OS1", 32),
    "OS1-64": ouster("OS1", 64),
    "OS1-128": ouster("OS1", 128),
    "OS1-256": ouster("OS1", 256),
    "OS0-128": ouster("OS0", 128),
    "OS2-128": ouster("OS2", 128),
    # Waymo top LiDAR: 64 channels, -17.6..2.4 deg, ~2650 columns per rev
    "waymo64": LidarSpec("waymo64", 64, -17.6, 2.4, 2650, 75.0),
}


@dataclass(frozen=True)
class AzimuthWindow:
    """The part of a revolution that falls into a camera image."""

    hfov_deg: float
    az_min_deg: float
    az_max_deg: float
    columns: int  # number of azimuth steps inside the camera HFOV


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
    channel: NDArray[np.int32]  # (N,) channel index, 0 = top
    column: NDArray[np.int32]  # (N,) azimuth step index on the 360 deg grid
    azimuth_deg: NDArray[np.float32]  # (N,)
    range_m: NDArray[np.float32]  # (N,) measured range (with noise)
    pixel: NDArray[np.int32]  # (N, 2) hit pixel (u, v) in the depth map


def _ray_directions(
    spec: LidarSpec, az_min: float, az_max: float
) -> tuple[FloatArray, NDArray[np.int32], NDArray[np.int32], FloatArray]:
    """Unit directions (R, 3) in the sensor frame, on the global grid.

    Azimuth samples are multiples of the step so the columns are the same
    ones a real scan would have; returns directions, channel index, column
    index and azimuth in degrees.
    """
    res = spec.azimuth_res_deg
    cols = np.arange(
        int(np.ceil(az_min / res)), int(np.floor(az_max / res)) + 1
    )
    az = cols * res
    el = spec.elevations_deg
    az_g, el_g = np.meshgrid(np.deg2rad(az), np.deg2rad(el))
    ch_g = np.broadcast_to(np.arange(spec.channels)[:, None], az_g.shape)
    col_g = np.broadcast_to(cols[None, :] % spec.horizontal_steps, az_g.shape)
    # azimuth about the -y (up) axis, elevation towards -y
    x = np.cos(el_g) * np.sin(az_g)
    y = -np.sin(el_g)
    z = np.cos(el_g) * np.cos(az_g)
    dirs = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    return (
        dirs,
        ch_g.reshape(-1).astype(np.int32),
        col_g.reshape(-1).astype(np.int32),
        np.rad2deg(az_g).reshape(-1),
    )


def camera_hfov_deg(camera: PinholeCamera) -> float:
    """Horizontal field of view of a pinhole camera in degrees."""
    return float(2.0 * np.rad2deg(np.arctan((camera.width / 2.0) / camera.fx)))


def azimuth_window(
    camera: PinholeCamera,
    spec: LidarSpec,
    camera_from_sensor: NDArray[np.floating] | None = None,
) -> AzimuthWindow:
    """Azimuth range (sensor frame) and column count covering the image."""
    pose = (
        np.eye(4)
        if camera_from_sensor is None
        else np.asarray(camera_from_sensor)
    )
    hfov = camera_hfov_deg(camera)
    columns = int(round(hfov / 360.0 * spec.horizontal_steps))
    # azimuths (sensor frame) of the image frustum's side edges over the
    # sensor's range: a translated sensor sees the image window under a
    # wider, shifted range of azimuths than the camera does
    xs = np.array(
        [(0 - camera.cx) / camera.fx, (camera.width - camera.cx) / camera.fx]
    )
    ys = np.array(
        [(0 - camera.cy) / camera.fy, (camera.height - camera.cy) / camera.fy]
    )
    zs = np.geomspace(max(spec.min_range_m, 0.1), spec.max_range_m, 32)
    x, y, z = np.meshgrid(xs, ys, zs, indexing="ij")
    p_cam = np.stack([x * z, y * z, z], -1).reshape(-1, 3)
    p_sen = (p_cam - pose[:3, 3]) @ pose[:3, :3]  # R^T (p - t)
    az = np.rad2deg(np.arctan2(p_sen[:, 0], p_sen[:, 2]))
    # unwrap around the image centre's azimuth so a sensor looking
    # backwards (image at +-180 deg) gets one contiguous window
    centre = np.array([0.0, 0.0, float(np.median(zs))]) - pose[:3, 3]
    centre = centre @ pose[:3, :3]
    az0 = float(np.rad2deg(np.arctan2(centre[0], centre[2])))
    rel = (az - az0 + 180.0) % 360.0 - 180.0
    return AzimuthWindow(
        hfov, az0 + float(rel.min()), az0 + float(rel.max()), columns
    )


WALL_JUMP_M = 0.5  # depth steps beyond this are silhouettes: no normal there
SIDE_DEPTH_M = 0.25  # how far behind the front a side-entry return may lie


def surface_normals(
    depth: FloatArray, camera: PinholeCamera, wall_m: float = WALL_JUMP_M
) -> FloatArray:
    """Unit normals (H, W, 3) of the depth surface, facing the camera.

    Central differences of the back-projected points; at a silhouette the
    difference is taken on the continuous side, and pixels whose both
    sides jump by more than ``wall_m`` get a nan normal (no surface there).
    """
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    p = np.stack(
        [
            depth * (u - camera.cx) / camera.fx,
            depth * (v - camera.cy) / camera.fy,
            depth,
        ],
        -1,
    )

    def diff(axis: int) -> FloatArray:
        fwd = np.roll(p, -1, axis) - p
        bwd = p - np.roll(p, 1, axis)
        jf = np.abs(np.roll(depth, -1, axis) - depth)
        jb = np.abs(depth - np.roll(depth, 1, axis))
        use_f = jf <= jb
        d = np.where(use_f[..., None], fwd, bwd)
        bad = np.minimum(jf, jb) > wall_m
        return np.asarray(np.where(bad[..., None], np.nan, d))

    n = np.cross(diff(1), diff(0))
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    n = n / np.where(norm > 0, norm, np.nan)
    # every depth-map surface faces the camera: orient n against the view ray
    flip = np.sum(n * p, -1, keepdims=True) > 0
    return np.asarray(np.where(flip, -n, n), dtype=np.float64)


def _gap_at(
    origin: FloatArray,
    dirs: FloatArray,
    depth: FloatArray,
    camera: PinholeCamera,
    t: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Signed distance behind the depth surface and that surface's depth."""
    h, w = depth.shape
    p = origin + t[:, None] * dirs
    z = np.where(p[:, 2] > 0, p[:, 2], np.nan)
    u = np.rint(camera.fx * p[:, 0] / z + camera.cx)
    v = np.rint(camera.fy * p[:, 1] / z + camera.cy)
    inside = np.isfinite(z) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    idx = np.where(inside, np.nan_to_num(v * w + u), 0).astype(np.int64)
    surf = np.where(inside, depth.reshape(-1)[idx], np.nan)
    return np.asarray(p[:, 2] - surf), np.asarray(surf)


def _refine(
    origin: FloatArray,
    dirs: FloatArray,
    depth: FloatArray,
    camera: PinholeCamera,
    t_lo: FloatArray,
    t_hi: float,
    gap_lo: FloatArray,
    gap_hi: FloatArray,
    iterations: int = 6,
) -> FloatArray:
    """Bisect the crossing between ``t_lo`` (in front) and ``t_hi`` (behind)
    and interpolate the surface inside the final interval."""
    lo = t_lo.copy()
    hi = np.full(len(dirs), float(t_hi))
    g_lo, g_hi = gap_lo.copy(), gap_hi.copy()
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        g_mid, _ = _gap_at(origin, dirs, depth, camera, mid)
        behind = g_mid > 0
        hi = np.where(behind, mid, hi)
        g_hi = np.where(behind, g_mid, g_hi)
        lo = np.where(behind, lo, mid)
        g_lo = np.where(behind, g_lo, g_mid)
    frac = -g_lo / np.maximum(g_hi - g_lo, 1e-9)
    return np.asarray(lo + np.clip(frac, 0.0, 1.0) * (hi - lo))


def _march(
    origin: FloatArray,
    dirs: FloatArray,
    depth: FloatArray,
    camera: PinholeCamera,
    spec: LidarSpec,
    steps: int,
    normals: FloatArray | None = None,
) -> tuple[FloatArray, NDArray[np.bool_], NDArray[np.int64]]:
    """First crossing of each ray with a surface facing the sensor.

    Returns the ray parameter ``t`` (metres along the unit direction), a hit
    mask, and the hit pixel index (row-major) for each ray. With
    ``normals`` a crossing counts only where the surface faces the ray
    (dot(direction, normal) < 0): a sensor away from the camera cannot see
    surfaces that face away from it, even if the camera did.
    """
    h, w = depth.shape
    ts = np.geomspace(spec.min_range_m, spec.max_range_m, steps)
    n = len(dirs)
    t_hit = np.full(n, np.nan)
    hit = np.zeros(n, dtype=bool)
    pix = np.zeros(n, dtype=np.int64)
    prev_gap = np.full(n, np.nan)
    prev_t = np.full(n, np.nan)
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
        # refine the crossing by bisection on the depth surface: a ray
        # entering a silhouette from the side lands on the silhouette line,
        # i.e. within the body's thickness of the unseen side surface
        t_est = _refine(origin, dirs, depth, camera, prev_t, t, prev_gap, gap)
        # a ray entering a silhouette from the side lands on the silhouette
        # line: within a body's thickness behind the front it stands in for
        # the side surface the camera never saw, farther behind it is a
        # wall (the ray would have hit the body) and returns nothing
        gap_hit, _ = _gap_at(origin, dirs, depth, camera, t_est)
        crossing &= (gap_hit >= -0.05) & (gap_hit <= SIDE_DEPTH_M)
        # pixel of the crossing point itself (not of the previous sample)
        p_hit = origin + t_est[:, None] * dirs
        zh = np.where(p_hit[:, 2] > 0, p_hit[:, 2], np.nan)
        uh = np.clip(
            np.rint(camera.fx * p_hit[:, 0] / zh + camera.cx), 0, w - 1
        )
        vh = np.clip(
            np.rint(camera.fy * p_hit[:, 1] / zh + camera.cy), 0, h - 1
        )
        pix_hit = np.where(crossing, np.nan_to_num(vh * w + uh), 0).astype(
            np.int64
        )
        if normals is not None:
            n_hit = normals.reshape(-1, 3)[pix_hit]
            facing = np.sum(n_hit * dirs, -1) < 0  # nan normals never face
            crossing &= facing
        t_hit[crossing] = t_est[crossing]
        pix[crossing] = pix_hit[crossing]
        hit |= crossing
        prev_gap = np.where(inside, gap, np.nan)
        prev_t = np.full(n, t)
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
    window = azimuth_window(camera, spec, pose)
    margin = 2.0  # rays slightly outside the image still march (offsets)
    dirs_s, channel, column, az = _ray_directions(
        spec, window.az_min_deg - margin, window.az_max_deg + margin
    )
    dirs = dirs_s @ pose[:3, :3].T
    origin = pose[:3, 3]
    depth = np.asarray(depth_m, dtype=np.float64)
    normals = surface_normals(depth, camera)
    t_hit, hit, pix = _march(origin, dirs, depth, camera, spec, steps, normals)
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
        channel=channel[hit],
        column=column[hit],
        azimuth_deg=az[hit].astype(np.float32),
        range_m=t.astype(np.float32),
        pixel=pixel.astype(np.int32),
    )


def select_mask(scan: LidarScan, mask: NDArray[np.bool_]) -> LidarScan:
    """Keep only the returns whose hit pixel lies inside ``mask``."""
    keep = mask[scan.pixel[:, 1], scan.pixel[:, 0]]
    return LidarScan(
        points=scan.points[keep],
        channel=scan.channel[keep],
        column=scan.column[keep],
        azimuth_deg=scan.azimuth_deg[keep],
        range_m=scan.range_m[keep],
        pixel=scan.pixel[keep],
    )


def with_channels(spec: LidarSpec, channels: int) -> LidarSpec:
    """Same sensor with a different channel count (resolution ablation)."""
    family = spec.name.split("-")[0]
    return replace(spec, name=f"{family}-{channels}", channels=channels)


def sensor_from_camera(camera_from_sensor: NDArray[np.floating]) -> FloatArray:
    """Inverse pose, for expressing camera-frame points in the sensor."""
    return invert_se3(camera_from_sensor)
