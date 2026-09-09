"""LiDAR simulation against synthetic depth maps, and occlusions."""

from __future__ import annotations

import numpy as np

from lidar_bedlam.geometry.camera import PinholeCamera
from lidar_bedlam.lidar.augment import OcclusionConfig, occlude
from lidar_bedlam.lidar.simulate import (
    PRESETS,
    LidarSpec,
    azimuth_window,
    ouster,
    select_mask,
    sensor_pose,
    simulate,
    with_channels,
)

CAM = PinholeCamera.from_hfov(60.0, 320, 180)
NOISELESS = LidarSpec("test-32", 32, -15.0, 15.0, 360, range_noise_std_m=0.0)


def _wall(z: float) -> np.ndarray:
    return np.full((CAM.height, CAM.width), z)


def test_wall_from_camera_origin() -> None:
    scan = simulate(_wall(5.0), CAM, NOISELESS, rng=np.random.default_rng(0))
    assert len(scan.points) > 500
    assert np.allclose(scan.points[:, 2], 5.0, atol=0.02)
    # every channel of the spec that looks into the image is present
    assert scan.channel.min() >= 0 and scan.channel.max() < NOISELESS.channels
    assert scan.column.min() >= 0 and scan.column.max() < 360
    # ranges are consistent with the geometry (r = z / cos)
    r = np.linalg.norm(scan.points, axis=1)
    assert np.allclose(r, scan.range_m, atol=1e-4)


def test_wall_with_sensor_offset() -> None:
    pose = sensor_pose(np.array([0.2, -0.4, -0.3]), pitch_deg=3.0)
    scan = simulate(_wall(6.0), CAM, NOISELESS, pose, np.random.default_rng(0))
    assert len(scan.points) > 300
    assert np.allclose(scan.points[:, 2], 6.0, atol=0.03)
    # points emanate from the sensor origin: azimuth 0, beam through centre
    d = scan.points - pose[:3, 3]
    assert np.all(d[:, 2] > 0)


def test_occluder_in_front_of_wall() -> None:
    depth = _wall(10.0)
    depth[60:120, 140:180] = 4.0  # a box in the middle
    scan = simulate(depth, CAM, NOISELESS, rng=np.random.default_rng(0))
    near = scan.points[:, 2] < 5.0
    assert near.any() and (~near).any()
    assert np.allclose(scan.points[near, 2], 4.0, atol=0.02)
    mask = np.zeros_like(depth, dtype=bool)
    mask[60:120, 140:180] = True
    sel = select_mask(scan, mask)
    assert np.allclose(sel.points[:, 2], 4.0, atol=0.02)
    assert len(sel.points) == near.sum()


def test_channel_count_scales_density() -> None:
    depth = _wall(8.0)
    n32 = len(simulate(depth, CAM, with_channels(NOISELESS, 32)).points)
    n64 = len(simulate(depth, CAM, with_channels(NOISELESS, 64)).points)
    assert 1.8 < n64 / n32 < 2.2


def test_azimuth_window_columns() -> None:
    cam90 = PinholeCamera(50.0, 50.0, 49.5, 49.5, 100, 100)  # 90 deg HFOV
    w = azimuth_window(cam90, ouster("OS1", 64, steps=1024))
    assert abs(w.hfov_deg - 90.0) < 1e-9 and w.columns == 256
    assert azimuth_window(cam90, ouster("OS0", 32, steps=2048)).columns == 512
    assert ouster("OS2", 128).elevation_max_deg == 11.25


def test_presets_and_dropout() -> None:
    for name, spec in PRESETS.items():
        assert spec.name == name and spec.channels > 0
    spec = LidarSpec(
        "d-16", 16, -10, 10, 360, range_noise_std_m=0.0, dropout=0.5
    )
    full = simulate(
        _wall(5.0),
        CAM,
        LidarSpec("f-16", 16, -10, 10, 360, range_noise_std_m=0.0),
    )
    half = simulate(_wall(5.0), CAM, spec, rng=np.random.default_rng(1))
    assert 0.35 < len(half.points) / len(full.points) < 0.65


def test_occlude_keeps_minimum_and_removes_something() -> None:
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(500, 3)) * [0.3, 0.9, 0.2]
    cfg = OcclusionConfig(
        p_halfspace=1.0, p_box=1.0, p_height_cut=1.0, p_subsample=1.0
    )
    keep = occlude(pts, cfg, rng)
    assert cfg.min_points <= keep.sum() < 500
    tiny = occlude(pts[:5], cfg, rng)
    assert tiny.all()
