"""Shard records and sensor placement."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lidar_bedlam.generate.records import (
    CROP,
    MAX_POINTS,
    Record,
    Scan,
    Shard,
    write_shard,
)
from lidar_bedlam.geometry.camera import WAYMO_CAM_TO_OPENCV, se3
from lidar_bedlam.lidar.placement import (
    BallPlacement,
    lidar_to_simulator_axes,
    rig_lidar_pose,
)
from lidar_bedlam.rigs.schema import Sensor, SensorRig


def _record(i: int, n_pts: int) -> Record:
    rng = np.random.default_rng(i)
    scan = Scan(
        points=rng.normal(size=(n_pts, 3)).astype(np.float32),
        channel=np.arange(n_pts, dtype=np.int16) % 64,
        column=np.arange(n_pts, dtype=np.int16),
        channels=64,
        steps=1024,
        sensor_pose=np.eye(4),
    )
    return Record(
        key=f"k{i}", dataset="bedlam",
        image=np.full((CROP, CROP, 3), i, np.uint8),
        image_aug=np.zeros((CROP, CROP, 3), np.uint8),
        mask=np.zeros((CROP, CROP), bool),
        intrinsics=np.eye(3), crop_origin=np.zeros(3),
        has_image=True, has_smpl=True,
        global_orient=np.zeros(3), body_pose=np.zeros(69),
        betas=np.zeros(10), transl=np.array([0, 0, 8.0]),
        joints3d=np.zeros((24, 3)), joints3d_valid=np.ones(24, bool),
        joint_convention="smpl24", kp2d=np.zeros((24, 3)),
        box3d=np.zeros(7), distance_scale=1.0,
        scans={"main_0": scan, "check": scan} if i else {"main_0": scan},
    )  # fmt: skip


def test_shard_roundtrip(tmp_path: Path) -> None:
    recs = [_record(0, 10), _record(1, MAX_POINTS + 50)]
    path = tmp_path / "s.npz"
    write_shard(recs, path)
    shard = Shard(path)
    assert len(shard) == 2 and shard.variants == ["check", "main_0"]
    assert shard.array("image")[1, 0, 0, 0] == 1
    s0 = shard.scan(0, "main_0")
    assert len(s0.points) == 10 and s0.channels == 64 and s0.steps == 1024
    s1 = shard.scan(1, "main_0")
    assert len(s1.points) == MAX_POINTS  # truncated to the padding size
    assert shard.scan(0, "check").points.shape == (0, 3)  # absent variant
    assert np.allclose(shard.array("transl")[0], [0, 0, 8.0])


def test_ball_placement_bounds() -> None:
    rng = np.random.default_rng(0)
    ball = BallPlacement(r_max_m=1.0, tilt_max_deg=5.0)
    radii = np.array(
        [np.linalg.norm(ball.sample(rng)[:3, 3]) for _ in range(500)]
    )
    assert radii.max() <= 1.0 and radii.min() >= 0.0
    assert 0.35 < radii.mean() < 0.65  # radius uniform, not volume-uniform


def test_rig_lidar_pose_and_axes() -> None:
    # camera at (1, 0, 2) looking along +x (Waymo axes), LiDAR at (1, 0, 2.5)
    cam = Sensor(
        "CAM", "camera", se3(np.eye(3), np.array([1.0, 0.0, 2.0])),
        optical_from_sensor=WAYMO_CAM_TO_OPENCV,
    )  # fmt: skip
    lid = Sensor("LID", "lidar", se3(np.eye(3), np.array([1.0, 0.0, 2.5])))
    rig = SensorRig("t", "car", "x fwd", (cam, lid))
    pose = rig_lidar_pose(rig, "CAM", "LID")
    # 0.5 m above the camera == -0.5 along OpenCV y
    assert np.allclose(pose[:3, 3], [0.0, -0.5, 0.0])
    sim = lidar_to_simulator_axes(pose)
    # simulator forward (+z column) must be the LiDAR's x (forward)
    assert np.allclose(sim[:3, 2], WAYMO_CAM_TO_OPENCV @ [1, 0, 0])


def test_shard_views_match_npz_and_rows_are_copies(tmp_path: Path) -> None:
    from test_shard_dataset import _record

    path = tmp_path / "v_00000.npz"
    write_shard([_record(i) for i in range(4)], path)
    shard = Shard(path)
    ref = np.load(path, allow_pickle=False)
    for name in ref.files:
        assert np.array_equal(shard.array(name), ref[name]), name
    assert not shard.array("image").flags.writeable
    row = shard.row("image", 2)
    assert row.flags.writeable and row.base is None
    scan = shard.scan(1, "main_0")
    assert scan.points.dtype == np.float32 and len(scan.points) == 300
