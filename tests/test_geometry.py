"""Cameras, crops, rotations and 3D boxes."""

from __future__ import annotations

import numpy as np

from lidar_bedlam.geometry.boxes import (
    aabb_from_points,
    heading_yaw,
    iou3d,
    oriented_box_from_points,
)
from lidar_bedlam.geometry.camera import (
    WAYMO_CAM_TO_OPENCV,
    PinholeCamera,
    invert_se3,
    se3,
    transform_points,
)
from lidar_bedlam.geometry.crop import square_crop_from_bbox
from lidar_bedlam.geometry.rotations import (
    axis_angle_to_matrix,
    euler_to_matrix,
    matrix_to_axis_angle,
)


def test_project_unproject_roundtrip() -> None:
    cam = PinholeCamera.from_hfov(52.0, 64, 32)
    depth = np.full((32, 64), 3.0)
    pts = cam.unproject_depth(depth)
    uv = cam.project(pts)
    v, u = np.mgrid[0:32, 0:64]
    assert np.allclose(uv[:, 0], u.reshape(-1))
    assert np.allclose(uv[:, 1], v.reshape(-1))
    assert np.allclose(pts[:, 2], 3.0)


def test_project_behind_camera_is_nan() -> None:
    cam = PinholeCamera(100, 100, 50, 50, 100, 100)
    uv = cam.project(np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]]))
    assert np.isnan(uv[0]).all() and np.allclose(uv[1], [50, 50])


def test_distortion_is_identity_at_centre() -> None:
    cam = PinholeCamera(100, 100, 50, 50, 100, 100, k1=0.1, p1=0.01)
    assert np.allclose(cam.project(np.array([[0.0, 0.0, 2.0]])), [[50, 50]])


def test_cropped_camera_projects_consistently() -> None:
    cam = PinholeCamera(800, 800, 320, 240, 640, 480)
    p = np.array([[0.3, -0.2, 4.0]])
    spec = square_crop_from_bbox(np.array([100, 50, 300, 350]), 256)
    uv_full = cam.project(p)
    uv_crop = spec.camera(cam).project(p)
    assert np.allclose(spec.to_crop(uv_full), uv_crop)


def test_se3_inverse_and_transform() -> None:
    r = euler_to_matrix("xyz", np.array([10.0, 20.0, 30.0]))
    t = se3(r, np.array([1.0, 2.0, 3.0]))
    p = np.random.default_rng(0).normal(size=(5, 3))
    back = transform_points(invert_se3(t), transform_points(t, p))
    assert np.allclose(back, p)


def test_waymo_axis_map_is_rotation() -> None:
    m = WAYMO_CAM_TO_OPENCV
    assert np.allclose(m @ m.T, np.eye(3)) and np.isclose(np.linalg.det(m), 1)
    # Waymo forward (x) becomes OpenCV z, Waymo up (z) becomes OpenCV -y
    assert np.allclose(m @ [1, 0, 0], [0, 0, 1])
    assert np.allclose(m @ [0, 0, 1], [0, -1, 0])


def test_axis_angle_roundtrip() -> None:
    aa = np.array([[0.1, -0.4, 0.9], [0.0, 0.0, 0.0]])
    assert np.allclose(matrix_to_axis_angle(axis_angle_to_matrix(aa)), aa)


def test_iou_identical_and_disjoint() -> None:
    a = np.array([0, 0, 0, 2, 1, 1, 0.3])
    b = a.copy()
    b[0] += 10
    assert np.isclose(iou3d(a, a, up_axis=1), 1.0)
    assert np.isclose(iou3d(a, b, up_axis=1), 0.0)


def test_iou_axis_aligned_known_value() -> None:
    a = np.array([0, 0, 0, 2, 2, 2, 0.0])
    b = np.array([1, 0, 0, 2, 2, 2, 0.0])  # half overlap along x
    assert np.isclose(iou3d(a, b, up_axis=1), 4.0 / 12.0)


def test_iou_rotated_square_equals_one() -> None:
    a = np.array([0, 0, 0, 2, 1, 2, 0.0])
    b = np.array([0, 0, 0, 2, 1, 2, np.pi / 2])
    assert np.isclose(iou3d(a, b, up_axis=1), 1.0)


def test_oriented_box_recovers_rotated_points() -> None:
    rng = np.random.default_rng(1)
    local = rng.uniform(-0.5, 0.5, size=(500, 3)) * [2.0, 1.0, 0.5]
    local[[0, 1]] = [[-1, -0.5, -0.25], [1, 0.5, 0.25]]  # exact extents
    yaw = 0.7
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])
    pts = local @ rot.T + [3.0, -1.0, 8.0]
    box = oriented_box_from_points(pts, yaw, up_axis=1)
    assert np.allclose(box[:3], [3.0, -1.0, 8.0], atol=1e-6)
    assert np.allclose(box[3:6], [2.0, 1.0, 0.5], atol=1e-6)
    assert np.isclose(heading_yaw(rot @ [1, 0, 0], 1), yaw)
    assert np.allclose(aabb_from_points(pts)[6], 0.0)
