"""Rig export writes PNG / HTML / GLB / tree files."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lidar_bedlam.geometry.camera import PinholeCamera, se3
from lidar_bedlam.geometry.rotations import euler_to_matrix
from lidar_bedlam.rigs.export import export_rig, rig_mesh
from lidar_bedlam.rigs.schema import Sensor, SensorRig


def _rig() -> SensorRig:
    cam_pose = se3(
        euler_to_matrix("xyz", np.array([-90.0, 0.0, -90.0])),
        np.array([1.0, 0.0, 1.5]),
    )
    return SensorRig(
        "demo",
        "car",
        "x fwd",
        (
            Sensor("LIDAR", "lidar", se3(np.eye(3), np.array([0, 0, 2.0]))),
            Sensor(
                "CAM",
                "camera",
                cam_pose,
                intrinsics=PinholeCamera(800, 800, 320, 240, 640, 480),
            ),  # fmt: skip
        ),
    )


def test_rig_mesh_has_geometry() -> None:
    scene = rig_mesh(_rig())
    # base triad (3) + 2 links + 2x3 axes + 8 frustum rods
    assert len(scene.geometry) == 3 + 2 + 6 + 8
    assert scene.bounds is not None and scene.bounds[1][2] > 1.5


def test_export_rig_writes_files(tmp_path: Path) -> None:
    out = export_rig(_rig(), tmp_path, "demo")
    for key in ("png", "html", "glb", "txt"):
        assert out[key].exists() and out[key].stat().st_size > 0
    assert (tmp_path / "plotly.min.js").exists()
    assert "demo (car)" in out["txt"].read_text()
    assert out["html"].stat().st_size < 200_000  # references shared js
