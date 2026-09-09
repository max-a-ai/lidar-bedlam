"""Sensor rig loaders and drawing (real files, skipped if absent)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lidar_bedlam.rigs.plot import rig_traces, rigs_figure
from lidar_bedlam.rigs.schema import Sensor, SensorRig, check_rigid

WAYMO = Path("data/waymo_perception")
NUSC = Path("data/nuscenes")
S4D = Path("data/sloper4d/seq008_running_001")


def test_check_rigid_rejects_reflection() -> None:
    bad = np.diag([1.0, 1.0, -1.0, 1.0])
    with pytest.raises(ValueError, match="reflection"):
        check_rigid(bad, "x")


def test_tree_text_and_traces_synthetic() -> None:
    cam = Sensor("CAM", "camera", np.eye(4))
    rig = SensorRig("demo", "car", "x fwd", (cam,))
    assert "demo (car)" in rig.tree_text() and "CAM" in rig.tree_text()
    assert len(rig_traces(rig)) > 4
    assert rigs_figure([rig]).layout.height == 650


@pytest.mark.skipif(not WAYMO.exists(), reason="Waymo data absent")
def test_waymo_rig() -> None:
    from lidar_bedlam.rigs.waymo import load_waymo_rig

    rig = load_waymo_rig(WAYMO)
    assert len(rig.by_kind("lidar")) == 5 and len(rig.by_kind("camera")) == 5
    top = next(s for s in rig.sensors if s.name == "LIDAR_TOP")
    assert 1.5 < top.position[2] < 2.5  # roof height
    front = next(s for s in rig.sensors if s.name == "CAM_FRONT")
    assert front.intrinsics is not None and front.intrinsics.width == 1920
    assert np.allclose(front.rotation[:, 0], [1, 0, 0], atol=0.05)  # x fwd


@pytest.mark.skipif(not NUSC.exists(), reason="nuScenes data absent")
def test_nuscenes_rig() -> None:
    from lidar_bedlam.rigs.nuscenes import load_nuscenes_rig

    rig = load_nuscenes_rig(NUSC)
    assert len(rig.by_kind("camera")) == 6 and len(rig.by_kind("lidar")) == 1
    cam = next(s for s in rig.sensors if s.name == "CAM_FRONT")
    assert cam.intrinsics is not None and cam.intrinsics.fx > 1000
    # OpenCV camera: optical axis (z) points forward in the ego frame
    assert cam.rotation[0, 2] > 0.95


@pytest.mark.skipif(not S4D.exists(), reason="SLOPER4D data absent")
def test_sloper4d_rig() -> None:
    from lidar_bedlam.rigs.sloper4d import load_sloper4d_rig

    rig = load_sloper4d_rig(S4D)
    assert rig.mount == "helmet" and len(rig.sensors) == 2
    cam = rig.by_kind("camera")[0]
    assert np.linalg.norm(cam.position) < 0.3  # camera near the LiDAR
    assert cam.rotation[0, 2] > 0.95  # optical axis forward
