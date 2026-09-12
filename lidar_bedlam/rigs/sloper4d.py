"""SLOPER4D head-mounted rig (Ouster OS1-128 + action camera).

``dataset_params.json`` gives ``lidar2cam`` (LiDAR -> camera, OpenCV
camera axes) and the RGB intrinsics. The base frame is the LiDAR itself
(x forward, y left, z up), since no vehicle exists.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import numpy as np

from lidar_bedlam.geometry.camera import PinholeCamera, invert_se3
from lidar_bedlam.rigs.schema import Sensor, SensorRig, check_rigid


def _rgb_info(params: dict[str, Any]) -> dict[str, Any]:
    info = params["RGB_info"]
    if isinstance(info, str):  # some sequences store it as a repr string
        info = ast.literal_eval(info)
    return dict(info)


def load_sloper4d_rig(seq_dir: Path) -> SensorRig:
    """Rig of one SLOPER4D sequence directory."""
    with open(seq_dir / "dataset_params.json") as fh:
        params = json.load(fh)
    info = _rgb_info(params)
    lidar2cam = np.asarray(info["lidar2cam"], dtype=np.float64)
    check_rigid(lidar2cam, "sloper4d lidar2cam")
    fx, fy, cx, cy = (float(v) for v in info["intrinsics"])
    k = PinholeCamera(fx, fy, cx, cy, int(info["width"]), int(info["height"]))
    sensors = (
        Sensor(
            name="LIDAR_HEAD",
            kind="lidar",
            base_from_sensor=np.eye(4),
            notes="Ouster OS1-128 on the helmet; base frame",
        ),  # fmt: skip
        Sensor(
            name="CAM_HEAD",
            kind="camera",
            base_from_sensor=invert_se3(lidar2cam),
            intrinsics=k,
            notes="DJI action camera, OpenCV axes",
        ),  # fmt: skip
    )
    return SensorRig(
        dataset="SLOPER4D",
        mount="helmet",
        base_frame="head LiDAR: x forward, y left, z up",
        sensors=sensors,
        source=f"{seq_dir.name}/dataset_params.json",
    )
