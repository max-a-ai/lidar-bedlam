"""Waymo Open Dataset (v2 parquet) sensor rig.

``camera_calibration`` and ``lidar_calibration`` hold, per segment, the
sensor-to-vehicle 4x4 extrinsics. Vehicle frame: x forward, y left, z up,
origin on the rear axle at ground level. Waymo camera frames use x forward,
y left, z up (not OpenCV).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lidar_bedlam.geometry.camera import WAYMO_CAM_TO_OPENCV, PinholeCamera
from lidar_bedlam.rigs.schema import Sensor, SensorRig, check_rigid

CAMERA_NAMES = {
    1: "FRONT",
    2: "FRONT_LEFT",
    3: "FRONT_RIGHT",
    4: "SIDE_LEFT",
    5: "SIDE_RIGHT",
}
LIDAR_NAMES = {
    1: "TOP",
    2: "FRONT",
    3: "SIDE_LEFT",
    4: "SIDE_RIGHT",
    5: "REAR",
}
CAM = "[CameraCalibrationComponent]"
LID = "[LiDARCalibrationComponent]"


def load_waymo_rig(
    root: Path, split: str = "validation", segment: str | None = None
) -> SensorRig:
    """Rig of one segment (the first of the split unless given)."""
    import pyarrow.parquet as pq

    cam_dir = root / split / "camera_calibration"
    files = sorted(cam_dir.glob("*.parquet"))
    if segment is not None:
        files = [f for f in files if f.stem == segment]
    if not files:
        msg = f"no camera_calibration parquet under {cam_dir}"
        raise FileNotFoundError(msg)
    seg = files[0].stem
    sensors: list[Sensor] = []
    for row in pq.read_table(
        root / split / "lidar_calibration" / f"{seg}.parquet"
    ).to_pylist():
        t = np.array(
            row[f"{LID}.extrinsic.transform"], dtype=np.float64
        ).reshape(4, 4)
        lo = float(np.rad2deg(row[f"{LID}.beam_inclination.min"]))
        hi = float(np.rad2deg(row[f"{LID}.beam_inclination.max"]))
        name = LIDAR_NAMES[int(row["key.laser_name"])]
        check_rigid(t, f"waymo lidar {name}")
        sensors.append(
            Sensor(
                name=f"LIDAR_{name}",
                kind="lidar",
                base_from_sensor=t,
                notes=f"beam inclination {lo:.1f}..{hi:.1f} deg",
            )  # fmt: skip
        )
    for row in pq.read_table(files[0]).to_pylist():
        t = np.array(
            row[f"{CAM}.extrinsic.transform"], dtype=np.float64
        ).reshape(4, 4)
        name = CAMERA_NAMES[int(row["key.camera_name"])]
        check_rigid(t, f"waymo camera {name}")
        intr = f"{CAM}.intrinsic."
        k = PinholeCamera(
            fx=float(row[intr + "f_u"]),
            fy=float(row[intr + "f_v"]),
            cx=float(row[intr + "c_u"]),
            cy=float(row[intr + "c_v"]),
            width=int(row[f"{CAM}.width"]),
            height=int(row[f"{CAM}.height"]),
            k1=float(row[intr + "k1"]),
            k2=float(row[intr + "k2"]),
            p1=float(row[intr + "p1"]),
            p2=float(row[intr + "p2"]),
            k3=float(row[intr + "k3"]),
        )
        sensors.append(
            Sensor(
                name=f"CAM_{name}",
                kind="camera",
                base_from_sensor=t,
                intrinsics=k,
                optical_from_sensor=WAYMO_CAM_TO_OPENCV,
                notes="camera axes x forward, y left, z up (Waymo)",
            )  # fmt: skip
        )
    return SensorRig(
        dataset="Waymo Open",
        mount="car",
        base_frame="vehicle: x fwd, y left, z up, origin rear axle / ground",
        sensors=tuple(sensors),
        source=f"{split}/{{camera,lidar}}_calibration/{seg}.parquet",
    )
