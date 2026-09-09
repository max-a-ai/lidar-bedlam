"""nuScenes-format sensor rigs (nuScenes itself and our own exports).

Each ``calibrated_sensor`` entry stores the sensor pose in the ego frame as
a translation and a quaternion ``[w, x, y, z]``; cameras carry a 3x3
intrinsic matrix. Ego frame: x forward, y left, z up. nuScenes cameras use
the OpenCV convention. Image sizes are read from ``sample_data`` (nuScenes
1600x900, our AVA / FUSE-Bike exports 2200x1200).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from lidar_bedlam.geometry.camera import PinholeCamera, se3
from lidar_bedlam.rigs.schema import Sensor, SensorRig, check_rigid

DEFAULT_IMAGE_SIZE = (1600, 900)
KIND_ORDER = {"lidar": 0, "radar": 1, "camera": 2}


def _image_sizes(meta: Path) -> dict[str, tuple[int, int]]:
    """Image (width, height) per calibrated_sensor token from sample_data."""
    path = meta / "sample_data.json"
    if not path.exists():
        return {}
    with open(path) as fh:
        rows: list[dict[str, Any]] = json.load(fh)
    sizes: dict[str, tuple[int, int]] = {}
    for r in rows:
        tok = str(r["calibrated_sensor_token"])
        if tok not in sizes and int(r.get("width", 0)) > 0:
            sizes[tok] = (int(r["width"]), int(r["height"]))
    return sizes


def load_nuscenes_rig(
    root: Path,
    version: str = "v1.0-mini",
    dataset: str = "nuScenes",
    mount: str = "car",
    base_frame: str = "ego: x fwd, y left, z up, origin rear axle / ground",
) -> SensorRig:
    """Rig from the first calibrated entry of every sensor channel."""
    meta = root / version
    with open(meta / "sensor.json") as fh:
        channels = {s["token"]: s for s in json.load(fh)}
    with open(meta / "calibrated_sensor.json") as fh:
        calibrated = json.load(fh)
    sizes = _image_sizes(meta)
    seen: dict[str, Sensor] = {}
    for c in calibrated:
        sensor = channels[c["sensor_token"]]
        name = str(sensor["channel"])
        if name in seen:
            continue
        rot = Rotation.from_quat(c["rotation"], scalar_first=True).as_matrix()
        t = se3(np.asarray(rot), np.asarray(c["translation"], dtype=float))
        check_rigid(t, f"{dataset} {name}")
        kind = str(sensor["modality"])
        intrinsics = None
        k = c.get("camera_intrinsic") or []
        if kind == "camera" and k:
            km = np.asarray(k, dtype=np.float64)
            w, h = sizes.get(str(c["token"]), DEFAULT_IMAGE_SIZE)
            intrinsics = PinholeCamera(
                km[0, 0], km[1, 1], km[0, 2], km[1, 2], w, h
            )
        seen[name] = Sensor(
            name=name,
            kind=kind,
            base_from_sensor=t,
            intrinsics=intrinsics,
            notes="OpenCV camera axes" if kind == "camera" else "",
        )
    sensors = sorted(
        seen.values(), key=lambda s: (KIND_ORDER.get(s.kind, 9), s.name)
    )
    return SensorRig(
        dataset=dataset,
        mount=mount,
        base_frame=base_frame,
        sensors=tuple(sensors),
        source=f"{root.name}/{version}/calibrated_sensor.json",
    )
