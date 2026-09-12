"""Waymo Open Dataset pedestrians with 3D keypoints (pre-extracted crops).

Uses the ``waymo_pose_complete_4`` extraction made for LIF-Net:
``<subset>/<context>_labels.pkl`` (dict image_id -> record) and
``<subset>/images/<image_id>.jpg`` (full 1920x1280 camera frames).
Records hold, in the *vehicle* frame: ``lidar`` (N, 3) pedestrian points,
``keypoints_3d_arr`` (15, 4), ``bb_3d`` (centre, size, heading about +z);
``extrinsic`` is the camera-to-vehicle transform (Waymo camera axes) and
``intrinsic`` is ``[f_u, f_v, c_u, c_v, k1, k2, p1, p2, k3]``.
No SMPL ground truth exists; ``keypoints_2d_arr`` (15, 3) are full-image
pixels + occlusion flag.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from lidar_bedlam.data.base import SampleSource
from lidar_bedlam.data.schema import Sample, SampleMeta
from lidar_bedlam.geometry.boxes import heading_yaw
from lidar_bedlam.geometry.camera import (
    CAMERA_UP_AXIS,
    WAYMO_CAM_TO_OPENCV,
    PinholeCamera,
    invert_se3,
    se3,
    transform_points,
)
from lidar_bedlam.utils.io import read_image

FloatArray = NDArray[np.float64]
NUM_JOINTS = 15


def vehicle_to_opencv(camera_to_vehicle: NDArray[np.floating]) -> FloatArray:
    """4x4 transform from the Waymo vehicle frame to the OpenCV camera."""
    to_waymo_cam = invert_se3(camera_to_vehicle)
    axes = se3(WAYMO_CAM_TO_OPENCV, np.zeros(3))
    return np.asarray(axes @ to_waymo_cam, dtype=np.float64)


def box_vehicle_to_camera(
    bb_3d: dict[str, float], vehicle_to_cam: NDArray[np.floating]
) -> FloatArray:
    """Waymo box (centre, length/width/height, heading) -> camera 7-box."""
    centre = np.array(
        [bb_3d["center_x"], bb_3d["center_y"], bb_3d["center_z"]]
    )
    heading = np.array(
        [np.cos(bb_3d["heading"]), np.sin(bb_3d["heading"]), 0.0]
    )
    r = np.asarray(vehicle_to_cam, dtype=np.float64)[:3, :3]
    c = transform_points(vehicle_to_cam, centre[None])[0]
    yaw = heading_yaw(r @ heading, CAMERA_UP_AXIS)
    # box axes: length along the heading, height along up (-y), width across
    size = np.array([bb_3d["length"], bb_3d["height"], bb_3d["width"]])
    return np.concatenate([c, size, [yaw]])


class WaymoSource(SampleSource):
    """Pedestrian crops with GT 3D keypoints and LiDAR points."""

    name = "waymo"

    def __init__(
        self,
        root: Path,
        split: str = "train",
        subsets: tuple[str, ...] = ("3D", "3D_2D"),
        min_points: int = 1,
    ) -> None:
        self.root = root
        contexts = set((root / f"{split}_scenarios.txt").read_text().split())
        self._records: list[tuple[str, str, dict[str, Any]]] = []
        for subset in subsets:
            for pkl in sorted((root / subset).glob("*_labels.pkl")):
                context = pkl.name[: -len("_labels.pkl")]
                if context not in contexts:
                    continue
                with open(pkl, "rb") as fh:
                    labels: dict[str, dict[str, Any]] = pickle.load(fh)
                for image_id, rec in labels.items():
                    if "lidar" not in rec or len(rec["lidar"]) < min_points:
                        continue
                    if not (
                        root / subset / "images" / f"{image_id}.jpg"
                    ).exists():
                        continue
                    self._records.append((subset, image_id, rec))

    def __len__(self) -> int:
        return len(self._records)

    def meta(self, index: int) -> SampleMeta:
        subset, image_id, rec = self._records[index]
        frame, cam, obj = image_id.split("_", 2)
        return SampleMeta(
            self.name, str(rec.get("context", subset)), f"{frame}_{cam}", obj
        )

    def load(self, index: int) -> Sample:
        subset, image_id, rec = self._records[index]
        image = read_image(self.root / subset / "images" / f"{image_id}.jpg")
        h, w = image.shape[:2]
        fu, fv, cu, cv, k1, k2, p1, p2, k3 = (
            float(v) for v in rec["intrinsic"]
        )
        camera = PinholeCamera(fu, fv, cu, cv, w, h, k1, k2, p1, p2, k3)
        to_cam = vehicle_to_opencv(
            np.asarray(rec["extrinsic"], dtype=np.float64)
        )
        points = transform_points(to_cam, rec["lidar"])
        bb = rec["bb_2d"]
        bbox = np.array(
            [
                bb["center_x"] - bb["width"] / 2.0,
                bb["center_y"] - bb["height"] / 2.0,
                bb["center_x"] + bb["width"] / 2.0,
                bb["center_y"] + bb["height"] / 2.0,
            ]
        )
        kp3d = np.asarray(rec["keypoints_3d_arr"], dtype=np.float64)
        valid = np.ones(NUM_JOINTS, dtype=bool)
        valid[list(rec.get("mask_3d", []))] = False
        valid &= np.any(kp3d[:, :3] != 0, axis=1)
        joints = transform_points(to_cam, kp3d[:, :3])
        kp2d = None
        kp2d_arr = np.asarray(rec.get("keypoints_2d_arr", np.zeros(0)))
        if kp2d_arr.size and np.any(kp2d_arr[:, :2] != 0):
            kp2d = np.asarray(kp2d_arr, dtype=np.float64).copy()
            present = np.ones(NUM_JOINTS, dtype=bool)
            present[list(rec.get("mask_2d", []))] = False
            present &= np.any(kp2d[:, :2] != 0, axis=1)
            kp2d[:, 2] = present.astype(np.float64)
        return Sample(
            meta=self.meta(index),
            image=image,
            camera=camera,
            bbox_xyxy=bbox,
            points=np.ascontiguousarray(points, dtype=np.float32),
            joints3d=joints,
            joints3d_valid=valid,
            joint_convention="waymo15",
            kp2d=kp2d,
            box3d=box_vehicle_to_camera(rec["bb_3d"], to_cam),
        )
