"""TUMTraf (A9 release r02, the S110 intersection) pedestrians as samples.

One scene directory holds ``point_clouds/<lidar>/*.pcd`` (Ouster, ascii
PCD), ``labels_point_clouds/<lidar>/*.json`` (OpenLABEL: one frame per
file with 3D cuboids in that LiDAR's frame) and ``images/<camera>/*.jpg``
(two Basler cameras on the same gantry, 1920 x 1200, rectified).

The OpenLABEL ``coordinate_systems`` tree carries a ``matrix4x4`` per
child; it maps parent coordinates into the child frame (the frame's own
``transforms`` list holds the inverse, named ``<child>_to_<parent>``), so
the camera entry is ``camera_from_lidar`` directly and its camera axes are
OpenCV's. Cuboids are ``(x, y, z, qx, qy, qz, qw, length, width, height)``.

A pedestrian becomes a sample for the camera in which all eight box
corners project inside the image; the LiDAR returns inside the (slightly
enlarged) cuboid are the person's points. There is no body label: the
source exists for qualitative inference.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from lidar_bedlam.data.base import SampleSource
from lidar_bedlam.data.nusc_schema import quat_to_rot
from lidar_bedlam.data.schema import Sample, SampleMeta
from lidar_bedlam.geometry.boxes import heading_yaw
from lidar_bedlam.geometry.camera import (
    CAMERA_UP_AXIS,
    PinholeCamera,
    transform_points,
)
from lidar_bedlam.geometry.crop import bbox_from_points_2d
from lidar_bedlam.utils.io import read_image
from lidar_bedlam.utils.pcd import read_pcd

FloatArray = NDArray[np.float64]

TUMTRAF_ROOT = Path("/home/max/nas_drive/publicdatasets/A9 Dataset")
DEFAULT_SCENE = TUMTRAF_ROOT / "a9_dataset_r02_s01"
DEFAULT_LIDAR = "s110_lidar_ouster_south"


@dataclass(frozen=True)
class Candidate:
    """One pedestrian cuboid seen by one camera."""

    label_file: Path
    object_id: str
    camera: str
    image_file: str
    cuboid: FloatArray  # (10,) x y z qx qy qz qw l w h, LiDAR frame
    num_points: int
    bbox: FloatArray  # (4,) xyxy in the image


def _frame(label_file: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """The OpenLABEL document and its single frame."""
    doc = json.loads(label_file.read_text())["openlabel"]
    frame = next(iter(doc["frames"].values()))
    return doc, frame


def camera_calibration(
    doc: dict[str, Any], camera: str
) -> tuple[PinholeCamera, FloatArray]:
    """Intrinsics and ``camera_from_lidar`` of ``camera`` from a document."""
    stream = doc["streams"][camera]["stream_properties"]["intrinsics_pinhole"]
    k = np.asarray(stream["camera_matrix_3x4"], dtype=np.float64)
    cam = PinholeCamera(
        k[0, 0], k[1, 1], k[0, 2], k[1, 2],
        int(stream["width_px"]), int(stream["height_px"]),
    )  # fmt: skip
    pose = doc["coordinate_systems"][camera]["pose_wrt_parent"]["matrix4x4"]
    return cam, np.asarray(pose, dtype=np.float64).reshape(4, 4)


def cuboid_corners(cuboid: FloatArray) -> FloatArray:
    """The 8 LiDAR-frame corners of an OpenLABEL cuboid."""
    x, y, z, qx, qy, qz, qw, length, width, height = (float(v) for v in cuboid)
    rot = quat_to_rot([qw, qx, qy, qz])
    sx = np.array([1, 1, 1, 1, -1, -1, -1, -1], dtype=np.float64) * length / 2
    sy = np.array([1, 1, -1, -1, 1, 1, -1, -1], dtype=np.float64) * width / 2
    sz = np.array([1, -1, 1, -1, 1, -1, 1, -1], dtype=np.float64) * height / 2
    local = np.stack([sx, sy, sz], 1)
    corners: FloatArray = np.asarray(
        local @ rot.T + np.array([x, y, z]), dtype=np.float64
    )
    return corners


def points_in_cuboid(
    points: FloatArray, cuboid: FloatArray, margin_m: float = 0.15
) -> NDArray[np.bool_]:
    """Mask of LiDAR-frame points inside the cuboid grown by ``margin_m``."""
    x, y, z, qx, qy, qz, qw, length, width, height = (float(v) for v in cuboid)
    rot = quat_to_rot([qw, qx, qy, qz])
    local = (points - np.array([x, y, z])) @ rot
    half = np.array([length, width, height]) / 2 + margin_m
    inside: NDArray[np.bool_] = np.all(np.abs(local) <= half, axis=1)
    return inside


def _project(cam: PinholeCamera, pts_cam: FloatArray) -> FloatArray:
    z = pts_cam[:, 2]
    uv = np.full((len(pts_cam), 2), np.nan)
    ok = z > 0.1
    uv[ok, 0] = cam.fx * pts_cam[ok, 0] / z[ok] + cam.cx
    uv[ok, 1] = cam.fy * pts_cam[ok, 1] / z[ok] + cam.cy
    return uv


@lru_cache(maxsize=4)
def _scan(path: str) -> FloatArray:
    pts = read_pcd(path, fields=("x", "y", "z"))
    return np.asarray(pts[np.isfinite(pts).all(1)], dtype=np.float64)


class TumTrafSource(SampleSource):
    """Pedestrians of one TUMTraf r02 scene through one LiDAR's labels."""

    name = "tumtraf"

    def __init__(
        self,
        scene: Path = DEFAULT_SCENE,
        lidar: str = DEFAULT_LIDAR,
        min_points: int = 30,
        min_box_height_px: int = 64,
        every: int = 1,
        cameras: tuple[str, ...] = (
            "s110_camera_basler_south1_8mm",
            "s110_camera_basler_south2_8mm",
        ),
    ) -> None:
        self.scene = scene
        self.lidar = lidar
        self.candidates: list[Candidate] = []
        files = sorted((scene / "labels_point_clouds" / lidar).glob("*.json"))
        for label_file in files[::every]:
            doc, frame = _frame(label_file)
            images = frame["frame_properties"].get("image_file_names", [])
            for cam_name in cameras:
                image_file = next((f for f in images if cam_name in f), None)
                if image_file is None:
                    continue
                cam, cam_from_lidar = camera_calibration(doc, cam_name)
                for oid, obj in frame["objects"].items():
                    od = obj["object_data"]
                    if od["type"] != "PEDESTRIAN":
                        continue
                    cub = np.asarray(od["cuboid"]["val"], dtype=np.float64)
                    nums = {
                        a["name"]: a["val"]
                        for a in od["cuboid"]
                        .get("attributes", {})
                        .get("num", [])
                    }
                    n_pts = int(nums.get("num_points", 0))
                    if n_pts < min_points:
                        continue
                    corners_cam = transform_points(
                        cam_from_lidar, cuboid_corners(cub)
                    )
                    if (corners_cam[:, 2] <= 0.5).any():
                        continue
                    uv = _project(cam, corners_cam)
                    if not (
                        (uv[:, 0] >= 0).all()
                        and (uv[:, 0] < cam.width).all()
                        and (uv[:, 1] >= 0).all()
                        and (uv[:, 1] < cam.height).all()
                    ):
                        continue
                    bbox = bbox_from_points_2d(uv, cam.width, cam.height)
                    if bbox is None or bbox[3] - bbox[1] < min_box_height_px:
                        continue
                    self.candidates.append(
                        Candidate(
                            label_file,
                            oid,
                            cam_name,
                            image_file,
                            cub,
                            n_pts,
                            bbox,
                        )
                    )

    def __len__(self) -> int:
        return len(self.candidates)

    def meta(self, index: int) -> SampleMeta:
        c = self.candidates[index]
        return SampleMeta(
            dataset=self.name,
            sequence=f"{self.scene.name}/{c.camera}",
            frame=c.label_file.stem,
            person=c.object_id,
        )

    def category(self, index: int) -> str:
        """Object type plus the labelled return count."""
        return f"pedestrian ({self.candidates[index].num_points} pts)"

    def load(self, index: int) -> Sample:
        c = self.candidates[index]
        doc, _ = _frame(c.label_file)
        cam, cam_from_lidar = camera_calibration(doc, c.camera)
        image = read_image(self.scene / "images" / c.camera / c.image_file)
        pcd = (
            self.scene
            / "point_clouds"
            / self.lidar
            / f"{c.label_file.stem}.pcd"
        )
        scan = _scan(str(pcd))
        inside = points_in_cuboid(scan, c.cuboid)
        pts_cam = transform_points(cam_from_lidar, scan[inside])
        centre = transform_points(cam_from_lidar, c.cuboid[None, :3])[0]
        rot = cam_from_lidar[:3, :3] @ quat_to_rot(
            [c.cuboid[6], c.cuboid[3], c.cuboid[4], c.cuboid[5]]
        )
        yaw = heading_yaw(rot[:, 0], CAMERA_UP_AXIS)
        length, width, height = (float(v) for v in c.cuboid[7:10])
        box3d = np.concatenate([centre, [length, height, width], [yaw]])
        return Sample(
            meta=self.meta(index),
            image=image,
            camera=cam,
            bbox_xyxy=c.bbox,
            points=np.ascontiguousarray(pts_cam, dtype=np.float32),
            box3d=box3d,
        )
