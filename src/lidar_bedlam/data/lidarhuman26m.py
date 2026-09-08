"""LiDARHuman26M (LiDARCap) loader.

The release ships person crops (``images/<scene>/<frame>.png``) cut from
1920x1080 frames at ``lidarhuman26M_top_left.json`` offsets, the LiDAR
points of the person (``labels/3d/segment``) and SMPL params
(``labels/3d/pose``), both in the LiDAR frame. The LiDAR-to-camera
extrinsics and the camera intrinsics are the fixed values published with
LiDARCap (one rig for all scenes).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lidar_bedlam.body.smpl import SmplModel, SmplParams, transform_smpl_params
from lidar_bedlam.data.base import SampleSource
from lidar_bedlam.data.schema import Sample, SampleMeta
from lidar_bedlam.geometry.camera import PinholeCamera, transform_points
from lidar_bedlam.io import read_image, read_ply_xyz

FULL_WIDTH, FULL_HEIGHT = 1920, 1080

# Intrinsics + Brown-Conrady distortion (k1, k2, p1, p2, k3) of the rig.
CAMERA = PinholeCamera(
    fx=956.32709662202160,
    fy=956.87763573729683,
    cx=962.09910493679433,
    cy=590.26610775785059,
    width=FULL_WIDTH,
    height=FULL_HEIGHT,
    k1=-6.1100617222502205e-03,
    k2=3.0647823796371827e-02,
    p1=3.3304524444662654e-04,
    p2=-4.4038460096976607e-04,
    k3=-2.5974982760794661e-02,
)

# LiDAR (x forward, y left, z up) -> OpenCV camera.
LIDAR_TO_CAMERA = np.array(
    [
        [-0.0043368991524, -0.99998911867, -0.0017186757713, 0.016471385748],
        [-0.0052925495236, 0.0017416212982, -0.99998447772, 0.080050847871],
        [0.99997658984, -0.0043277356572, -0.0053000451695, -0.049279053295],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


class LidarHuman26mSource(SampleSource):
    """One sample per crop image of the chosen split."""

    name = "lidarhuman26m"

    def __init__(
        self, root: Path, smpl_model: SmplModel, split: str = "train"
    ) -> None:
        self.root = root
        self.smpl_model = smpl_model
        split_file = root / ("test.txt" if split == "test" else "train.txt")
        scenes = [
            s.strip() for s in split_file.read_text().split() if s.strip()
        ]
        with open(root / "lidarhuman26M_top_left.json") as fh:
            self._top_left: dict[str, list[int]] = json.load(fh)
        self._index: list[tuple[str, str]] = []
        for scene in scenes:
            for png in sorted((root / "images" / scene).glob("*.png")):
                if f"{scene}/{png.stem}" in self._top_left:
                    self._index.append((scene, png.stem))

    def __len__(self) -> int:
        return len(self._index)

    def meta(self, index: int) -> SampleMeta:
        scene, frame = self._index[index]
        return SampleMeta(self.name, scene, frame)

    def load(self, index: int) -> Sample:
        scene, frame = self._index[index]
        image = read_image(self.root / "images" / scene / f"{frame}.png")
        h, w = image.shape[:2]
        tlx, tly = (float(v) for v in self._top_left[f"{scene}/{frame}"])
        # the crop is a window of the full image: shift the principal point
        camera = PinholeCamera(
            CAMERA.fx, CAMERA.fy, CAMERA.cx - tlx, CAMERA.cy - tly, w, h,
            CAMERA.k1, CAMERA.k2, CAMERA.p1, CAMERA.p2, CAMERA.k3,
        )  # fmt: skip
        with open(
            self.root / "labels" / "3d" / "pose" / scene / f"{frame}.json"
        ) as fh:
            gt = json.load(fh)
        smpl_lidar = SmplParams.from_pose72(
            np.asarray(gt["pose"]),
            np.asarray(gt["beta"]),
            np.asarray(gt["trans"]),
        )
        smpl = transform_smpl_params(
            smpl_lidar,
            LIDAR_TO_CAMERA,
            self.smpl_model.rest_pelvis(smpl_lidar.betas),
        )
        pts_lidar = read_ply_xyz(
            self.root / "labels" / "3d" / "segment" / scene / f"{frame}.ply"
        )
        points = transform_points(LIDAR_TO_CAMERA, pts_lidar)
        return Sample(
            meta=self.meta(index),
            image=image,
            camera=camera,
            bbox_xyxy=np.array([0.0, 0.0, w - 1.0, h - 1.0]),
            points=np.ascontiguousarray(points, dtype=np.float32),
            smpl=smpl,
        )
