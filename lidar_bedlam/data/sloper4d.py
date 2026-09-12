"""SLOPER4D loader (LiDAR + RGB + SMPL, head-mounted rig).

Reads ``<seq>/<seq>_labels.pkl``. Everything in the pickle is in the
SLOPER4D world frame (metres, z up); ``RGB_frames.cam_pose`` is the
world-to-camera transform (OpenCV convention) of every RGB frame.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from lidar_bedlam.body.smpl import SmplModel, SmplParams, transform_smpl_params
from lidar_bedlam.data.base import SampleSource
from lidar_bedlam.data.schema import Sample, SampleMeta
from lidar_bedlam.geometry.camera import PinholeCamera, transform_points
from lidar_bedlam.utils.io import read_image


def _skel_2d(raw: object) -> NDArray[np.float64] | None:
    """COCO-17 keypoints (17, 3): some sequences store 4 values per joint."""
    arr = np.asarray(raw, dtype=np.float64)
    if arr.size == 0 or arr.size % 17:
        return None
    arr = arr.reshape(17, -1)
    if arr.shape[1] < 3:
        return None
    return np.ascontiguousarray(arr[:, :3])


class Sloper4dSource(SampleSource):
    """One sample per RGB frame that has a bbox and person points."""

    name = "sloper4d"

    def __init__(
        self,
        root: Path,
        smpl_model: SmplModel,
        sequences: list[str] | None = None,
    ) -> None:
        self.root = root
        self.smpl_model = smpl_model
        self.sequences = sequences or sorted(
            p.name
            for p in root.iterdir()
            if (p / f"{p.name}_labels.pkl").exists()
        )
        self._cache: dict[str, dict[str, Any]] = {}
        self._index: list[tuple[str, int]] = []
        for seq in self.sequences:
            data = self._labels(seq)
            frames = data["RGB_frames"]
            for i, (bbox, pts) in enumerate(
                zip(frames["bbox"], frames["human_points"], strict=True)
            ):
                if len(bbox) == 4 and len(pts) > 0:
                    self._index.append((seq, i))

    def _labels(self, seq: str) -> dict[str, Any]:
        if seq not in self._cache:
            with open(self.root / seq / f"{seq}_labels.pkl", "rb") as fh:
                self._cache[seq] = pickle.load(fh)
        return self._cache[seq]

    def __len__(self) -> int:
        return len(self._index)

    def meta(self, index: int) -> SampleMeta:
        seq, i = self._index[index]
        name = self._labels(seq)["RGB_frames"]["file_basename"][i]
        return SampleMeta(self.name, seq, Path(name).stem)

    def load(self, index: int) -> Sample:
        seq, i = self._index[index]
        data = self._labels(seq)
        frames = data["RGB_frames"]
        info = data["RGB_info"]
        fx, fy, cx, cy = (float(v) for v in info["intrinsics"])
        k1, k2, p1, p2, k3 = (float(v) for v in info["dist"])
        camera = PinholeCamera(
            fx, fy, cx, cy, int(info["width"]), int(info["height"]),
            k1, k2, p1, p2, k3,
        )  # fmt: skip
        world_to_cam = np.asarray(frames["cam_pose"][i], dtype=np.float64)
        points = transform_points(world_to_cam, frames["human_points"][i])
        smpl_world = SmplParams.from_pose72(
            np.asarray(frames["smpl_pose"][i]),
            np.asarray(frames["beta"][i]),
            np.asarray(frames["global_trans"][i]),
        )
        smpl = transform_smpl_params(
            smpl_world,
            world_to_cam,
            self.smpl_model.rest_pelvis(smpl_world.betas),
        )
        image_path = (
            self.root
            / seq
            / "rgb_data"
            / f"{seq}_imgs"
            / frames["file_basename"][i]
        )
        kp2d = _skel_2d(frames["skel_2d"][i])
        return Sample(
            meta=self.meta(index),
            image=read_image(image_path),
            camera=camera,
            bbox_xyxy=np.asarray(frames["bbox"][i], dtype=np.float64),
            points=np.ascontiguousarray(points, dtype=np.float32),
            smpl=smpl,
            kp2d=kp2d,
            extra={"kp2d_convention_coco17": np.zeros(0)},
        )
