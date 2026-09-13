"""3DPW loader: frames with gendered SMPL labels in the camera frame.

3DPW ships ``sequenceFiles/<split>/<seq>.pkl`` with per-actor ``poses``
(F, 72), ``betas`` (10), ``trans`` (F, 3) in the world frame,
``cam_poses`` (F, 4, 4) world-to-camera (verified against ``poses2d``),
``cam_intrinsics`` (3, 3) and ``genders``. Images are
``imageFiles/<seq>/image_XXXXX.jpg``. There is no depth and no LiDAR: the
mesh-based generator renders both from the labelled meshes.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from lidar_bedlam.body.smpl import SmplModel, SmplParams, transform_smpl_params
from lidar_bedlam.geometry.camera import PinholeCamera
from lidar_bedlam.utils.io import read_image

GENDERS = ("male", "female", "neutral")


@dataclass(frozen=True)
class ActorLabel:
    """One actor in one frame, camera frame."""

    actor: int
    gender: str
    smpl: SmplParams


@dataclass(frozen=True)
class Frame:
    """Everything the mesh generator needs for one frame."""

    seq: str
    frame: int
    image_path: Path
    camera: PinholeCamera
    actors: list[ActorLabel]


class ThreeDPWSource:
    """Frames of one split, one entry per (sequence, frame)."""

    name = "3dpw"

    def __init__(
        self,
        root: Path,
        split: str,
        smpl_models: dict[str, SmplModel],
        frame_stride: int = 5,
        sequences: list[str] | None = None,
    ) -> None:
        self.root = root
        self.split = split
        self.models = smpl_models
        seq_dir = root / "sequenceFiles" / split
        self.sequences = sequences or sorted(
            p.stem for p in seq_dir.glob("*.pkl")
        )
        self._cache: dict[str, dict[str, Any]] = {}
        self._index: list[tuple[str, int]] = []
        for seq in self.sequences:
            data = self._data(seq)
            n = len(data["img_frame_ids"])
            valid = np.all(np.asarray(data["campose_valid"]), axis=0)
            for i in range(0, n, frame_stride):
                if bool(valid[i]):
                    self._index.append((seq, i))

    def _data(self, seq: str) -> dict[str, Any]:
        if seq not in self._cache:
            path = self.root / "sequenceFiles" / self.split / f"{seq}.pkl"
            with open(path, "rb") as fh:
                self._cache[seq] = pickle.load(fh, encoding="latin1")
        return self._cache[seq]

    def __len__(self) -> int:
        return len(self._index)

    def frame(self, index: int) -> Frame:
        """Labels of frame ``index`` transformed into the camera frame."""
        seq, i = self._index[index]
        data = self._data(seq)
        k = np.asarray(data["cam_intrinsics"], dtype=np.float64)
        world_to_cam = np.asarray(data["cam_poses"][i], dtype=np.float64)
        frame_id = int(data["img_frame_ids"][i])
        image_path = (
            self.root / "imageFiles" / seq / f"image_{frame_id:05d}.jpg"
        )
        actors = []
        for a, gender in enumerate(data["genders"]):
            g = {"m": "male", "f": "female"}.get(str(gender), str(gender))
            model = self.models[g]
            world = SmplParams.from_pose72(
                np.asarray(data["poses"][a][i], dtype=np.float64),
                np.asarray(data["betas"][a][:10], dtype=np.float64),
                np.asarray(data["trans"][a][i], dtype=np.float64),
            )
            cam = transform_smpl_params(
                world, world_to_cam, model.rest_pelvis(world.betas)
            )
            actors.append(ActorLabel(a, g, cam))
        return Frame(seq, frame_id, image_path, self._camera(k), actors)

    def _camera(self, k: NDArray[np.float64]) -> PinholeCamera:
        # 3DPW frames are 1080 x 1920 (portrait); the principal point of the
        # intrinsics tells which orientation the sequence has
        w, h = (1080, 1920) if k[1, 2] > k[0, 2] else (1920, 1080)
        return PinholeCamera(k[0, 0], k[1, 1], k[0, 2], k[1, 2], w, h)

    def read_image(self, frame: Frame) -> NDArray[np.uint8]:
        """The RGB frame."""
        return read_image(frame.image_path)
