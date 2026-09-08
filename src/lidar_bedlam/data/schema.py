"""The common sample schema every dataset loader produces.

Everything 3D is expressed in the OpenCV frame of the image's camera
(x right, y down, z forward), in metres. The up direction is -y, so 3D
boxes are yawed about the y axis (``CAMERA_UP_AXIS``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from lidar_bedlam.body.smpl import SmplParams
from lidar_bedlam.geometry.camera import PinholeCamera
from lidar_bedlam.geometry.crop import CropSpec

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SampleMeta:
    """Where a sample comes from."""

    dataset: str
    sequence: str
    frame: str
    person: str = "0"

    @property
    def key(self) -> str:
        """Unique string id."""
        return f"{self.dataset}/{self.sequence}/{self.frame}/{self.person}"


@dataclass
class Sample:
    """One person in one frame, ready for cropping and batching."""

    meta: SampleMeta
    image: NDArray[np.uint8]  # full RGB image (H, W, 3)
    camera: PinholeCamera  # intrinsics of ``image``
    bbox_xyxy: FloatArray  # (4,) person box in full-image pixels
    points: NDArray[np.float32]  # (N, 3) person LiDAR points, camera frame
    smpl: SmplParams | None = None  # camera frame
    joints3d: FloatArray | None = None  # (J, 3) camera frame
    joints3d_valid: NDArray[np.bool_] | None = None  # (J,)
    joint_convention: str = ""  # e.g. "smpl24", "waymo15", "coco17"
    kp2d: FloatArray | None = None  # (J, 3) full-image px + confidence
    box3d: FloatArray | None = None  # (7,) cx cy cz dx dy dz yaw, cam frame
    mask: NDArray[np.bool_] | None = None  # (H, W) person mask
    extra: dict[str, FloatArray] = field(default_factory=dict)


@dataclass
class CroppedSample:
    """A :class:`Sample` after square cropping (what the model consumes)."""

    meta: SampleMeta
    image: NDArray[np.uint8]  # (S, S, 3)
    crop: CropSpec
    camera: PinholeCamera  # intrinsics of the crop
    points: NDArray[np.float32]  # (N, 3)
    smpl: SmplParams | None
    joints3d: FloatArray | None
    joints3d_valid: NDArray[np.bool_] | None
    joint_convention: str
    kp2d: FloatArray | None  # (J, 3) crop pixels + confidence
    box3d: FloatArray | None
    mask: NDArray[np.bool_] | None  # (S, S)


WAYMO15_JOINT_NAMES = (
    "nose", "left_shoulder", "left_elbow", "left_wrist", "left_hip",
    "left_knee", "left_ankle", "right_shoulder", "right_elbow",
    "right_wrist", "right_hip", "right_knee", "right_ankle", "forehead",
    "head_center",
)  # fmt: skip
