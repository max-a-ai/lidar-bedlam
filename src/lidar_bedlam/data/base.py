"""Shared pipeline from a raw :class:`Sample` to a :class:`CroppedSample`.

Every dataset loader implements :class:`SampleSource`; the rest of the
pipeline (SMPL joints, 3D box, crop, keypoint transfer, point sampling)
is dataset-agnostic and lives here so it cannot diverge between datasets.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from lidar_bedlam.body.smpl import SmplModel
from lidar_bedlam.data.schema import CroppedSample, Sample, SampleMeta
from lidar_bedlam.geometry.boxes import heading_yaw, oriented_box_from_points
from lidar_bedlam.geometry.camera import CAMERA_UP_AXIS
from lidar_bedlam.geometry.crop import (
    crop_image,
    crop_mask,
    square_crop_from_bbox,
)
from lidar_bedlam.geometry.rotations import axis_angle_to_matrix

FloatArray = NDArray[np.float64]

# SMPL rest pose faces +z with +y up; the heading is the rotated +z axis.
SMPL_FORWARD = np.array([0.0, 0.0, 1.0])


class SampleSource(ABC):
    """A dataset that can produce raw samples by index."""

    name: str

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def meta(self, index: int) -> SampleMeta:
        """Cheap identification of a sample without loading it."""

    @abstractmethod
    def load(self, index: int) -> Sample:
        """Load one raw sample (full image, camera-frame points, labels)."""


def add_smpl_derived(sample: Sample, model: SmplModel) -> None:
    """Fill joints and the 3D box from SMPL params if they are missing."""
    if sample.smpl is None:
        return
    verts, joints = model.forward(sample.smpl)
    if sample.joints3d is None:
        sample.joints3d = joints
        sample.joints3d_valid = np.ones(len(joints), dtype=bool)
        sample.joint_convention = "smpl24"
    if sample.box3d is None:
        forward = (
            axis_angle_to_matrix(sample.smpl.global_orient) @ SMPL_FORWARD
        )
        yaw = heading_yaw(forward, CAMERA_UP_AXIS)
        sample.box3d = oriented_box_from_points(verts, yaw, CAMERA_UP_AXIS)
    sample.extra["vertices"] = verts


def crop_sample(
    sample: Sample, out_size: int, padding: float = 1.2
) -> CroppedSample:
    """Square-crop the image, mask and 2D keypoints around the person."""
    spec = square_crop_from_bbox(sample.bbox_xyxy, out_size, padding)
    kp2d = None
    if sample.kp2d is not None:
        kp2d = sample.kp2d.copy()
        kp2d[:, :2] = spec.to_crop(kp2d[:, :2])
    mask = crop_mask(sample.mask, spec) if sample.mask is not None else None
    return CroppedSample(
        meta=sample.meta,
        image=crop_image(sample.image, spec),
        crop=spec,
        camera=spec.camera(sample.camera),
        points=sample.points,
        smpl=sample.smpl,
        joints3d=sample.joints3d,
        joints3d_valid=sample.joints3d_valid,
        joint_convention=sample.joint_convention,
        kp2d=kp2d,
        box3d=sample.box3d,
        mask=mask,
    )


def sample_points(
    points: NDArray[np.float32], n: int, rng: np.random.Generator
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    """Random subset (or padding with repeats) to exactly ``n`` points.

    Returns the points and a validity mask that is False for padded rows.
    An empty cloud yields zeros with an all-False mask.
    """
    m = len(points)
    if m == 0:
        return np.zeros((n, 3), dtype=np.float32), np.zeros(n, dtype=bool)
    if m >= n:
        idx = rng.choice(m, n, replace=False)
        return points[idx], np.ones(n, dtype=bool)
    idx = np.concatenate([np.arange(m), rng.choice(m, n - m, replace=True)])
    valid = np.zeros(n, dtype=bool)
    valid[:m] = True
    return points[idx], valid
