"""``torch.utils.data.Dataset`` over one or more :class:`SampleSource`.

Every item is a flat dict of fixed-shape tensors (plus a few strings), so
the default collate function works. Missing labels are zero-filled and
flagged with ``has_*`` booleans, which the losses respect.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from lidar_bedlam.body.smpl import SmplModel
from lidar_bedlam.data.base import (
    SampleSource,
    add_smpl_derived,
    crop_sample,
    sample_points,
)
from lidar_bedlam.data.schema import CroppedSample

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MAX_JOINTS = 24

Item = dict[str, torch.Tensor | str]


class HumanPoseDataset(Dataset[Item]):
    """Concatenation of sources with cropping and tensor conversion."""

    def __init__(
        self,
        sources: Sequence[SampleSource],
        smpl_model: SmplModel | None,
        out_size: int = 256,
        n_points: int = 1024,
        padding: float = 1.2,
        seed: int = 0,
    ) -> None:
        self.sources = list(sources)
        self.smpl_model = smpl_model
        self.out_size = out_size
        self.n_points = n_points
        self.padding = padding
        self._rng = np.random.default_rng(seed)
        self._offsets = np.cumsum([0] + [len(s) for s in self.sources])

    def __len__(self) -> int:
        return int(self._offsets[-1])

    def _locate(self, index: int) -> tuple[SampleSource, int]:
        src = int(np.searchsorted(self._offsets, index, side="right") - 1)
        return self.sources[src], index - int(self._offsets[src])

    def cropped(self, index: int) -> CroppedSample:
        """The cropped sample before tensor conversion (for inspection)."""
        source, local = self._locate(index)
        sample = source.load(local)
        if self.smpl_model is not None:
            add_smpl_derived(sample, self.smpl_model)
        return crop_sample(sample, self.out_size, self.padding)

    def __getitem__(self, index: int) -> Item:
        return self.to_tensors(self.cropped(index))

    def to_tensors(self, s: CroppedSample) -> Item:
        """Flatten a cropped sample into fixed-shape tensors."""
        img = (
            s.image.astype(np.float32) / 255.0 - IMAGENET_MEAN
        ) / IMAGENET_STD
        pts, pts_valid = sample_points(s.points, self.n_points, self._rng)
        item: Item = {
            "key": s.meta.key,
            "dataset": s.meta.dataset,
            "image": torch.from_numpy(
                np.ascontiguousarray(img.transpose(2, 0, 1))
            ),
            "points": torch.from_numpy(pts),
            "points_valid": torch.from_numpy(pts_valid),
            "intrinsics": torch.from_numpy(s.camera.matrix).float(),
            "crop_origin": torch.tensor(
                [s.crop.x0, s.crop.y0, s.crop.side]
            ).float(),
        }
        self._add_smpl(item, s)
        self._add_joints(item, s)
        self._add_box(item, s)
        mask = (
            s.mask if s.mask is not None else np.zeros(s.image.shape[:2], bool)
        )
        item["mask"] = torch.from_numpy(mask)
        item["has_mask"] = torch.tensor(s.mask is not None)
        return item

    def _add_smpl(self, item: Item, s: CroppedSample) -> None:
        z = np.zeros
        p = s.smpl
        item["has_smpl"] = torch.tensor(p is not None)
        item["global_orient"] = _f(p.global_orient if p else z(3))
        item["body_pose"] = _f(p.body_pose if p else z(69))
        item["betas"] = _f(p.betas if p else z(10))
        item["transl"] = _f(p.transl if p else z(3))

    def _add_joints(self, item: Item, s: CroppedSample) -> None:
        joints = np.zeros((MAX_JOINTS, 3))
        valid = np.zeros(MAX_JOINTS, dtype=bool)
        if s.joints3d is not None and s.joints3d_valid is not None:
            n = len(s.joints3d)
            joints[:n] = s.joints3d
            valid[:n] = s.joints3d_valid
        item["joints3d"] = _f(joints)
        item["joints3d_valid"] = torch.from_numpy(valid)
        item["joint_convention"] = s.joint_convention
        kp = np.zeros((MAX_JOINTS, 3))
        if s.kp2d is not None:
            kp[: len(s.kp2d)] = s.kp2d
        item["kp2d"] = _f(kp)
        item["has_kp2d"] = torch.tensor(s.kp2d is not None)

    def _add_box(self, item: Item, s: CroppedSample) -> None:
        item["box3d"] = _f(s.box3d if s.box3d is not None else np.zeros(7))
        item["has_box3d"] = torch.tensor(s.box3d is not None)


def _f(x: np.ndarray[tuple[int, ...], np.dtype[np.floating]]) -> torch.Tensor:
    return torch.from_numpy(np.asarray(x, dtype=np.float32))
