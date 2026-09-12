"""Real-data records (Waymo, SLOPER4D) in the same shard format.

Waymo: person points are the LiDAR returns whose projection falls inside
the SAM 3 mask of the crop (``resources/data/generated/waymo_masks``),
falling back
to all points of the labelled 3D box when no mask exists. SLOPER4D ships
segmented person points. Channel and column ids are unknown for real
scans (-1). One clean and one augmented crop per record, like synthetic
data. Real records have no SMPL (Waymo) or full SMPL (SLOPER4D).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from lidar_bedlam.body.smpl import SmplModel
from lidar_bedlam.data.base import SampleSource, add_smpl_derived
from lidar_bedlam.data.image_augment import ImageAugmentConfig, augment_image
from lidar_bedlam.data.schema import Sample
from lidar_bedlam.generate.records import CROP, Record, Scan, write_shard
from lidar_bedlam.geometry.crop import (
    crop_image,
    crop_mask,
    square_crop_from_bbox,
)

FloatArray = NDArray[np.float64]


def points_in_mask(
    sample: Sample, mask: NDArray[np.bool_]
) -> NDArray[np.float32]:
    """Subset of the sample's points whose projection lies in ``mask``."""
    uv = sample.camera.project(sample.points.astype(np.float64))
    ok = np.isfinite(uv).all(axis=1)
    u = np.rint(uv[:, 0]).astype(int)
    v = np.rint(uv[:, 1]).astype(int)
    h, w = mask.shape
    inb = ok & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    keep = np.zeros(len(uv), dtype=bool)
    keep[inb] = mask[v[inb], u[inb]]
    return np.asarray(sample.points[keep], dtype=np.float32)


class RealRecordBuilder:
    """Turns loader samples into records."""

    def __init__(
        self,
        smpl: SmplModel,
        mask_dir: Path | None = None,
        min_box_h_px: float = 90.0,
        min_box_w_px: float = 35.0,
        min_points: int = 30,
        seed: int = 0,
    ) -> None:
        self.smpl = smpl
        self.mask_dir = mask_dir
        self.min_box_h = min_box_h_px
        self.min_box_w = min_box_w_px
        self.min_points = min_points
        self.rng = np.random.default_rng(seed)
        self.image_aug = ImageAugmentConfig(
            bbox_scale_std=0.0, bbox_shift_std=0.0
        )
        self.stats: dict[str, int] = {
            "samples": 0,
            "rejected_box": 0,
            "rejected_points": 0,
            "mask_used": 0,
            "records": 0,
        }

    def _waymo_mask(self, sample: Sample) -> NDArray[np.bool_] | None:
        if self.mask_dir is None or sample.meta.dataset != "waymo":
            return None
        # key waymo/<subset>/<frame>_<cam>/<obj> -> file <frame>_<cam>_<obj>
        _, subset, frame_cam, obj = sample.meta.key.split("/", 3)
        path = self.mask_dir / subset / f"{frame_cam}_{obj}.npz"
        if not path.exists():
            return None
        mask = np.asarray(np.load(path)["mask"], dtype=bool)
        return mask if mask.shape == sample.image.shape[:2] else None

    def record(self, sample: Sample) -> Record | None:
        """Build one record or None if the sample fails the visibility rule."""
        self.stats["samples"] += 1
        x0, y0, x1, y1 = sample.bbox_xyxy
        if y1 - y0 < self.min_box_h or x1 - x0 < self.min_box_w:
            self.stats["rejected_box"] += 1
            return None
        mask = self._waymo_mask(sample)
        points = sample.points
        if mask is not None and mask.any():
            selected = points_in_mask(sample, mask)
            if len(selected) >= self.min_points:
                points = selected
                self.stats["mask_used"] += 1
                sample.mask = mask
        if len(points) < self.min_points:
            self.stats["rejected_points"] += 1
            return None
        add_smpl_derived(sample, self.smpl)
        spec = square_crop_from_bbox(sample.bbox_xyxy, CROP, 1.2)
        cam = spec.camera(sample.camera)
        img = crop_image(sample.image, spec)
        img_aug, _ = augment_image(
            img, np.array([0, 0, CROP, CROP]), self.image_aug, self.rng
        )
        joints = np.zeros((24, 3))
        valid = np.zeros(24, dtype=bool)
        if sample.joints3d is not None and sample.joints3d_valid is not None:
            n = len(sample.joints3d)
            joints[:n] = sample.joints3d
            valid[:n] = sample.joints3d_valid
        kp = np.zeros((24, 3))
        if sample.kp2d is not None:
            n = min(len(sample.kp2d), 24)
            kp[:n, :2] = spec.to_crop(sample.kp2d[:n, :2])
            kp[:n, 2] = sample.kp2d[:n, 2]
        smpl = sample.smpl
        n_pts = len(points)
        self.stats["records"] += 1
        return Record(
            key=sample.meta.key,
            dataset=sample.meta.dataset,
            image=img,
            image_aug=img_aug,
            mask=crop_mask(sample.mask, spec)
            if sample.mask is not None
            else np.zeros((CROP, CROP), bool),
            intrinsics=cam.matrix,
            crop_origin=np.array([spec.x0, spec.y0, spec.side]),
            has_image=True,
            has_smpl=smpl is not None,
            global_orient=smpl.global_orient if smpl else np.zeros(3),
            body_pose=smpl.body_pose if smpl else np.zeros(69),
            betas=smpl.betas if smpl else np.zeros(10),
            transl=smpl.transl if smpl else np.zeros(3),
            joints3d=joints,
            joints3d_valid=valid,
            joint_convention=sample.joint_convention,
            kp2d=kp,
            box3d=sample.box3d if sample.box3d is not None else np.zeros(7),
            distance_scale=1.0,
            scans={
                "real": Scan(
                    points=np.asarray(points, dtype=np.float32),
                    channel=np.full(n_pts, -1, np.int16),
                    column=np.full(n_pts, -1, np.int16),
                    channels=0,
                    steps=0,
                    sensor_pose=np.eye(4),
                )
            },
        )


def build_real_shards(
    source: SampleSource,
    builder: RealRecordBuilder,
    out_dir: Path,
    prefix: str,
    shard_size: int = 512,
    indices: list[int] | None = None,
) -> dict[str, Any]:
    """Write all (or the given) samples of a source as shards."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = list(range(len(source))) if indices is None else indices
    buffer: list[Record] = []
    shard_id = 0
    for i in ids:
        rec = builder.record(source.load(i))
        if rec is not None:
            buffer.append(rec)
        if len(buffer) >= shard_size:
            write_shard(buffer, out_dir / f"{prefix}_{shard_id:05d}.npz")
            buffer, shard_id = [], shard_id + 1
    if buffer:
        write_shard(buffer, out_dir / f"{prefix}_{shard_id:05d}.npz")
        shard_id += 1
    stats: dict[str, Any] = dict(builder.stats)
    stats["shards"] = shard_id
    stats["source"] = source.name
    (out_dir / f"{prefix}_stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def dump_config(obj: Any, path: Path) -> None:
    """Write a dataclass config as json next to the shards."""
    path.write_text(json.dumps(asdict(obj), indent=2))
