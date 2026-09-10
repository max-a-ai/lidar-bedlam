"""Fixed-shape sample records and npz shards.

One :class:`Record` is one person in one frame with everything the model
needs, already cropped and in the camera frame. Shards hold ``N`` records
as stacked arrays (``np.savez``), so a training dataset can memory-map one
shard at a time without Python object overhead. LiDAR scans come in named
variants (``main_0``, ``main_1``, ``rig_waymo``, ...) padded to
``MAX_POINTS`` with a per-variant count.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

MAX_POINTS = 2048
CROP = 256
SCAN_FIELDS = ("points", "channel", "column")


@dataclass
class Scan:
    """One simulated (or real) person point cloud with channel/column ids."""

    points: NDArray[np.float32]  # (n, 3) camera frame
    channel: NDArray[np.int16]  # (n,)
    column: NDArray[np.int16]  # (n,)
    channels: int  # sensor channel count
    steps: int  # azimuth steps per revolution
    sensor_pose: NDArray[np.float64]  # camera_from_sensor 4x4


@dataclass
class Record:
    """One person sample."""

    key: str
    dataset: str
    image: NDArray[np.uint8]  # (CROP, CROP, 3)
    image_aug: NDArray[np.uint8]  # augmented copy
    mask: NDArray[np.bool_]  # (CROP, CROP)
    intrinsics: NDArray[np.float64]  # (3, 3) of the crop
    crop_origin: NDArray[np.float64]  # (3,) x0, y0, side in the full image
    has_image: bool
    has_smpl: bool
    global_orient: NDArray[np.float64]  # (3,)
    body_pose: NDArray[np.float64]  # (69,)
    betas: NDArray[np.float64]  # (10,)
    transl: NDArray[np.float64]  # (3,)
    joints3d: NDArray[np.float64]  # (24, 3)
    joints3d_valid: NDArray[np.bool_]  # (24,)
    joint_convention: str
    kp2d: NDArray[np.float64]  # (24, 3) crop pixels + conf
    box3d: NDArray[np.float64]  # (7,)
    distance_scale: float  # virtual-distance factor (1 = as rendered)
    scans: dict[str, Scan] = field(default_factory=dict)


def _pad(
    scan: Scan,
) -> tuple[NDArray[np.float16], NDArray[np.int16], NDArray[np.int16], int]:
    n = min(len(scan.points), MAX_POINTS)
    pts = np.zeros((MAX_POINTS, 3), np.float16)
    ch = np.full(MAX_POINTS, -1, np.int16)
    col = np.full(MAX_POINTS, -1, np.int16)
    pts[:n] = scan.points[:n]
    ch[:n] = scan.channel[:n]
    col[:n] = scan.column[:n]
    return pts, ch, col, n


def write_shard(records: list[Record], path: Path) -> None:
    """Stack records into one compressed npz."""
    if not records:
        msg = "empty shard"
        raise ValueError(msg)
    variants = sorted({v for r in records for v in r.scans})
    arrays: dict[str, Any] = {
        "key": np.array([r.key for r in records]),
        "dataset": np.array([r.dataset for r in records]),
        "image": np.stack([r.image for r in records]),
        "image_aug": np.stack([r.image_aug for r in records]),
        "mask": np.stack([r.mask for r in records]),
        "intrinsics": np.stack([r.intrinsics for r in records]).astype(
            np.float32
        ),
        "crop_origin": np.stack([r.crop_origin for r in records]).astype(
            np.float32
        ),
        "has_image": np.array([r.has_image for r in records]),
        "has_smpl": np.array([r.has_smpl for r in records]),
        "global_orient": np.stack([r.global_orient for r in records]).astype(
            np.float32
        ),
        "body_pose": np.stack([r.body_pose for r in records]).astype(
            np.float32
        ),
        "betas": np.stack([r.betas for r in records]).astype(np.float32),
        "transl": np.stack([r.transl for r in records]).astype(np.float32),
        "joints3d": np.stack([r.joints3d for r in records]).astype(np.float32),
        "joints3d_valid": np.stack([r.joints3d_valid for r in records]),
        "joint_convention": np.array([r.joint_convention for r in records]),
        "kp2d": np.stack([r.kp2d for r in records]).astype(np.float32),
        "box3d": np.stack([r.box3d for r in records]).astype(np.float32),
        "distance_scale": np.array(
            [r.distance_scale for r in records], np.float32
        ),
        "scan_variants": np.array(variants),
    }
    for v in variants:
        padded = [
            _pad(r.scans[v])
            if v in r.scans
            else (
                np.zeros((MAX_POINTS, 3), np.float16),
                np.full(MAX_POINTS, -1, np.int16),
                np.full(MAX_POINTS, -1, np.int16),
                0,
            )
            for r in records
        ]
        arrays[f"scan/{v}/points"] = np.stack([p[0] for p in padded])
        arrays[f"scan/{v}/channel"] = np.stack([p[1] for p in padded])
        arrays[f"scan/{v}/column"] = np.stack([p[2] for p in padded])
        arrays[f"scan/{v}/count"] = np.array([p[3] for p in padded], np.int32)
        arrays[f"scan/{v}/channels"] = np.array(
            [r.scans[v].channels if v in r.scans else 0 for r in records],
            np.int16,
        )
        arrays[f"scan/{v}/steps"] = np.array(
            [r.scans[v].steps if v in r.scans else 0 for r in records],
            np.int16,
        )
        arrays[f"scan/{v}/sensor_pose"] = np.stack(
            [
                r.scans[v].sensor_pose if v in r.scans else np.eye(4)
                for r in records
            ]
        ).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


class Shard:
    """Read access to one shard (lazy npz)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._z = np.load(path, allow_pickle=False)
        self.n = int(len(self._z["key"]))
        self.variants: list[str] = [str(v) for v in self._z["scan_variants"]]

    def __len__(self) -> int:
        return self.n

    def array(self, name: str) -> NDArray[Any]:
        """A whole stacked array (cached by numpy's NpzFile)."""
        return np.asarray(self._z[name])

    def scan(self, index: int, variant: str) -> Scan:
        """One person scan of one variant."""
        n = int(self._z[f"scan/{variant}/count"][index])
        return Scan(
            points=self._z[f"scan/{variant}/points"][index, :n].astype(
                np.float32
            ),
            channel=self._z[f"scan/{variant}/channel"][index, :n],
            column=self._z[f"scan/{variant}/column"][index, :n],
            channels=int(self._z[f"scan/{variant}/channels"][index]),
            steps=int(self._z[f"scan/{variant}/steps"][index]),
            sensor_pose=self._z[f"scan/{variant}/sensor_pose"][index].astype(
                np.float64
            ),
        )
