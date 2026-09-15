"""Shared helpers for the baseline runners.

The runners execute inside the baselines' own conda environments, so this
module depends on numpy and PIL only: it reads our evaluation shards, builds
the 256 px input crops the image methods expect, converts weak-perspective
cameras to a translation in the (crop) camera frame and writes predictions
as one npz per split. Scoring happens in our environment
(``lidar_bedlam/scripts/score_baselines.py``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CROP = 256
TIGHT_MARGIN = 1.2  # keypoint box -> square crop side, HMR2-style padding
MIN_SIDE = 64.0


def list_shards(root: Path, pattern: str) -> list[Path]:
    """Shard files matching ``pattern`` (side files skipped)."""
    return sorted(
        p
        for p in root.glob(pattern)
        if not p.name.endswith("stats.npz") and ".fit." not in p.name
    )


def load_shard(path: Path, points: bool = False) -> dict[str, Any]:
    """The arrays the baselines need (images RGB uint8, crop camera K)."""
    with np.load(path) as z:
        out: dict[str, Any] = {
            "key": z["key"],
            "dataset": z["dataset"],
            "image": z["image"],
            "intrinsics": z["intrinsics"].astype(np.float64),
            "kp2d": z["kp2d"].astype(np.float64),
        }
        if points:
            # real scans where recorded; simulated shards (3DPW mesh-LiDAR)
            # carry main_0 / main_1 at the target resolutions: take main_0
            variant = "real"
            if "scan/real/count" not in z.files:
                mains = sorted(
                    v for v in z["scan_variants"] if str(v).startswith("main_")
                )
                variant = (
                    str(mains[0]) if mains else str(z["scan_variants"][0])
                )
            count = z[f"scan/{variant}/count"]
            pts = z[f"scan/{variant}/points"]
            out["points"] = [
                pts[i, : int(count[i])].astype(np.float32)
                for i in range(len(count))
            ]
    return out


def keypoint_box(kp2d: FloatArray) -> tuple[float, float, float]:
    """(cx, cy, side) of an HMR2-style square box around the valid joints."""
    valid = kp2d[:, 2] > 0
    if valid.sum() < 2:
        return CROP / 2.0, CROP / 2.0, float(CROP)
    xy = kp2d[valid, :2]
    lo, hi = xy.min(0), xy.max(0)
    extent = float(max(hi[0] - lo[0], hi[1] - lo[1]))
    side = max(MIN_SIDE, TIGHT_MARGIN * extent)
    c = (lo + hi) / 2.0
    return float(c[0]), float(c[1]), side


def crop_patch(
    image: NDArray[np.uint8], cx: float, cy: float, side: float
) -> NDArray[np.uint8]:
    """Square window of ``side`` px centred at (cx, cy), resampled to 256."""
    from PIL import Image

    x0, y0 = cx - side / 2.0, cy - side / 2.0
    box = (x0, y0, x0 + side, y0 + side)
    im = Image.fromarray(image).transform(
        (CROP, CROP), Image.Transform.EXTENT, box, Image.Resampling.BILINEAR
    )
    return np.asarray(im, dtype=np.uint8)


def normalise(images: NDArray[np.uint8]) -> NDArray[np.float32]:
    """uint8 (N, H, W, 3) RGB -> float32 (N, 3, H, W), ImageNet statistics."""
    x = (images.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(x.transpose(0, 3, 1, 2))


def full_translation(
    cam: FloatArray,
    box_center: FloatArray,
    box_size: FloatArray,
    intrinsics: FloatArray,
) -> FloatArray:
    """Weak-perspective (s, tx, ty) on a square patch -> camera translation.

    Same algebra as HMR2's ``cam_crop_to_full`` but with the true principal
    point instead of the image centre (our crops are not centred).
    """
    s, tx, ty = cam[:, 0], cam[:, 1], cam[:, 2]
    f = intrinsics[:, 0, 0]
    bs = box_size * s + 1e-9
    tz = 2.0 * f / bs
    full_tx = tx + 2.0 * (box_center[:, 0] - intrinsics[:, 0, 2]) / bs
    full_ty = ty + 2.0 * (box_center[:, 1] - intrinsics[:, 1, 2]) / bs
    return np.stack([full_tx, full_ty, tz], axis=-1)


@dataclass
class Predictions:
    """Accumulated outputs of one method on one split."""

    key: list[str] = field(default_factory=list)
    vertices: list[NDArray[np.float16]] = field(default_factory=list)
    transl: list[FloatArray] = field(default_factory=list)
    global_orient: list[FloatArray] = field(default_factory=list)
    betas: list[FloatArray] = field(default_factory=list)
    extra: dict[str, list[Any]] = field(default_factory=dict)

    def add(
        self,
        key: NDArray[np.str_],
        vertices: FloatArray,
        transl: FloatArray,
        global_orient: FloatArray,
        betas: FloatArray,
        **extra: Any,
    ) -> None:
        """Append one batch (vertices in the camera frame, metres)."""
        self.key.extend(str(k) for k in key)
        self.vertices.append(vertices.astype(np.float16))
        self.transl.append(np.asarray(transl, dtype=np.float64))
        self.global_orient.append(np.asarray(global_orient, dtype=np.float64))
        self.betas.append(np.asarray(betas, dtype=np.float64))
        for k, v in extra.items():
            self.extra.setdefault(k, []).append(np.asarray(v))

    def save(self, path: Path, **meta: Any) -> None:
        """Write the npz the scorer reads."""
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, Any] = {
            "key": np.array(self.key),
            "vertices": np.concatenate(self.vertices),
            "transl": np.concatenate(self.transl),
            "global_orient": np.concatenate(self.global_orient),
            "betas": np.concatenate(self.betas),
        }
        for k, v in self.extra.items():
            arrays[k] = np.concatenate(v)
        for k, v in meta.items():
            arrays[f"meta/{k}"] = np.array(v)
        np.savez(path, **arrays)


def smpl_mirror_map(v_template: NDArray[np.floating]) -> NDArray[np.int64]:
    """``sym[j]`` = the template vertex mirror-symmetric to vertex ``j``.

    SMPL's template is left/right symmetric up to a millimetre, so the
    nearest template vertex of the x-reflected template is the partner.
    """
    from scipy.spatial import cKDTree

    mirrored = np.asarray(v_template, dtype=np.float64).copy()
    mirrored[:, 0] *= -1.0
    _, idx = cKDTree(np.asarray(v_template, dtype=np.float64)).query(mirrored)
    return np.asarray(idx, dtype=np.int64)


def mirror_back(
    vertices: FloatArray,
    transl: FloatArray,
    global_orient: FloatArray,
    sym: NDArray[np.int64],
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Undo a left/right mirrored input on predictions (camera frame).

    Reflects x, then relabels vertices with the symmetry map so the
    predicted left side lands on the true left side again; the axis-angle
    orientation of a reflected rotation is (rx, -ry, -rz).
    """
    v = np.asarray(vertices, dtype=np.float64).copy()
    v[..., 0] *= -1.0
    v = v[:, sym]
    t = np.asarray(transl, dtype=np.float64).copy()
    t[:, 0] *= -1.0
    aa = np.asarray(global_orient, dtype=np.float64).copy()
    aa[:, 1:] *= -1.0
    return v, t, aa


class Timer:
    """Wall-clock seconds spent inside the model, and samples seen."""

    def __init__(self) -> None:
        self.seconds = 0.0
        self.samples = 0
        self._t0 = 0.0

    def start(self) -> None:
        self._t0 = time.perf_counter()

    def stop(self, n: int) -> None:
        self.seconds += time.perf_counter() - self._t0
        self.samples += n

    @property
    def rate(self) -> float:
        return self.samples / max(self.seconds, 1e-9)
