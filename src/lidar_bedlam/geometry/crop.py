"""Square person crops from a 2D bounding box."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from lidar_bedlam.geometry.camera import PinholeCamera

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class CropSpec:
    """A square crop window in full-image pixel coordinates."""

    x0: float
    y0: float
    side: float
    out_size: int

    @property
    def scale(self) -> float:
        """Resize factor from the crop window to the output image."""
        return self.out_size / self.side

    def to_crop(self, uv: NDArray[np.floating]) -> FloatArray:
        """Map full-image pixels (N, 2) into crop pixels."""
        return (
            np.asarray(uv, dtype=np.float64) - [self.x0, self.y0]
        ) * self.scale

    def camera(self, cam: PinholeCamera) -> PinholeCamera:
        """Intrinsics valid for the cropped image."""
        return cam.cropped(self.x0, self.y0, self.scale, self.out_size)


def square_crop_from_bbox(
    bbox_xyxy: NDArray[np.floating], out_size: int, padding: float = 1.2
) -> CropSpec:
    """Square window centred on the bbox, side = padding * max extent."""
    x0, y0, x1, y1 = (float(v) for v in bbox_xyxy)
    side = max(x1 - x0, y1 - y0) * padding
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    return CropSpec(cx - side / 2.0, cy - side / 2.0, side, out_size)


def bbox_from_points_2d(
    uv: NDArray[np.floating], width: int, height: int
) -> FloatArray | None:
    """xyxy box around finite pixels (N, 2), clipped to the image, or None."""
    p = np.asarray(uv, dtype=np.float64)
    p = p[np.isfinite(p).all(axis=1)]
    if len(p) == 0:
        return None
    x0, y0 = np.clip(p.min(axis=0), 0, [width - 1, height - 1])
    x1, y1 = np.clip(p.max(axis=0), 0, [width - 1, height - 1])
    if x1 <= x0 or y1 <= y0:
        return None
    return np.array([x0, y0, x1, y1], dtype=np.float64)


def crop_image(image: NDArray[np.uint8], spec: CropSpec) -> NDArray[np.uint8]:
    """Crop and resize an RGB image (H, W, 3); outside pixels are black."""
    box = (spec.x0, spec.y0, spec.x0 + spec.side, spec.y0 + spec.side)
    out = Image.fromarray(image).transform(
        (spec.out_size, spec.out_size),
        Image.Transform.EXTENT,
        box,
        resample=Image.Resampling.BILINEAR,
    )
    return np.asarray(out, dtype=np.uint8)


def crop_mask(mask: NDArray[np.bool_], spec: CropSpec) -> NDArray[np.bool_]:
    """Crop and resize a binary mask (H, W) with nearest sampling."""
    box = (spec.x0, spec.y0, spec.x0 + spec.side, spec.y0 + spec.side)
    out = Image.fromarray(mask.astype(np.uint8) * 255).transform(
        (spec.out_size, spec.out_size),
        Image.Transform.EXTENT,
        box,
        resample=Image.Resampling.NEAREST,
    )
    return np.asarray(out) > 0
