"""Pinhole camera model and rigid transforms (OpenCV convention).

Camera frame: x right, y down, z forward. Pixel (u, v) = (0, 0) is the
centre of the top-left pixel. Optional Brown-Conrady distortion
(k1, k2, p1, p2, k3) is applied in :meth:`PinholeCamera.project`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PinholeCamera:
    """Intrinsics of a pinhole camera."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    k3: float = 0.0

    @classmethod
    def from_hfov(
        cls, hfov_deg: float, width: int, height: int
    ) -> PinholeCamera:
        """Square-pixel camera from a horizontal field of view in degrees."""
        f = (width / 2.0) / np.tan(np.deg2rad(hfov_deg) / 2.0)
        return cls(f, f, width / 2.0 - 0.5, height / 2.0 - 0.5, width, height)

    @property
    def matrix(self) -> FloatArray:
        """3x3 intrinsic matrix K."""
        return np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    @property
    def has_distortion(self) -> bool:
        """True if any distortion coefficient is non-zero."""
        return any(
            c != 0.0 for c in (self.k1, self.k2, self.p1, self.p2, self.k3)
        )

    def _distort(
        self, x: FloatArray, y: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        r2 = x * x + y * y
        radial = 1.0 + self.k1 * r2 + self.k2 * r2**2 + self.k3 * r2**3
        xd = x * radial + 2 * self.p1 * x * y + self.p2 * (r2 + 2 * x * x)
        yd = y * radial + self.p1 * (r2 + 2 * y * y) + 2 * self.p2 * x * y
        return xd, yd

    def project(self, points: NDArray[np.floating]) -> FloatArray:
        """Project camera-frame points (N, 3) to pixels (N, 2).

        Points with z <= 0 are mapped to NaN.
        """
        p = np.asarray(points, dtype=np.float64)
        z = p[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            x = p[:, 0] / z
            y = p[:, 1] / z
        if self.has_distortion:
            x, y = self._distort(x, y)
        uv = np.stack([self.fx * x + self.cx, self.fy * y + self.cy], axis=-1)
        uv[z <= 0] = np.nan
        return uv

    def unproject_depth(
        self,
        depth: NDArray[np.floating],
        valid: NDArray[np.bool_] | None = None,
    ) -> FloatArray:
        """Back-project a planar z-depth map (H, W) to points (N, 3).

        ``valid`` selects the pixels to keep (row-major order). Depth must be
        the distance along the optical axis, in the unit of the output.
        Distortion is ignored (rendered depth maps are undistorted).
        """
        h, w = depth.shape
        v, u = np.mgrid[0:h, 0:w]
        z = np.asarray(depth, dtype=np.float64)
        x = (u - self.cx) / self.fx * z
        y = (v - self.cy) / self.fy * z
        pts = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        if valid is None:
            return pts
        return pts[valid.reshape(-1)]

    def cropped(
        self, x0: float, y0: float, scale: float, size: int
    ) -> PinholeCamera:
        """Intrinsics after cropping at (x0, y0) and resizing by ``scale``.

        Distortion coefficients are dropped: they are defined on the
        original image and do not transfer to the crop.
        """
        return PinholeCamera(
            fx=self.fx * scale,
            fy=self.fy * scale,
            cx=(self.cx - x0) * scale,
            cy=(self.cy - y0) * scale,
            width=size,
            height=size,
        )


def se3(
    rotation: NDArray[np.floating], translation: NDArray[np.floating]
) -> FloatArray:
    """Build a 4x4 rigid transform from R (3, 3) and t (3,)."""
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = rotation
    t[:3, 3] = translation
    return t


def invert_se3(transform: NDArray[np.floating]) -> FloatArray:
    """Invert a rigid 4x4 transform without a general matrix inverse."""
    r = np.asarray(transform, dtype=np.float64)[:3, :3]
    t = np.asarray(transform, dtype=np.float64)[:3, 3]
    return se3(r.T, -r.T @ t)


def transform_points(
    transform: NDArray[np.floating], points: NDArray[np.floating]
) -> FloatArray:
    """Apply a 4x4 transform to points (N, 3)."""
    p = np.asarray(points, dtype=np.float64)
    r = np.asarray(transform, dtype=np.float64)[:3, :3]
    t = np.asarray(transform, dtype=np.float64)[:3, 3]
    return np.asarray(p @ r.T + t, dtype=np.float64)


# Waymo camera frame (x forward, y left, z up) -> OpenCV (x right, y down,
# z forward). Rows are the OpenCV axes expressed in the Waymo frame.
WAYMO_CAM_TO_OPENCV: FloatArray = np.array(
    [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]], dtype=np.float64
)

# Up axis of the OpenCV camera frame is -y; boxes rotate about index 1.
CAMERA_UP_AXIS = 1
