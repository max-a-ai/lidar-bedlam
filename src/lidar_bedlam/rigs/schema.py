"""Sensor rigs: where every LiDAR / camera sits relative to a base frame.

A :class:`SensorRig` is a star graph: the base link (vehicle, helmet, ...)
in the middle and one edge per sensor carrying ``base_from_sensor``, the
4x4 pose of the sensor frame expressed in the base frame. Camera frames
keep the dataset's own axis convention; ``optical_from_sensor`` maps them
to OpenCV (x right, y down, z forward) so frustums and projections agree.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from lidar_bedlam.geometry.camera import PinholeCamera
from lidar_bedlam.geometry.rotations import matrix_to_axis_angle

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Sensor:
    """One sensor and its pose in the rig's base frame."""

    name: str
    kind: str  # "lidar" | "camera" | "imu"
    base_from_sensor: FloatArray  # 4x4
    intrinsics: PinholeCamera | None = None
    optical_from_sensor: FloatArray = field(default_factory=lambda: np.eye(3))
    notes: str = ""

    @property
    def position(self) -> FloatArray:
        """Sensor origin in the base frame."""
        return np.asarray(self.base_from_sensor[:3, 3], dtype=np.float64)

    @property
    def rotation(self) -> FloatArray:
        """Sensor axes as columns, in the base frame."""
        return np.asarray(self.base_from_sensor[:3, :3], dtype=np.float64)


@dataclass(frozen=True)
class SensorRig:
    """A named platform with its sensors."""

    dataset: str
    mount: str  # "car", "helmet", "backpack", "infrastructure", ...
    base_frame: str  # e.g. "vehicle (x forward, y left, z up)"
    sensors: tuple[Sensor, ...]
    source: str = ""  # file the calibration came from

    @property
    def title(self) -> str:
        """Figure title: dataset and mount."""
        return f"{self.dataset} ({self.mount})"

    def by_kind(self, kind: str) -> list[Sensor]:
        """Sensors of one kind."""
        return [s for s in self.sensors if s.kind == kind]

    def tree_text(self) -> str:
        """The rig as an indented connection tree with poses."""
        lines = [f"{self.title}", f"  base: {self.base_frame}"]
        if self.source:
            lines.append(f"  source: {self.source}")
        lines.append("  base_link")
        for i, s in enumerate(self.sensors):
            branch = "└─" if i == len(self.sensors) - 1 else "├─"
            aa = matrix_to_axis_angle(s.rotation)
            angle = float(np.rad2deg(np.linalg.norm(aa)))
            x, y, z = s.position
            extra = ""
            if s.intrinsics is not None:
                k = s.intrinsics
                extra = f", {k.width}x{k.height} px, f={k.fx:.0f}"
            lines.append(
                f"    {branch} {s.kind:6s} {s.name:16s} "
                f"t=({x:+.2f}, {y:+.2f}, {z:+.2f}) m, "
                f"rot {angle:5.1f} deg{extra}"
            )
        return "\n".join(lines)


def check_rigid(transform: NDArray[np.floating], name: str) -> None:
    """Raise if a 4x4 matrix is not a proper rigid transform."""
    t = np.asarray(transform, dtype=np.float64)
    r = t[:3, :3]
    if t.shape != (4, 4) or not np.allclose(r @ r.T, np.eye(3), atol=1e-5):
        msg = f"{name}: extrinsic is not a rigid transform"
        raise ValueError(msg)
    if np.linalg.det(r) < 0 or not np.allclose(t[3], [0, 0, 0, 1]):
        msg = f"{name}: extrinsic has a reflection or bad last row"
        raise ValueError(msg)
