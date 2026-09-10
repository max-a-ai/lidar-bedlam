"""Where the simulated LiDAR sits relative to the (unchanged) camera.

Two sources of placements: random draws in a ball around the camera
(radius uniform in ``[0, r_max]``, direction uniform, small tilt), and the
fixed offsets of a real rig (LiDAR pose expressed in the OpenCV frame of
one of the rig's cameras).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from lidar_bedlam.geometry.camera import (
    WAYMO_CAM_TO_OPENCV,
    invert_se3,
    se3,
)
from lidar_bedlam.lidar.simulate import sensor_pose
from lidar_bedlam.rigs.schema import SensorRig

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class BallPlacement:
    """Random placement: radius uniform in [0, r_max], tilt up to max."""

    r_max_m: float = 1.0
    tilt_max_deg: float = 5.0

    def sample(self, rng: np.random.Generator) -> FloatArray:
        """camera_from_sensor 4x4."""
        radius = rng.uniform(0.0, self.r_max_m)
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        angles = rng.uniform(-self.tilt_max_deg, self.tilt_max_deg, size=3)
        return sensor_pose(direction * radius, *angles)


def rig_lidar_pose(rig: SensorRig, camera: str, lidar: str) -> FloatArray:
    """LiDAR pose in the OpenCV frame of a rig camera (camera_from_sensor).

    The LiDAR keeps its own axes; the simulator only uses its origin and
    orientation relative to the camera, both taken from the calibration.
    """
    cam = next(s for s in rig.sensors if s.name == camera)
    lid = next(s for s in rig.sensors if s.name == lidar)
    optical_from_base = se3(cam.optical_from_sensor, np.zeros(3)) @ invert_se3(
        cam.base_from_sensor
    )
    return np.asarray(
        optical_from_base @ lid.base_from_sensor, dtype=np.float64
    )


def lidar_to_simulator_axes(
    camera_from_lidar: NDArray[np.floating],
) -> FloatArray:
    """Re-express a LiDAR pose in the simulator's sensor axes.

    Real LiDAR frames use x forward, y left, z up; the simulator's sensor
    frame follows the camera (x right, y down, z forward). The fixed axis
    map keeps the physical mounting, so the scan sweeps the same way.
    """
    pose = np.asarray(camera_from_lidar, dtype=np.float64).copy()
    pose[:3, :3] = pose[:3, :3] @ WAYMO_CAM_TO_OPENCV.T
    return pose
