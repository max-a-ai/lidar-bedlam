"""Rotation conversions (numpy). Convention: axis-angle <-> 3x3 matrices."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

FloatArray = NDArray[np.float64]


def axis_angle_to_matrix(aa: NDArray[np.floating]) -> FloatArray:
    """Convert axis-angle vectors (..., 3) to rotation matrices (..., 3, 3)."""
    flat = np.asarray(aa, dtype=np.float64).reshape(-1, 3)
    mats = Rotation.from_rotvec(flat).as_matrix()
    return np.asarray(mats, dtype=np.float64).reshape(*aa.shape[:-1], 3, 3)


def matrix_to_axis_angle(mat: NDArray[np.floating]) -> FloatArray:
    """Convert rotation matrices (..., 3, 3) to axis-angle vectors (..., 3)."""
    flat = np.asarray(mat, dtype=np.float64).reshape(-1, 3, 3)
    vecs = Rotation.from_matrix(flat).as_rotvec()
    return np.asarray(vecs, dtype=np.float64).reshape(*mat.shape[:-2], 3)


def euler_to_matrix(seq: str, angles_deg: NDArray[np.floating]) -> FloatArray:
    """Rotation matrix from Euler angles in degrees (scipy ``seq`` string)."""
    return np.asarray(
        Rotation.from_euler(seq, angles_deg, degrees=True).as_matrix(),
        dtype=np.float64,
    )


def yaw_from_matrix(mat: NDArray[np.floating], up_axis: int) -> float:
    """Heading angle (rad) of a rotation around ``up_axis`` (0=x, 1=y, 2=z).

    The heading is the angle of the rotated forward axis projected onto the
    plane orthogonal to ``up_axis``, measured from the first in-plane axis.
    """
    axes = [i for i in range(3) if i != up_axis]
    forward = np.asarray(mat, dtype=np.float64)[:, axes[0]]
    return float(np.arctan2(forward[axes[1]], forward[axes[0]]))
