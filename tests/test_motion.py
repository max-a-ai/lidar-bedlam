"""Ego speed settings and LiDAR rolling shutter."""

from __future__ import annotations

import numpy as np
import pytest

from lidar_bedlam.lidar.motion import (
    STATIONARY,
    WAYMO_URBAN,
    EgoMotion,
    SpeedSetting,
    apply_rolling_shutter,
    column_times,
    sweep_duration_s,
)
from lidar_bedlam.lidar.simulate import AzimuthWindow, LidarScan, ouster

WINDOW = AzimuthWindow(
    hfov_deg=90.0, az_min_deg=-45.0, az_max_deg=45.0, columns=256
)


def _scan(az: np.ndarray) -> LidarScan:
    n = len(az)
    pts = np.tile([0.0, 0.0, 10.0], (n, 1)).astype(np.float32)
    zeros = np.zeros(n, np.int32)
    return LidarScan(
        points=pts,
        channel=zeros,
        column=zeros,
        azimuth_deg=az.astype(np.float32),
        range_m=np.full(n, 10.0, np.float32),
        pixel=np.zeros((n, 2), np.int32),
    )


def test_speed_setting_steps_and_sampling() -> None:
    with pytest.raises(ValueError, match="multiples"):
        SpeedSetting(15.0, 0.0)
    rng = np.random.default_rng(0)
    assert STATIONARY.sample_mps(rng) == 0.0
    assert SpeedSetting(30.0, 0.0).sample_mps(rng) == pytest.approx(30 / 3.6)
    v = np.array([WAYMO_URBAN.sample_mps(rng) for _ in range(2000)]) * 3.6
    assert v.min() >= 0 and v.max() <= 70 + 1e-6 and 15 < v.mean() < 30


def test_column_times_span_the_window() -> None:
    motion = EgoMotion(speed_mps=5.0, spin_hz=10.0, reference="start")
    t = column_times(np.array([-45.0, 0.0, 45.0]), motion, WINDOW)
    assert t[0] == pytest.approx(0.0)
    assert t[2] == pytest.approx(
        sweep_duration_s(WINDOW, ouster("OS1", 64), 10.0)
    )
    assert t[2] == pytest.approx(0.025)  # 90 deg of a 0.1 s revolution
    centred = column_times(
        np.array([0.0]), EgoMotion(5.0, spin_hz=10.0), WINDOW
    )
    assert centred[0] == pytest.approx(0.0)


def test_rolling_shutter_shifts_columns_against_motion() -> None:
    az = np.array([-45.0, 0.0, 45.0])
    motion = EgoMotion(speed_mps=10.0, direction=(0.0, 0.0, 1.0), spin_hz=10.0)
    out = apply_rolling_shutter(_scan(az), motion, WINDOW)
    # centre column unchanged; earlier columns appear further, later nearer
    assert out.points[1, 2] == pytest.approx(10.0)
    assert out.points[0, 2] == pytest.approx(10.0 + 10.0 * 0.0125)
    assert out.points[2, 2] == pytest.approx(10.0 - 10.0 * 0.0125)
    assert apply_rolling_shutter(_scan(az), EgoMotion(), WINDOW) is not None


def test_yaw_rate_rotates_about_up() -> None:
    az = np.array([45.0])
    motion = EgoMotion(speed_mps=0.0, yaw_rate_deg_s=40.0, spin_hz=10.0)
    out = apply_rolling_shutter(_scan(az), motion, WINDOW)
    p = out.points[0]
    assert p[1] == pytest.approx(0.0) and np.linalg.norm(p) == pytest.approx(
        10.0
    )
    assert abs(p[0]) > 0.05  # rotated in the x-z plane
