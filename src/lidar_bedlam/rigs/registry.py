"""All known rigs, keyed by a short slug, loaded from the ``data/`` links."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from lidar_bedlam.rigs.nuscenes import load_nuscenes_rig
from lidar_bedlam.rigs.schema import SensorRig
from lidar_bedlam.rigs.sloper4d import load_sloper4d_rig
from lidar_bedlam.rigs.waymo import load_waymo_rig


def _ava(data: Path) -> SensorRig:
    return load_nuscenes_rig(
        data / "ava",
        dataset="AVA",
        mount="car",
        base_frame="own car export: ego x fwd, y left, z up at the roof LiDAR",
    )


def _fusebike(data: Path) -> SensorRig:
    return load_nuscenes_rig(
        data / "fusebike",
        dataset="FUSE-Bike",
        mount="bicycle",
        base_frame="own bicycle export: ego x fwd, y left, z up at top LiDAR",
    )


LOADERS: dict[str, Callable[[Path], SensorRig]] = {
    "waymo": lambda d: load_waymo_rig(d / "waymo_perception"),
    "nuscenes": lambda d: load_nuscenes_rig(d / "nuscenes"),
    "sloper4d": lambda d: load_sloper4d_rig(
        d / "sloper4d" / "seq008_running_001"
    ),
    "ava": _ava,
    "fusebike": _fusebike,
}


def load_all_rigs(data: Path) -> tuple[dict[str, SensorRig], dict[str, str]]:
    """Load every rig whose data exists; returns rigs and skip reasons."""
    rigs: dict[str, SensorRig] = {}
    skipped: dict[str, str] = {}
    for slug, loader in LOADERS.items():
        try:
            rigs[slug] = loader(data)
        except (FileNotFoundError, OSError, KeyError, ValueError) as exc:
            skipped[slug] = f"{type(exc).__name__}: {exc}"
    return rigs, skipped
