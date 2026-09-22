"""nuScenes pedestrians and cyclists as model samples.

One metadata directory holds every scene of a split, so a
:class:`~lidar_bedlam.data.nusc_schema.Scene` is built per scene token and
the source concatenates them. Reading is the shared nuScenes-schema code.

Differences from the car data, both handled upstream: nuScenes images are
already rectified (no ``camera_distortion`` key, read as zero) and the
cameras are near level, so ``box3d``'s level-camera yaw convention holds.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from lidar_bedlam.data.nusc_schema import NuscSchemaSource, Scene

#: cyclists: in nuScenes the ridden-cycle box contains the rider
NUSC_CYCLIST = ("vehicle.bicycle", "vehicle.motorcycle")

NUSCENES_ROOT = Path("/home/max/nas_drive/publicdatasets/nuscenes")


def nuscenes_scenes(
    root: Path | str = NUSCENES_ROOT,
    version: str = "v1.0-mini",
    limit: int | None = None,
) -> list[Scene]:
    """One :class:`Scene` per nuScenes scene token, cheapest split first."""
    index = Scene(root, version)
    tokens = index.scene_tokens()
    if limit is not None:
        tokens = tokens[:limit]
    return [Scene(root, version, scene_token=t) for t in tokens]


class NuScenesSource(NuscSchemaSource):
    """Annotated pedestrians and cyclists from nuScenes."""

    name = "nuscenes"

    def __init__(
        self,
        scenes: Sequence[Scene] | None = None,
        categories: tuple[str, ...] = ("human",),
        min_points: int = 50,
        min_box_height_px: float = 64.0,
        version: str = "v1.0-mini",
        limit_scenes: int | None = None,
    ) -> None:
        super().__init__(
            list(scenes)
            if scenes is not None
            else nuscenes_scenes(version=version, limit=limit_scenes),
            categories=categories,
            min_points=min_points,
            min_box_height_px=min_box_height_px,
        )
