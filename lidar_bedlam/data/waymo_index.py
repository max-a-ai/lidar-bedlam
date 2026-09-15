"""Index of our Waymo pseudo-GT records for matching other label sets.

Our shards keep only the crop and the camera-frame labels, so anything
that has to be compared against a foreign label set is looked up in the
extraction pickles: the frame timestamp, the person's box centre in the
Waymo vehicle frame and in the world frame, and the transforms that take
either frame into the camera of the record.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from lidar_bedlam.data.waymo import vehicle_to_opencv
from lidar_bedlam.generate.records import Shard
from lidar_bedlam.geometry.camera import invert_se3

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class RecordIndex:
    """One row per record of our shards, in the order they were read."""

    key: NDArray[np.str_]
    context: NDArray[np.str_]
    timestamp: NDArray[np.int64]
    world: FloatArray  # (N, 3) box centre, Waymo world frame
    vehicle: FloatArray  # (N, 3) box centre, Waymo vehicle frame
    world_to_camera: FloatArray  # (N, 4, 4)
    vehicle_to_camera: FloatArray  # (N, 4, 4)

    def __len__(self) -> int:
        return int(len(self.key))


def accepted_keys(
    shards: Path, pattern: str = "waymo_train_*.npz"
) -> set[str]:
    """Keys of the records that carry an accepted pseudo-GT fit."""
    keys: set[str] = set()
    for path in sorted(shards.glob(pattern)):
        if ".fit." in path.name:
            continue
        shard = Shard(path)
        rows = zip(shard.array("key"), shard.array("has_smpl"), strict=True)
        keys |= {str(k) for k, ok in rows if ok}
    return keys


def build(
    labels: Path,
    keys: set[str],
    subsets: tuple[str, ...] = ("3D", "3D_2D"),
    progress: Any = None,
) -> RecordIndex:
    """Read the extraction pickles and index every record in ``keys``."""
    rows: dict[str, list[Any]] = {
        "key": [], "context": [], "timestamp": [], "world": [],
        "vehicle": [], "world_to_camera": [], "vehicle_to_camera": [],
    }  # fmt: skip
    for subset in subsets:
        for pkl in sorted((labels / subset).glob("*_labels.pkl")):
            context = pkl.name[: -len("_labels.pkl")]
            with pkl.open("rb") as fh:
                records: dict[str, dict[str, Any]] = pickle.load(fh)
            for image_id, rec in records.items():
                frame, cam, obj = image_id.split("_", 2)
                key = f"waymo/{subset}/{frame}_{cam}/{obj}"
                if key not in keys:
                    continue
                box = rec["bb_3d"]
                centre = np.array(
                    [box["center_x"], box["center_y"], box["center_z"], 1.0]
                )
                to_world = np.asarray(
                    rec["frame_pose_transform"], dtype=np.float64
                ).reshape(4, 4)
                to_camera = vehicle_to_opencv(
                    np.asarray(rec["extrinsic"], dtype=np.float64)
                )
                rows["key"].append(key)
                rows["context"].append(context)
                rows["timestamp"].append(int(frame))
                rows["world"].append((to_world @ centre)[:3])
                rows["vehicle"].append(centre[:3])
                rows["world_to_camera"].append(
                    to_camera @ invert_se3(to_world)
                )
                rows["vehicle_to_camera"].append(to_camera)
            if progress is not None:
                progress(subset, len(rows["key"]))
    return RecordIndex(
        key=np.asarray(rows["key"]),
        context=np.asarray(rows["context"]),
        timestamp=np.asarray(rows["timestamp"], dtype=np.int64),
        world=np.asarray(rows["world"], dtype=np.float64),
        vehicle=np.asarray(rows["vehicle"], dtype=np.float64),
        world_to_camera=np.asarray(rows["world_to_camera"], dtype=np.float64),
        vehicle_to_camera=np.asarray(
            rows["vehicle_to_camera"], dtype=np.float64
        ),
    )
