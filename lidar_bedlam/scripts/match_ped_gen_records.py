"""Match our Waymo pseudo-GT records to the ped_gen pedestrian tracks.

The two label sets share no identifier: our records are keyed by the
Waymo frame timestamp and the *camera* object id, the ped_gen tracks by
their own scene index and instance index (whose ids are base64 Waymo
*laser* ids). They do share the Waymo world frame, so a record is matched
to the instance frame closest to it in world coordinates.

Our world position comes from ``frame_pose_transform @ bb_3d`` in the
extraction labels, theirs from ``obj_to_world`` in each scene's
``instances/instances_info.json``. A match within ``--max-distance`` is
the same person in the same frame; the distance is kept so the notebook
can show it.

    uv run python lidar_bedlam/scripts/match_ped_gen_records.py

Writes ``resources/data/generated/ped_gen/waymo_match.npz`` with, per
matched record, our key and context, their scene, object and frame, the
world distance and the 4x4 world-to-camera transform of our record.
"""

from __future__ import annotations

import argparse
import base64
import json
import pickle
import sys
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from lidar_bedlam.data.waymo import vehicle_to_opencv
from lidar_bedlam.generate.records import Shard
from lidar_bedlam.geometry.camera import invert_se3

FloatArray = NDArray[np.float64]

LABELS = Path("resources/data/waymo_perception/waymo_pose_complete_4")
PED_GEN = Path.home() / "nas_drive/methods/ped_gen/waymo/processed"
SHARDS = Path("resources/data/generated/real/v1_pseudo")
OUT = Path("resources/data/generated/ped_gen/waymo_match.npz")


def shard_keys(shards: Path, pattern: str) -> set[str]:
    """Keys of the accepted pseudo-GT records."""
    keys: set[str] = set()
    for path in sorted(shards.glob(pattern)):
        if ".fit." in path.name:
            continue
        shard = Shard(path)
        accepted = shard.array("has_smpl")
        rows = zip(shard.array("key"), accepted, strict=True)
        keys |= {str(k) for k, ok in rows if ok}
    return keys


def our_index(
    labels: Path, keys: set[str], subsets: tuple[str, ...]
) -> dict[str, list[Any]]:
    """World position and world-to-camera transform of every record."""
    out: dict[str, list[Any]] = {
        "key": [], "context": [], "world": [], "world_to_camera": [],
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
                ) @ invert_se3(to_world)
                out["key"].append(key)
                out["context"].append(context)
                out["world"].append((to_world @ centre)[:3])
                out["world_to_camera"].append(to_camera)
            sys.stdout.write(f"\r{subset}: {len(out['key'])} records")
            sys.stdout.flush()
    sys.stdout.write("\n")
    return out


def their_index(base: Path) -> dict[str, list[Any]]:
    """World position of every pedestrian instance frame with a track."""
    tracks: dict[str, set[str]] = {}
    for path in (base / "dataset_waymo_pretrain").glob("*.json"):
        scene, obj = path.stem.split("_")
        tracks.setdefault(scene, set()).add(obj)
    out: dict[str, list[Any]] = {
        "scene": [], "object": [], "frame": [], "world": [], "laser_id": [],
    }  # fmt: skip
    for scene_dir in sorted(d for d in (base / "training").iterdir()):
        info_path = scene_dir / "instances" / "instances_info.json"
        have = tracks.get(scene_dir.name, set())
        if not info_path.exists() or not have:
            continue
        with info_path.open() as fh:
            info: dict[str, dict[str, Any]] = json.load(fh)
        for obj, entry in info.items():
            if entry["class_name"] != "Pedestrian" or obj not in have:
                continue
            laser = str(
                uuid.UUID(bytes=base64.urlsafe_b64decode(entry["id"] + "=="))
            )
            ann = entry["frame_annotations"]
            for frame, pose in zip(
                ann["frame_idx"], ann["obj_to_world"], strict=True
            ):
                out["scene"].append(scene_dir.name)
                out["object"].append(obj)
                out["frame"].append(int(frame))
                out["world"].append(np.asarray(pose, dtype=np.float64)[:3, 3])
                out["laser_id"].append(laser)
        sys.stdout.write(f"\rscene {scene_dir.name}: {len(out['scene'])}")
        sys.stdout.flush()
    sys.stdout.write("\n")
    return out


def main(argv: list[str] | None = None) -> int:
    """CLI."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", type=Path, default=LABELS)
    ap.add_argument("--ped-gen", type=Path, default=PED_GEN)
    ap.add_argument("--shards", type=Path, default=SHARDS)
    ap.add_argument("--pattern", default="waymo_train_*.npz")
    ap.add_argument("--subsets", nargs="+", default=["3D", "3D_2D"])
    ap.add_argument("--max-distance", type=float, default=0.3)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    keys = shard_keys(args.shards, args.pattern)
    sys.stdout.write(f"{len(keys)} accepted records in the shards\n")
    ours = our_index(args.labels, keys, tuple(args.subsets))
    theirs = their_index(args.ped_gen)
    if not ours["key"] or not theirs["scene"]:
        sys.stderr.write("nothing to match\n")
        return 1
    theirs_world = np.asarray(theirs["world"])
    found = cKDTree(theirs_world).query(np.asarray(ours["world"]), k=1)
    dist = np.asarray(found[0], dtype=np.float64)
    idx = np.asarray(found[1], dtype=np.int64)
    hit = dist <= args.max_distance
    sel = np.nonzero(hit)[0]
    matched = {
        "key": np.asarray(ours["key"])[sel],
        "context": np.asarray(ours["context"])[sel],
        "world": np.asarray(ours["world"])[sel],
        "world_to_camera": np.asarray(ours["world_to_camera"])[sel],
        "scene": np.asarray(theirs["scene"])[idx[sel]],
        "object": np.asarray(theirs["object"])[idx[sel]],
        "frame": np.asarray(theirs["frame"])[idx[sel]],
        "laser_id": np.asarray(theirs["laser_id"])[idx[sel]],
        "distance_m": dist[sel],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **matched)  # type: ignore[arg-type]
    names = zip(matched["scene"], matched["object"], strict=True)
    tracks = {f"{s}_{o}" for s, o in names}
    sys.stdout.write(
        f"matched {int(hit.sum())} of {len(dist)} records to "
        f"{len(tracks)} ped_gen tracks "
        f"(median {np.median(matched['distance_m']) * 100:.1f} cm), "
        f"wrote {args.out}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
