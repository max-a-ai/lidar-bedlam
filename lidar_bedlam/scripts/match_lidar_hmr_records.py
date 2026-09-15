"""Match our Waymo pseudo-GT records to LiDAR-HMR's published fits.

LiDAR-HMR ships its Waymo pseudo ground truth as ``save_data/waymov2``
pickles; every record names the Waymo segment, the frame timestamp and a
laser object id, and holds the fitted mesh in the Waymo *vehicle* frame.
Our records carry the same timestamps but a *camera* object id, so a
record is matched to the fit in the same frame whose mesh centre is
closest to its box centre on the ground plane.

Their mesh is reproduced exactly (0.0 mm) by our SMPL model from the
stored parameters, with the 63-vector body pose padded to 69, so only the
parameters and the transform into our camera frame are cached.

    uv run python lidar_bedlam/scripts/match_lidar_hmr_records.py \\
        --splits train

Writes ``resources/data/generated/lidar_hmr/waymo_match.npz``.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from lidar_bedlam.data import waymo_index

FloatArray = NDArray[np.float64]

LABELS = Path("resources/data/waymo_perception/waymo_pose_complete_4")
HMR = Path("resources/data/external/lidar_hmr_waymov2")
SHARDS = Path("resources/data/generated/real/v1_pseudo")
OUT = Path("resources/data/generated/lidar_hmr/waymo_match.npz")
BODY_POSE = 69


def their_records(path: Path) -> dict[str, list[Any]]:
    """Timestamp, vehicle-frame mesh centre and parameters of one file."""
    with path.open("rb") as fh:
        records: list[dict[str, Any]] = pickle.load(fh)
    out: dict[str, list[Any]] = {
        "timestamp": [], "centre": [], "global_orient": [], "body_pose": [],
        "betas": [], "transl": [], "index": [],
    }  # fmt: skip
    for i, rec in enumerate(records):
        mesh = rec["mesh_dict"]
        pose = np.zeros(BODY_POSE, dtype=np.float64)
        stored = np.asarray(mesh["body_pose"], dtype=np.float64).ravel()
        pose[: len(stored)] = stored
        out["timestamp"].append(int(rec["location_dict"]["time"]))
        out["centre"].append(
            np.asarray(rec["smpl_verts"], dtype=np.float64).mean(0)
        )
        out["global_orient"].append(
            np.asarray(mesh["global_orient"], dtype=np.float64).ravel()
        )
        out["body_pose"].append(pose)
        out["betas"].append(np.asarray(mesh["betas"], dtype=np.float64))
        out["transl"].append(
            np.asarray(mesh["transl"], dtype=np.float64).ravel()
        )
        out["index"].append(i)
    return out


def main(argv: list[str] | None = None) -> int:
    """CLI."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", type=Path, default=LABELS)
    ap.add_argument("--lidar-hmr", type=Path, default=HMR)
    ap.add_argument("--splits", nargs="+", default=["train"])
    ap.add_argument("--shards", type=Path, default=SHARDS)
    ap.add_argument("--subsets", nargs="+", default=["3D", "3D_2D"])
    ap.add_argument("--max-distance", type=float, default=0.5)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)

    keys = waymo_index.accepted_keys(args.shards)
    sys.stdout.write(f"{len(keys)} accepted records in the shards\n")

    seen = {"n": 0}

    def note(subset: str, n: int) -> None:
        if n >= seen["n"] + 500:  # one line per 500 records, not per file
            seen["n"] = n
            sys.stdout.write(f"\r{subset}: {n} records indexed")
            sys.stdout.flush()

    ours = waymo_index.build(
        args.labels, keys, tuple(args.subsets), progress=note
    )
    sys.stdout.write("\n")

    theirs: dict[str, list[Any]] = {}
    split_of: list[str] = []
    for split in args.splits:
        got = their_records(args.lidar_hmr / f"{split}.pkl")
        for name, values in got.items():
            theirs.setdefault(name, []).extend(values)
        split_of += [split] * len(got["index"])
        sys.stdout.write(f"{split}.pkl: {len(got['index'])} fits\n")
    centre = np.asarray(theirs["centre"], dtype=np.float64)
    stamp = np.asarray(theirs["timestamp"], dtype=np.int64)

    # a fit belongs to a record only inside the same frame; within a frame
    # the person is picked on the ground plane, where the box centre and
    # the mesh centre agree (they differ vertically by construction)
    rows: list[tuple[int, int, float]] = []
    for frame in np.unique(ours.timestamp):
        mine = np.nonzero(ours.timestamp == frame)[0]
        cand = np.nonzero(stamp == frame)[0]
        if not len(cand):
            continue
        tree = cKDTree(centre[cand][:, :2])
        found = tree.query(ours.vehicle[mine][:, :2], k=1)
        dist = np.asarray(found[0], dtype=np.float64)
        near = np.asarray(found[1], dtype=np.int64)
        for row, d, j in zip(mine, dist, near, strict=True):
            if d <= args.max_distance:
                rows.append((int(row), int(cand[j]), float(d)))
    if not rows:
        sys.stderr.write("nothing matched\n")
        return 1
    ours_rows = np.asarray([r[0] for r in rows], dtype=np.int64)
    their_rows = np.asarray([r[1] for r in rows], dtype=np.int64)
    matched = {
        "key": ours.key[ours_rows],
        "context": ours.context[ours_rows],
        "vehicle_to_camera": ours.vehicle_to_camera[ours_rows],
        "distance_m": np.asarray([r[2] for r in rows], dtype=np.float64),
        "split": np.asarray(split_of)[their_rows],
        "record": np.asarray(theirs["index"], dtype=np.int64)[their_rows],
        "global_orient": np.asarray(theirs["global_orient"])[their_rows],
        "body_pose": np.asarray(theirs["body_pose"])[their_rows],
        "betas": np.asarray(theirs["betas"])[their_rows],
        "transl": np.asarray(theirs["transl"])[their_rows],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **matched)  # type: ignore[arg-type]
    sys.stdout.write(
        f"matched {len(rows)} of {len(ours)} records "
        f"(median {np.median(matched['distance_m']) * 100:.1f} cm), "
        f"wrote {args.out}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
