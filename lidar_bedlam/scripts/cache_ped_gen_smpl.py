"""Cache the ped_gen Waymo SMPL tracks as one compact npz.

The pseudo ground truth of the pedestrian-generation project lives in
about 400 GiB of per-track JSON on the NAS (one 4-184 MB file per
pedestrian track; every frame carries its full 6890-vertex mesh), which
reads at roughly 7 MB/s over sshfs and is far too slow for a notebook.

The meshes are redundant: a frame's ``vertices`` are exactly
``SMPL(body_pose, betas)`` centred, rotated by ``ped_orientation`` and
moved to the mesh centroid (checked to 0.0 mm). This script samples
tracks, keeps a few frames of each, and stores only the parameters, the
centroid, the joints and a few reference vertices, so a notebook can
rebuild the meshes locally and verify the reconstruction.

    uv run python lidar_bedlam/scripts/cache_ped_gen_smpl.py \\
        --tracks 90 --frames 3 --workers 4

With ``--match`` it caches exactly the frames that
``match_ped_gen_records.py`` paired with our own pseudo-GT records, and
stores their meshes in the camera frame of our record, so both label sets
can be drawn on the same LiDAR returns. Their track frame is the ego
frame of the track's first frame with the axes permuted
(``ego x, y, z = their z, x, y``); the ego trajectory in the file is
compared against the scene's ego poses and the residual is stored with
every sample.

    uv run python lidar_bedlam/scripts/cache_ped_gen_smpl.py \\
        --match resources/data/generated/ped_gen/waymo_match.npz \\
        --workers 4
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

SRC = Path.home() / (
    "nas_drive/methods/ped_gen/waymo/processed/dataset_waymo_pretrain"
)
OUT = Path("resources/data/generated/ped_gen/waymo_smpl_cache.npz")
MATCH_OUT = Path("resources/data/generated/ped_gen/waymo_smpl_matched.npz")
SCENES = Path.home() / "nas_drive/methods/ped_gen/waymo/processed/training"
# their track frame -> ego frame of the track's first frame
AXES = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
EGO_CHECK_FRAMES = 12
# vertices kept per frame so the notebook can check its reconstruction
REF_IDX = np.linspace(0, 6889, 64).astype(np.int64)


def _frames(total: int, want: int) -> list[int]:
    """``want`` frame indices spread over a track of ``total`` frames."""
    if total <= want:
        return list(range(total))
    return [int(round(x)) for x in np.linspace(0, total - 1, want)]


def read_track(path: Path, want: int) -> list[dict[str, Any]]:
    """Samples of one track file (everything the cache stores)."""
    with path.open() as fh:
        track = json.load(fh)
    verts = track["vertices"]
    out: list[dict[str, Any]] = []
    for t in _frames(len(verts), want):
        v = np.asarray(verts[t], dtype=np.float64)
        out.append(
            {
                "track": path.stem,
                "scene": str(track["scene_id"]),
                "object": str(track["object_id"]),
                "frame": int(track["frame_of_existence"][t]),
                "frame_index": t,
                "n_frames": len(verts),
                "body_pose": np.asarray(track["body_pose"][t], np.float32),
                "betas": np.asarray(track["betas"][t], np.float32),
                "orient": np.asarray(track["ped_orientation"][t], np.float32),
                "centroid": v.mean(0).astype(np.float32),
                "verts_ref": v[REF_IDX].astype(np.float32),
                "joints": np.asarray(track["motion"][t], np.float32),
                "ped_traj": np.asarray(track["ped_trajectory"][t], np.float32),
                "ego_pos": np.asarray(track["ego_motion"][t], np.float32),
                "description": str(track["description"][0].split("#")[0]),
            }
        )
    return out


def ego_pose(scene: str, frame: int, scenes: Path) -> NDArray[np.float64]:
    """4x4 ego-to-world pose of one scene frame."""
    text = (scenes / scene / "ego_pose" / f"{frame:03d}.txt").read_text()
    rows = [
        [float(x) for x in line.split()]
        for line in text.splitlines()
        if line.strip()
    ]
    return np.asarray(rows, dtype=np.float64).reshape(4, 4)


def ego_residual(
    track: dict[str, Any], scene: str, scenes: Path
) -> tuple[NDArray[np.float64], float]:
    """Ego pose of the track's first frame, and the axis-change residual.

    The residual is the largest disagreement between the ego trajectory
    stored in the track and the scene's ego poses, in metres. It is 0 for
    a track whose frame really is the permuted ego frame.
    """
    frames = list(track["frame_of_existence"])
    first = ego_pose(scene, frames[0], scenes)
    picks = np.unique(
        np.linspace(0, len(frames) - 1, EGO_CHECK_FRAMES).astype(int)
    )
    inv = np.linalg.inv(first)
    theirs = np.asarray([track["ego_motion"][i] for i in picks])
    wanted = np.asarray(
        [
            (inv @ np.append(ego_pose(scene, frames[i], scenes)[:3, 3], 1.0))[
                :3
            ]
            for i in picks
        ]
    )
    residual = float(np.abs((AXES @ theirs.T).T - wanted).max())
    return first, residual


def read_matched(
    path: Path, scene: str, wants: list[dict[str, Any]], scenes: Path
) -> list[dict[str, Any]]:
    """Cache entries for the matched frames of one track.

    Their mesh is expressed in the camera frame of our matching record:
    a vertex is ``(rest - mean) @ orient.T + centroid`` exactly as for our
    own records, so the notebook draws both the same way.
    """
    with path.open() as fh:
        track = json.load(fh)
    frames = list(track["frame_of_existence"])
    first, residual = ego_residual(track, scene, scenes)
    out: list[dict[str, Any]] = []
    for want in wants:
        if want["frame"] not in frames:
            continue
        t = frames.index(want["frame"])
        to_camera = np.asarray(want["world_to_camera"], dtype=np.float64)
        rot = to_camera[:3, :3] @ first[:3, :3] @ AXES
        shift = to_camera[:3, :3] @ first[:3, 3] + to_camera[:3, 3]
        verts = np.asarray(track["vertices"][t], dtype=np.float64)
        joints = np.asarray(track["motion"][t], dtype=np.float64)
        out.append(
            {
                "key": str(want["key"]),
                "track": path.stem,
                "scene": scene,
                "object": str(track["object_id"]),
                "frame": int(want["frame"]),
                "frame_index": t,
                "n_frames": len(frames),
                "body_pose": np.asarray(track["body_pose"][t], np.float32),
                "betas": np.asarray(track["betas"][t], np.float32),
                "orient": (
                    rot @ np.asarray(track["ped_orientation"][t], np.float64)
                ).astype(np.float32),
                "centroid": (rot @ verts.mean(0) + shift).astype(np.float32),
                "verts_ref": (verts[REF_IDX] @ rot.T + shift).astype(
                    np.float32
                ),
                "joints": (joints @ rot.T + shift).astype(np.float32),
                "distance_m": np.float32(want["distance_m"]),
                "ego_residual_m": np.float32(residual),
                "description": str(track["description"][0].split("#")[0]),
            }
        )
    return out


def _matched_worker(
    args: tuple[Path, str, list[dict[str, Any]], Path],
) -> list[dict[str, Any]]:
    """``read_matched`` for the process pool (never raises)."""
    path, scene, wants, scenes = args
    try:
        return read_matched(path, scene, wants, scenes)
    except Exception as exc:  # noqa: BLE001 - one bad track must not stop
        sys.stderr.write(f"{path.name}: {type(exc).__name__}: {exc}\n")
        return []


def matched_jobs(
    match: Path, src: Path, scenes: Path
) -> list[tuple[Path, str, list[dict[str, Any]], Path]]:
    """One job per matched track, carrying the frames wanted from it."""
    m = np.load(match)
    jobs: dict[str, list[dict[str, Any]]] = {}
    for i in range(len(m["key"])):
        name = f"{m['scene'][i]}_{m['object'][i]}"
        jobs.setdefault(name, []).append(
            {
                "key": str(m["key"][i]),
                "frame": int(m["frame"][i]),
                "world_to_camera": m["world_to_camera"][i],
                "distance_m": float(m["distance_m"][i]),
            }
        )
    return [
        (src / f"{name}.json", name.split("_")[0], wants, scenes)
        for name, wants in sorted(jobs.items())
        if (src / f"{name}.json").exists()
    ]


def _worker(args: tuple[Path, int]) -> list[dict[str, Any]]:
    """``read_track`` for the process pool (never raises)."""
    path, want = args
    try:
        return read_track(path, want)
    except Exception as exc:  # noqa: BLE001 - one bad track must not stop
        sys.stderr.write(f"{path.name}: {type(exc).__name__}: {exc}\n")
        return []


def stack(samples: list[dict[str, Any]]) -> dict[str, NDArray[Any]]:
    """Stack the per-sample dicts into arrays, sorted by track."""
    samples = sorted(samples, key=lambda s: (s["track"], s["frame_index"]))
    out: dict[str, NDArray[Any]] = {}
    for name in samples[0]:
        out[name] = np.asarray([s[name] for s in samples])
    return out


def run_matched(args: argparse.Namespace) -> int:
    """Cache the frames paired with our records by the matcher."""
    jobs = matched_jobs(args.match, args.src, args.scenes)
    if not jobs:
        sys.stderr.write(f"no matched tracks under {args.src}\n")
        return 1
    out = MATCH_OUT if args.out == OUT else args.out
    samples: list[dict[str, Any]] = []
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for got in pool.map(_matched_worker, jobs):
            samples += got
            done += 1
            sys.stdout.write(
                f"\r{done}/{len(jobs)} tracks, {len(samples)} samples"
            )
            sys.stdout.flush()
    sys.stdout.write("\n")
    if not samples:
        sys.stderr.write("no samples read\n")
        return 1
    arrays = stack(samples)
    arrays["source"] = np.asarray(str(args.src))
    arrays["ref_idx"] = REF_IDX
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **arrays)  # type: ignore[arg-type]
    good = arrays["ego_residual_m"] < 0.05
    sys.stdout.write(
        f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB): {len(samples)} "
        f"frames from {len({s['track'] for s in samples})} tracks, "
        f"{int(good.sum())} with an ego residual under 5 cm\n"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--tracks", type=int, default=90)
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-bytes", type=int, default=1_000_000)
    ap.add_argument("--match", type=Path, default=None)
    ap.add_argument("--scenes", type=Path, default=SCENES)
    args = ap.parse_args(argv)
    if args.match is not None:
        return run_matched(args)
    paths = sorted(
        p
        for p in args.src.glob("*.json")
        if p.stat().st_size >= args.min_bytes
    )
    if not paths:
        sys.stderr.write(f"no track files under {args.src}\n")
        return 1
    rng = np.random.default_rng(args.seed)
    picks = [paths[i] for i in rng.permutation(len(paths))[: args.tracks]]
    samples: list[dict[str, Any]] = []
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for got in pool.map(_worker, [(p, args.frames) for p in picks]):
            samples += got
            done += 1
            sys.stdout.write(
                f"\r{done}/{len(picks)} tracks, {len(samples)} samples"
            )
            sys.stdout.flush()
    sys.stdout.write("\n")
    if not samples:
        sys.stderr.write("no samples read\n")
        return 1
    arrays = stack(samples)
    arrays["source"] = np.asarray(str(args.src))
    arrays["ref_idx"] = REF_IDX
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **arrays)  # type: ignore[arg-type]
    size_mb = args.out.stat().st_size / 1e6
    sys.stdout.write(
        f"wrote {args.out} ({size_mb:.1f} MB): {len(samples)} samples "
        f"from {len({s['track'] for s in samples})} tracks\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
