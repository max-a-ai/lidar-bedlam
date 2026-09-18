"""Mirror ``outputs/<run>/metrics.jsonl`` files to wandb, live.

Compute nodes have no internet, so the trainer writes every logged row to
``metrics.jsonl`` and this script, running on the login node, pushes new
rows to wandb (online) under the run's own name. Runs are resumed by id,
so restarts of this script or of the job continue the same wandb run.
Offsets are kept in ``outputs/<run>/.mirror_offset``.

    uv run python lidar_bedlam/scripts/wandb_mirror.py [--root outputs]
    uv run python lidar_bedlam/scripts/wandb_mirror.py --once   # one pass
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import wandb

# wandb stages one ``run-<stamp>-<id>`` tree per ``init``. The login-node
# loop re-runs this script with --once every couple of minutes, so every
# pass stages a fresh tree for each active run; left alone they cost a few
# hundred inodes each and exhausted the 500k file quota of /home/hpc.
# Every tree is dropped once its run is finished, and trees a crashed pass
# left behind are swept after this many hours.
STAGING_MAX_AGE_H = 6.0


def _mirror_root() -> Path | None:
    """The staging root, when the loop set one (``WANDB_MIRROR_DIR``)."""
    root = os.environ.get("WANDB_MIRROR_DIR")
    return Path(root).resolve() if root else None


def _drop_staging(staging: Path) -> None:
    """Remove one staging tree, never anything outside the mirror root."""
    root = _mirror_root()
    if root is None:
        return  # staging lives in the run's own directory: leave it alone
    try:
        staging.resolve().relative_to(root)
    except ValueError:
        return
    shutil.rmtree(staging, ignore_errors=True)


def _close(run: Any) -> None:
    """Finish ``run`` and drop the staging tree wandb wrote for it."""
    staging = Path(run.dir).parent  # <root>/wandb/run-<stamp>-<id>/files
    run.finish()  # blocks until the upload completes
    _drop_staging(staging)


def sweep_staging(max_age_h: float = STAGING_MAX_AGE_H) -> int:
    """Drop staging trees a crashed pass never finished; returns the count.

    Trees of the current pass are minutes old and never match.
    """
    root = _mirror_root()
    if root is None:
        return 0
    cutoff = time.time() - max_age_h * 3600.0
    dropped = 0
    for path in sorted((root / "wandb").glob("run-*")):
        if not path.is_dir():
            continue
        try:
            stale = path.stat().st_mtime < cutoff
        except OSError:  # vanished under us
            continue
        if stale:
            shutil.rmtree(path, ignore_errors=True)
            dropped += 1
    return dropped


def _run_config(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "config.json"
    if not path.exists():
        return {}
    data: dict[str, Any] = json.loads(path.read_text())
    return data


def _open_run(run_dir: Path, cfg: dict[str, Any]) -> Any:
    # a run deleted on wandb cannot be resumed under its id: the retry
    # gets a fresh id (kept in .mirror_id so later passes resume it)
    id_file = run_dir / ".mirror_id"
    run_id = id_file.read_text().strip() if id_file.exists() else run_dir.name
    for attempt in range(2):
        try:
            run = wandb.init(
                entity=cfg.get("wandb_entity", "erik_hm"),
                project=cfg.get("wandb_project", "lidar-bedlam"),
                name=run_dir.name,
                id=run_id,
                resume="allow",
                config=cfg,
                # wandb writes hundreds of files per run: keep them out of
                # inode-limited workspaces (WANDB_MIRROR_DIR)
                dir=os.environ.get("WANDB_MIRROR_DIR", str(run_dir)),
                reinit="create_new",
            )
            break
        except wandb.errors.CommError as exc:
            if attempt or "deleted" not in str(exc):
                raise
            run_id = f"{run_dir.name}-m{int(time.time()) % 100000}"
            id_file.write_text(run_id)
            sys.stdout.write(
                f"{run_dir.name}: deleted on wandb, new id {run_id}\n"
            )
    # epochs on the x axis of every chart; the raw step stays available
    run.define_metric("train/epoch")
    run.define_metric("*", step_metric="train/epoch")
    return run


def _with_images(row: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """``image/*`` values are relative PNG paths: upload them as images."""
    out: dict[str, Any] = {}
    for k, v in row.items():
        if k.startswith("image/") and isinstance(v, str):
            path = run_dir / v
            if path.exists():
                out[k] = wandb.Image(str(path))
        else:
            out[k] = v
    return out


def mirror_once(root: Path, open_runs: dict[str, Any]) -> int:
    """Push the new rows of every run under ``root``; returns rows sent."""
    sent = 0
    for metrics in sorted(root.glob("*/metrics.jsonl")):
        run_dir = metrics.parent
        offset_file = run_dir / ".mirror_offset"
        if (run_dir / ".mirror_done").exists():
            continue  # finished and fully mirrored
        try:
            offset = (
                int(offset_file.read_text()) if offset_file.exists() else 0
            )
        except ValueError:  # partially written offset: retry next pass
            continue
        size = metrics.stat().st_size
        done = (run_dir / "DONE").exists()
        if size <= offset and not done:
            continue
        if run_dir.name not in open_runs:
            try:
                open_runs[run_dir.name] = _open_run(
                    run_dir, _run_config(run_dir)
                )
            except Exception as exc:  # one broken run must not stop the pass
                sys.stdout.write(f"{run_dir.name}: skipped ({exc})\n")
                continue
        run = open_runs[run_dir.name]
        with open(metrics) as fh:
            fh.seek(offset)
            for line in fh:
                if not line.endswith("\n"):
                    break  # partial write, next pass
                row = json.loads(line)
                # a resumed run refuses steps it already holds: the trainer
                # writes the validation row (with its figures) a minute
                # after the training row of the same step, so it often
                # arrives one pass later; it then goes to the next free
                # step (the epoch x axis keeps it in place)
                step = max(int(row["step"]), int(run.step))
                run.log(_with_images(row, run_dir), step=step)
                offset += len(line.encode())
                sent += 1
        tmp = offset_file.with_suffix(".tmp")
        tmp.write_text(str(offset))
        tmp.replace(offset_file)
        if done:
            _close(run)
            del open_runs[run_dir.name]
            (run_dir / ".mirror_done").write_text(str(offset))
            sys.stdout.write(f"{run_dir.name}: finished\n")
    return sent


def main() -> int:
    """Entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("outputs"))
    ap.add_argument("--interval", type=float, default=120.0)
    ap.add_argument("--once", action="store_true")
    ap.add_argument(
        "--staging-max-age",
        type=float,
        default=STAGING_MAX_AGE_H,
        help="drop staging trees a crashed pass left behind, in hours",
    )
    args = ap.parse_args()
    open_runs: dict[str, Any] = {}
    while True:
        stale = sweep_staging(args.staging_max_age)
        if stale:
            sys.stdout.write(f"swept {stale} stale staging trees\n")
            sys.stdout.flush()
        sent = mirror_once(args.root, open_runs)
        if sent:
            sys.stdout.write(
                f"{time.strftime('%H:%M:%S')} mirrored {sent} rows "
                f"({len(open_runs)} live)\n"
            )
            sys.stdout.flush()
        if args.once:
            break
        time.sleep(args.interval)
    for run in open_runs.values():
        _close(run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
