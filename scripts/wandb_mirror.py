"""Mirror ``outputs/<run>/metrics.jsonl`` files to wandb, live.

Compute nodes have no internet, so the trainer writes every logged row to
``metrics.jsonl`` and this script, running on the login node, pushes new
rows to wandb (online) under the run's own name. Runs are resumed by id,
so restarts of this script or of the job continue the same wandb run.
Offsets are kept in ``outputs/<run>/.mirror_offset``.

    uv run python scripts/wandb_mirror.py [--root outputs] [--interval 120]
    uv run python scripts/wandb_mirror.py --once      # one pass, then exit
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import wandb


def _run_config(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "config.json"
    if not path.exists():
        return {}
    data: dict[str, Any] = json.loads(path.read_text())
    return data


def _open_run(run_dir: Path, cfg: dict[str, Any]) -> Any:
    run = wandb.init(
        entity=cfg.get("wandb_entity", "erik_hm"),
        project=cfg.get("wandb_project", "lidar-bedlam"),
        name=run_dir.name,
        id=run_dir.name,
        resume="allow",
        config=cfg,
        dir=str(run_dir),
        reinit="create_new",
    )
    # epochs on the x axis of every chart; the raw step stays available
    run.define_metric("train/epoch")
    run.define_metric("*", step_metric="train/epoch")
    return run


def mirror_once(root: Path, open_runs: dict[str, Any]) -> int:
    """Push the new rows of every run under ``root``; returns rows sent."""
    sent = 0
    for metrics in sorted(root.glob("*/metrics.jsonl")):
        run_dir = metrics.parent
        offset_file = run_dir / ".mirror_offset"
        offset = int(offset_file.read_text()) if offset_file.exists() else 0
        size = metrics.stat().st_size
        done = (run_dir / "DONE").exists()
        if size <= offset and not done:
            continue
        if run_dir.name not in open_runs:
            open_runs[run_dir.name] = _open_run(run_dir, _run_config(run_dir))
        run = open_runs[run_dir.name]
        with open(metrics) as fh:
            fh.seek(offset)
            for line in fh:
                if not line.endswith("\n"):
                    break  # partial write, next pass
                row = json.loads(line)
                run.log(row, step=int(row["step"]))
                offset += len(line.encode())
                sent += 1
        offset_file.write_text(str(offset))
        if done:
            run.finish()
            del open_runs[run_dir.name]
            sys.stdout.write(f"{run_dir.name}: finished\n")
    return sent


def main() -> int:
    """Entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("outputs"))
    ap.add_argument("--interval", type=float, default=120.0)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    open_runs: dict[str, Any] = {}
    while True:
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
        run.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
