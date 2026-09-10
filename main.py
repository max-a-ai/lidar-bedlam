"""Train the selective fusion model.

    torchrun --nproc_per_node 4 main.py --config configs/main_mixed.yaml \\
        --wandb-project lidar-bedlam --wandb-name main-mixed-000 \\
        --set optim.max_steps=17000

The run directory is ``<checkpoint_root>/<wandb-name>``; an existing
``last.pt`` there is resumed automatically, so re-launching with the same
name continues the run (that is what the Slurm chain does).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lidar_bedlam.train.config import load_config
from lidar_bedlam.train.loop import Trainer, new_run_name, seed_everything


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--wandb-project", required=True)
    ap.add_argument(
        "--wandb-name",
        required=True,
        help="run name; 'auto' enumerates <experiment>-NNN",
    )
    ap.add_argument("--wandb-entity", default=None)
    ap.add_argument(
        "--wandb-mode",
        default=None,
        choices=["online", "offline", "disabled"],
    )
    ap.add_argument("--set", nargs="*", default=[], help="section.key=value")
    ap.add_argument("--checkpoint-root", type=Path, default=None)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    cfg = load_config(args.config, args.set)
    cfg.wandb_project = args.wandb_project
    if args.wandb_entity:
        cfg.wandb_entity = args.wandb_entity
    if args.wandb_mode:
        cfg.wandb_mode = args.wandb_mode
    if args.checkpoint_root:
        cfg.checkpoint_root = str(args.checkpoint_root)
    root = Path(cfg.checkpoint_root)
    root.mkdir(parents=True, exist_ok=True)
    name = args.wandb_name
    if name == "auto":
        name = new_run_name(root, cfg.experiment)
    run_dir = root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg.seed)
    trainer = Trainer(cfg, run_dir, name)
    trainer.fit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
