"""Evaluate a checkpoint on validation shards.

    python scripts/lidar-bedlam-eval.py \\
        --checkpoint outputs/main-mixed-000/best.pt \\
        --config configs/main_mixed.yaml [--set data.val=...] \\
        --out results/main-mixed-000.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch

from lidar_bedlam.train.config import load_config
from lidar_bedlam.train.loop import Trainer


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    cfg.wandb_mode = "disabled"
    trainer = Trainer(cfg, args.checkpoint.parent, args.checkpoint.parent.name)
    state = torch.load(
        args.checkpoint, map_location=trainer.device, weights_only=False
    )
    trainer.model.load_state_dict(state["model"])
    results = trainer.evaluate(trainer.val_loaders())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": str(args.checkpoint),
        "step": int(state.get("step", -1)),
        "results": {k: asdict(v) for k, v in results.items()},
    }
    args.out.write_text(json.dumps(payload, indent=1))
    for name, r in results.items():
        sys.stdout.write(
            f"{name}: n={r.n} MPJPE {r.mpjpe:.1f} PA {r.pa_mpjpe:.1f} "
            f"transl {r.transl_err_m:.3f} m mAP {r.map:.3f}\n"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
