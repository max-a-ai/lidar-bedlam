"""Run PromptHMR (Wang et al., CVPR 2025) on our evaluation crops and write
camera-frame SMPL meshes.

The model is promptable: it takes the whole image, a person box and the
camera intrinsics, and regresses SMPL-X pose, shape and a metric
translation. We hand over our 256 px crop as the image, one box covering
it and the crop's true intrinsics, so placement is comparable with the
other image rows. SMPL-X vertices are mapped to SMPL with the
``smplx2smpl.pkl`` matrix that ships with the method.

Runs in the ``phmr_pt2.6`` conda environment created by the method's
``scripts/install.sh``; the checkout is ``third_party/PromptHMR`` and is
made the cwd so its relative asset paths resolve::

    ~/miniconda3/envs/phmr_pt2.6/bin/python \\
        lidar_bedlam/scripts/baselines/run_prompthmr.py \\
        --shards resources/data/generated/real/v1 \\
        --pattern "waymo_val_*.npz" \\
        --out outputs/baselines/prompthmr/waymo_val.npz
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from lidar_bedlam.scripts.baselines.common import (  # noqa: E402
    CROP,
    Predictions,
    Timer,
    list_shards,
    load_shard,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
PHMR = REPO_ROOT / "third_party/PromptHMR"
MODEL_DIR = "data/pretrain/phmr"
SMPLX2SMPL = "data/body_models/smplx2smpl.pkl"


def load_model() -> tuple[Any, Any]:
    """The image model and the SMPL-X to SMPL vertex matrix (6890, 10475)."""
    os.chdir(PHMR)
    sys.path.insert(0, str(PHMR))
    from prompt_hmr import (
        load_model_from_folder,
    )

    model = load_model_from_folder(MODEL_DIR)
    with open(SMPLX2SMPL, "rb") as fh:
        mapping = pickle.load(fh)
    matrix = mapping["matrix"] if isinstance(mapping, dict) else mapping
    return model, np.asarray(matrix, dtype=np.float64)


def run(args: argparse.Namespace) -> None:
    """Predict every record of the matching shards."""
    import torch
    from scipy.spatial.transform import Rotation

    shards = list_shards(args.shards, args.pattern)
    if not shards:
        msg = f"no shards for {args.pattern} in {args.shards}"
        raise SystemExit(msg)
    model, to_smpl = load_model()
    from prompt_hmr.models.inference import (
        prepare_batch,
    )

    box = torch.tensor([[0.0, 0.0, CROP - 1.0, CROP - 1.0]])
    preds = Predictions()
    timer = Timer()
    done = 0
    for shard in shards:
        data = load_shard(shard)
        n = len(data["key"])
        for start in range(0, n, args.batch_size):
            sl = slice(start, min(n, start + args.batch_size))
            inputs = [
                {
                    "image_cv": np.ascontiguousarray(data["image"][i]),
                    "boxes": box,
                    "cam_int": torch.from_numpy(
                        data["intrinsics"][i].astype(np.float32)
                    )[None],
                    "text": None,
                    "masks": None,
                }
                for i in range(sl.start, sl.stop)
            ]
            timer.start()
            with torch.no_grad(), torch.autocast("cuda"):
                batch = prepare_batch(inputs, img_size=args.img_size)
                outputs = model(batch, use_mean_hands=True)
            torch.cuda.synchronize()
            timer.stop(sl.stop - sl.start)
            vx = np.stack(
                [o["vertices"][0].float().cpu().numpy() for o in outputs]
            ).astype(np.float64)  # SMPL-X, camera frame, metres
            transl = np.stack(
                [o["transl"][0].float().cpu().numpy() for o in outputs]
            ).astype(np.float64)
            rot = np.stack(
                [o["rotmat"][0, 0].float().cpu().numpy() for o in outputs]
            ).astype(np.float64)
            betas = np.stack(
                [o["betas"][0].float().cpu().numpy() for o in outputs]
            ).astype(np.float64)
            v = np.einsum("sv,nvk->nsk", to_smpl, vx)
            preds.add(
                data["key"][sl],
                v,
                transl,
                Rotation.from_matrix(rot).as_rotvec(),
                betas,
            )
            done += sl.stop - sl.start
            if args.limit and done >= args.limit:
                break
        sys.stdout.write(
            f"{shard.name}: {timer.samples} done, {timer.rate:.1f} samples/s\n"
        )
        sys.stdout.flush()
        if args.limit and done >= args.limit:
            break
    preds.save(
        args.out,
        method="prompthmr",
        crop="full",
        mirror=0,
        img_size=args.img_size,
        seconds=timer.seconds,
        samples=timer.samples,
    )
    sys.stdout.write(f"wrote {args.out} ({timer.samples} records)\n")


def main(argv: list[str] | None = None) -> int:
    """CLI."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", type=Path, required=True)
    ap.add_argument("--pattern", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--img-size", type=int, default=896)
    ap.add_argument("--limit", type=int, default=0, help="records (0 = all)")
    args = ap.parse_args(argv)
    args.shards = args.shards.resolve()
    args.out = args.out.resolve()
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
