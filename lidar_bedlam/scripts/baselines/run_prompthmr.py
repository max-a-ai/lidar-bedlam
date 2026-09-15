"""Run PromptHMR (Wang et al., CVPR 2025) on our evaluation crops and write
camera-frame SMPL meshes.

The model is promptable: it takes the whole image, a person box and the
camera intrinsics, and regresses SMPL-X pose, shape and a metric
translation. We hand over our 256 px crop as the image and one box
covering it. The crop's true intrinsics (principal point hundreds of pixels
outside the crop, a 9 degree field of view) are nothing the model saw in
training and give poor depths, and a crop stretched to fill the whole
896 px input is a framing the model never saw either (145 vs 108 mm
root-relative on a 12-record check). So the crop is placed as a
``CANVAS_SIDE`` px window in the middle of a neutral 896 px canvas with the
box around it, under a default camera (focal = canvas size, principal
point at the centre). The depth is then rescaled by the ratio of the true
focal length (in canvas pixels) to the default one, which keeps every
pixel where it is (the weak-perspective route of the other image rows),
and the prediction is rotated into the true camera (the default camera is
the true camera turned towards the crop centre; an exact rotation). SMPL-X
vertices are mapped to SMPL with the ``smplx2smpl.pkl`` matrix that ships
with the method.

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
CANVAS = 896  # the model's input size
CANVAS_SIDE = 320  # the crop's size on the canvas
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
    from PIL import Image
    from prompt_hmr.models.inference import (
        prepare_batch,
    )

    x0 = (CANVAS - CANVAS_SIDE) // 2
    box = torch.tensor(
        [[x0, x0, x0 + CANVAS_SIDE - 1.0, x0 + CANVAS_SIDE - 1.0]],
        dtype=torch.float32,
    )
    k0 = np.array(
        [
            [CANVAS, 0.0, CANVAS / 2.0],
            [0.0, CANVAS, CANVAS / 2.0],
            [0, 0, 1.0],
        ],
        dtype=np.float32,
    )
    e_z = np.array([0.0, 0.0, 1.0])

    def on_canvas(img: Any) -> Any:
        small = np.array(
            Image.fromarray(np.ascontiguousarray(img)).resize(
                (CANVAS_SIDE, CANVAS_SIDE), Image.Resampling.BILINEAR
            )
        )
        can = np.full((CANVAS, CANVAS, 3), 128, np.uint8)
        can[x0 : x0 + CANVAS_SIDE, x0 : x0 + CANVAS_SIDE] = small
        return can

    preds = Predictions()
    timer = Timer()
    done = 0
    for shard in shards:
        data = load_shard(shard)
        n = len(data["key"])
        for start in range(0, n, args.batch_size):
            sl = slice(start, min(n, start + args.batch_size))
            ks = data["intrinsics"][sl].astype(np.float64)
            inputs = [
                {
                    "image_cv": on_canvas(data["image"][i]),
                    "boxes": box,
                    "cam_int": torch.from_numpy(k0)[None],
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
            # default focal -> true focal (in canvas pixels): the depth
            # scales, the pixel stays
            scale = ks[:, 0, 0] * (CANVAS_SIDE / CROP) / CANVAS
            new_t = transl.copy()
            new_t[:, 2] *= scale
            v = v - transl[:, None, :] + new_t[:, None, :]
            transl = new_t
            # default camera -> true camera: rotate e_z onto the ray through
            # the crop centre (true camera frame)
            aa = np.zeros((len(v), 3))
            for j in range(len(v)):
                k = ks[j]
                ray = np.array(
                    [
                        (CROP / 2.0 - k[0, 2]) / k[0, 0],
                        (CROP / 2.0 - k[1, 2]) / k[1, 1],
                        1.0,
                    ]
                )
                r_fix, _ = Rotation.align_vectors(
                    [ray / np.linalg.norm(ray)], [e_z]
                )
                m = r_fix.as_matrix()
                v[j] = v[j] @ m.T
                transl[j] = m @ transl[j]
                aa[j] = Rotation.from_matrix(m @ rot[j]).as_rotvec()
            preds.add(data["key"][sl], v, transl, aa, betas)
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
        camera="crop on a canvas, default focal, depth rescaled, rotated back",
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
