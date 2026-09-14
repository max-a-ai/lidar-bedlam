"""Run Human3R (CUT3R + Multi-HMR, monocular) on our evaluation crops and
write camera-frame SMPL meshes.

Runs in the ``human3r`` conda environment from the local Human3R checkout
(weights in ``src/human3r.pth``, body models in ``src/models``)::

    ~/miniconda3/envs/human3r/bin/python \\
        lidar_bedlam/scripts/baselines/run_human3r.py \\
        --shards resources/data/generated/real/v1 \\
        --pattern "waymo_val_*.npz" \\
        --out outputs/baselines/human3r/waymo_val.npz

Every crop is one single-frame sequence (fresh recurrent state), resized
to the model's 512 px input with the crop's intrinsics scaled along, so
the predicted translation is metric in the crop camera. Of the humans the
model finds, the one whose 2D pelvis is nearest the crop centre is taken;
the SMPL-X mesh is mapped to SMPL with the checkout's ``smplx2smpl``
matrix. Crops without a detection are skipped (counted in the meta).
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

REPO = Path.home() / "Documents/Human3R"
WEIGHTS = "src/human3r.pth"
INPUT = 512  # model image size


def load_model() -> tuple[Any, Any, Any, Any]:
    """Model, SMPL-X layer, SMPL-X -> SMPL matrix and SMPL model."""
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    from add_ckpt_path import add_path_to_dust3r

    add_path_to_dust3r(WEIGHTS)
    import smplx
    import torch
    from dust3r.utils.smpl_layer import SMPL_Layer
    from src.dust3r.model import ARCroco3DStereo

    model = ARCroco3DStereo.from_pretrained(WEIGHTS).cuda().eval()
    layer = SMPL_Layer(
        type="smplx",
        gender="neutral",
        num_betas=10,
        kid=False,
        person_center="head",
    ).cuda()
    with open("src/models/smplx2smpl.pkl", "rb") as fh:
        x2s = torch.from_numpy(
            pickle.load(fh)["matrix"].astype(np.float32)
        ).cuda()
    smpl = smplx.create("src/models", "smpl", gender="neutral").cuda()
    return model, layer, x2s, smpl


def make_view(
    image: np.ndarray[Any, Any], k: np.ndarray[Any, Any], res: int
) -> dict[str, Any]:
    """One view dict as Human3R's evaluation builds it (single frame)."""
    import torch
    from dust3r.utils.geometry import resize_camera_intrinsics
    from dust3r.utils.image import ImgNorm, pad_image
    from PIL import Image

    s = INPUT / CROP
    pil = Image.fromarray(image).resize(
        (INPUT, INPUT), Image.Resampling.BICUBIC
    )
    img = ImgNorm(pil).unsqueeze(0)  # (1, 3, 512, 512) in [-1, 1]
    k2 = k.copy()
    k2[:2] *= s
    kt = torch.from_numpy(k2).float().unsqueeze(0)
    return {
        "img": img,
        "true_shape": torch.tensor([[INPUT, INPUT]], dtype=torch.int32),
        "camera_intrinsics": kt,
        "img_mask": torch.tensor(True).unsqueeze(0),
        "ray_mask": torch.tensor(False).unsqueeze(0),
        "update": torch.tensor(True).unsqueeze(0),
        "reset": torch.tensor(False).unsqueeze(0),
        "idx": 0,
        "instance": "0",
        "img_mhmr": pad_image(img, res),
        "K_mhmr": resize_camera_intrinsics(kt, INPUT, INPUT, res),
    }


def run(args: argparse.Namespace) -> None:
    """Predict every record of the matching shards."""
    shards = list_shards(args.shards, args.pattern)
    if not shards:
        msg = f"no shards for {args.pattern} in {args.shards}"
        raise SystemExit(msg)
    model, layer, x2s, smpl = load_model()
    import roma
    import torch
    from dust3r.inference import inference_recurrent_lighter

    res = int(model.mhmr_img_res)
    with torch.no_grad():
        rest_pelvis = smpl().joints[0, 0].double().cpu().numpy()
        regressor = smpl.J_regressor.double().cpu().numpy()
    preds = Predictions()
    timer = Timer()
    missing = 0
    for shard in shards:
        data = load_shard(shard)
        for i in range(len(data["key"])):
            view = make_view(data["image"][i], data["intrinsics"][i], res)
            timer.start()
            with torch.no_grad():
                outputs, _ = inference_recurrent_lighter(
                    [view], model, "cuda", verbose=False
                )
                pred = outputs["pred"][0]
                n = int(pred.get("smpl_shape", torch.empty(1, 0, 10)).shape[1])
                if n == 0:
                    missing += 1
                    timer.stop(1)
                    continue
                loc = pred["smpl_loc"][0].float()  # (n, 2) in the mhmr image
                j = int(((loc - res / 2.0) ** 2).sum(-1).argmin())
                rotvec = roma.rotmat_to_rotvec(
                    pred["smpl_rotmat"][0][j]
                ).cuda()
                expr = pred.get("smpl_expression", [None])[0]
                out = layer(
                    rotvec.unsqueeze(0),
                    pred["smpl_shape"][0][j].unsqueeze(0).cuda(),
                    pred["smpl_transl"][0][j].unsqueeze(0).cuda(),
                    None,
                    None,
                    K=view["camera_intrinsics"].cuda(),
                    expression=None
                    if expr is None
                    else expr[j].unsqueeze(0).cuda(),
                )
                verts = (x2s @ out["smpl_v3d"][0]).double().cpu().numpy()
            torch.cuda.synchronize()
            timer.stop(1)
            joints = regressor @ verts
            preds.add(
                data["key"][i : i + 1],
                verts[None],
                (joints[0] - rest_pelvis)[None],
                rotvec[0].double().cpu().numpy()[None],
                np.zeros((1, 10)),
                n_humans=np.array([n]),
            )
        sys.stdout.write(
            f"{shard.name}: {timer.samples} done, {missing} without "
            f"detection, {timer.rate:.1f} samples/s\n"
        )
        sys.stdout.flush()
    preds.save(
        args.out,
        method="human3r",
        missing=missing,
        seconds=timer.seconds,
        samples=timer.samples,
    )
    sys.stdout.write(
        f"wrote {args.out}: {len(preds.key)} samples ({missing} without "
        f"detection), {timer.seconds:.1f} s model time ({timer.rate:.1f}/s)\n"
    )


def main(argv: list[str] | None = None) -> int:
    """CLI."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", type=Path, required=True)
    ap.add_argument("--pattern", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    args.shards = args.shards.resolve()
    args.out = args.out.resolve()
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
