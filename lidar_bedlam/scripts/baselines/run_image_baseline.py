"""Run an image-only SMPL regressor (TokenHMR, HMR2 / 4D Humans, CameraHMR)
on our evaluation crops and write camera-frame meshes.

Runs in the baseline's conda environment (``4D-humans`` works for all
three); the repo of the method is put on ``sys.path`` and made the cwd so
its relative asset paths resolve::

    ~/miniconda3/envs/4D-humans/bin/python \\
        lidar_bedlam/scripts/baselines/run_image_baseline.py \\
        --method tokenhmr --crop tight \\
        --shards resources/data/generated/real/v1 \\
        --pattern "waymo_val_*.npz" \\
        --out outputs/baselines/tokenhmr-tight/waymo_val.npz

``--crop full`` feeds our 256 px crop as is; ``--crop tight`` re-crops an
HMR2-style square box around the labelled 2D joints (what the methods saw
in training). Either way the weak-perspective camera is converted to a
translation with the crop's true intrinsics, so placement is comparable
with our model.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from lidar_bedlam.scripts.baselines.common import (  # noqa: E402
    CROP,
    Predictions,
    Timer,
    crop_patch,
    full_translation,
    keypoint_box,
    list_shards,
    load_shard,
    mirror_back,
    normalise,
    smpl_mirror_map,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
HOME = Path.home()
REPOS = {
    "hmr2": HOME / "Documents/4D-Humans",
    "tokenhmr": HOME / "Documents/tokenHMR",
    "camerahmr": REPO_ROOT / "third_party/CameraHMR",
}
SMPL_NEUTRAL = REPO_ROOT / "resources/data/generated/body_models/smpl"

# predict(img, box_center, box_size, K)
#   -> (cam, global_orient, body_pose, betas)
Predictor = Callable[..., tuple[Any, Any, Any, Any]]


def _enter(repo: Path, subdir: str = "") -> None:
    import os

    os.chdir(repo)
    sys.path.insert(0, str(repo / subdir) if subdir else str(repo))


def load_hmr2(repo: Path) -> Predictor:
    """HMR2.0 (4D Humans) release checkpoint."""
    _enter(repo)
    import torch
    from hmr2.models import (
        DEFAULT_CHECKPOINT,
        load_hmr2,
    )

    model, _ = load_hmr2(DEFAULT_CHECKPOINT)
    model = model.cuda().eval()

    def predict(img: Any, *_: Any) -> tuple[Any, Any, Any, Any]:
        with torch.no_grad():
            out = model({"img": img})
        p = out["pred_smpl_params"]
        return out["pred_cam"], p["global_orient"], p["body_pose"], p["betas"]

    return predict


def load_tokenhmr(repo: Path) -> Predictor:
    """TokenHMR release checkpoint (``data/checkpoints`` of the repo)."""
    _enter(repo, "tokenhmr")
    import torch
    from lib.models import load_tokenhmr

    # the shipped config has ``ckpt_path: null`` entries yacs cannot parse
    cfg_text = Path("data/checkpoints/model_config.yaml").read_text()
    cfg_path = Path(tempfile.mkdtemp()) / "model_config.yaml"
    cfg_path.write_text(
        "\n".join(
            ln for ln in cfg_text.splitlines() if not ln.endswith(": null")
        )
    )
    # the checkout's ``load_pretrained`` hardcodes a NAS copy of the weights;
    # load the identical local file instead
    import lib.models.tokenhmr as tokenhmr_module
    from lib.utils.misc import prepare_statedict

    def load_local(cfg: Any, backbone: Any, head: Any, *_: Any) -> Any:
        state = torch.load(
            "data/checkpoints/tokenhmr_model_latest.ckpt",
            map_location="cpu",
            weights_only=False,
        )["state_dict"]
        prepare_statedict(backbone, state, "backbone")
        prepare_statedict(head, state, "smpl_head")
        return backbone, head

    tokenhmr_module.load_pretrained = load_local
    model, _ = load_tokenhmr(
        checkpoint_path="data/checkpoints/tokenhmr_model_latest.ckpt",
        model_cfg=str(cfg_path),
        is_demo=True,
    )
    model = model.cuda().eval()

    def predict(img: Any, *_: Any) -> tuple[Any, Any, Any, Any]:
        with torch.no_grad():
            out = model({"img": img})
        p = out["pred_smpl_params"]
        return out["pred_cam"], p["global_orient"], p["body_pose"], p["betas"]

    return predict


def load_camerahmr(repo: Path) -> Predictor:
    """CameraHMR (SMPL) checkpoint; intrinsics are given, not estimated."""
    _enter(repo)
    import torch
    from core.camerahmr_model import (
        CameraHMR,
    )
    from core.constants import (
        CHECKPOINT_PATH,
    )

    model = CameraHMR.load_from_checkpoint(
        CHECKPOINT_PATH, strict=False, model_type="smpl"
    )
    model = model.cuda().eval()

    def predict(
        img: Any, box_center: Any, box_size: Any, intrinsics: Any
    ) -> tuple[Any, Any, Any, Any]:
        # the model measures the box from the image centre; passing twice the
        # principal point as the image size makes that the true optical axis
        img_size = torch.stack(
            [2.0 * intrinsics[:, 1, 2], 2.0 * intrinsics[:, 0, 2]], dim=-1
        )
        batch = {
            "img": img,
            "box_center": box_center,
            "box_size": box_size,
            "img_size": img_size,
            "cam_int": intrinsics,
        }
        with torch.no_grad():
            p, cam, _ = model(batch)
        return cam, p["global_orient"], p["body_pose"], p["betas"]

    return predict


LOADERS = {
    "hmr2": load_hmr2,
    "tokenhmr": load_tokenhmr,
    "camerahmr": load_camerahmr,
}


def run(args: argparse.Namespace) -> None:
    """Predict every record of the matching shards."""
    import smplx
    import torch
    from scipy.spatial.transform import Rotation

    shards = list_shards(args.shards, args.pattern)
    if not shards:
        msg = f"no shards for {args.pattern} in {args.shards}"
        raise SystemExit(msg)
    predict = LOADERS[args.method](REPOS[args.method])
    smpl = smplx.SMPLLayer(model_path=str(SMPL_NEUTRAL), num_betas=10).cuda()
    sym = smpl_mirror_map(smpl.v_template.double().cpu().numpy())
    preds = Predictions()
    timer = Timer()
    for shard in shards:
        data = load_shard(shard)
        if args.mirror:  # flip the crop, its joints and the principal point
            data["image"] = np.ascontiguousarray(data["image"][:, :, ::-1])
            data["kp2d"][:, :, 0] = CROP - data["kp2d"][:, :, 0]
            data["intrinsics"][:, 0, 2] = CROP - data["intrinsics"][:, 0, 2]
        n = len(data["key"])
        for start in range(0, n, args.batch_size):
            sl = slice(start, min(n, start + args.batch_size))
            images, centres, sides = [], [], []
            for i in range(sl.start, sl.stop):
                if args.crop == "tight":
                    cx, cy, side = keypoint_box(data["kp2d"][i])
                    images.append(crop_patch(data["image"][i], cx, cy, side))
                else:
                    cx, cy, side = CROP / 2.0, CROP / 2.0, float(CROP)
                    images.append(data["image"][i])
                centres.append((cx, cy))
                sides.append(side)
            img = torch.from_numpy(normalise(np.stack(images))).cuda()
            k = torch.from_numpy(data["intrinsics"][sl]).float().cuda()
            centre = torch.tensor(centres, dtype=torch.float32).cuda()
            side_t = torch.tensor(sides, dtype=torch.float32).cuda()
            timer.start()
            cam, go, bp, betas = predict(img, centre, side_t, k)
            with torch.no_grad():
                verts = smpl(
                    betas=betas.float(),
                    global_orient=go.float().reshape(-1, 1, 3, 3),
                    body_pose=bp.float().reshape(-1, 23, 3, 3),
                ).vertices
            torch.cuda.synchronize()
            timer.stop(sl.stop - sl.start)
            transl = full_translation(
                cam.double().cpu().numpy(),
                np.array(centres, dtype=np.float64),
                np.array(sides, dtype=np.float64),
                data["intrinsics"][sl],
            )
            v = verts.double().cpu().numpy() + transl[:, None, :]
            aa = Rotation.from_matrix(
                go.double().cpu().numpy().reshape(-1, 3, 3)
            ).as_rotvec()
            if args.mirror:
                v, transl, aa = mirror_back(v, transl, aa, sym)
            preds.add(
                data["key"][sl],
                v,
                transl,
                aa,
                betas.double().cpu().numpy(),
                cam=cam.double().cpu().numpy(),
                box_center=np.array(centres, dtype=np.float64),
                box_size=np.array(sides, dtype=np.float64),
            )
        sys.stdout.write(
            f"{shard.name}: {timer.samples} done, {timer.rate:.0f} samples/s\n"
        )
        sys.stdout.flush()
    preds.save(
        args.out,
        method=args.method,
        crop=args.crop,
        mirror=int(args.mirror),
        seconds=timer.seconds,
        samples=timer.samples,
    )
    sys.stdout.write(
        f"wrote {args.out}: {timer.samples} samples, "
        f"{timer.seconds:.1f} s model time ({timer.rate:.0f}/s)\n"
    )


def main(argv: list[str] | None = None) -> int:
    """CLI."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", choices=sorted(LOADERS), required=True)
    ap.add_argument("--crop", choices=["full", "tight"], default="tight")
    ap.add_argument("--shards", type=Path, required=True)
    ap.add_argument("--pattern", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument(
        "--mirror",
        action="store_true",
        help="left/right mirror the crop and un-mirror the mesh",
    )
    args = ap.parse_args(argv)
    args.shards = args.shards.resolve()
    args.out = args.out.resolve()
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
