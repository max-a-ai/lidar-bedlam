"""Stage 1 of the SAM 3D Body row: MHR meshes for every evaluation crop.

SAM 3D Body (Meta, DINOv3-H+) predicts the MHR body model. Each crop is
handed over as the whole image with one box covering it and the crop's
true intrinsics as the camera. The MHR vertices (18,439 x 3), the camera
translation and the record keys are written per split; stage 2
(``convert_sam3d_to_smpl.py``, MHR environment) fits SMPL to them and
writes the npz the scorer reads.

Runs in the ``sam_3d_body`` conda environment from the local checkout::

    cd ~/Documents/sam-3d-body && PYTHONPATH=. \\
        ~/miniconda3/envs/sam_3d_body/bin/python \\
        <repo>/lidar_bedlam/scripts/baselines/run_sam3d_body.py \\
        --shards <repo>/resources/data/generated/real/v1 \\
        --pattern "waymo_val_*.npz" \\
        --out <repo>/outputs/baselines/sam3d-body/waymo_val.mhr.npz
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from lidar_bedlam.scripts.baselines.common import (  # noqa: E402
    CROP,
    list_shards,
    load_shard,
)

HOME = Path.home()
CHECKPOINT = HOME / "Documents/sam-3d-body/checkpoints/sam-3d-body-dinov3"


def load_estimator() -> Any:
    """SAM 3D Body without detector, segmentor or FOV estimator."""
    import torch
    from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body

    model, cfg = load_sam_3d_body(
        str(CHECKPOINT / "model.ckpt"),
        device=torch.device("cuda"),
        mhr_path=str(CHECKPOINT / "assets/mhr_model.pt"),
    )
    return SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=cfg)


def run(args: argparse.Namespace) -> None:
    """Predict every record of the matching shards."""
    import torch

    shards = list_shards(args.shards, args.pattern)
    if not shards:
        msg = f"no shards for {args.pattern} in {args.shards}"
        raise SystemExit(msg)
    est = load_estimator()
    box = np.array([[0.0, 0.0, CROP - 1.0, CROP - 1.0]])
    keys: list[str] = []
    verts: list[np.ndarray[Any, Any]] = []
    cam_t: list[np.ndarray[Any, Any]] = []
    missing = 0
    t0 = time.time()
    done = 0
    for shard in shards:
        data = load_shard(shard)
        for i in range(len(data["key"])):
            img = np.ascontiguousarray(data["image"][i])
            k = torch.from_numpy(data["intrinsics"][i].astype(np.float32))
            outs = est.process_one_image(
                img, bboxes=box, cam_int=k[None], inference_type="full"
            )
            if not outs:
                missing += 1
                continue
            o = outs[0]
            keys.append(str(data["key"][i]))
            verts.append(np.asarray(o["pred_vertices"], np.float16))
            cam_t.append(np.asarray(o["pred_cam_t"], np.float64))
            done += 1
            if args.limit and done >= args.limit:
                break
        sys.stdout.write(
            f"{shard.name}: {done} done, {done / (time.time() - t0):.1f}"
            " samples/s\n"
        )
        sys.stdout.flush()
        if args.limit and done >= args.limit:
            break
    args.out.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {
        "key": np.array(keys),
        "mhr_vertices": np.stack(verts),
        "cam_t": np.stack(cam_t),
        "faces": np.asarray(est.faces, np.int64),
        "meta/method": np.array("sam3d-body"),
        "meta/seconds": np.array(time.time() - t0),
        "meta/samples": np.array(done),
        "meta/skipped_no_output": np.array(missing),
    }
    np.savez(args.out, **arrays)
    sys.stdout.write(f"wrote {args.out} ({done} records, {missing} skipped)\n")


def main(argv: list[str] | None = None) -> int:
    """CLI."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", type=Path, required=True)
    ap.add_argument("--pattern", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0, help="records (0 = all)")
    args = ap.parse_args(argv)
    args.shards = args.shards.resolve()
    args.out = args.out.resolve()
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
