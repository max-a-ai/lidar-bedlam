"""Person masks for the Waymo crops with SAM 3 (box prompt = GT 2D box).

Runs in the ``sam3`` conda environment, not in this repo's venv::

    ~/miniconda3/envs/sam3/bin/python scripts/sam3_waymo_masks.py \
        --root data/waymo_perception/waymo_pose_complete_4 \
        --out data/generated/waymo_masks --subsets 3D_2D

For every record with an image, one compressed npz
``<out>/<subset>/<image_id>.npz`` with ``mask`` (H, W) bool and ``score``.
Records of the same full image share one image embedding.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def load_records(
    root: Path, subsets: list[str]
) -> dict[tuple[str, str], dict[str, Any]]:
    """(subset, image_id) -> record for every labelled crop with an image."""
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for subset in subsets:
        for pkl in sorted((root / subset).glob("*_labels.pkl")):
            with open(pkl, "rb") as fh:
                labels = pickle.load(fh)
            for image_id, rec in labels.items():
                if (root / subset / "images" / f"{image_id}.jpg").exists():
                    out[(subset, image_id)] = rec
    return out


def main(argv: list[str] | None = None) -> int:
    """CLI."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--subsets", nargs="+", default=["3D_2D"])
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    import sam3
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    bpe = os.path.join(
        os.path.dirname(sam3.__file__),
        "..",
        "assets/bpe_simple_vocab_16e6.txt.gz",
    )
    model = build_sam3_image_model(
        bpe_path=bpe, enable_inst_interactivity=True
    )
    proc = Sam3Processor(model)

    records = load_records(args.root, args.subsets)
    by_image: dict[tuple[str, str], list[str]] = defaultdict(list)
    for subset, image_id in records:
        frame_cam = "_".join(image_id.split("_")[:2])  # timestamp_camera
        by_image[(subset, frame_cam)].append(image_id)
    todo = [(k, v) for k, v in by_image.items()]
    if args.limit:
        todo = todo[: args.limit]
    sys.stdout.write(f"{len(records)} records in {len(todo)} images\n")
    t0 = time.time()
    done = 0
    for n, ((subset, _frame_cam), image_ids) in enumerate(todo):
        pending = [
            i
            for i in image_ids
            if not (args.out / subset / f"{i}.npz").exists()
        ]
        if not pending:
            continue
        img_path = args.root / subset / "images" / f"{pending[0]}.jpg"
        state = proc.set_image(Image.open(img_path).convert("RGB"))
        boxes = []
        for image_id in pending:
            bb = records[(subset, image_id)]["bb_2d"]
            boxes.append(
                [
                    bb["center_x"] - bb["width"] / 2,
                    bb["center_y"] - bb["height"] / 2,
                    bb["center_x"] + bb["width"] / 2,
                    bb["center_y"] + bb["height"] / 2,
                ]
            )
        masks, scores, _ = model.predict_inst(
            state,
            point_coords=None,
            point_labels=None,
            box=np.array(boxes, dtype=np.float32),
            multimask_output=False,
        )
        h, w = np.asarray(masks).shape[-2:]
        masks = np.asarray(masks).reshape(len(pending), -1, h, w)[:, 0]
        scores = np.asarray(scores).reshape(len(pending), -1)[:, 0]
        for image_id, m, s in zip(pending, masks, scores, strict=True):
            (args.out / subset).mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                args.out / subset / f"{image_id}.npz",
                mask=np.asarray(m, dtype=bool),
                score=float(s),
            )
            done += 1
        if n % 50 == 0:
            dt = time.time() - t0
            sys.stdout.write(
                f"{n}/{len(todo)} images, {done} masks, {dt:.0f} s\n"
            )
            sys.stdout.flush()
    sys.stdout.write(f"done: {done} masks in {time.time() - t0:.0f} s\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
