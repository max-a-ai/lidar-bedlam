"""Frozen ViT-H tokens for every crop of every shard (clean + augmented).

Writes ``<shard>.tokens.npy`` (fp16, (N, 2, 256, 1280)) next to each shard.

Usage::

    uv run python scripts/precompute_tokens.py data/generated/synth/v1 \
        --checkpoint /path/to/tokenhmr_model_latest.ckpt --batch 64
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

from lidar_bedlam.data.shards import ShardDataset
from lidar_bedlam.data.torch_dataset import IMAGENET_MEAN, IMAGENET_STD
from lidar_bedlam.generate.records import Shard
from lidar_bedlam.models.vit import VIT_H, ViT, load_backbone_weights

DEFAULT_CKPT = Path(
    "/home/max/nas_drive/methods/max/data/checkpoints/tokenhmr/"
    "tokenhmr_model_latest/data/checkpoints/tokenhmr_model_latest.ckpt"
)


@torch.no_grad()
def tokens_for(
    vit: ViT,
    images: np.ndarray[tuple[int, ...], np.dtype[np.uint8]],
    batch: int,
    device: str,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float16]]:
    """(N, 256, 1280) fp16 tokens for uint8 crops (N, S, S, 3)."""
    out = []
    for s in range(0, len(images), batch):
        x = images[s : s + batch].astype(np.float32) / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        t = torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2))).to(
            device
        )
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=device == "cuda"
        ):
            tok, _ = vit(t)
        out.append(tok.half().cpu().numpy())
    return np.concatenate(out)


def main(argv: list[str] | None = None) -> int:
    """CLI."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("shard_dirs", nargs="+", type=Path)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vit = ViT(VIT_H)
    missing, unexpected = load_backbone_weights(vit, args.checkpoint)
    if missing or unexpected:
        sys.stderr.write(
            f"backbone keys: missing {len(missing)}, "
            f"unexpected {len(unexpected)}\n"
        )
        return 1
    vit = vit.to(device).eval()
    shards = sorted(
        p
        for d in args.shard_dirs
        for p in d.glob("*.npz")
        if not p.name.endswith("stats.npz")
    )
    t0 = time.time()
    for n, path in enumerate(shards):
        out = ShardDataset.tokens_path(path)
        if out.exists() and not args.overwrite:
            continue
        shard = Shard(path)
        clean = tokens_for(vit, shard.array("image"), args.batch, device)
        aug = tokens_for(vit, shard.array("image_aug"), args.batch, device)
        np.save(out, np.stack([clean, aug], axis=1))
        dt = time.time() - t0
        sys.stdout.write(
            f"{n + 1}/{len(shards)} {path.name}: {len(shard)} records, "
            f"{dt:.0f} s\n"
        )
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
