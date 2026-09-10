"""ShardDataset yields collate-ready items from generated shards."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from lidar_bedlam.data.shards import ShardDataset, ShardDatasetConfig
from lidar_bedlam.generate.records import CROP, Record, Scan, write_shard
from lidar_bedlam.lidar.augment import PointAugmentConfig


def _record(i: int) -> Record:
    rng = np.random.default_rng(i)
    pts = (rng.normal(size=(300, 3)) * 0.3 + [0, 0, 9.0]).astype(np.float32)

    def scan() -> Scan:
        return Scan(
            pts, np.arange(300, dtype=np.int16) % 64,
            np.arange(300, dtype=np.int16), 64, 1024, np.eye(4),
        )  # fmt: skip

    return Record(
        key=f"k{i}", dataset="bedlam",
        image=np.full((CROP, CROP, 3), 100, np.uint8),
        image_aug=np.full((CROP, CROP, 3), 50, np.uint8),
        mask=np.ones((CROP, CROP), bool), intrinsics=np.eye(3),
        crop_origin=np.zeros(3), has_image=True, has_smpl=True,
        global_orient=np.zeros(3), body_pose=np.zeros(69),
        betas=np.zeros(10), transl=np.array([0, 0, 9.0]),
        joints3d=np.zeros((24, 3)), joints3d_valid=np.ones(24, bool),
        joint_convention="smpl24", kp2d=np.zeros((24, 3)),
        box3d=np.zeros(7), distance_scale=1.0,
        scans={"main_0": scan(), "main_1": scan(), "check": scan()},
    )  # fmt: skip


def test_shard_dataset_items_and_tokens(tmp_path: Path) -> None:
    for s in range(2):
        write_shard(
            [_record(3 * s + j) for j in range(3)],
            tmp_path / f"w00_{s:05d}.npz",
        )
    paths = sorted(tmp_path.glob("*.npz"))
    tokens = np.zeros((3, 2, 4, 8), np.float16)
    tokens[:, 1] = 1.0
    np.save(ShardDataset.tokens_path(paths[0]), tokens)
    cfg = ShardDatasetConfig(
        variant="random_main", n_points=128, use_augmented_image=1.0,
        point_augment=PointAugmentConfig(),
    )  # fmt: skip
    ds = ShardDataset(paths, cfg)
    assert len(ds) == 6
    item = ds[1]
    points, image, tok = item["points"], item["image"], item["tokens"]
    assert isinstance(points, torch.Tensor) and points.shape == (128, 3)
    assert isinstance(image, torch.Tensor) and image.shape == (3, CROP, CROP)
    assert isinstance(tok, torch.Tensor)
    assert torch.equal(tok, torch.ones(4, 8, dtype=torch.float16))
    assert "tokens" not in ds[4]  # second shard has no token file
    batch = next(
        iter(DataLoader(torch.utils.data.Subset(ds, [0, 1, 2]), batch_size=3))
    )
    assert batch["transl"].shape == (3, 3) and batch["tokens"].shape == (
        3,
        4,
        8,
    )
    strict = ShardDatasetConfig(variant="check", require_tokens=True)
    try:
        ShardDataset(paths, strict)
    except FileNotFoundError as exc:
        assert "without tokens" in str(exc)
    else:
        raise AssertionError("expected missing-token error")
