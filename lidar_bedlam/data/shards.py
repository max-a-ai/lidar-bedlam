"""Torch dataset over generated shards (synthetic and real records).

Every item has the same keys as :class:`data.torch_dataset.HumanPoseDataset`
plus ``tokens`` when precomputed ViT tokens exist next to the shard
(``<shard>.tokens.npy``, fp16, shape (N, 2, 256, 1280): clean and augmented
copy). The LiDAR scan variant is chosen per dataset instance (``variant``),
so the ablations are a config change; ``"random_main"`` picks ``main_0`` or
``main_1`` per item. Point augmentation, if configured, is applied on the
fly to the chosen scan.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from lidar_bedlam.data.base import sample_points
from lidar_bedlam.data.torch_dataset import IMAGENET_MEAN, IMAGENET_STD, Item
from lidar_bedlam.generate.records import Shard
from lidar_bedlam.lidar.augment import PointAugmentConfig, augment_points


@dataclass(frozen=True)
class ShardDatasetConfig:
    """What to read from the shards."""

    variant: str = "random_main"
    n_points: int = 1024
    use_augmented_image: float = 0.5  # probability of the augmented copy
    point_augment: PointAugmentConfig | None = None
    require_tokens: bool = False
    seed: int = 0


class ShardDataset(Dataset[Item]):
    """All records of a list of shard files."""

    def __init__(self, shards: list[Path], cfg: ShardDatasetConfig) -> None:
        self.paths = list(shards)
        self.cfg = cfg
        self._shards: dict[int, Shard] = {}
        self._tokens: dict[
            int, np.memmap[tuple[int, ...], np.dtype[np.float16]]
        ] = {}
        self._rng = np.random.default_rng(cfg.seed)
        self._offsets = [0]
        for p in self.paths:
            self._offsets.append(self._offsets[-1] + len(Shard(p)))
        if cfg.require_tokens:
            missing = [
                p for p in self.paths if not self.tokens_path(p).exists()
            ]
            if missing:
                msg = (
                    f"{len(missing)} shards without tokens, e.g. {missing[0]}"
                )
                raise FileNotFoundError(msg)

    @staticmethod
    def tokens_path(shard: Path) -> Path:
        """Where the precomputed tokens of a shard live."""
        return shard.with_suffix(".tokens.npy")

    def __len__(self) -> int:
        return self._offsets[-1]

    def _locate(self, index: int) -> tuple[int, int]:
        s = int(np.searchsorted(self._offsets, index, side="right") - 1)
        return s, index - self._offsets[s]

    def _shard(self, s: int) -> Shard:
        if s not in self._shards:
            self._shards[s] = Shard(self.paths[s])
        return self._shards[s]

    def _token_array(
        self, s: int
    ) -> np.memmap[tuple[int, ...], np.dtype[np.float16]] | None:
        path = self.tokens_path(self.paths[s])
        if not path.exists():
            return None
        if s not in self._tokens:
            self._tokens[s] = np.load(path, mmap_mode="r")
        return self._tokens[s]

    def _variant(self, shard: Shard) -> str:
        if self.cfg.variant == "random_main":
            mains = [v for v in shard.variants if v.startswith("main_")]
            return str(self._rng.choice(mains)) if mains else shard.variants[0]
        return self.cfg.variant

    def __getitem__(self, index: int) -> Item:
        s, i = self._locate(index)
        shard = self._shard(s)
        use_aug = self._rng.random() < self.cfg.use_augmented_image
        image = shard.row("image_aug" if use_aug else "image", i)
        img = (image.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        scan = shard.scan(i, self._variant(shard))
        points = scan.points
        if self.cfg.point_augment is not None and len(points):
            points = augment_points(
                points,
                scan.channel,
                max(scan.channels, 1),
                self.cfg.point_augment,
                self._rng,
            )
        pts, valid = sample_points(
            points.astype(np.float32), self.cfg.n_points, self._rng
        )
        item: Item = {
            "key": str(shard.array("key")[i]),
            "dataset": str(shard.array("dataset")[i]),
            "image": torch.from_numpy(
                np.ascontiguousarray(img.transpose(2, 0, 1))
            ),
            "points": torch.from_numpy(pts),
            "points_valid": torch.from_numpy(valid),
            "intrinsics": torch.from_numpy(shard.row("intrinsics", i)).float(),
            "crop_origin": torch.from_numpy(
                shard.row("crop_origin", i)
            ).float(),
            "has_image": torch.tensor(bool(shard.array("has_image")[i])),
            "has_smpl": torch.tensor(bool(shard.array("has_smpl")[i])),
            "global_orient": torch.from_numpy(shard.row("global_orient", i)),
            "body_pose": torch.from_numpy(shard.row("body_pose", i)),
            "betas": torch.from_numpy(shard.row("betas", i)),
            "transl": torch.from_numpy(shard.row("transl", i)),
            "joints3d": torch.from_numpy(shard.row("joints3d", i)),
            "joints3d_valid": torch.from_numpy(shard.row("joints3d_valid", i)),
            "joint_convention": str(shard.array("joint_convention")[i]),
            "kp2d": torch.from_numpy(shard.row("kp2d", i)),
            "has_kp2d": torch.tensor(True),
            "box3d": torch.from_numpy(shard.row("box3d", i)),
            "has_box3d": torch.tensor(True),
            "mask": torch.from_numpy(shard.row("mask", i)),
            "has_mask": torch.tensor(True),
            "distance_scale": torch.tensor(
                float(shard.array("distance_scale")[i])
            ),
        }
        tokens = self._token_array(s)
        if tokens is not None:
            item["tokens"] = torch.from_numpy(
                np.array(tokens[i, 1 if use_aug else 0])
            )
        return item
