"""Fixed-ratio mixture of sources per batch, sharded across DDP ranks.

Every batch contains ``round(weight_i * batch)`` items of source ``i`` no
matter how large the sources are, so the real data keeps its share while
the synthetic pool grows. Items are drawn with replacement, so an "epoch"
is simply a number of steps.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
from torch.utils.data import ConcatDataset, Dataset, Sampler

from lidar_bedlam.data.torch_dataset import Item


class MixtureBatchSampler(Sampler[list[int]]):
    """Yields index lists into a ``ConcatDataset`` of the sources."""

    def __init__(
        self,
        sizes: Sequence[int],
        weights: Sequence[float],
        batch_size: int,
        num_batches: int,
        seed: int = 0,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        if len(sizes) != len(weights) or not sizes:
            msg = "sizes and weights must be non-empty and equally long"
            raise ValueError(msg)
        self.sizes = list(sizes)
        w = np.asarray(weights, dtype=np.float64)
        w = w / w.sum()
        per_rank = batch_size // world_size
        counts = np.floor(w * per_rank).astype(int)
        counts[np.argmax(w)] += per_rank - counts.sum()
        self.counts = [int(c) for c in counts]
        self.num_batches = num_batches
        self.rng = np.random.default_rng(seed * 1000 + rank)
        self.offsets = np.cumsum([0, *self.sizes[:-1]])

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[list[int]]:
        for _ in range(self.num_batches):
            batch: list[int] = []
            for src, (n, count) in enumerate(
                zip(self.sizes, self.counts, strict=True)
            ):
                if count == 0 or n == 0:
                    continue
                idx = self.rng.integers(0, n, size=count) + self.offsets[src]
                batch.extend(int(i) for i in idx)
            self.rng.shuffle(batch)
            yield batch


def concat(datasets: Sequence[Dataset[Item]]) -> ConcatDataset[Item]:
    """Concatenate the source datasets (order = sampler order)."""
    return ConcatDataset(list(datasets))
