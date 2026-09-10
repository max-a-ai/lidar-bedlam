"""Training plumbing: sampler ratios, config, gate modes, trainer smoke."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from test_shard_dataset import _record

from lidar_bedlam.data.shards import ShardDataset
from lidar_bedlam.generate.records import write_shard
from lidar_bedlam.models.selective_attention import (
    GATE_MODES,
    SelectiveDecoder,
)
from lidar_bedlam.models.vit import VIT_H
from lidar_bedlam.train.config import SourceConfig, load_config
from lidar_bedlam.train.loop import Trainer, new_run_name
from lidar_bedlam.train.sampler import MixtureBatchSampler

SMPL_DIR = Path("data/generated/body_models")


def test_mixture_sampler_fixed_counts_and_disjoint_ranks() -> None:
    sizes, weights = [10, 1000, 5], [0.4, 0.5, 0.1]
    kw = {"seed": 1, "world_size": 2}
    s0 = MixtureBatchSampler(sizes, weights, 20, 3, rank=0, **kw)
    s1 = MixtureBatchSampler(sizes, weights, 20, 3, rank=1, **kw)
    assert s0.counts == [4, 5, 1] and len(s0) == 3
    b0, b1 = next(iter(s0)), next(iter(s1))
    assert len(b0) == 10 and b0 != b1
    offsets = np.cumsum([0, *sizes[:-1]])
    src = np.searchsorted(offsets, np.array(b0), side="right") - 1
    assert sorted(np.bincount(src, minlength=3).tolist()) == [1, 4, 5]


def test_load_config_expands_env_and_applies_overrides(tmp_path: Path) -> None:
    os.environ["DATA_ROOT"] = str(tmp_path)
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(
        "experiment: x\nbody_models: ${DATA_ROOT}/bm\n"
        "data:\n  train:\n"
        "    - {name: a, dirs: ['${DATA_ROOT}/a'], weight: 2}\n"
        "optim:\n  max_steps: 10\n"
    )
    over = ["optim.lr=0.5", "model.gate_mode=hard", "optim.amp=false"]
    cfg = load_config(cfg_path, over)
    assert cfg.body_models == f"{tmp_path}/bm"
    assert cfg.data.train[0].dirs == [f"{tmp_path}/a"]
    assert cfg.data.train[0].weight == 2.0
    assert cfg.optim.max_steps == 10 and cfg.optim.lr == 0.5
    assert cfg.model.gate_mode == "hard" and cfg.optim.amp is False


def test_gate_modes() -> None:
    dec = SelectiveDecoder(dim=32, num_heads=4, num_layers=2)
    img, pts = torch.randn(2, 5, 32), torch.randn(2, 6, 32)
    expected = {"image_only": 1.0, "lidar_only": 0.0, "none": 0.5}
    for mode in GATE_MODES:
        dec.set_gate_mode(mode)
        feats, gates = dec(img, pts)
        assert feats.shape[0] == 2 and gates.shape[0] == 2
        if mode in expected:
            want = torch.full_like(gates, expected[mode])
            assert torch.allclose(gates, want)
    with pytest.raises(ValueError, match="unknown gate mode"):
        dec.set_gate_mode("bogus")


def test_new_run_name_enumerates(tmp_path: Path) -> None:
    assert new_run_name(tmp_path, "main") == "main-000"
    (tmp_path / "main-000").mkdir()
    (tmp_path / "main-007").mkdir()
    assert new_run_name(tmp_path, "main") == "main-008"
    assert new_run_name(tmp_path, "other") == "other-000"


def _write_shards(root: Path, n_shards: int = 2, n: int = 4) -> None:
    for s in range(n_shards):
        path = root / f"toy_{s:05d}.npz"
        write_shard([_record(n * s + j) for j in range(n)], path)
        shape = (n, 2, 256, VIT_H.embed_dim)
        tokens = np.random.default_rng(s).normal(size=shape)
        np.save(ShardDataset.tokens_path(path), tokens.astype(np.float16))


@pytest.mark.skipif(not SMPL_DIR.exists(), reason="SMPL model files absent")
def test_trainer_smoke_checkpoint_resume(tmp_path: Path) -> None:
    _write_shards(tmp_path / "shards")
    os.environ["DATA_ROOT"] = str(tmp_path)
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(f"experiment: toy\nbody_models: {SMPL_DIR}\n")
    cfg = load_config(cfg_path)
    src = SourceConfig("toy", [str(tmp_path / "shards")], "toy_*.npz")
    cfg.data.train = [src]
    cfg.data.val = [SourceConfig("toy_val", src.dirs, src.pattern)]
    cfg.data.num_workers = 0
    cfg.data.n_points = 64
    cfg.model.dim, cfg.model.num_layers, cfg.model.point_tokens = 64, 1, 8
    cfg.model.point_knn = 4
    cfg.optim.batch_size, cfg.optim.max_steps, cfg.optim.warmup_steps = 4, 3, 1
    cfg.optim.eval_every_steps = cfg.optim.checkpoint_every_steps = 2
    cfg.optim.amp = False
    cfg.wandb_mode = "disabled"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    trainer = Trainer(cfg, run_dir, "toy-000")
    trainer.wandb = _NoWandb()
    trainer.fit()
    assert trainer.step == 3
    state = torch.load(run_dir / "last.pt", weights_only=False)
    assert state["step"] == 3 and (run_dir / "best.pt").exists()
    assert (run_dir / "val_0000002.json").exists()
    cfg.optim.max_steps = 5
    again = Trainer(cfg, run_dir, "toy-000")
    again.wandb = _NoWandb()
    again.fit()
    assert again.step == 5


class _NoWandb:
    def log(self, *_: object, **__: object) -> None:
        pass

    def finish(self) -> None:
        pass
