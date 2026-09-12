"""The training loop: DDP-ready, resumable, wandb-logged.

Design:
- one process per GPU (``torchrun``); rank 0 logs and checkpoints
- ``last.pt`` every ``checkpoint_every_steps`` and on SIGUSR1 (Slurm sends
  it before the wall-time limit), ``best.pt`` on the best validation
  translation error; ``resume()`` restores model, optimiser, schedule, step
  and sampler seed, so a re-queued job continues where it stopped
- wandb runs in the mode of the config (``offline`` on clusters without
  internet; sync from the login node afterwards)
"""

from __future__ import annotations

import json
import math
import os
import signal
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from lidar_bedlam.body.smpl import SmplModel
from lidar_bedlam.data.shards import ShardDataset, ShardDatasetConfig
from lidar_bedlam.data.torch_dataset import Item
from lidar_bedlam.lidar.augment import PointAugmentConfig
from lidar_bedlam.losses.smpl import FusionLoss, LossWeights
from lidar_bedlam.metrics.protocol import (
    MetricSummary,
    sample_metrics,
    summarize,
)
from lidar_bedlam.models.fusion import ModelConfig, SelectiveFusionModel
from lidar_bedlam.models.selective_attention import SelectiveDecoder
from lidar_bedlam.models.vit import VIT_H
from lidar_bedlam.train.config import SourceConfig, TrainConfig, to_dict
from lidar_bedlam.train.sampler import MixtureBatchSampler, concat

TENSOR_KEYS = (
    "image", "tokens", "points", "points_valid", "intrinsics", "has_image",
    "has_smpl", "global_orient", "body_pose", "betas", "transl", "joints3d",
    "joints3d_valid", "kp2d", "has_kp2d", "box3d", "has_box3d",
)  # fmt: skip


def source_shards(src: SourceConfig) -> list[Path]:
    """Shard files of a source (sorted, optionally truncated)."""
    files = sorted(p for d in src.dirs for p in Path(d).glob(src.pattern))
    files = [f for f in files if not f.name.endswith("stats.npz")]
    if src.max_shards:
        files = files[: src.max_shards]
    return files


def build_dataset(
    src: SourceConfig, cfg: TrainConfig, train: bool
) -> ShardDataset:
    """ShardDataset of one source."""
    ds_cfg = ShardDatasetConfig(
        variant=src.variant,
        n_points=cfg.data.n_points,
        use_augmented_image=cfg.data.use_augmented_image if train else 0.0,
        point_augment=PointAugmentConfig()
        if (train and cfg.data.point_augment)
        else None,
        require_tokens=cfg.data.require_tokens,
        seed=cfg.seed,
    )
    return ShardDataset(source_shards(src), ds_cfg)


def build_model(cfg: TrainConfig) -> SelectiveFusionModel:
    """Model from the config, with the gate mode applied."""
    m = cfg.model
    model = SelectiveFusionModel(
        ModelConfig(
            vit=VIT_H,
            dim=m.dim,
            num_heads=m.num_heads,
            num_layers=m.num_layers,
            point_tokens=m.point_tokens,
            point_knn=m.point_knn,
            smpl_model_dir=Path(cfg.body_models),
            init_depth_m=m.init_depth_m,
            use_backbone=m.use_backbone,
        )  # fmt: skip
    )
    decoder = model.decoder
    assert isinstance(decoder, SelectiveDecoder)
    decoder.set_gate_mode(m.gate_mode)
    return model


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move tensors, keep strings."""
    return {
        k: (v.to(device, non_blocking=True) if isinstance(v, Tensor) else v)
        for k, v in batch.items()
    }


def tensors_only(batch: dict[str, Any]) -> dict[str, Tensor]:
    """The tensor entries (what the model and losses take)."""
    return {k: v for k, v in batch.items() if isinstance(v, Tensor)}


class Trainer:
    """Owns model, data, optimiser and the run directory."""

    def __init__(self, cfg: TrainConfig, run_dir: Path, run_name: str) -> None:
        self.cfg = cfg
        self.run_dir = run_dir
        self.run_name = run_name
        self.rank = int(os.environ.get("RANK", "0"))
        self.world = int(os.environ.get("WORLD_SIZE", "1"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.device = (
            torch.device("cuda", self.local_rank)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        if self.world > 1:
            torch.distributed.init_process_group(
                "nccl" if self.device.type == "cuda" else "gloo"
            )
            torch.cuda.set_device(self.device)
        torch.manual_seed(cfg.seed + self.rank)
        self.model = build_model(cfg).to(self.device)
        self.ddp: nn.Module = (
            DistributedDataParallel(
                self.model,
                device_ids=[self.local_rank]
                if self.device.type == "cuda"
                else None,
            )
            if self.world > 1
            else self.model
        )
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            params, lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay
        )
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=cfg.optim.amp and self.device.type == "cuda"
        )
        self.loss_fn = FusionLoss(LossWeights(**asdict(cfg.loss)))
        self.smpl_eval = SmplModel(Path(cfg.body_models))
        self.step = 0
        self.steps_per_epoch = 1
        self.best = math.inf
        self.stop_requested = False
        self.wandb: Any = None
        signal.signal(signal.SIGUSR1, self._on_signal)
        n_train = sum(p.numel() for p in params) / 1e6
        self._log(
            f"rank {self.rank}/{self.world} device {self.device} "
            f"trainable params {n_train:.1f} M"
        )

    # -- plumbing --------------------------------------------------------

    @property
    def is_main(self) -> bool:
        """Whether this process logs and checkpoints."""
        return self.rank == 0

    def _log(self, msg: str) -> None:
        if self.is_main:
            print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    def _on_signal(self, *_: object) -> None:
        self._log("SIGUSR1 received: checkpoint and stop")
        self.stop_requested = True

    def lr_at(self, step: int) -> float:
        """Linear warm-up then cosine decay to 5 % of the peak."""
        o = self.cfg.optim
        if step < o.warmup_steps:
            return o.lr * (step + 1) / o.warmup_steps
        t = (step - o.warmup_steps) / max(1, o.max_steps - o.warmup_steps)
        return o.lr * (
            0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))
        )

    # -- data ------------------------------------------------------------

    def train_loader(
        self, start_step: int, sets: list[ShardDataset]
    ) -> DataLoader[Item]:
        """Mixture loader over ``sets`` that resumes at ``start_step``."""
        cfg = self.cfg
        sampler = MixtureBatchSampler(
            [len(d) for d in sets], [s.weight for s in cfg.data.train],
            cfg.optim.batch_size, cfg.optim.max_steps - start_step,
            seed=cfg.seed + start_step, rank=self.rank, world_size=self.world,
        )  # fmt: skip
        self._log(
            "train sources: "
            + ", ".join(
                f"{s.name}={len(d)}"
                for s, d in zip(cfg.data.train, sets, strict=True)
            )
        )
        return DataLoader(
            concat(sets),
            batch_sampler=sampler,
            num_workers=cfg.data.num_workers,
            pin_memory=True,
            persistent_workers=cfg.data.num_workers > 0,
        )

    def val_loaders(self) -> dict[str, DataLoader[Item]]:
        """One loader per validation source (rank 0 only evaluates)."""
        out: dict[str, DataLoader[Item]] = {}
        for s in self.cfg.data.val:
            ds = build_dataset(s, self.cfg, train=False)
            n = min(len(ds), self.cfg.data.val_max_samples)
            subset = torch.utils.data.Subset(ds, list(range(n)))
            out[s.name] = DataLoader(
                subset,
                batch_size=64,
                num_workers=min(4, self.cfg.data.num_workers),
            )
        return out

    # -- checkpoints -----------------------------------------------------

    def save(self, name: str) -> None:
        """Write a checkpoint (rank 0)."""
        if not self.is_main:
            return
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "step": self.step,
            "best": self.best,
            "config": to_dict(self.cfg),
            "run_name": self.run_name,
        }
        tmp = self.run_dir / f"{name}.tmp"
        torch.save(state, tmp)
        tmp.replace(self.run_dir / f"{name}.pt")
        self._log(f"saved {name}.pt at step {self.step}")

    def resume(self) -> bool:
        """Load ``last.pt`` if it exists; returns whether it did."""
        path = self.run_dir / "last.pt"
        if not path.exists():
            return False
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scaler.load_state_dict(state["scaler"])
        self.step = int(state["step"])
        self.best = float(state["best"])
        self._log(f"resumed from step {self.step}")
        return True

    # -- evaluation ------------------------------------------------------

    @torch.no_grad()
    def evaluate(
        self, loaders: dict[str, DataLoader[Item]]
    ) -> dict[str, MetricSummary]:
        """Metrics per validation source."""
        self.model.eval()
        results: dict[str, MetricSummary] = {}
        for name, loader in loaders.items():
            samples = []
            for batch in loader:
                b = to_device(batch, self.device)
                pred = self.model(tensors_only(b))
                samples.extend(sample_metrics(pred, b, self.smpl_eval))
            results[name] = summarize(samples)
        self.model.train()
        return results

    # -- training --------------------------------------------------------

    def fit(self) -> None:
        """Run to ``max_steps`` / ``max_epochs`` (or until SIGUSR1)."""
        cfg = self.cfg
        self.resume()
        sets = [build_dataset(s, cfg, train=True) for s in cfg.data.train]
        self.steps_per_epoch = max(
            1, math.ceil(sum(len(d) for d in sets) / cfg.optim.batch_size)
        )
        if cfg.optim.max_epochs > 0:
            cfg.optim.max_steps = math.ceil(
                cfg.optim.max_epochs * self.steps_per_epoch
            )
        self._log(
            f"{self.steps_per_epoch} steps per epoch, "
            f"{cfg.optim.max_steps / self.steps_per_epoch:.1f} epochs "
            f"= {cfg.optim.max_steps} steps"
        )
        if self.step >= cfg.optim.max_steps:
            self._log("already finished")
            return
        if self.is_main:
            (self.run_dir / "config.json").write_text(
                json.dumps(to_dict(cfg), indent=1)
            )
            if self.wandb is None:
                self.wandb = self._init_wandb()
        loader = self.train_loader(self.step, sets)
        val = self.val_loaders() if self.is_main else {}
        self.model.train()
        t0 = time.time()
        for batch in loader:
            if self.stop_requested:
                break
            b = to_device(batch, self.device)
            lr = self.lr_at(self.step)
            for g in self.optimizer.param_groups:
                g["lr"] = lr
            with torch.autocast(
                self.device.type,
                dtype=torch.bfloat16,
                enabled=cfg.optim.amp and self.device.type == "cuda",
            ):
                pred = self.ddp(tensors_only(b))
            pred = {k: v.float() for k, v in pred.items()}
            total, parts = self.loss_fn(pred, tensors_only(b))
            self.optimizer.zero_grad(set_to_none=True)
            scaled: Tensor = self.scaler.scale(total)
            scaled.backward()  # type: ignore[no-untyped-call]
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), cfg.optim.grad_clip
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.step += 1
            if self.step % 50 == 0:
                dt = (time.time() - t0) / 50
                mem = (
                    torch.cuda.max_memory_allocated(self.device) / 2**30
                    if self.device.type == "cuda"
                    else 0.0
                )
                self._log(
                    f"step {self.step}/{cfg.optim.max_steps} "
                    f"ep {self.epoch:.2f} "
                    f"loss {float(total):.4f} lr {lr:.2e} "
                    f"{dt:.2f} s/step "
                    f"{cfg.optim.batch_size / dt:.0f} samples/s "
                    f"peak {mem:.1f} GiB/gpu"
                )
                t0 = time.time()
                self._wandb_log(
                    {
                        "train/loss": float(total),
                        "train/lr": lr,
                        "perf/step_s": dt,
                        "perf/samples_per_s": cfg.optim.batch_size / dt,
                        "perf/peak_mem_gib": mem,
                        **{f"train/{k}": float(v) for k, v in parts.items()},
                    }
                )
            if (
                self.step % cfg.optim.eval_every_steps == 0
                or self.step == cfg.optim.max_steps
            ):
                self._eval_and_track(val)
            if self.step % cfg.optim.checkpoint_every_steps == 0:
                self.save("last")
        self.save("last")
        if self.is_main and self.step >= cfg.optim.max_steps:
            (self.run_dir / "DONE").write_text(f"{self.step}\n")
            self._log("finished")
        if self.wandb is not None:
            self.wandb.finish()
        if self.world > 1:
            torch.distributed.destroy_process_group()

    def _eval_and_track(self, val: dict[str, DataLoader[Item]]) -> None:
        if not self.is_main or not val:
            return
        results = self.evaluate(val)
        log: dict[str, float] = {}
        for name, r in results.items():
            log.update(
                {
                    f"val/{name}/mpjpe": r.mpjpe,
                    f"val/{name}/pa_mpjpe": r.pa_mpjpe,
                    f"val/{name}/transl_err": r.transl_err_m,
                    f"val/{name}/map": r.map,
                    f"val/{name}/mean_iou": r.mean_iou,
                }
            )
            self._log(
                f"val {name}: n={r.n} mpjpe {r.mpjpe:.1f} "
                f"pa {r.pa_mpjpe:.1f} transl {r.transl_err_m:.3f} m "
                f"mAP {r.map:.3f}"
            )
        self._wandb_log(log)
        (self.run_dir / f"val_{self.step:07d}.json").write_text(
            json.dumps({k: asdict(v) for k, v in results.items()}, indent=1)
        )
        primary = self.cfg.data.val[0].name
        score = (
            results[primary].transl_err_m if primary in results else math.inf
        )
        if score < self.best:
            self.best = score
            self.save("best")

    @property
    def epoch(self) -> float:
        """Passes over the training records so far."""
        return self.step / self.steps_per_epoch

    def _init_wandb(self) -> Any:
        if self.cfg.wandb_mode == "disabled":
            return None
        import wandb

        return wandb.init(
            entity=self.cfg.wandb_entity,
            project=self.cfg.wandb_project,
            name=self.run_name,
            id=self.run_name,
            resume="allow",
            mode=_wandb_mode(self.cfg.wandb_mode),
            config=to_dict(self.cfg),
            dir=str(self.run_dir),
        )

    def _wandb_log(self, values: dict[str, float]) -> None:
        """Append to ``metrics.jsonl`` (mirrored to wandb from a node with
        internet, see ``scripts/wandb_mirror.py``) and to wandb if live."""
        if not self.is_main:
            return
        row = {"step": self.step, "train/epoch": self.epoch, **values}
        with open(self.run_dir / "metrics.jsonl", "a") as fh:
            fh.write(json.dumps(row) + "\n")
        if self.wandb is not None:
            self.wandb.log(row, step=self.step)


def _wandb_mode(mode: str) -> Literal["online", "offline", "disabled"]:
    if mode not in ("online", "offline", "disabled"):
        msg = f"unknown wandb mode {mode!r}"
        raise ValueError(msg)
    return mode  # type: ignore[return-value]


def new_run_name(root: Path, experiment: str) -> str:
    """``<experiment>-NNN`` with the next free number under ``root``."""
    existing = [p.name for p in root.glob(f"{experiment}-[0-9][0-9][0-9]")]
    nums = [int(n.rsplit("-", 1)[1]) for n in existing]
    return f"{experiment}-{(max(nums) + 1 if nums else 0):03d}"


def seed_everything(seed: int) -> None:
    """numpy + torch seeds."""
    np.random.seed(seed)
    torch.manual_seed(seed)
