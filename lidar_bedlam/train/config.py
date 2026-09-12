"""Training configuration (YAML + ``key=value`` overrides)."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SourceConfig:
    """One data source: shard files and its share of every batch."""

    name: str
    dirs: list[str]
    pattern: str = "*.npz"
    weight: float = 1.0  # fraction of each batch (normalised over sources)
    variant: str = "random_main"  # LiDAR scan variant
    max_shards: int = 0  # 0 = all (used by the data-scaling ablation)


@dataclass
class DataConfig:
    """Train and validation sources."""

    train: list[SourceConfig] = field(default_factory=list)
    val: list[SourceConfig] = field(default_factory=list)
    n_points: int = 1024
    use_augmented_image: float = 0.5
    point_augment: bool = True
    require_tokens: bool = True
    num_workers: int = 8
    val_max_samples: int = 4000


@dataclass
class ModelSection:
    """Model hyper-parameters that the trainer exposes."""

    dim: int = 512
    num_heads: int = 8
    num_layers: int = 4
    point_tokens: int = 128
    point_knn: int = 16
    init_depth_m: float = 8.0
    use_backbone: bool = False
    gate_mode: str = (
        "learned"  # learned | none | hard | image_only | lidar_only
    )


@dataclass
class OptimConfig:
    """Optimiser and schedule."""

    batch_size: int = 256  # global batch across all ranks
    lr: float = 1e-4
    weight_decay: float = 0.05
    warmup_steps: int = 500
    max_steps: int = 17000
    # epoch = one pass over all training records (sum of the sources);
    # > 0 overrides max_steps = ceil(max_epochs * records / batch_size)
    max_epochs: float = 0.0
    grad_clip: float = 1.0
    amp: bool = True
    eval_every_steps: int = 1000
    checkpoint_every_steps: int = 500


@dataclass
class LossSection:
    """Loss weights (see losses.smpl.LossWeights)."""

    global_orient: float = 1.0
    body_pose: float = 1.0
    betas: float = 0.005
    joints3d: float = 5.0
    kp2d: float = 1.0
    transl: float = 5.0
    box3d: float = 1.0
    box_conf: float = 1.0


@dataclass
class TrainConfig:
    """Everything a run needs."""

    experiment: str = "main-mixed"
    seed: int = 0
    body_models: str = "resources/data/generated/body_models"
    checkpoint_root: str = "outputs"  # one directory per run
    wandb_entity: str = "erik_hm"
    wandb_project: str = "lidar-bedlam"
    wandb_mode: str = "offline"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelSection = field(default_factory=ModelSection)
    optim: OptimConfig = field(default_factory=OptimConfig)
    loss: LossSection = field(default_factory=LossSection)


def _build(cls: type[Any], raw: dict[str, Any]) -> Any:
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in raw:
            continue
        value = raw[f.name]
        target = f.type if not isinstance(f.type, str) else None
        if f.name in ("train", "val"):
            kwargs[f.name] = [_build(SourceConfig, v) for v in value]
        elif (
            isinstance(value, dict)
            and target is not None
            and is_dataclass(target)
        ):
            kwargs[f.name] = _build(target, value)
        else:
            kwargs[f.name] = value
    return cls(**kwargs)


_SECTIONS = {
    "data": DataConfig,
    "model": ModelSection,
    "optim": OptimConfig,
    "loss": LossSection,
}


DEFAULT_DATA_ROOT = "resources/data/generated"


def _expand(value: Any) -> Any:
    """``${VAR}`` expansion in every string of the raw config."""
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    return value


def load_config(path: Path, overrides: list[str] | None = None) -> TrainConfig:
    """Read YAML, expand ``${DATA_ROOT}`` etc., apply overrides."""
    os.environ.setdefault("DATA_ROOT", DEFAULT_DATA_ROOT)
    with open(path) as fh:
        raw = _expand(yaml.safe_load(fh) or {})
    for section, cls in _SECTIONS.items():
        if section in raw:
            raw[section] = _build(cls, raw[section])
    cfg: TrainConfig = _build(TrainConfig, raw)
    for item in overrides or []:
        key, _, value = item.partition("=")
        _apply_override(cfg, key.split("."), value)
    return cfg


def _apply_override(obj: Any, path: list[str], value: str) -> None:
    for p in path[:-1]:
        obj = getattr(obj, p)
    current = getattr(obj, path[-1])
    parsed: Any = yaml.safe_load(value)
    if isinstance(current, bool):
        parsed = bool(parsed)
    elif isinstance(current, int) and not isinstance(parsed, bool):
        parsed = int(parsed)
    elif isinstance(current, float):
        parsed = float(parsed)
    setattr(obj, path[-1], parsed)


def to_dict(cfg: TrainConfig) -> dict[str, Any]:
    """Plain dict (for wandb and checkpoints)."""
    return asdict(cfg)
