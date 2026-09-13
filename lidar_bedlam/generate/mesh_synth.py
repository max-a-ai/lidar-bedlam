"""Camera + simulated-LiDAR records for datasets that have SMPL labels but
no depth (3DPW).

Per frame the labelled meshes are pushed outwards along their normals by a
random clothing offset, rasterised into a planar depth map plus an id
image, and handed to the same LiDAR scan plan and record builder as the
BEDLAM pipeline. The image is real, the SMPL label is real, only the
LiDAR is simulated; the records carry ``dataset="3dpw"``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from lidar_bedlam.body.smpl import SmplModel, SmplParams
from lidar_bedlam.data.threedpw import Frame, ThreeDPWSource
from lidar_bedlam.generate.records import Record, write_shard
from lidar_bedlam.generate.synth import SynthConfig, SynthGenerator
from lidar_bedlam.geometry.camera import PinholeCamera
from lidar_bedlam.lidar.mesh_depth import offset_mesh, render_depth
from lidar_bedlam.lidar.simulate import LidarScan, LidarSpec, simulate

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class MeshSynthConfig:
    """Clothing offset range (metres), sampled per actor and frame."""

    offset_min_m: float = 0.01
    offset_max_m: float = 0.04


class MeshLidarGenerator(SynthGenerator):
    """SynthGenerator whose depth comes from rasterised SMPL meshes."""

    def __init__(
        self,
        source: ThreeDPWSource | None,
        smpl_models: dict[str, SmplModel],
        cfg: SynthConfig,
        mesh_cfg: MeshSynthConfig,
        data_dir: Path,
        seed_offset: int = 0,
    ) -> None:
        super().__init__(
            None, smpl_models["neutral"], cfg, data_dir, seed_offset, "3dpw"
        )
        self.pw = source
        self.models = smpl_models
        self.mesh_cfg = mesh_cfg

    def frame_records(self, index: int) -> list[Record]:
        """Records of frame ``index`` of the 3DPW split."""
        assert self.pw is not None
        frame = self.pw.frame(index)
        if not frame.image_path.exists():  # 3DPW frame ids have gaps
            self.stats.frames_missing_image += 1
            return []
        image = self.pw.read_image(frame)
        return self.records_from_frame(frame, image, self.pw.split)

    def records_from_frame(
        self, frame: Frame, image: NDArray[np.uint8], split: str = "train"
    ) -> list[Record]:
        """Records from labels and the RGB frame (no file access)."""
        self.stats.frames += 1
        self.stats.persons_total += len(frame.actors)
        meshes: list[tuple[FloatArray, NDArray[np.int64]]] = []
        for actor in frame.actors:
            model = self.models[actor.gender]
            verts, _ = model.forward(actor.smpl)
            offset = float(
                self.rng.uniform(
                    self.mesh_cfg.offset_min_m, self.mesh_cfg.offset_max_m
                )
            )
            meshes.append(
                (offset_mesh(verts, model.faces, offset), model.faces)
            )
        depth, ids = render_depth(meshes, frame.camera)
        scans: dict[str, tuple[LidarScan, LidarSpec, FloatArray]] = {}
        for name, (spec, pose) in self.scan_plan().items():
            scan = simulate(
                depth.astype(np.float64), frame.camera, spec, pose, self.rng
            )
            scans[name] = (scan, spec, pose)
        records = []
        for k, actor in enumerate(frame.actors):
            mask = ids == k
            if not mask.any():
                continue
            self.smpl = self.models[actor.gender]  # joints and box per gender
            rec = self._person_record(
                split,
                frame.seq,
                f"{frame.frame:05d}",
                f"{actor.actor:02d}",
                actor.smpl,
                frame.camera,
                image,
                mask,
                mask,
                scans,
                0.0,
            )
            if rec is not None:
                records.append(rec)
        self.stats.records += len(records)
        self.stats.per_group[frame.seq] = self.stats.per_group.get(
            frame.seq, 0
        ) + len(records)
        return records


def generate_mesh(
    gen: MeshLidarGenerator, indices: list[int], out_dir: Path, prefix: str
) -> dict[str, Any]:
    """Process frames, writing shards of ``cfg.shard_size`` records."""
    import json
    from dataclasses import asdict

    out_dir.mkdir(parents=True, exist_ok=True)
    buffer: list[Record] = []
    shard_id = 0
    for i in indices:
        buffer.extend(gen.frame_records(i))
        while len(buffer) >= gen.cfg.shard_size:
            write_shard(
                buffer[: gen.cfg.shard_size],
                out_dir / f"{prefix}_{shard_id:05d}.npz",
            )
            buffer = buffer[gen.cfg.shard_size :]
            shard_id += 1
    if buffer:
        write_shard(buffer, out_dir / f"{prefix}_{shard_id:05d}.npz")
        shard_id += 1
    stats = asdict(gen.stats)
    stats["shards"] = shard_id
    stats["config"] = asdict(gen.cfg)
    stats["mesh_config"] = asdict(gen.mesh_cfg)
    (out_dir / f"{prefix}_stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def label_from_arrays(
    pose72: FloatArray, betas: FloatArray, transl: FloatArray
) -> SmplParams:
    """Convenience for tests and notebooks."""
    return SmplParams.from_pose72(pose72, betas, transl)


__all__ = [
    "MeshLidarGenerator",
    "MeshSynthConfig",
    "PinholeCamera",
    "generate_mesh",
    "label_from_arrays",
]
