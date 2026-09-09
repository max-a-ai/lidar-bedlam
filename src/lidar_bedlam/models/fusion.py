"""The selective-attention LiDAR-camera SMPL model.

image crop -> ViT tokens ─┐
                          ├─ SelectiveDecoder (joint-group queries) ─ heads
person points -> tokens ──┘

Heads: per joint group 6D rotations for its SMPL joints; root query also
predicts the pelvis pixel (normalised crop coords) and log-depth, which
give the camera-frame translation through the crop intrinsics; shape
query predicts betas. A differentiable SMPL layer turns this into joints,
vertices, projected 2D joints and the 3D box, so every loss in
``losses.smpl`` applies.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from lidar_bedlam.body.smpl import NUM_BETAS, NUM_BODY_JOINTS
from lidar_bedlam.models.point_encoder import PointTokenizer
from lidar_bedlam.models.rotation import IDENTITY_ROT6D, rot6d_to_matrix
from lidar_bedlam.models.selective_attention import (
    JOINT_GROUPS,
    SelectiveDecoder,
)
from lidar_bedlam.models.vit import VIT_H, ViT, ViTConfig

NUM_JOINTS = NUM_BODY_JOINTS + 1


@dataclass
class ModelConfig:
    """Hyper-parameters of the fusion model."""

    vit: ViTConfig = VIT_H
    dim: int = 512
    num_heads: int = 8
    num_layers: int = 4
    point_tokens: int = 128
    point_knn: int = 16
    smpl_model_dir: Path | None = None
    init_depth_m: float = 8.0
    freeze_backbone: bool = True


class SmplHeads(nn.Module):
    """Map group features to SMPL parameters."""

    def __init__(self, dim: int, init_depth_m: float) -> None:
        super().__init__()
        self.rot_heads = nn.ModuleDict()
        for g in JOINT_GROUPS:
            if not g.joints:
                continue
            head = nn.Linear(dim, 6 * len(g.joints))
            nn.init.zeros_(head.weight)
            with torch.no_grad():
                head.bias.copy_(IDENTITY_ROT6D.repeat(len(g.joints)))
            self.rot_heads[g.name] = head
        self.transl_head = nn.Linear(dim, 3)  # (u_n, v_n, log z)
        nn.init.zeros_(self.transl_head.weight)
        with torch.no_grad():
            self.transl_head.bias.copy_(
                torch.tensor([0.0, 0.0, float(np.log(init_depth_m))])
            )
        self.shape_head = nn.Linear(dim, NUM_BETAS)
        nn.init.zeros_(self.shape_head.weight)
        nn.init.zeros_(self.shape_head.bias)
        self.root_index = [g.name for g in JOINT_GROUPS].index("root")
        self.shape_index = [g.name for g in JOINT_GROUPS].index("shape")

    def forward(self, feats: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Rotations (B, 24, 3, 3), betas (B, 10), raw transl (B, 3)."""
        b = feats.shape[0]
        rot6d = feats.new_zeros(b, NUM_JOINTS, 6)
        for idx, g in enumerate(JOINT_GROUPS):
            if not g.joints:
                continue
            rot6d[:, list(g.joints)] = self.rot_heads[g.name](
                feats[:, idx]
            ).reshape(b, len(g.joints), 6)
        rots = rot6d_to_matrix(rot6d)
        betas = self.shape_head(feats[:, self.shape_index])
        raw_t = self.transl_head(feats[:, self.root_index])
        return rots, betas, raw_t


def translation_from_raw(raw: Tensor, intrinsics: Tensor, size: int) -> Tensor:
    """(u_n, v_n, log z) + crop intrinsics -> camera-frame pelvis position.

    ``u_n, v_n`` are offsets from the crop centre in units of half the crop
    size, so 0 means the pelvis projects to the crop centre.
    """
    z = torch.exp(raw[:, 2])
    u = intrinsics[:, 0, 2] + raw[:, 0] * size / 2.0
    v = intrinsics[:, 1, 2] + raw[:, 1] * size / 2.0
    x = (u - intrinsics[:, 0, 2]) / intrinsics[:, 0, 0] * z
    y = (v - intrinsics[:, 1, 2]) / intrinsics[:, 1, 1] * z
    return torch.stack([x, y, z], -1)


def project(points: Tensor, intrinsics: Tensor) -> Tensor:
    """Pinhole projection of (B, N, 3) with K (B, 3, 3) -> (B, N, 2)."""
    z = points[..., 2].clamp_min(1e-3)
    u = (
        intrinsics[:, None, 0, 0] * points[..., 0] / z
        + intrinsics[:, None, 0, 2]
    )
    v = (
        intrinsics[:, None, 1, 1] * points[..., 1] / z
        + intrinsics[:, None, 1, 2]
    )
    return torch.stack([u, v], -1)


def box_from_vertices(verts: Tensor, global_orient: Tensor) -> Tensor:
    """Oriented 7-box (B, 7) around vertices, heading from the pelvis.

    Up axis is -y; the heading is the rotated SMPL forward axis (+z).
    """
    fwd = global_orient[:, :, 2]  # R @ [0, 0, 1]
    yaw = torch.atan2(fwd[:, 2], fwd[:, 0])
    c, s = torch.cos(yaw)[:, None], torch.sin(yaw)[:, None]
    x, y, z = verts[..., 0], verts[..., 1], verts[..., 2]
    a = c * x + s * z  # along heading
    b = -s * x + c * z  # across
    lo_a, hi_a = a.amin(1), a.amax(1)
    lo_b, hi_b = b.amin(1), b.amax(1)
    lo_y, hi_y = y.amin(1), y.amax(1)
    ca, cb = (lo_a + hi_a) / 2, (lo_b + hi_b) / 2
    cx = c[:, 0] * ca - s[:, 0] * cb
    cz = s[:, 0] * ca + c[:, 0] * cb
    cy = (lo_y + hi_y) / 2
    return torch.stack(
        [cx, cy, cz, hi_a - lo_a, hi_y - lo_y, hi_b - lo_b, yaw], -1
    )


class SelectiveFusionModel(nn.Module):
    """Image + LiDAR -> SMPL pose, shape and 3D placement."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = ViT(cfg.vit)
        if cfg.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
        self.img_proj = nn.Linear(cfg.vit.embed_dim, cfg.dim)
        self.points = PointTokenizer(
            cfg.dim, cfg.point_tokens, cfg.point_knn, num_heads=cfg.num_heads
        )
        self.decoder = SelectiveDecoder(cfg.dim, cfg.num_heads, cfg.num_layers)
        self.heads = SmplHeads(cfg.dim, cfg.init_depth_m)
        self.smpl: nn.Module | None = None
        if cfg.smpl_model_dir is not None:
            import smplx

            self.smpl = smplx.create(
                str(cfg.smpl_model_dir), model_type="smpl", gender="neutral",
                num_betas=NUM_BETAS, use_pose_blendshapes=True,
                create_global_orient=False, create_body_pose=False,
                create_betas=False, create_transl=False,
            )  # fmt: skip
            for p in self.smpl.parameters():
                p.requires_grad_(False)

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Predict from ``image``, ``points`` and crop ``intrinsics``."""
        tokens, _ = self.backbone(batch["image"])
        img = self.img_proj(tokens)
        pts, _ = self.points(batch["points"])
        feats, gates = self.decoder(img, pts)
        rots, betas, raw_t = self.heads(feats)
        size = batch["image"].shape[-1]
        transl = translation_from_raw(raw_t, batch["intrinsics"], size)
        out: dict[str, Tensor] = {
            "global_orient": rots[:, :1],
            "body_pose": rots[:, 1:],
            "betas": betas,
            "transl": transl,
            "gates": gates,
        }
        if self.smpl is not None:
            self._add_smpl_outputs(out, batch["intrinsics"])
        return out

    def _add_smpl_outputs(self, out: dict[str, Tensor], k: Tensor) -> None:
        assert self.smpl is not None
        # smplx.SMPL with rotation matrices (pose2rot=False); transl is added
        # by smplx after the pelvis-centred rotation, matching our convention
        smpl_out = self.smpl(
            betas=out["betas"],
            global_orient=out["global_orient"],
            body_pose=out["body_pose"],
            transl=out["transl"],
            pose2rot=False,
        )
        joints = smpl_out.joints[:, :NUM_JOINTS]
        out["vertices"] = smpl_out.vertices
        out["joints3d"] = joints
        out["kp2d"] = project(joints, k)
        out["box3d"] = box_from_vertices(
            smpl_out.vertices, out["global_orient"][:, 0]
        )
