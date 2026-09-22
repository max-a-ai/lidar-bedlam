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
    # "camera": pelvis as (u, v, log z) in the crop, unprojected with K;
    # "points": centroid of the valid input points plus a metric offset
    # (samples without points fall back to the camera parameterisation)
    transl_anchor: str = "camera"
    freeze_backbone: bool = True
    use_backbone: bool = True  # False: train from precomputed tokens only
    crop_size: int = 256
    # "rot6d": a 6-D rotation head per joint group (the default);
    # "token": the 21 body rotations are decoded by the frozen TokenHMR
    # tokenizer from a latent that a query head predicts (TokenPoseHead),
    # so every body pose lies on the codebook manifold
    pose_head: str = "rot6d"
    tokenizer_path: Path | None = None


class TokenPoseHead(nn.Module):
    """Predict the tokenizer latent (B, C, T) from the group features.

    ``num_tokens`` learnable queries cross-attend to the decoder's joint
    group features in a small transformer, then a linear map gives each
    query its code vector. The latent is quantised to the nearest codebook
    entry with a straight-through estimator, so the frozen decoder always
    sees a codebook pose while the gradient reaches the queries.
    """

    def __init__(
        self, dim: int, num_heads: int, code_dim: int, num_tokens: int
    ) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_tokens, dim) * 0.02)
        layer = nn.TransformerDecoderLayer(
            dim, num_heads, dim_feedforward=2 * dim, dropout=0.0,
            batch_first=True, norm_first=True,
        )  # fmt: skip
        self.decoder = nn.TransformerDecoder(layer, num_layers=2)
        self.proj = nn.Linear(dim, code_dim)

    def forward(self, feats: Tensor) -> Tensor:
        """Group features (B, G, dim) -> latent (B, C, T)."""
        q = self.queries[None].expand(feats.shape[0], -1, -1)
        out: Tensor = self.proj(self.decoder(q, feats).float())
        return out.permute(0, 2, 1)


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
        self.offset_head = nn.Linear(dim, 3)  # metres from the point centroid
        nn.init.zeros_(self.offset_head.weight)
        nn.init.zeros_(self.offset_head.bias)
        self.shape_head = nn.Linear(dim, NUM_BETAS)
        nn.init.zeros_(self.shape_head.weight)
        nn.init.zeros_(self.shape_head.bias)
        # padded-box residual (Waymo convention): centre offset (m) and
        # log size scale on top of the mesh-extent box
        self.box_head = nn.Linear(dim, 6)
        nn.init.zeros_(self.box_head.weight)
        nn.init.zeros_(self.box_head.bias)
        self.root_index = [g.name for g in JOINT_GROUPS].index("root")
        self.shape_index = [g.name for g in JOINT_GROUPS].index("shape")

    def forward(self, feats: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Rotations (B, 24, 3, 3), betas (B, 10), raw transl (B, 3) and
        the metric offset from the point centroid (B, 3)."""
        b = feats.shape[0]
        # float32 heads regardless of autocast: rotations and the SMPL
        # forward that follows are numerically sensitive
        rot6d = feats.new_zeros(b, NUM_JOINTS, 6, dtype=torch.float32)
        for idx, g in enumerate(JOINT_GROUPS):
            if not g.joints:
                continue
            rot6d[:, list(g.joints)] = (
                self.rot_heads[g.name](feats[:, idx])
                .float()
                .reshape(b, len(g.joints), 6)
            )
        rots = rot6d_to_matrix(rot6d)
        betas = self.shape_head(feats[:, self.shape_index]).float()
        root = feats[:, self.root_index]
        raw_t = self.transl_head(root).float()
        offset = self.offset_head(root).float()
        return rots, betas, raw_t, offset

    def box_residual(self, feats: Tensor) -> tuple[Tensor, Tensor]:
        """Centre offset (B, 3) in metres and log size scale (B, 3) of the
        padded box relative to the mesh-extent box."""
        r = self.box_head(feats[:, self.root_index]).float()
        return r[:, :3], r[:, 3:]


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


def point_centroid(
    points: Tensor, valid: Tensor | None
) -> tuple[Tensor, Tensor]:
    """Mean of the valid points (B, 3) and whether any were valid (B,)."""
    if valid is None:
        valid = torch.ones(
            points.shape[:2], dtype=torch.bool, device=points.device
        )
    w = valid.to(points.dtype)[..., None]
    n = w.sum(1)
    centroid = (points * w).sum(1) / n.clamp(min=1.0)
    return centroid, n.squeeze(-1) > 0


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

    coco_regressor: Tensor | None  # (17, 6890) COCO joint regressor buffer

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone: ViT | None = ViT(cfg.vit) if cfg.use_backbone else None
        if self.backbone is not None and cfg.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
        self.img_proj = nn.Linear(cfg.vit.embed_dim, cfg.dim)
        # stands in for the image tokens of LiDAR-only samples
        self.no_image = nn.Parameter(torch.zeros(1, 1, cfg.dim))
        # clothing offset of the LiDAR returns outside the body surface,
        # learned by the chamfer term (metres)
        self.clothing_offset = nn.Parameter(torch.tensor(0.02))
        self.iou_head = nn.Linear(cfg.dim, 1)
        self.points = PointTokenizer(
            cfg.dim, cfg.point_tokens, cfg.point_knn, num_heads=cfg.num_heads
        )
        self.decoder = SelectiveDecoder(cfg.dim, cfg.num_heads, cfg.num_layers)
        self.heads = SmplHeads(cfg.dim, cfg.init_depth_m)
        self.tokenizer: nn.Module | None = None
        self.token_head: TokenPoseHead | None = None
        if cfg.pose_head == "token":
            from lidar_bedlam.body.pose_tokenizer import PoseTokenizer

            if cfg.tokenizer_path is None:
                msg = "pose_head 'token' needs tokenizer_path"
                raise ValueError(msg)
            tok = PoseTokenizer.load(cfg.tokenizer_path)
            self.tokenizer = tok
            self.token_head = TokenPoseHead(
                cfg.dim, cfg.num_heads, tok.code_dim, tok.num_tokens
            )
        elif cfg.pose_head != "rot6d":
            msg = f"unknown pose_head {cfg.pose_head!r}"
            raise ValueError(msg)
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
        # COCO-17 joints regressed from the mesh: supervision and evaluation
        # for datasets with keypoint labels (Waymo)
        reg = (
            cfg.smpl_model_dir / "J_regressor_coco.npy"
            if cfg.smpl_model_dir is not None
            else None
        )
        self.register_buffer(
            "coco_regressor",
            torch.from_numpy(np.load(reg)).float()
            if reg is not None and reg.exists()
            else None,
            persistent=False,
        )

    def image_tokens(self, batch: dict[str, Tensor]) -> Tensor:
        """Projected image tokens from ``tokens`` or from the backbone."""
        if "tokens" in batch:
            tokens = batch["tokens"].float()
        elif self.backbone is not None:
            tokens, _ = self.backbone(batch["image"])
        else:
            msg = "model without backbone needs precomputed 'tokens'"
            raise KeyError(msg)
        img: Tensor = self.img_proj(tokens)
        if "has_image" in batch:
            keep = batch["has_image"].to(img.dtype)[:, None, None]
            img = keep * img + (1 - keep) * self.no_image.expand_as(img)
        return img

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Predict from ``image`` / ``tokens``, ``points``, ``intrinsics``."""
        img = self.image_tokens(batch)
        pts, _ = self.points(batch["points"])
        feats, gates = self.decoder(img, pts)
        rots, betas, raw_t, offset = self.heads(feats)
        commit: Tensor | None = None
        if self.token_head is not None and self.tokenizer is not None:
            latent = self.token_head(feats)
            q, _ = self.tokenizer.quantize(latent)  # type: ignore[operator]
            commit = ((latent - q.detach()) ** 2).mean()
            body = self.tokenizer.decode(  # type: ignore[operator]
                latent + (q - latent).detach()
            )
            rots = torch.cat([rots[:, :1], body, rots[:, 22:]], 1)
        transl = translation_from_raw(
            raw_t, batch["intrinsics"], self.cfg.crop_size
        )
        if self.cfg.transl_anchor == "points":
            centroid, has_pts = point_centroid(
                batch["points"].float(), batch.get("points_valid")
            )
            transl = torch.where(has_pts[:, None], centroid + offset, transl)
        elif self.cfg.transl_anchor != "camera":
            msg = f"unknown transl_anchor {self.cfg.transl_anchor!r}"
            raise ValueError(msg)
        conf: Tensor = torch.sigmoid(
            self.iou_head(feats[:, self.heads.root_index]).squeeze(-1)
        )
        out: dict[str, Tensor] = {
            "global_orient": rots[:, :1],
            "body_pose": rots[:, 1:],
            "betas": betas,
            "transl": transl,
            "gates": gates,
            "box_conf": conf,
            "clothing_offset": self.clothing_offset.clamp(0.0, 0.1),
        }
        if commit is not None:
            out["token_commit"] = commit
        if self.smpl is not None:
            self._add_smpl_outputs(out, batch["intrinsics"])
            out["box3d_padded"] = self._padded_box(out, feats)
        return out

    def _padded_box(self, out: dict[str, Tensor], feats: Tensor) -> Tensor:
        """The Waymo-convention box: the mesh-extent box, its vertices
        detached, plus the learned residual. Its loss reaches the box head
        and the translation, never the pose or the shape."""
        centre_off, log_scale = self.heads.box_residual(feats)
        box = out["box3d"].detach()
        transl = out["transl"]
        centre = box[:, :3] - transl.detach() + transl + centre_off
        size = box[:, 3:6] * torch.exp(log_scale)
        return torch.cat([centre, size, box[:, 6:7]], -1)

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
        if self.coco_regressor is not None:
            coco = torch.einsum(
                "jv,bvc->bjc", self.coco_regressor, smpl_out.vertices
            )
            out["joints_coco"] = coco
            out["kp2d_coco"] = project(coco, k)
        out["box3d"] = box_from_vertices(
            smpl_out.vertices, out["global_orient"][:, 0]
        )
