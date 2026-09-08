"""Training losses on batches produced by :mod:`data.torch_dataset`.

All functions take a prediction dict and the batch dict, respect the
``has_*`` / ``*_valid`` flags, and return per-batch scalars. Everything
is in the crop camera frame (metres, crop pixels).

Prediction keys (all float tensors, batch first):
``global_orient`` (B, 3, 3) or (B, 3), ``body_pose`` (B, 23, 3, 3) or
(B, 69), ``betas`` (B, 10), ``transl`` (B, 3), ``joints3d`` (B, J, 3),
``kp2d`` (B, J, 2), ``box3d`` (B, 7).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor


def rodrigues(aa: Tensor) -> Tensor:
    """Axis-angle (..., 3) to rotation matrices (..., 3, 3)."""
    angle = torch.linalg.norm(aa, dim=-1, keepdim=True).clamp_min(1e-8)
    axis = aa / angle
    x, y, z = axis.unbind(-1)
    zero = torch.zeros_like(x)
    k = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], dim=-1).reshape(
        *aa.shape[:-1], 3, 3
    )
    eye = torch.eye(3, dtype=aa.dtype, device=aa.device)
    sin = torch.sin(angle)[..., None]
    cos = torch.cos(angle)[..., None]
    return eye + sin * k + (1 - cos) * (k @ k)


def as_rotmats(x: Tensor, n_joints: int) -> Tensor:
    """Accept axis-angle or rotation matrices; return (B, n_joints, 3, 3)."""
    if x.shape[-1] == 3 and x.dim() >= 3 and x.shape[-2] == 3:
        return x.reshape(x.shape[0], n_joints, 3, 3)
    return rodrigues(x.reshape(x.shape[0], n_joints, 3))


def _masked_mean(err: Tensor, mask: Tensor) -> Tensor:
    """Mean of ``err`` over the entries where ``mask`` is True (or 0)."""
    m = mask.to(err.dtype)
    denom = m.sum().clamp_min(1.0)
    return (err * m).sum() / denom


def smpl_param_loss(
    pred: dict[str, Tensor], batch: dict[str, Tensor]
) -> dict[str, Tensor]:
    """MSE on rotation matrices (orient, body pose) and on betas."""
    has = batch["has_smpl"]
    go_p = as_rotmats(pred["global_orient"], 1)
    go_g = as_rotmats(batch["global_orient"], 1)
    bp_p = as_rotmats(pred["body_pose"], 23)
    bp_g = as_rotmats(batch["body_pose"], 23)
    per_sample_go = ((go_p - go_g) ** 2).flatten(1).mean(1)
    per_sample_bp = ((bp_p - bp_g) ** 2).flatten(1).mean(1)
    per_sample_betas = ((pred["betas"] - batch["betas"]) ** 2).mean(1)
    return {
        "global_orient": _masked_mean(per_sample_go, has),
        "body_pose": _masked_mean(per_sample_bp, has),
        "betas": _masked_mean(per_sample_betas, has),
    }


def joints3d_loss(pred: Tensor, batch: dict[str, Tensor]) -> Tensor:
    """L1 on 3D joints in the camera frame, masked by joint validity."""
    gt = batch["joints3d"][:, : pred.shape[1]]
    valid = batch["joints3d_valid"][:, : pred.shape[1]]
    err = (pred - gt).abs().sum(-1)
    return _masked_mean(err, valid)


def kp2d_loss(
    pred_px: Tensor, batch: dict[str, Tensor], crop_size: int
) -> Tensor:
    """L1 on 2D keypoints in crop units, over confident keypoints."""
    gt = batch["kp2d"][:, : pred_px.shape[1]]
    conf = gt[..., 2] * batch["has_kp2d"][:, None].to(gt.dtype)
    err = (pred_px - gt[..., :2]).abs().sum(-1) / crop_size
    return _masked_mean(err, conf > 0) if conf.sum() > 0 else err.sum() * 0


def translation_loss(pred: Tensor, batch: dict[str, Tensor]) -> Tensor:
    """L1 on the SMPL translation (3D placement) in the camera frame."""
    err = (pred - batch["transl"]).abs().sum(-1)
    return _masked_mean(err, batch["has_smpl"])


def box3d_loss(pred: Tensor, batch: dict[str, Tensor]) -> Tensor:
    """Centre L1 + size L1 + heading (1 - cos) on the 7-vector 3D box."""
    gt = batch["box3d"]
    centre = (pred[:, :3] - gt[:, :3]).abs().sum(-1)
    size = (pred[:, 3:6] - gt[:, 3:6]).abs().sum(-1)
    yaw = 1.0 - torch.cos(pred[:, 6] - gt[:, 6])
    return _masked_mean(centre + size + yaw, batch["has_box3d"])


@dataclass
class LossWeights:
    """Relative weights of the loss terms."""

    global_orient: float = 1.0
    body_pose: float = 1.0
    betas: float = 0.005
    joints3d: float = 5.0
    kp2d: float = 1.0
    transl: float = 5.0
    box3d: float = 1.0
    crop_size: int = 256
    extra: dict[str, float] = field(default_factory=dict)


class FusionLoss:
    """Weighted sum of all terms; returns the total and the parts."""

    def __init__(self, weights: LossWeights | None = None) -> None:
        self.w = weights or LossWeights()

    def __call__(
        self, pred: dict[str, Tensor], batch: dict[str, Tensor]
    ) -> tuple[Tensor, dict[str, Tensor]]:
        parts = smpl_param_loss(pred, batch)
        parts["joints3d"] = joints3d_loss(pred["joints3d"], batch)
        parts["transl"] = translation_loss(pred["transl"], batch)
        parts["box3d"] = box3d_loss(pred["box3d"], batch)
        if "kp2d" in pred:
            parts["kp2d"] = kp2d_loss(pred["kp2d"], batch, self.w.crop_size)
        total = torch.zeros(
            (), dtype=pred["betas"].dtype, device=pred["betas"].device
        )
        for name, value in parts.items():
            total = total + getattr(self.w, name) * value
        return total, parts
