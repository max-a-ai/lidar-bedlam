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

import numpy as np
import torch
from torch import Tensor

from lidar_bedlam.data.schema import WAYMO15_TO_COCO17


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


_WAYMO_SEL = torch.from_numpy(np.nonzero(WAYMO15_TO_COCO17 >= 0)[0])
_COCO_SEL = torch.from_numpy(WAYMO15_TO_COCO17[WAYMO15_TO_COCO17 >= 0])


def _by_convention(
    pred_smpl: Tensor,
    pred_coco: Tensor | None,
    gt: Tensor,
    valid: Tensor,
    conv: Tensor | None,
) -> tuple[Tensor, Tensor]:
    """Per-sample error rows and masks, comparing SMPL-24 rows to the SMPL
    joints and waymo15 rows to the COCO joints regressed from the mesh."""
    n = pred_smpl.shape[1]
    err = (pred_smpl - gt[:, :n]).abs().sum(-1)
    mask = valid[:, :n].clone()
    if conv is None or pred_coco is None:
        return err, mask
    waymo = conv == 1
    if not bool(waymo.any()):
        return err, mask
    wsel = _WAYMO_SEL.to(gt.device)
    csel = _COCO_SEL.to(gt.device)
    err_w = (pred_coco[:, csel] - gt[:, wsel]).abs().sum(-1)
    mask_w = valid[:, wsel] & waymo[:, None]
    mask = mask & ~waymo[:, None]
    return torch.cat([err, err_w], 1), torch.cat([mask, mask_w], 1)


def joints3d_loss(pred: dict[str, Tensor], batch: dict[str, Tensor]) -> Tensor:
    """L1 on 3D joints in the camera frame, convention aware."""
    err, mask = _by_convention(
        pred["joints3d"],
        pred.get("joints_coco"),
        batch["joints3d"],
        batch["joints3d_valid"],
        batch.get("joint_convention_id"),
    )
    return _masked_mean(err, mask)


def kp2d_loss(
    pred: dict[str, Tensor], batch: dict[str, Tensor], crop_size: int
) -> Tensor:
    """L1 on 2D keypoints in crop units, over confident keypoints."""
    gt = batch["kp2d"]
    conf = (gt[..., 2] > 0) & batch["has_kp2d"][:, None]
    err, mask = _by_convention(
        pred["kp2d"],
        pred.get("kp2d_coco"),
        gt[..., :2],
        conf,
        batch.get("joint_convention_id"),
    )
    err = err / crop_size
    return _masked_mean(err, mask) if mask.any() else err.sum() * 0


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


def box_conf_loss(
    pred_conf: Tensor, pred_box: Tensor, batch: dict[str, Tensor]
) -> Tensor:
    """L1 between the predicted confidence and the true 3D IoU (no grad)."""
    from lidar_bedlam.geometry.boxes import iou3d
    from lidar_bedlam.geometry.camera import CAMERA_UP_AXIS

    pb = pred_box.detach().cpu().double().numpy()
    gb = batch["box3d"].detach().cpu().double().numpy()
    target = torch.tensor(
        [iou3d(a, b, CAMERA_UP_AXIS) for a, b in zip(pb, gb, strict=True)],
        dtype=pred_conf.dtype,
        device=pred_conf.device,
    )
    return _masked_mean((pred_conf - target).abs(), batch["has_box3d"])


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
    box_conf: float = 1.0
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
        parts["joints3d"] = joints3d_loss(pred, batch)
        parts["transl"] = translation_loss(pred["transl"], batch)
        parts["box3d"] = box3d_loss(pred["box3d"], batch)
        if "kp2d" in pred:
            parts["kp2d"] = kp2d_loss(pred, batch, self.w.crop_size)
        if "box_conf" in pred and "box3d" in pred:
            parts["box_conf"] = box_conf_loss(
                pred["box_conf"], pred["box3d"], batch
            )
        total = torch.zeros(
            (), dtype=pred["betas"].dtype, device=pred["betas"].device
        )
        for name, value in parts.items():
            total = total + getattr(self.w, name) * value
        return total, parts
