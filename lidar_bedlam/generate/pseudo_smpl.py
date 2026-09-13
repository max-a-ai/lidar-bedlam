"""Pseudo ground-truth SMPL for keypoint-only datasets (Waymo).

A trained fusion model initialises the parameters; a short optimisation
then fits pose, shape and translation to the record's evidence:

- the 3D keypoints (COCO joints regressed from the mesh vs the 13 shared
  Waymo keypoints),
- the 2D keypoints (projected with the crop intrinsics),
- the LiDAR returns, which should lie a few centimetres *outside* the
  mesh (clothing), so their nearest-vertex distance is pulled to
  ``lidar_offset_m``,
- priors that keep the pose close to the initial estimate and the shape
  small.

Records whose fitted 3D keypoint error exceeds ``max_error_m`` keep
``has_smpl=False``; the others get the fitted parameters as labels while
their real keypoints stay the joint supervision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation
from torch import Tensor

from lidar_bedlam.data.schema import WAYMO15_TO_COCO17
from lidar_bedlam.models.fusion import SelectiveFusionModel, project
from lidar_bedlam.models.rotation import matrix_to_rot6d, rot6d_to_matrix


@dataclass(frozen=True)
class PseudoFitConfig:
    """Optimisation settings."""

    iters: int = 200
    lr: float = 0.02
    w_joints3d: float = 1.0
    w_kp2d: float = 0.5
    w_lidar: float = 0.5
    lidar_offset_m: float = 0.03
    w_pose_prior: float = 0.05
    w_betas: float = 0.01
    max_error_m: float = 0.08
    crop_size: int = 256


@dataclass
class FitResult:
    """Fitted parameters of one batch (numpy)."""

    global_orient: NDArray[np.float64]  # (B, 3) axis-angle
    body_pose: NDArray[np.float64]  # (B, 69)
    betas: NDArray[np.float64]  # (B, 10)
    transl: NDArray[np.float64]  # (B, 3)
    error_m: NDArray[np.float64]  # (B,) mean 3D keypoint error after fit
    error_init_m: NDArray[np.float64]  # (B,) before the fit
    accepted: NDArray[np.bool_]


_WSEL = np.nonzero(WAYMO15_TO_COCO17 >= 0)[0]
_CSEL = WAYMO15_TO_COCO17[_WSEL]


class PseudoSmplFitter:
    """Batched SMPL fitting on top of a trained fusion model."""

    def __init__(
        self,
        model: SelectiveFusionModel,
        cfg: PseudoFitConfig,
        device: torch.device,
    ) -> None:
        assert model.smpl is not None and model.coco_regressor is not None
        self.model = model.to(device).eval()
        self.cfg = cfg
        self.device = device
        self.wsel = torch.from_numpy(_WSEL).to(device)
        self.csel = torch.from_numpy(_CSEL).to(device)

    def _mesh(
        self, rot6d: Tensor, betas: Tensor, transl: Tensor
    ) -> tuple[Tensor, Tensor]:
        rots = rot6d_to_matrix(rot6d.reshape(-1, 6)).reshape(
            rot6d.shape[0], 24, 3, 3
        )
        assert self.model.smpl is not None
        out = self.model.smpl(
            betas=betas,
            global_orient=rots[:, :1],
            body_pose=rots[:, 1:],
            transl=transl,
            pose2rot=False,
        )
        verts: Tensor = out.vertices
        coco = torch.einsum("jv,bvc->bjc", self.model.coco_regressor, verts)
        return verts, coco

    def _keypoint_error(
        self, coco: Tensor, batch: dict[str, Tensor]
    ) -> Tensor:
        gt = batch["joints3d"][:, self.wsel]
        valid = batch["joints3d_valid"][:, self.wsel].float()
        err = (coco[:, self.csel] - gt).norm(dim=-1)
        mean: Tensor = (err * valid).sum(1) / valid.sum(1).clamp_min(1)
        return mean

    @torch.no_grad()
    def initialise(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Parameters predicted by the model (6D rotations)."""
        out = self.model(batch)
        rots = torch.cat([out["global_orient"], out["body_pose"]], 1)
        return {
            "rot6d": matrix_to_rot6d(rots.reshape(-1, 3, 3)).reshape(
                rots.shape[0], 24, 6
            ),
            "betas": out["betas"],
            "transl": out["transl"],
        }

    def fit(self, batch: dict[str, Tensor]) -> FitResult:
        """Optimise one batch; tensors must already be on the device."""
        cfg = self.cfg
        init = self.initialise(batch)
        rot6d = init["rot6d"].clone().requires_grad_(True)
        betas = init["betas"].clone().requires_grad_(True)
        transl = init["transl"].clone().requires_grad_(True)
        rot_init = init["rot6d"].clone()
        with torch.no_grad():
            _, coco0 = self._mesh(rot6d, betas, transl)
            err_init = self._keypoint_error(coco0, batch)
        opt = torch.optim.Adam([rot6d, betas, transl], lr=cfg.lr)
        pts = batch["points"]
        pts_valid = batch["points_valid"].float()
        kp = batch["kp2d"][:, self.wsel]
        kp_conf = (kp[..., 2] > 0).float() * batch["has_kp2d"][:, None].float()
        for _ in range(cfg.iters):
            opt.zero_grad(set_to_none=True)
            verts, coco = self._mesh(rot6d, betas, transl)
            loss = cfg.w_joints3d * self._keypoint_error(coco, batch).mean()
            uv = project(coco[:, self.csel], batch["intrinsics"])
            kp_err = (uv - kp[..., :2]).abs().sum(-1) / cfg.crop_size
            loss = loss + cfg.w_kp2d * (
                kp_err * kp_conf
            ).sum() / kp_conf.sum().clamp_min(1)
            dist = torch.cdist(pts, verts).min(-1).values
            lid = (dist - cfg.lidar_offset_m).abs()
            loss = loss + cfg.w_lidar * (
                lid * pts_valid
            ).sum() / pts_valid.sum().clamp_min(1)
            loss = (
                loss
                + cfg.w_pose_prior
                * ((rot6d[:, 1:] - rot_init[:, 1:]) ** 2).mean()
            )
            loss = loss + cfg.w_betas * (betas**2).mean()
            loss.backward()  # type: ignore[no-untyped-call]
            opt.step()
        with torch.no_grad():
            _, coco = self._mesh(rot6d, betas, transl)
            err = self._keypoint_error(coco, batch)
            rots = rot6d_to_matrix(rot6d.reshape(-1, 6)).cpu().numpy()
        aa = Rotation.from_matrix(rots).as_rotvec().reshape(-1, 24, 3)
        err_np = err.cpu().numpy().astype(np.float64)
        return FitResult(
            global_orient=aa[:, 0],
            body_pose=aa[:, 1:].reshape(-1, 69),
            betas=betas.detach().cpu().numpy().astype(np.float64),
            transl=transl.detach().cpu().numpy().astype(np.float64),
            error_m=err_np,
            error_init_m=err_init.cpu().numpy().astype(np.float64),
            accepted=err_np <= cfg.max_error_m,
        )


def summarize(results: list[FitResult]) -> dict[str, Any]:
    """Acceptance rate and error statistics over all fitted records."""
    err = np.concatenate([r.error_m for r in results])
    init = np.concatenate([r.error_init_m for r in results])
    acc = np.concatenate([r.accepted for r in results])
    return {
        "records": int(len(err)),
        "accepted": int(acc.sum()),
        "acceptance_rate": float(acc.mean()),
        "keypoint_error_init_mm": float(init.mean() * 1000),
        "keypoint_error_fit_mm": float(err.mean() * 1000),
        "keypoint_error_fit_accepted_mm": float(err[acc].mean() * 1000)
        if acc.any()
        else float("nan"),
    }
