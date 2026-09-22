"""Pseudo ground truth v2: accurate *and* realistic SMPL labels for records
that carry keypoints and a scan but no mesh (Waymo).

The first fitter (``pseudo_smpl``) optimised the joint rotations directly
and matched the 13 keypoints with bodies that look wrong: nothing held
the shape, the torso or the joints no keypoint observes. This one keeps
every candidate inside the space of human poses and shapes:

- **pose** lives in the latent space of TokenHMR's pose tokenizer
  (``body.pose_tokenizer``), a VQ-VAE trained on AMASS and MOYO. The
  optimiser moves the latent, the decoder turns it into a body pose, and
  a commitment term keeps the latent close to the codebook the decoder
  was trained on. Hands stay at rest.
- **shape** is one set of betas per track (all records of one Waymo
  object), with an L2 prior.
- **initialisation** is a plausible body: LiDAR-HMR's published fit of
  the same frame, or TokenHMR's image prediction, never a T-pose. A
  trust-region term penalises drift from it.
- **evidence**: the 3D keypoints, the 2D keypoints, the LiDAR returns
  pulled a clothing offset outside the mesh, and the ground: the box
  bottom of the Waymo label, which the lowest body point may not sink
  below and should not float above.
- **temporal smoothness** between records of one track less than
  ``smooth_gap_s`` apart.

After the fit every record gets three residuals (keypoint error, median
point-to-mesh distance, prior energy of the pose under the BEDLAM
statistics), a gate on all three, and a confidence in [0, 1] the
training loss weights the mesh terms with. Rejected records keep their
keypoints only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation
from torch import Tensor

from lidar_bedlam.body.pose_tokenizer import PoseTokenizer
from lidar_bedlam.body.smpl import NUM_BETAS
from lidar_bedlam.data.schema import WAYMO15_TO_COCO17
from lidar_bedlam.losses.pose_prior import PosePrior
from lidar_bedlam.models.fusion import project
from lidar_bedlam.models.rotation import matrix_to_rot6d, rot6d_to_matrix

FloatArray = NDArray[np.float64]
_WSEL = np.nonzero(WAYMO15_TO_COCO17 >= 0)[0]
_CSEL = WAYMO15_TO_COCO17[_WSEL]
NUM_BODY = 21  # tokenised joints; 22 and 23 are the hands


@dataclass(frozen=True)
class FitV2Config:
    """Weights, schedule and gate of the fit."""

    iters: int = 150
    lr: float = 0.01
    w_joints3d: float = 1.0
    # hip centre (COCO 11, 12 mean) against the labelled hip centre: the
    # root the evaluation and the training placement loss are measured on
    w_root: float = 1.0
    w_kp2d: float = 0.5
    w_lidar: float = 0.5
    lidar_offset_m: float = 0.03
    w_ground: float = 1.0
    ground_tolerance_m: float = 0.05
    w_trust_pose: float = 0.5
    w_trust_orient: float = 0.5
    w_trust_transl: float = 0.1
    w_commit: float = 0.1
    w_smooth: float = 0.5
    smooth_gap_s: float = 0.25
    w_betas: float = 0.01
    max_kp_error_m: float = 0.08
    max_chamfer_m: float = 0.06
    max_prior_energy: float = 6.0
    conf_kp_sigma_m: float = 0.05
    conf_chamfer_sigma_m: float = 0.04
    crop_size: int = 256
    vertex_stride: int = 4  # every k-th vertex in the point-to-mesh term
    # SMPL joints no Waymo keypoint observes and no data term constrains:
    # set to the rest rotation after decoding (wrists 20, 21; feet 10, 11).
    # The ankles stay free (ground contact and the ankle keypoint hold
    # them), the head too (nose, forehead and head centre are labelled).
    neutral_joints: tuple[int, ...] = (10, 11, 20, 21)
    # constant offset per Waymo keypoint (13 shared joints, in the order of
    # ``WAYMO15_TO_COCO17 >= 0``) from the regressed COCO joint to the
    # annotated point, in the pelvis-centred body frame, metres. Waymo
    # annotates the hips wider (trochanter) and the shoulders narrower than
    # the COCO regressor places them, and the knees a little lower; measured
    # on 1,900 confident pseudo-GT v2 fits (2026-09-22).
    joint_offset_m: tuple[tuple[float, float, float], ...] = (
        (0.000, -0.003, -0.005),  # nose
        (-0.020, 0.003, 0.003),  # left shoulder
        (-0.001, 0.000, 0.001),  # left elbow
        (-0.001, -0.003, 0.000),  # left wrist
        (0.023, 0.000, 0.002),  # left hip
        (-0.001, -0.016, -0.002),  # left knee
        (0.001, 0.001, -0.001),  # left ankle
        (0.019, 0.005, 0.004),  # right shoulder
        (0.001, -0.004, 0.001),  # right elbow
        (-0.001, 0.000, -0.002),  # right wrist
        (-0.027, -0.001, 0.004),  # right hip
        (0.000, -0.012, -0.003),  # right knee
        (-0.002, 0.007, 0.000),  # right ankle
    )


@dataclass
class FitBatch:
    """Records of one or more tracks, numpy, in each record's camera frame.

    ``track`` indexes the betas shared within a track; ``time_s`` orders
    the records of a track for the smoothness term.
    """

    keys: list[str]
    track: NDArray[np.int64]
    time_s: FloatArray
    joints3d: FloatArray  # (N, 15, 3)
    joints3d_valid: NDArray[np.bool_]  # (N, 15)
    kp2d: FloatArray  # (N, 15, 3) crop pixels + confidence
    intrinsics: FloatArray  # (N, 3, 3)
    points: FloatArray  # (N, P, 3)
    points_valid: NDArray[np.bool_]  # (N, P)
    box3d: FloatArray  # (N, 7) cx cy cz dx dy dz yaw, camera frame
    init_global_orient: FloatArray  # (N, 3) axis-angle
    init_body_pose: FloatArray  # (N, 69)
    init_betas: FloatArray  # (N, 10)
    init_transl: FloatArray  # (N, 3)


@dataclass
class FitV2Result:
    """Fitted parameters, residuals, gate and confidence per record."""

    global_orient: FloatArray
    body_pose: FloatArray
    betas: FloatArray
    transl: FloatArray
    kp_error_m: FloatArray
    chamfer_m: FloatArray
    prior_energy: FloatArray
    accepted: NDArray[np.bool_]
    confidence: FloatArray


class PseudoFitterV2:
    """Batched fit in the tokenizer's latent space."""

    def __init__(
        self,
        body_models: Path,
        tokenizer: Path,
        device: torch.device,
        cfg: FitV2Config | None = None,
    ) -> None:
        import smplx

        self.cfg = cfg or FitV2Config()
        self.device = device
        self.smpl = smplx.create(
            str(body_models), model_type="smpl", gender="neutral",
            num_betas=NUM_BETAS, use_pose_blendshapes=True,
            create_global_orient=False, create_body_pose=False,
            create_betas=False, create_transl=False,
        ).to(device)  # fmt: skip
        for p in self.smpl.parameters():
            p.requires_grad_(False)
        self.coco = (
            torch.from_numpy(np.load(body_models / "J_regressor_coco.npy"))
            .float()
            .to(device)
        )
        self.tok = PoseTokenizer.load(tokenizer).to(device)
        self.prior = PosePrior(
            body_models / "pose_prior_bedlam.npz",
            joints=tuple(range(1, NUM_BODY + 1)),
        )
        self.wsel = torch.from_numpy(_WSEL).to(device)
        self.csel = torch.from_numpy(_CSEL).to(device)

    # -- pieces ----------------------------------------------------------

    def _body(self, decoded: Tensor) -> Tensor:
        """Full body pose (N, 23, 3, 3): the decoded 21 joints with the
        unobserved joints at rest, plus the hands at rest."""
        n = decoded.shape[0]
        eye = torch.eye(3, device=decoded.device)
        body = decoded.clone()
        for j in self.cfg.neutral_joints:
            body[:, j - 1] = eye
        hands = eye.expand(n, 2, 3, 3)
        return torch.cat([body, hands], 1)

    def _mesh(
        self, orient: Tensor, body: Tensor, betas: Tensor, transl: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Vertices and COCO joints; ``body`` (N, 23, 3, 3)."""
        out = self.smpl(
            betas=betas,
            global_orient=orient,
            body_pose=body,
            transl=transl,
            pose2rot=False,
        )
        verts: Tensor = out.vertices
        coco = torch.einsum("jv,bvc->bjc", self.coco, verts)
        return verts, coco

    def _kp_error(
        self,
        coco: Tensor,
        gt: Tensor,
        valid: Tensor,
        orient: Tensor | None = None,
    ) -> Tensor:
        """Mean distance of the regressed joints to the labelled keypoints;
        with ``orient`` (B, 3, 3) the joints carry the convention offsets."""
        pred = coco[:, self.csel]
        if orient is not None:
            off = torch.as_tensor(
                self.cfg.joint_offset_m, dtype=coco.dtype, device=coco.device
            )
            pred = pred + torch.einsum("bij,kj->bki", orient, off)
        err = (pred - gt[:, self.wsel]).norm(dim=-1)
        v = valid[:, self.wsel].float()
        out: Tensor = (err * v).sum(1) / v.sum(1).clamp_min(1)
        return out

    def _root_error(self, coco: Tensor, gt: Tensor, valid: Tensor) -> Tensor:
        """Distance of the hip centres, zero where a hip is unlabelled."""
        both = (valid[:, 4] & valid[:, 10]).float()
        p_root = (coco[:, 11] + coco[:, 12]) / 2.0
        g_root = (gt[:, 4] + gt[:, 10]) / 2.0
        out: Tensor = (p_root - g_root).norm(dim=-1) * both
        return out

    def _to_device(self, b: FitBatch) -> dict[str, Tensor]:
        def f(a: Any) -> Tensor:
            return torch.as_tensor(
                np.asarray(a, dtype=np.float32), device=self.device
            )

        def m(a: Any) -> Tensor:
            return torch.as_tensor(np.asarray(a), device=self.device)

        # smoothness pairs: consecutive records of one track, close in time.
        # Decided here in float64: Waymo timestamps (and even their offsets
        # within a batch that mixes segments) do not survive float32.
        time_s = np.asarray(b.time_s, dtype=np.float64)
        same = b.track[1:] == b.track[:-1]
        close = np.abs(time_s[1:] - time_s[:-1]) < self.cfg.smooth_gap_s
        return {
            "track": m(b.track),
            "pair": f((same & close).astype(np.float32)),
            "joints3d": f(b.joints3d),
            "joints3d_valid": m(b.joints3d_valid),
            "kp2d": f(b.kp2d),
            "intrinsics": f(b.intrinsics),
            "points": f(b.points),
            "points_valid": m(b.points_valid),
            "box3d": f(b.box3d),
            "init_go": f(b.init_global_orient),
            "init_bp": f(b.init_body_pose),
            "init_betas": f(b.init_betas),
            "init_transl": f(b.init_transl),
        }

    # -- the fit ---------------------------------------------------------

    def fit(self, batch: FitBatch) -> FitV2Result:
        """Optimise every record of ``batch``."""
        cfg = self.cfg
        t = self._to_device(batch)
        n = len(batch.keys)
        rot_go = (
            torch.from_numpy(
                Rotation.from_rotvec(batch.init_global_orient).as_matrix()
            )
            .float()
            .to(self.device)
        )
        rot_bp = (
            torch.from_numpy(
                Rotation.from_rotvec(
                    batch.init_body_pose.reshape(-1, 3)
                ).as_matrix()
            )
            .float()
            .to(self.device)
            .reshape(n, 23, 3, 3)
        )
        with torch.no_grad():
            latent0 = self.tok.encode(rot_bp[:, :NUM_BODY])
            body0_6d = matrix_to_rot6d(self.tok.decode(latent0))
        go6d0 = matrix_to_rot6d(rot_go)
        latent = latent0.clone().requires_grad_(True)
        go6d = go6d0.clone().requires_grad_(True)
        transl = t["init_transl"].clone().requires_grad_(True)
        n_tracks = int(t["track"].max().item()) + 1
        betas_track = torch.zeros(n_tracks, NUM_BETAS, device=self.device)
        betas_track.index_add_(0, t["track"], t["init_betas"])
        counts = torch.zeros(n_tracks, device=self.device).index_add_(
            0, t["track"], torch.ones(n, device=self.device)
        )
        betas_track = (betas_track / counts[:, None]).requires_grad_(True)
        ground_y = t["box3d"][:, 1] + t["box3d"][:, 4] / 2.0  # y is down
        sub = torch.arange(0, 6890, cfg.vertex_stride, device=self.device)
        kp = t["kp2d"][:, self.wsel]
        kp_conf = (kp[..., 2] > 0).float()
        pts_valid = t["points_valid"].float()
        pair = t["pair"]
        opt = torch.optim.Adam([latent, go6d, transl, betas_track], lr=cfg.lr)
        for _ in range(cfg.iters):
            opt.zero_grad(set_to_none=True)
            decoded = self.tok.decode(latent)
            body = self._body(decoded)
            betas = betas_track[t["track"]]
            orient = rot6d_to_matrix(go6d)
            verts, coco = self._mesh(orient[:, None], body, betas, transl)
            loss = (
                cfg.w_joints3d
                * self._kp_error(
                    coco, t["joints3d"], t["joints3d_valid"], orient
                ).mean()
            )
            loss = loss + cfg.w_root * self._root_error(
                coco, t["joints3d"], t["joints3d_valid"]
            ).mean()
            uv = project(coco[:, self.csel], t["intrinsics"])
            kp_err = (uv - kp[..., :2]).abs().sum(-1) / cfg.crop_size
            loss = loss + cfg.w_kp2d * (
                kp_err * kp_conf
            ).sum() / kp_conf.sum().clamp_min(1)
            dist = torch.cdist(t["points"], verts[:, sub]).min(-1).values
            lid = (dist - cfg.lidar_offset_m).abs()
            loss = loss + cfg.w_lidar * (
                lid * pts_valid
            ).sum() / pts_valid.sum().clamp_min(1)
            low = verts[..., 1].max(-1).values  # lowest body point
            sink = torch.relu(low - ground_y)
            hover = torch.relu(ground_y - low - cfg.ground_tolerance_m)
            loss = loss + cfg.w_ground * (sink**2 + 0.1 * hover**2).mean()
            body6d = matrix_to_rot6d(decoded)
            loss = loss + cfg.w_trust_pose * ((body6d - body0_6d) ** 2).mean()
            loss = loss + cfg.w_trust_orient * ((go6d - go6d0) ** 2).mean()
            loss = (
                loss
                + cfg.w_trust_transl
                * ((transl - t["init_transl"]) ** 2).sum(-1).mean()
            )
            q, _ = self.tok.quantize(latent.detach())
            loss = loss + cfg.w_commit * ((latent - q) ** 2).mean()
            if n > 1 and pair.any():
                dp = ((body6d[1:] - body6d[:-1]) ** 2).flatten(1).mean(1)
                dt = ((transl[1:] - transl[:-1]) ** 2).sum(-1)
                loss = (
                    loss + cfg.w_smooth * ((dp + dt) * pair).sum() / pair.sum()
                )
            loss = loss + cfg.w_betas * (betas_track**2).mean()
            loss.backward()  # type: ignore[no-untyped-call]
            opt.step()
        return self._finish(batch, t, latent, go6d, transl, betas_track)

    @torch.no_grad()
    def _finish(
        self,
        batch: FitBatch,
        t: dict[str, Tensor],
        latent: Tensor,
        go6d: Tensor,
        transl: Tensor,
        betas_track: Tensor,
    ) -> FitV2Result:
        cfg = self.cfg
        n = len(batch.keys)
        full = self._body(self.tok.decode(latent))
        betas = betas_track[t["track"]]
        orient = rot6d_to_matrix(go6d)
        verts, coco = self._mesh(orient[:, None], full, betas, transl)
        kp_err = self._kp_error(coco, t["joints3d"], t["joints3d_valid"])
        dist = torch.cdist(t["points"], verts).min(-1).values
        dist = torch.where(t["points_valid"], dist, torch.nan)
        chamfer = dist.nanmedian(-1).values
        chamfer = torch.nan_to_num(chamfer, nan=float("inf"))
        energy = self.prior(full)
        kp_np = kp_err.cpu().numpy().astype(np.float64)
        ch_np = chamfer.cpu().numpy().astype(np.float64)
        en_np = energy.cpu().numpy().astype(np.float64)
        accepted = (
            (kp_np <= cfg.max_kp_error_m)
            & (ch_np <= cfg.max_chamfer_m)
            & (en_np <= cfg.max_prior_energy)
        )
        conf = np.exp(-((kp_np / cfg.conf_kp_sigma_m) ** 2)) * np.exp(
            -((np.minimum(ch_np, 1.0) / cfg.conf_chamfer_sigma_m) ** 2)
        )
        conf = np.where(accepted, conf, 0.0)
        aa = (
            Rotation.from_matrix(
                full.reshape(-1, 3, 3).cpu().numpy().astype(np.float64)
            )
            .as_rotvec()
            .reshape(n, 69)
        )
        go = Rotation.from_matrix(
            orient.cpu().numpy().astype(np.float64)
        ).as_rotvec()
        return FitV2Result(
            global_orient=go,
            body_pose=aa,
            betas=betas.cpu().numpy().astype(np.float64),
            transl=transl.cpu().numpy().astype(np.float64),
            kp_error_m=kp_np,
            chamfer_m=ch_np,
            prior_energy=en_np,
            accepted=accepted,
            confidence=conf.astype(np.float64),
        )


def summarize(results: list[FitV2Result]) -> dict[str, Any]:
    """Acceptance rate and residual statistics over all fitted records."""
    kp = np.concatenate([r.kp_error_m for r in results])
    ch = np.concatenate([r.chamfer_m for r in results])
    en = np.concatenate([r.prior_energy for r in results])
    acc = np.concatenate([r.accepted for r in results])
    conf = np.concatenate([r.confidence for r in results])
    finite = np.isfinite(ch)
    return {
        "records": int(len(kp)),
        "accepted": int(acc.sum()),
        "acceptance_rate": float(acc.mean()),
        "keypoint_error_mm": float(kp.mean() * 1000),
        "keypoint_error_accepted_mm": float(kp[acc].mean() * 1000)
        if acc.any()
        else float("nan"),
        "chamfer_median_mm": float(np.median(ch[finite]) * 1000),
        "prior_energy_mean": float(en.mean()),
        "confidence_mean_accepted": float(conf[acc].mean())
        if acc.any()
        else float("nan"),
        "rejected_by_keypoints": int((kp > 0.08).sum()),
        "rejected_by_chamfer": int((ch > 0.06).sum()),
        "rejected_by_prior": int((en > 6.0).sum()),
    }
