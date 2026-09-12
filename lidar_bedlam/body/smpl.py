"""SMPL body model wrapper and frame changes of SMPL parameters.

The model files are licensed and never live in the repo; pass the directory
that contains a chumpy-free ``smpl/SMPL_NEUTRAL.pkl`` (see
:mod:`convert_smpl` for the conversion of the original release file).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from lidar_bedlam.geometry.rotations import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
)

FloatArray = NDArray[np.float64]

NUM_BETAS = 10
NUM_BODY_JOINTS = 23
NUM_VERTICES = 6890

SMPL_JOINT_NAMES = (
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot",
    "right_foot", "neck", "left_collar", "right_collar", "head",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hand", "right_hand",
)  # fmt: skip

COCO_JOINT_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
    "right_knee", "left_ankle", "right_ankle",
)  # fmt: skip


@dataclass(frozen=True)
class SmplParams:
    """SMPL parameters of one person, axis-angle, in some frame (metres)."""

    global_orient: FloatArray  # (3,)
    body_pose: FloatArray  # (69,)
    betas: FloatArray  # (10,)
    transl: FloatArray  # (3,)

    @classmethod
    def from_pose72(
        cls,
        pose: NDArray[np.floating],
        betas: NDArray[np.floating],
        transl: NDArray[np.floating],
    ) -> SmplParams:
        """Build from the common 72-vector pose layout."""
        p = np.asarray(pose, dtype=np.float64).reshape(72)
        return cls(
            global_orient=p[:3].copy(),
            body_pose=p[3:].copy(),
            betas=np.asarray(betas, dtype=np.float64).reshape(NUM_BETAS)[
                :NUM_BETAS
            ],
            transl=np.asarray(transl, dtype=np.float64).reshape(3),
        )


class SmplModel:
    """Thin wrapper around ``smplx.SMPL`` (neutral, 10 betas)."""

    def __init__(self, model_dir: Path, device: str = "cpu") -> None:
        import smplx

        self.device = torch.device(device)
        self._model = smplx.create(
            str(model_dir),
            model_type="smpl",
            gender="neutral",
            num_betas=NUM_BETAS,
        ).to(self.device)
        self._model.eval()
        for p in self._model.parameters():
            p.requires_grad_(False)
        self.faces: NDArray[np.int64] = np.asarray(
            self._model.faces, dtype=np.int64
        )
        self.coco_regressor: FloatArray | None = None
        coco = model_dir / "J_regressor_coco.npy"
        if coco.exists():
            self.coco_regressor = np.load(coco).astype(np.float64)

    def rest_pelvis(self, betas: NDArray[np.floating]) -> FloatArray:
        """Pelvis joint of the shaped rest pose, i.e. the rotation centre."""
        b = torch.as_tensor(
            np.asarray(betas, dtype=np.float32).reshape(1, NUM_BETAS),
            device=self.device,
        )
        with torch.no_grad():
            out = self._model(betas=b)
        return np.asarray(out.joints[0, 0].cpu().numpy(), dtype=np.float64)

    def forward(self, params: SmplParams) -> tuple[FloatArray, FloatArray]:
        """Posed vertices (6890, 3) and the 24 SMPL joints (24, 3)."""
        f32 = np.float32

        def t(x: FloatArray) -> torch.Tensor:
            return torch.as_tensor(
                np.asarray(x, dtype=f32)[None], device=self.device
            )

        with torch.no_grad():
            out = self._model(
                betas=t(params.betas),
                global_orient=t(params.global_orient),
                body_pose=t(params.body_pose),
                transl=t(params.transl),
            )
        verts = np.asarray(out.vertices[0].cpu().numpy(), dtype=np.float64)
        joints = np.asarray(
            out.joints[0, : len(SMPL_JOINT_NAMES)].cpu().numpy(),
            dtype=np.float64,
        )
        return verts, joints

    def coco_joints(self, vertices: NDArray[np.floating]) -> FloatArray:
        """17 COCO joints regressed from vertices (needs the regressor)."""
        if self.coco_regressor is None:
            msg = "J_regressor_coco.npy not found next to the model"
            raise FileNotFoundError(msg)
        return np.asarray(self.coco_regressor @ vertices, dtype=np.float64)


def transform_smpl_params(
    params: SmplParams,
    transform: NDArray[np.floating],
    rest_pelvis: NDArray[np.floating],
) -> SmplParams:
    """Express SMPL params in a new frame given a 4x4 rigid transform.

    SMPL rotates the body about the shaped rest pelvis ``J0`` and then adds
    ``transl``. For a frame change ``p' = R p + t`` the new parameters are
    ``R_go' = R R_go`` and ``transl' = R (J0 + transl) + t - J0``.
    """
    tf = np.asarray(transform, dtype=np.float64)
    r, t = tf[:3, :3], tf[:3, 3]
    j0 = np.asarray(rest_pelvis, dtype=np.float64)
    r_go = axis_angle_to_matrix(params.global_orient)
    new_orient = matrix_to_axis_angle(r @ r_go)
    new_transl = r @ (j0 + params.transl) + t - j0
    return SmplParams(
        global_orient=new_orient,
        body_pose=params.body_pose.copy(),
        betas=params.betas.copy(),
        transl=new_transl,
    )
