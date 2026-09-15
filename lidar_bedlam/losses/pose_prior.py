"""Pose prior for joints that no label supervises.

Waymo labels 13 keypoints: the wrist and ankle positions pin the elbow
and knee rotations, but the wrist, hand, ankle, foot and head rotations
of a Waymo record get no gradient at all, and the hands and feet drift
into implausible poses. The prior pulls those rotations towards the
per-joint mean rotation of the BEDLAM poses, scaled by the spread of the
BEDLAM poses around it (``scripts/pose_prior_stats.py``): per joint the
squared geodesic angle to the mean over its standard deviation, averaged
over the prior joints, applied to the rows without a full SMPL label.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor

# SMPL joints without a Waymo keypoint on the child segment: ankles, feet,
# head, wrists, hands (body_pose index = joint index - 1)
DEFAULT_PRIOR_JOINTS = (7, 8, 10, 11, 15, 20, 21, 22, 23)


class PosePrior:
    """Per-joint Gaussian on the geodesic angle to a mean rotation."""

    def __init__(
        self, stats: Path, joints: tuple[int, ...] = DEFAULT_PRIOR_JOINTS
    ) -> None:
        z = np.load(stats)
        self.joints = [j - 1 for j in joints]  # body_pose indices
        self.mean = torch.from_numpy(z["mean_rot"][self.joints]).float()
        # BEDLAM keeps the hand joints at rest (sigma 0): a floor keeps the
        # prior finite and lets those joints move a few degrees
        sigma = np.maximum(z["sigma"][self.joints], np.radians(3.0))
        self.sigma = torch.from_numpy(sigma).float()

    def __call__(self, body_pose: Tensor) -> Tensor:
        """Per-sample prior energy for ``body_pose`` (B, 23, 3, 3)."""
        mean = self.mean.to(body_pose.device, body_pose.dtype)
        sigma = self.sigma.to(body_pose.device, body_pose.dtype)
        rel = body_pose[:, self.joints] @ mean.transpose(1, 2)[None]
        tr = rel.diagonal(dim1=-2, dim2=-1).sum(-1)
        angle = torch.acos(((tr - 1.0) / 2.0).clamp(-1.0 + 1e-6, 1.0 - 1e-6))
        return ((angle / sigma) ** 2).mean(-1)
