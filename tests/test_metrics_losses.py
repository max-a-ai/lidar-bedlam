"""Metrics and losses behave on synthetic data."""

from __future__ import annotations

import numpy as np
import torch

from lidar_bedlam.losses.smpl import FusionLoss, rodrigues
from lidar_bedlam.metrics.detection import (
    average_precision,
    box_detection_metrics,
    error_by_distance,
    translation_error,
)
from lidar_bedlam.metrics.pose import mpjpe, pa_mpjpe, procrustes_align


def test_mpjpe_and_pa_mpjpe() -> None:
    rng = np.random.default_rng(0)
    gt = rng.normal(size=(24, 3))
    pred = gt + 0.01
    assert np.isclose(mpjpe(pred, gt), np.sqrt(3) * 10.0)
    rot = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    moved = 1.3 * gt @ rot.T + [1.0, 2.0, 3.0]
    assert pa_mpjpe(moved, gt) < 1e-6
    valid = np.ones(24, bool)
    valid[:4] = False
    assert np.allclose(procrustes_align(moved, gt, valid), gt)


def test_translation_error_and_bins() -> None:
    e = translation_error(
        np.zeros((3, 3)), np.array([[3, 4, 0], [0, 0, 1], [0, 0, 0]])
    )
    assert np.allclose(e, [5.0, 1.0, 0.0])
    bins = error_by_distance(
        e, np.array([5.0, 15.0, 25.0]), (0.0, 10.0, 20.0, np.inf)
    )
    assert bins == {"0-10m": 5.0, "10-20m": 1.0, ">20m": 0.0}


def test_average_precision() -> None:
    assert np.isclose(
        average_precision(np.array([True, True]), np.ones(2)), 1.0
    )
    # best-ranked wrong, then right: recall 0.5 at precision 0.5
    ap = average_precision(np.array([False, True]), np.array([0.9, 0.1]))
    assert np.isclose(ap, 0.25)


def test_box_detection_metrics_perfect() -> None:
    boxes = np.array(
        [[0, 0, 5, 0.6, 1.7, 0.4, 0.2], [1, 0, 9, 0.6, 1.7, 0.4, -1.0]]
    )
    res = box_detection_metrics(boxes, boxes)
    assert np.isclose(res.map, 1.0) and np.isclose(res.mean_iou, 1.0)


def test_rodrigues_matches_scipy() -> None:
    from lidar_bedlam.geometry.rotations import axis_angle_to_matrix

    aa = np.array([[0.2, -0.5, 1.1], [0.0, 0.0, 0.0]])
    r = rodrigues(torch.tensor(aa)).numpy()
    assert np.allclose(r, axis_angle_to_matrix(aa), atol=1e-6)


def _batch(b: int = 2) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    return {
        "has_smpl": torch.tensor([True, False]),
        "global_orient": torch.randn(b, 3, generator=g),
        "body_pose": torch.randn(b, 69, generator=g) * 0.3,
        "betas": torch.randn(b, 10, generator=g),
        "transl": torch.randn(b, 3, generator=g),
        "joints3d": torch.randn(b, 24, 3, generator=g),
        "joints3d_valid": torch.ones(b, 24, dtype=torch.bool),
        "kp2d": torch.rand(b, 24, 3, generator=g) * 256,
        "has_kp2d": torch.tensor([True, True]),
        "box3d": torch.randn(b, 7, generator=g),
        "has_box3d": torch.tensor([True, True]),
    }


def test_fusion_loss_zero_on_ground_truth() -> None:
    batch = _batch()
    pred = {
        "global_orient": batch["global_orient"],
        "body_pose": batch["body_pose"],
        "betas": batch["betas"],
        "transl": batch["transl"],
        "joints3d": batch["joints3d"],
        "kp2d": batch["kp2d"][..., :2],
        "box3d": batch["box3d"],
    }
    total, parts = FusionLoss()(pred, batch)
    assert total.item() < 1e-6 and all(v.item() < 1e-6 for v in parts.values())


def test_fusion_loss_respects_flags() -> None:
    batch = _batch()
    pred = {
        "global_orient": torch.zeros(2, 3),
        "body_pose": torch.zeros(2, 69),
        "betas": batch["betas"] + 1.0,
        "transl": batch["transl"] + 1.0,
        "joints3d": batch["joints3d"],
        "box3d": batch["box3d"],
    }
    _, parts = FusionLoss()(pred, batch)
    # only sample 0 has SMPL, so betas error is exactly 1 (MSE of +1)
    assert np.isclose(parts["betas"].item(), 1.0)
    assert np.isclose(parts["transl"].item(), 3.0)
    assert "kp2d" not in parts


def test_joint_losses_use_coco_joints_for_waymo_rows() -> None:
    from lidar_bedlam.data.schema import WAYMO15_TO_COCO17

    batch = _batch()
    coco = torch.randn(2, 17, 3)
    sel = WAYMO15_TO_COCO17 >= 0
    # sample 1 carries Waymo keypoints: rows in waymo15 order
    batch["joints3d"] = batch["joints3d"].clone()
    batch["joints3d"][1, :15] = 0.0
    batch["joints3d"][1, : int(sel.sum())] = coco[1, WAYMO15_TO_COCO17[sel]]
    batch["joints3d_valid"] = batch["joints3d_valid"].clone()
    batch["joints3d_valid"][1] = False
    batch["joints3d_valid"][1, : int(sel.sum())] = True
    batch["joint_convention_id"] = torch.tensor([0, 1])
    pred = {
        "global_orient": batch["global_orient"],
        "body_pose": batch["body_pose"],
        "betas": batch["betas"],
        "transl": batch["transl"],
        "joints3d": batch["joints3d"] + 5.0,  # SMPL joints wrong on purpose
        "joints_coco": coco,
        "box3d": batch["box3d"],
    }
    _, parts = FusionLoss()(pred, batch)
    # sample 0 (smpl24): error 15 on each valid joint; sample 1 (waymo15):
    # 0 on its 13 shared COCO joints; the loss is the mean over all of them
    n0 = int(batch["joints3d_valid"][0].sum())
    n1 = int(sel.sum())
    expected = 15.0 * n0 / (n0 + n1)
    assert np.isclose(parts["joints3d"].item(), expected, atol=1e-4)
    pred["joints_coco"] = coco + 1.0
    _, parts = FusionLoss()(pred, batch)
    assert parts["joints3d"].item() > expected + 1e-3
