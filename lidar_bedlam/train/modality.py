"""Sensor-modality dropout: train one model for camera + LiDAR, camera
only and LiDAR only.

A record carries ``has_image`` (False for LiDAR-only sources) and
``points_valid`` (all False for an empty cloud). Dropping a modality sets
the same flags the data would set when the sensor is absent, so the model
sees exactly the inference-time input: ``has_image`` False makes the
fusion model substitute its ``no_image`` token, ``has_points`` False its
``no_points`` token, and the point-anchored translation falls back to
the camera parameterisation. A sample never loses both modalities.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

MODALITIES = ("both", "image", "points")


def _present(batch: dict[str, Any]) -> tuple[Tensor, Tensor]:
    valid = batch["points_valid"]
    has_pts = valid.any(1)
    has_img = batch.get("has_image")
    if has_img is None:
        has_img = torch.ones_like(has_pts)
    return has_img.bool(), has_pts


def _apply(
    batch: dict[str, Any], drop_img: Tensor, drop_pts: Tensor
) -> dict[str, Any]:
    has_img, has_pts = _present(batch)
    out = dict(batch)
    out["has_image"] = has_img & ~drop_img
    keep_pts = ~drop_pts
    out["points_valid"] = batch["points_valid"] & keep_pts[:, None]
    out["points"] = (
        batch["points"] * keep_pts.to(batch["points"].dtype)[:, None, None]
    )
    out["has_points"] = has_pts & keep_pts
    return out


def drop_modalities(
    batch: dict[str, Any],
    p_image: float,
    p_points: float,
    generator: torch.Generator | None = None,
) -> dict[str, Any]:
    """Randomly drop the image or the points of each sample, never both.

    Each modality is dropped with its own probability; when both are drawn
    for a sample that has both, one of them is kept at random. A modality
    the record lacks already is never counted as dropped, so a LiDAR-only
    record keeps its points.
    """
    if p_image <= 0 and p_points <= 0:
        return batch
    has_img, has_pts = _present(batch)
    b = has_img.shape[0]
    dev = has_img.device
    u = torch.rand(b, 3, generator=generator, device=dev)
    drop_img = (u[:, 0] < p_image) & has_img & has_pts
    drop_pts = (u[:, 1] < p_points) & has_pts & has_img
    both = drop_img & drop_pts
    keep_image = u[:, 2] < 0.5
    drop_img = drop_img & ~(both & keep_image)
    drop_pts = drop_pts & ~(both & ~keep_image)
    return _apply(batch, drop_img, drop_pts)


def force_modality(batch: dict[str, Any], modality: str) -> dict[str, Any]:
    """Keep only ``modality`` (``image`` or ``points``); ``both`` is a no-op.

    Used for the validation variants: the same records evaluated as
    camera-only and as LiDAR-only. A record that lacks the kept modality
    keeps what it has.
    """
    if modality not in MODALITIES:
        msg = f"unknown modality {modality!r}; choose from {MODALITIES}"
        raise ValueError(msg)
    if modality == "both":
        return batch
    has_img, has_pts = _present(batch)
    if modality == "image":
        return _apply(batch, torch.zeros_like(has_img), has_pts & has_img)
    return _apply(batch, has_img & has_pts, torch.zeros_like(has_pts))
