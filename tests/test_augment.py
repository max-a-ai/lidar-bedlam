"""Point-cloud and image augmentations."""

from __future__ import annotations

import numpy as np

from lidar_bedlam.data.image_augment import (
    ImageAugmentConfig,
    augment_image,
    color_jitter,
    cover_side,
    gaussian_blur,
    jitter_bbox,
    jpeg_recompress,
    random_erase,
)
from lidar_bedlam.lidar.augment import (
    PointAugmentConfig,
    SensorCover,
    add_outliers,
    augment_points,
    axis_cut_fraction,
    cover_mask,
    drop_channels,
    height_cut_fraction,
    jitter,
    miscalibrated_pose,
    range_jitter,
)


def _body(n: int = 400) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.uniform(-0.5, 0.5, size=(n, 3)) * [0.6, 1.8, 0.4] + [
        0,
        0,
        8.0,
    ]


def test_height_and_axis_cuts() -> None:
    p = _body()
    legs_gone = height_cut_fraction(p, 0.5, from_bottom=True)
    assert 0.4 < legs_gone.mean() < 0.6
    assert p[legs_gone, 1].max() < p[:, 1].max() - 0.8
    left_gone = axis_cut_fraction(p, 0, 0.25, from_min=True)
    assert 0.65 < left_gone.mean() < 0.85


def test_jitter_and_outliers() -> None:
    rng = np.random.default_rng(1)
    p = _body()
    assert np.array_equal(jitter(p, 0.0, rng), p.astype(np.float32))
    j = jitter(p, 0.02, rng)
    assert 0.01 < np.abs(j - p).mean() < 0.03
    rj = range_jitter(p, np.zeros(3), 0.05, rng)
    # noise is along the ray: cross product with the ray is ~0
    d = rj - p
    ray = p / np.linalg.norm(p, axis=1, keepdims=True)
    assert np.abs(np.cross(d, ray)).max() < 1e-4
    more = add_outliers(p, 0.1, 0.3, rng)
    assert len(more) == len(p) + 40


def test_cover_and_channel_dropout() -> None:
    rng = np.random.default_rng(2)
    ch = np.repeat(np.arange(64), 10)
    az = np.tile(np.linspace(-26, 26, 10), 64)
    keep = cover_mask(ch, az, 64, -26, 26, SensorCover(left=0.5, top=0.25))
    assert np.all(az[keep] >= 0) and np.all(ch[keep] >= 16)
    dk = drop_channels(ch, 64, 0.25, rng)
    assert len(np.unique(ch[dk])) == 48


def test_miscalibration_is_small_rigid_transform() -> None:
    rng = np.random.default_rng(3)
    pose = miscalibrated_pose(np.eye(4), 1.0, 0.02, rng)
    r = pose[:3, :3]
    assert np.allclose(r @ r.T, np.eye(3), atol=1e-9)
    assert np.linalg.norm(pose[:3, 3]) < 0.1


def test_augment_points_pipeline() -> None:
    rng = np.random.default_rng(4)
    p = _body()
    out = augment_points(
        p, np.arange(len(p)) % 64, 64, PointAugmentConfig(), rng
    )
    assert out.dtype == np.float32 and out.ndim == 2 and len(out) >= 16


def test_image_augmentations() -> None:
    rng = np.random.default_rng(5)
    img = np.full((64, 96, 3), 128, dtype=np.uint8)
    assert cover_side(img, "left", 0.5)[:, :48].max() == 0
    assert cover_side(img, "bottom", 0.25)[48:].max() == 0
    erased = random_erase(img, np.array([10, 10, 80, 60]), (0.5, 0.5), rng)
    assert (erased == 0).any() and erased[:5, :5].min() == 128
    bright = color_jitter(img, 1.5, 1.0, 1.0)
    assert bright.mean() > img.mean()
    assert gaussian_blur(img, 1.0).shape == img.shape
    assert jpeg_recompress(img, 50).shape == img.shape
    box = jitter_bbox(np.array([0, 0, 100, 200]), 0.0, 0.0, rng)
    assert np.allclose(box, [0, 0, 100, 200])
    out, jbox = augment_image(
        img, np.array([10, 10, 80, 60]), ImageAugmentConfig(), rng
    )
    assert out.shape == img.shape and jbox.shape == (4,)
