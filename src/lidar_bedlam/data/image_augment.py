"""Image-side augmentations (numpy / PIL only).

Camera cover and random erasing model occlusion of the image; colour
jitter, blur and JPEG re-encoding model appearance differences between
BEDLAM renders and real cameras; bbox jitter models detector noise.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageEnhance, ImageFilter

UInt8Array = NDArray[np.uint8]
FloatArray = NDArray[np.float64]


def erase_rect(
    image: UInt8Array, x0: int, y0: int, x1: int, y1: int, value: int = 0
) -> UInt8Array:
    """Return a copy with the rectangle [x0, x1) x [y0, y1) set to value."""
    out = image.copy()
    h, w = out.shape[:2]
    out[max(y0, 0) : min(y1, h), max(x0, 0) : min(x1, w)] = value
    return out


def cover_side(image: UInt8Array, side: str, frac: float) -> UInt8Array:
    """Cover a fraction of the image from one side (lens obstruction)."""
    h, w = image.shape[:2]
    if frac <= 0:
        return image.copy()
    if side == "left":
        return erase_rect(image, 0, 0, int(w * frac), h)
    if side == "right":
        return erase_rect(image, w - int(w * frac), 0, w, h)
    if side == "top":
        return erase_rect(image, 0, 0, w, int(h * frac))
    if side == "bottom":
        return erase_rect(image, 0, h - int(h * frac), w, h)
    msg = f"unknown side {side!r}"
    raise ValueError(msg)


def random_erase(
    image: UInt8Array,
    bbox_xyxy: NDArray[np.floating],
    frac: tuple[float, float],
    rng: np.random.Generator,
) -> UInt8Array:
    """Erase a random rectangle inside the person box (object occlusion)."""
    x0, y0, x1, y1 = (float(v) for v in bbox_xyxy)
    bw, bh = x1 - x0, y1 - y0
    ew, eh = bw * rng.uniform(*frac), bh * rng.uniform(*frac)
    ex = rng.uniform(x0, max(x0, x1 - ew))
    ey = rng.uniform(y0, max(y0, y1 - eh))
    return erase_rect(
        image, int(ex), int(ey), int(ex + ew), int(ey + eh), value=0
    )


def color_jitter(
    image: UInt8Array,
    brightness: float,
    contrast: float,
    saturation: float,
) -> UInt8Array:
    """Multiply brightness / contrast / saturation (1.0 = unchanged)."""
    im = Image.fromarray(image)
    im = ImageEnhance.Brightness(im).enhance(brightness)
    im = ImageEnhance.Contrast(im).enhance(contrast)
    im = ImageEnhance.Color(im).enhance(saturation)
    return np.asarray(im, dtype=np.uint8)


def gaussian_blur(image: UInt8Array, sigma_px: float) -> UInt8Array:
    """Gaussian blur (defocus / motion approximation)."""
    if sigma_px <= 0:
        return image.copy()
    im = Image.fromarray(image).filter(ImageFilter.GaussianBlur(sigma_px))
    return np.asarray(im, dtype=np.uint8)


def jpeg_recompress(image: UInt8Array, quality: int) -> UInt8Array:
    """Re-encode as JPEG (real cameras are compressed, renders are not)."""
    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"), dtype=np.uint8)


def jitter_bbox(
    bbox_xyxy: NDArray[np.floating],
    scale_std: float,
    shift_std: float,
    rng: np.random.Generator,
) -> FloatArray:
    """Perturb a box like a detector would (scale and centre, relative)."""
    x0, y0, x1, y1 = (float(v) for v in bbox_xyxy)
    cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
    s = float(np.exp(rng.normal(0.0, scale_std)))
    cx += rng.normal(0.0, shift_std) * w
    cy += rng.normal(0.0, shift_std) * h
    w, h = w * s, h * s
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


@dataclass(frozen=True)
class ImageAugmentConfig:
    """Training-time image augmentation."""

    p_erase: float = 0.3
    erase_frac: tuple[float, float] = (0.2, 0.5)
    p_cover: float = 0.1
    cover_frac: tuple[float, float] = (0.1, 0.4)
    brightness: tuple[float, float] = (0.7, 1.3)
    contrast: tuple[float, float] = (0.7, 1.3)
    saturation: tuple[float, float] = (0.7, 1.3)
    p_blur: float = 0.2
    blur_sigma_px: tuple[float, float] = (0.5, 2.0)
    p_jpeg: float = 0.3
    jpeg_quality: tuple[int, int] = (40, 90)
    bbox_scale_std: float = 0.1
    bbox_shift_std: float = 0.05


def augment_image(
    image: UInt8Array,
    bbox_xyxy: NDArray[np.floating],
    cfg: ImageAugmentConfig,
    rng: np.random.Generator,
) -> tuple[UInt8Array, FloatArray]:
    """Apply the configured augmentations; returns image and jittered box."""
    out = image
    if rng.random() < cfg.p_erase:
        out = random_erase(out, bbox_xyxy, cfg.erase_frac, rng)
    if rng.random() < cfg.p_cover:
        side = str(rng.choice(["left", "right", "top", "bottom"]))
        out = cover_side(out, side, rng.uniform(*cfg.cover_frac))
    out = color_jitter(
        out,
        rng.uniform(*cfg.brightness),
        rng.uniform(*cfg.contrast),
        rng.uniform(*cfg.saturation),
    )
    if rng.random() < cfg.p_blur:
        out = gaussian_blur(out, rng.uniform(*cfg.blur_sigma_px))
    if rng.random() < cfg.p_jpeg:
        out = jpeg_recompress(out, int(rng.integers(*cfg.jpeg_quality)))
    box = jitter_bbox(bbox_xyxy, cfg.bbox_scale_std, cfg.bbox_shift_std, rng)
    return out, box
