"""Colour-constancy correction (manuscript section 4.4).

Shades-of-Gray illuminant estimation with a Minkowski norm of p = 6, followed by
a von Kries diagonal transform that maps the estimated illuminant to neutral grey.

The method assumes a single, spatially uniform illuminant. It cannot correct
mixed lighting, strong shadows, flash specularities, device tone curves or
compression artefacts.
"""


from __future__ import annotations

import numpy as np

MINKOWSKI_P = 6
__all__ = ["estimate_illuminant", "von_kries", "shades_of_gray", "resize_short_side"]


def _as_float(image) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError("expected an H x W x 3 RGB image")
    arr = arr[..., :3].astype(np.float64)
    return arr / 255.0 if arr.max() > 1.0 else arr


def estimate_illuminant(image, p: int = MINKOWSKI_P) -> np.ndarray:
    """Estimate the illuminant as the Minkowski-p mean of each channel."""
    arr = _as_float(image)
    flat = arr.reshape(-1, 3)
    if p <= 0:
        raise ValueError("p must be positive")
    est = np.power(np.mean(np.power(flat, p), axis=0), 1.0 / p)
    norm = np.linalg.norm(est)
    return est / norm if norm > 0 else np.full(3, 1 / np.sqrt(3))


def von_kries(image, illuminant) -> np.ndarray:
    """Scale each channel so the estimated illuminant becomes neutral grey."""
    arr = _as_float(image)
    illuminant = np.asarray(illuminant, dtype=float)
    gain = (1.0 / np.sqrt(3.0)) / np.where(illuminant > 1e-8, illuminant, 1e-8)
    return np.clip(arr * gain, 0.0, 1.0)


def shades_of_gray(image, p: int = MINKOWSKI_P) -> np.ndarray:
    """Full correction: estimate the illuminant, then apply the diagonal transform."""
    return von_kries(image, estimate_illuminant(image, p=p))


def resize_short_side(image, short_side: int = 512):
    """Resize so the shorter side is `short_side`, preserving aspect ratio."""
    from PIL import Image

    img = image if isinstance(image, Image.Image) else Image.fromarray(
        (np.asarray(image) * 255).astype(np.uint8)
        if np.asarray(image).max() <= 1.0 else np.asarray(image).astype(np.uint8)
    )
    w, h = img.size
    scale = short_side / min(w, h)
    if scale >= 1.0:
        return img
    return img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BICUBIC)
