"""Paired augmentations: the moiré input and its ground truth always get the same transform.

Getting this wrong (e.g. cropping the two images at different positions) silently
ruins training, so every function here takes both images and one random generator.
"""

from __future__ import annotations

import numpy as np


def check_pair(a: np.ndarray, b: np.ndarray) -> None:
    """Raise if the two images don't have the same shape."""
    if a.shape != b.shape:
        raise ValueError(f"pair shapes differ: {a.shape} vs {b.shape}")


def random_crop_pair(
    a: np.ndarray, b: np.ndarray, size: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Crop the same ``size`` x ``size`` window from both images."""
    check_pair(a, b)
    h, w = a.shape[:2]
    if h < size or w < size:
        raise ValueError(f"image {w}x{h} is smaller than the crop size {size}")
    y = int(rng.integers(0, h - size + 1))
    x = int(rng.integers(0, w - size + 1))
    return a[y : y + size, x : x + size], b[y : y + size, x : x + size]


def flip_rot_pair(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Random horizontal flip, vertical flip and 90-degree rotation, the same for both images."""
    if rng.random() < 0.5:
        a, b = a[:, ::-1], b[:, ::-1]
    if rng.random() < 0.5:
        a, b = a[::-1], b[::-1]
    k = int(rng.integers(0, 4))
    if k:
        a, b = np.rot90(a, k), np.rot90(b, k)
    return np.ascontiguousarray(a), np.ascontiguousarray(b)


def augment_pair(
    a: np.ndarray,
    b: np.ndarray,
    rng: np.random.Generator,
    crop: int | None = None,
    flips: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Training augmentation: optional random crop, then optional flips/rotation."""
    check_pair(a, b)
    if crop is not None:
        a, b = random_crop_pair(a, b, crop, rng)
    if flips:
        a, b = flip_rot_pair(a, b, rng)
    return a, b
