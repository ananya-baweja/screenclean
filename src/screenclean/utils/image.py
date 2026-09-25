"""Small image conversions shared by every module.

Conventions: float32 RGB in [0, 1]; numpy images are HWC, torch tensors are BCHW.
OpenCV's BGR order only appears at I/O boundaries (``rgb_to_bgr`` / ``bgr_to_rgb``).
"""

from __future__ import annotations

import cv2
import numpy as np


def to_float01(img: np.ndarray) -> np.ndarray:
    """Convert uint8 / uint16 / float input to float32 in [0, 1]."""
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    if img.dtype == np.uint16:
        return img.astype(np.float32) / 65535.0
    return np.clip(img.astype(np.float32), 0.0, 1.0)


def to_uint8(img: np.ndarray) -> np.ndarray:
    """Convert a float image in [0, 1] (or uint8) to uint8 with rounding."""
    if img.dtype == np.uint8:
        return img
    return (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def rgb_to_bgr(img: np.ndarray) -> np.ndarray:
    """Swap channel order RGB -> BGR (for OpenCV I/O)."""
    return np.ascontiguousarray(img[..., ::-1])


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    """Swap channel order BGR -> RGB (for OpenCV I/O)."""
    return np.ascontiguousarray(img[..., ::-1])


def resize_max_side(img: np.ndarray, max_side: int) -> np.ndarray:
    """Downscale so the longer side is at most ``max_side`` (INTER_AREA). Never upscales."""
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return img
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def hwc_to_bchw(img: np.ndarray):
    """Turn an HWC numpy image into a 1xCxHxW torch tensor (torch imported lazily)."""
    import torch

    return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))[None]


def bchw_to_hwc(t) -> np.ndarray:
    """Turn the first image of a BCHW torch tensor into an HWC float32 numpy array."""
    return t[0].detach().float().cpu().numpy().transpose(1, 2, 0)
