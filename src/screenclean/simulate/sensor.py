"""The sensor: optical blur, point sampling through a Bayer colour filter, auto-exposure and noise.

A phone sensor has no optical anti-aliasing filter, and each pixel only sees one colour
(the Bayer pattern). Sampling the screen's fine subpixel grid this way folds its high
frequencies down into low ones: that is the moiré.
"""

from __future__ import annotations

import cv2
import numpy as np

CFA = {  # colour index (0 R, 1 G, 2 B) at positions (0,0) (0,1) / (1,0) (1,1)
    "RGGB": ((0, 1), (1, 2)),
    "BGGR": ((2, 1), (1, 0)),
    "GRBG": ((1, 0), (2, 1)),
    "GBRG": ((1, 2), (0, 1)),
}


def blur(raster: np.ndarray, sigma_px: float) -> np.ndarray:
    """Gaussian blur (lens PSF + pixel aperture), ``sigma_px`` in raster samples."""
    if sigma_px < 0.05:
        return raster
    return cv2.GaussianBlur(raster, (0, 0), sigmaX=sigma_px, sigmaY=sigma_px, borderType=cv2.BORDER_REFLECT)


def sample(raster: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    """Point-sample the (already blurred) raster at the given raster coordinates. No anti-aliasing."""
    return cv2.remap(raster, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def mosaic(rgb: np.ndarray, pattern: str) -> np.ndarray:
    """Keep one colour per pixel according to the Bayer ``pattern``: (H, W, 3) -> (H, W)."""
    h, w = rgb.shape[:2]
    idx = np.array(CFA[pattern])
    ch = np.tile(idx, (-(-h // 2), -(-w // 2)))[:h, :w]
    return np.take_along_axis(rgb, ch[..., None], axis=2)[..., 0]


def expose(raw: np.ndarray, percentile: float, target: float, floor: float = 0.0) -> np.ndarray:
    """Auto-exposure: scale so the given percentile of the scene lands at ``target``.

    ``floor`` limits the gain for dark scenes (e.g. dark-mode pages): the reference level is
    never taken below it, so a near-black background isn't pushed to mid-grey; bright text may clip.
    """
    ref = max(float(np.percentile(raw, percentile)), floor, 1e-6)
    return raw * (target / ref)


def add_noise(raw: np.ndarray, rng: np.random.Generator, photons: float, read_noise: float) -> np.ndarray:
    """Shot noise (Poisson, ``photons`` at full scale) plus Gaussian read noise, clipped to [0, 1]."""
    shot = rng.poisson(np.clip(raw, 0, None) * photons) / photons
    return np.clip(shot + rng.normal(0.0, read_noise, raw.shape), 0.0, 1.0).astype(np.float32)
