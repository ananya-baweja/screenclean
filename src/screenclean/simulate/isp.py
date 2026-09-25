"""The phone's image processing: demosaic, white balance, tone curve, sharpening and JPEG.

Demosaicing fills in the two missing colours at every pixel and is easily fooled by the
moiré pattern (false colour). Sharpening then makes the bands stronger. So the simulated
photo gets the same kinds of artefacts as a real one.
"""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

from screenclean.simulate.sensor import mosaic

DEMOSAIC = {"bilinear": "", "ea": "_EA", "vng": "_VNG"}


@lru_cache(maxsize=8)
def bayer_code(pattern: str, method: str) -> int:
    """The OpenCV conversion code for our pattern name, found by trying all four on a test image.

    OpenCV names Bayer patterns after the second row, which is easy to get wrong, so this is
    checked instead of assumed.
    """
    rng = np.random.default_rng(0)
    rgb = cv2.GaussianBlur(rng.integers(0, 256, (64, 64, 3)).astype(np.uint8), (0, 0), 3)
    raw = mosaic(rgb, pattern)
    best, best_err = None, np.inf
    for name in ("BG", "GB", "RG", "GR"):
        code = getattr(cv2, f"COLOR_Bayer{name}2RGB{DEMOSAIC[method]}")
        err = np.abs(cv2.cvtColor(raw, code)[8:-8, 8:-8].astype(int) - rgb[8:-8, 8:-8]).mean()
        if err < best_err:
            best, best_err = code, err
    return best


def demosaic(raw: np.ndarray, pattern: str, method: str) -> np.ndarray:
    """Bayer raw in [0, 1] -> RGB float in [0, 1] (8-bit internally, as VNG requires)."""
    raw8 = np.clip(raw * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return cv2.cvtColor(raw8, bayer_code(pattern, method)).astype(np.float32) / 255.0


def white_balance(rgb: np.ndarray, gains: np.ndarray) -> np.ndarray:
    return rgb * gains.reshape(1, 1, 3).astype(np.float32)


def tone(rgb: np.ndarray, gamma: float, contrast: float) -> np.ndarray:
    """Linear -> display values: gamma, then a gentle S-curve around mid-grey."""
    x = np.clip(rgb, 0, 1) ** (1.0 / gamma)
    return np.clip(0.5 + (x - 0.5) * contrast, 0, 1)


def sharpen(rgb: np.ndarray, amount: float, sigma: float) -> np.ndarray:
    if amount <= 0:
        return rgb
    return np.clip(rgb + amount * (rgb - cv2.GaussianBlur(rgb, (0, 0), sigma)), 0, 1)


def jpeg(rgb: np.ndarray, quality: int) -> tuple[bytes, np.ndarray]:
    """Encode as JPEG (4:2:0, like phone cameras); return the bytes and the decoded uint8 RGB image."""
    u8 = np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
    ok, enc = cv2.imencode(
        ".jpg", cv2.cvtColor(u8, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
    )
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    data = enc.tobytes()
    decoded = cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return data, decoded
