"""The screen: turn page pixels into the light emitted by a grid of coloured subpixels.

Each page pixel is drawn as ``up`` x ``up`` raster samples. Inside a pixel, only the subpixels
emit light: red, green and blue stripes (or a PenTile-like diamond layout), separated by a
black matrix. This fine, perfectly periodic structure is what the camera later undersamples,
which produces moiré.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

LAYOUTS = ("rgb", "bgr", "pentile")


def _stripe_regions(layout: str, fill_x: float, fill_y: float):
    """For stripe layouts: per channel, the lit rectangle (x0, x1, y0, y1) inside a unit pixel."""
    order = {"rgb": (0, 1, 2), "bgr": (2, 1, 0)}[layout]
    y0, y1 = (1 - fill_y) / 2, (1 + fill_y) / 2
    regions = {}
    for slot, ch in enumerate(order):
        cx = (slot + 0.5) / 3
        regions[ch] = (cx - fill_x / 6, cx + fill_x / 6, y0, y1)
    return regions


def _inside(
    layout: str, fx: np.ndarray, fy: np.ndarray, px: np.ndarray, py: np.ndarray, fill_x: float, fill_y: float
) -> np.ndarray:
    """Boolean (…, 3): is the point (fx, fy) inside pixel (px, py) lit for each channel?"""
    out = np.zeros(fx.shape + (3,), bool)
    if layout in ("rgb", "bgr"):
        for ch, (x0, x1, y0, y1) in _stripe_regions(layout, fill_x, fill_y).items():
            out[..., ch] = (fx >= x0) & (fx < x1) & (fy >= y0) & (fy < y1)
        return out
    # PenTile-like: every pixel has a small green diamond; red and blue alternate in a checkerboard.
    s = 0.5 * fill_x
    green = np.abs(fx - 0.72) + np.abs(fy - 0.5) * (0.5 / max(fill_y, 1e-3)) < 0.2 * fill_x
    big = np.abs(fx - 0.3) + np.abs(fy - 0.5) * (0.5 / max(fill_y, 1e-3)) < s * 0.55
    red_pixel = (px + py) % 2 == 0
    out[..., 0] = big & red_pixel
    out[..., 2] = big & ~red_pixel
    out[..., 1] = green
    return out


@lru_cache(maxsize=64)
def subpixel_mask(layout: str, up: int, fill_x: float, fill_y: float, oversample: int = 8) -> np.ndarray:
    """Coverage of every raster sample for a 2 x 2 block of pixels: shape (2*up, 2*up, 3).

    Each channel is scaled so its mean over the block is 1: a uniform page of value v then
    emits v on average, and the mask only adds the subpixel structure.
    """
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}")
    n = 2 * up * oversample
    t = (np.arange(n) + 0.5) / (up * oversample)  # position in pixel units over 2 pixels
    X, Y = np.meshgrid(t, t)
    px, py = np.floor(X).astype(int), np.floor(Y).astype(int)
    lit = _inside(layout, X - px, Y - py, px, py, fill_x, fill_y).astype(np.float32)
    cov = lit.reshape(2 * up, oversample, 2 * up, oversample, 3).mean(axis=(1, 3))
    return (cov / np.maximum(cov.mean(axis=(0, 1), keepdims=True), 1e-6)).astype(np.float32)


def render_display(
    page_lin: np.ndarray, layout: str, up: int, fill_x: float, fill_y: float, origin: tuple[int, int] = (0, 0)
) -> np.ndarray:
    """Emitted light of a page region: (h*up, w*up, 3) float32.

    ``page_lin`` is linear light in [0, 1]; ``origin`` is the region's top-left page pixel
    (keeps the PenTile checkerboard consistent with the full page).
    """
    h, w = page_lin.shape[:2]
    mask = subpixel_mask(layout, up, round(fill_x, 3), round(fill_y, 3))
    if origin[0] % 2 or origin[1] % 2:
        mask = np.roll(mask, shift=(-(origin[1] % 2) * up, -(origin[0] % 2) * up), axis=(0, 1))
    big = np.repeat(np.repeat(page_lin.astype(np.float32), up, axis=0), up, axis=1)
    reps = (-(-h * up // mask.shape[0]), -(-w * up // mask.shape[1]), 1)
    return big * np.tile(mask, reps)[: h * up, : w * up]
