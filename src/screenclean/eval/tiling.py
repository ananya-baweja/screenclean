"""Tiled processing for images too large to handle in one piece (4K photos on a small GPU).

The image is cut into overlapping tiles. Each output tile is blended in with weights
that ramp linearly across the overlap, so tile borders don't show.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def tile_starts(length: int, tile: int, overlap: int) -> list[int]:
    """Start positions covering ``[0, length)`` with tiles of ``tile`` overlapping by at least ``overlap``."""
    if tile >= length:
        return [0]
    step = tile - overlap
    if step <= 0:
        raise ValueError("overlap must be smaller than the tile size")
    starts = list(range(0, length - tile, step))
    starts.append(length - tile)  # last tile flush with the edge
    return starts


def _ramp(n: int, overlap: int, at_start: bool, at_end: bool) -> np.ndarray:
    """1-D blending weights: linear ramps over ``overlap`` pixels on inner edges, 1 elsewhere."""
    w = np.ones(n, np.float32)
    r = min(overlap, n // 2)
    if r > 0:
        ramp = (np.arange(r, dtype=np.float32) + 0.5) / r
        if not at_start:
            w[:r] = ramp
        if not at_end:
            w[n - r :] = ramp[::-1]
    return w


def tiled_apply(
    img: np.ndarray, fn: Callable[[np.ndarray], np.ndarray], tile: int = 1024, overlap: int = 64
) -> np.ndarray:
    """Apply ``fn`` (HWC float -> HWC float, same size) tile by tile and blend the results."""
    h, w = img.shape[:2]
    if h <= tile and w <= tile:
        return fn(img)
    out = np.zeros(img.shape, np.float32)
    weight = np.zeros((h, w, 1), np.float32)
    ys, xs = tile_starts(h, tile, overlap), tile_starts(w, tile, overlap)
    for y in ys:
        for x in xs:
            th, tw = min(tile, h - y), min(tile, w - x)
            res = fn(img[y : y + th, x : x + tw])
            if res.shape[:2] != (th, tw):
                raise ValueError(f"tile function changed the tile size: {(th, tw)} -> {res.shape[:2]}")
            wy = _ramp(th, overlap, at_start=y == 0, at_end=y + th == h)
            wx = _ramp(tw, overlap, at_start=x == 0, at_end=x + tw == w)
            wt = (wy[:, None] * wx[None, :])[..., None]
            out[y : y + th, x : x + tw] += res * wt
            weight[y : y + th, x : x + tw] += wt
    return out / np.maximum(weight, 1e-8)
