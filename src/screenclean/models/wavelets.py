"""Orthonormal 2-D Haar wavelet transform as network layers.

``haar_dwt`` splits every 2 x 2 block of pixels into four sub-bands at half resolution:

- LL: the block average (a smaller copy of the image),
- LH, HL: horizontal and vertical detail,
- HH: diagonal detail.

With the factor 1/2 the transform is orthonormal: it keeps the image's energy and is inverted
exactly by ``haar_idwt``. Both use only slicing, arithmetic and ``pixel_shuffle`` (DepthToSpace in
ONNX), so they export cleanly and run in ONNX Runtime, including in a browser.

Channel layout: ``(B, C, H, W) -> (B, 4C, H/2, W/2)`` as ``[LL | LH | HL | HH]``, each C channels.

Note: a Haar step followed by a learned 1x1 convolution computes the same family of functions as a
2 x 2 stride-2 convolution (both are linear maps of each 2 x 2 block). The wavelet view gives a
lossless, energy-preserving starting point and names the frequency bands.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def haar_dwt(x: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) with even H and W -> (B, 4C, H/2, W/2): [LL, LH, HL, HH]."""
    if not torch.jit.is_tracing() and (x.shape[-2] % 2 or x.shape[-1] % 2):
        raise ValueError(f"haar_dwt needs even height and width, got {tuple(x.shape[-2:])}")
    a = x[..., 0::2, 0::2]  # top-left of each 2x2 block
    b = x[..., 0::2, 1::2]  # top-right
    c = x[..., 1::2, 0::2]  # bottom-left
    d = x[..., 1::2, 1::2]  # bottom-right
    ll = (a + b + c + d) * 0.5
    lh = (a - b + c - d) * 0.5  # left minus right: vertical edges
    hl = (a + b - c - d) * 0.5  # top minus bottom: horizontal edges
    hh = (a - b - c + d) * 0.5
    return torch.cat([ll, lh, hl, hh], dim=1)


def haar_idwt(y: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`haar_dwt`: (B, 4C, H, W) -> (B, C, 2H, 2W)."""
    if not torch.jit.is_tracing() and y.shape[1] % 4:
        raise ValueError(f"haar_idwt needs a multiple of 4 channels, got {y.shape[1]}")
    ll, lh, hl, hh = torch.chunk(y, 4, dim=1)
    a = (ll + lh + hl + hh) * 0.5
    b = (ll - lh + hl - hh) * 0.5
    c = (ll + lh - hl - hh) * 0.5
    d = (ll - lh - hl + hh) * 0.5
    bsz, ch, h, w = a.shape
    # pixel_shuffle wants, per channel, the 4 sub-pixels in order (0,0), (0,1), (1,0), (1,1)
    return F.pixel_shuffle(torch.stack([a, b, c, d], dim=2).reshape(bsz, ch * 4, h, w), 2)


class HaarDWT(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return haar_dwt(x)


class HaarIDWT(nn.Module):
    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return haar_idwt(y)
