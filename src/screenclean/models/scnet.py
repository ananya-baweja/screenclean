"""ScreenCleanNet: a small U-Net that removes moiré from photos of screens.

::

    x ── reflect-pad to a multiple of 16 ── conv3x3 3 -> w0
      encoder level i (widths w0..w3): NAFBlock x n_enc ─────────────── skip (added) ─┐
      down: Haar DWT (C -> 4C at half size) -> 1x1 conv -> next width               │
      bottleneck (1/16 size): dilated NAFBlocks (5x5 depthwise, dilations 1, 2, 3, 4) │
      up: 1x1 conv (C -> 4C') -> inverse Haar DWT (double size) + skip ─────────────┘ -> NAFBlock x n_dec
    └ conv3x3 w0 -> 3 ── y = x + residual ── remove the padding

The network predicts a *correction* added to the input, so a network that outputs zero leaves
the photo unchanged, and it only has to learn what moiré looks like. There is no FFT inside, so
it exports to ONNX and runs in a browser; the FFT is only used in the training loss.

Variants (``models/registry.py``) change widths and block counts, or swap the wavelet steps for
a 2x2 stride-2 conv (down) and 1x1 conv + PixelShuffle (up), or drop the dilation.
Outputs are not clamped here (training needs the gradients); inference code clamps to [0, 1].
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn

from screenclean.models.blocks import NAFBlock, dilated_naf_block
from screenclean.models.wavelets import HaarDWT, HaarIDWT


@dataclass(frozen=True)
class NetSpec:
    widths: tuple[int, ...] = (32, 64, 128, 256)  # encoder levels (full, 1/2, 1/4, 1/8 size)
    bottleneck: int = 256  # width at 1/16 size
    enc_blocks: int = 2
    dec_blocks: int = 1
    mid_dilations: tuple[int, ...] = (1, 2, 3, 4)  # one dilated block per entry
    mid_kernel: int = 5
    down: str = "haar"  # "haar" or "conv" (2x2, stride 2)
    up: str = "haar"  # "haar" or "shuffle" (PixelShuffle)

    def as_dict(self) -> dict:
        return asdict(self)


class Down(nn.Module):
    def __init__(self, c_in: int, c_out: int, kind: str):
        super().__init__()
        if kind == "haar":
            self.op = nn.Sequential(HaarDWT(), nn.Conv2d(4 * c_in, c_out, 1))
        elif kind == "conv":
            self.op = nn.Conv2d(c_in, c_out, 2, stride=2)
        else:
            raise ValueError(f"unknown down {kind!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Up(nn.Module):
    def __init__(self, c_in: int, c_out: int, kind: str):
        super().__init__()
        if kind not in ("haar", "shuffle"):
            raise ValueError(f"unknown up {kind!r}")
        self.conv = nn.Conv2d(c_in, 4 * c_out, 1)
        self.expand = HaarIDWT() if kind == "haar" else nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.expand(self.conv(x))


class ScreenCleanNet(nn.Module):
    def __init__(self, spec: NetSpec | None = None):
        super().__init__()
        self.spec = spec = spec or NetSpec()
        w = spec.widths
        nxt = [*w[1:], spec.bottleneck]
        self.intro = nn.Conv2d(3, w[0], 3, padding=1)
        self.encoders = nn.ModuleList(
            nn.Sequential(*(NAFBlock(c) for _ in range(spec.enc_blocks))) for c in w
        )
        self.downs = nn.ModuleList(Down(c, n, spec.down) for c, n in zip(w, nxt, strict=True))
        self.middle = nn.Sequential(
            *(dilated_naf_block(spec.bottleneck, d, spec.mid_kernel) for d in spec.mid_dilations)
        )
        self.ups = nn.ModuleList(Up(n, c, spec.up) for c, n in reversed(list(zip(w, nxt, strict=True))))
        self.decoders = nn.ModuleList(
            nn.Sequential(*(NAFBlock(c) for _ in range(spec.dec_blocks))) for c in reversed(w)
        )
        self.ending = nn.Conv2d(w[0], 3, 3, padding=1)
        nn.init.zeros_(self.ending.weight)  # start as the identity: the untrained model returns the photo
        nn.init.zeros_(self.ending.bias)
        self.multiple = 2 ** len(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        pad_h, pad_w = (-h) % self.multiple, (-w) % self.multiple
        if pad_h or pad_w:
            mode = "reflect" if pad_h < h and pad_w < w else "replicate"  # reflect needs pad < size
            x = F.pad(x, (0, pad_w, 0, pad_h), mode=mode)
        feat = self.intro(x)
        skips = []
        for enc, down in zip(self.encoders, self.downs, strict=True):
            feat = enc(feat)
            skips.append(feat)
            feat = down(feat)
        feat = self.middle(feat)
        for up, dec, skip in zip(self.ups, self.decoders, reversed(skips), strict=True):
            feat = dec(up(feat) + skip)
        return (x + self.ending(feat))[..., :h, :w]
