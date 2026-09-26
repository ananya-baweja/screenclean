"""Building blocks from NAFNet ("Simple Baselines for Image Restoration", Chen et al., ECCV 2022).

A NAFBlock has two residual halves and no ReLU or GELU: the only non-linearity is SimpleGate,
which splits the channels in two and multiplies the halves.

1. LayerNorm2d -> 1x1 conv (C -> 2C) -> depthwise conv -> SimpleGate -> simplified channel
   attention -> 1x1 conv -> x + beta * out
2. LayerNorm2d -> 1x1 conv (C -> 2C) -> SimpleGate -> 1x1 conv -> x + gamma * out

beta and gamma start at zero, so every block starts as the identity and a deep network trains
stably. A dilated depthwise conv (kernel 5, dilation d) widens the view cheaply: in the bottleneck,
at 1/16 resolution, dilations 1-4 let each output see hundreds of input pixels, the scale of moiré bands.
"""

from __future__ import annotations

import torch
from torch import nn


class LayerNorm2d(nn.Module):
    """Normalise over channels at every pixel (statistics in float32, safe under fp16 autocast)."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        mu = xf.mean(dim=1, keepdim=True)
        var = (xf - mu).pow(2).mean(dim=1, keepdim=True)
        y = (xf - mu) / torch.sqrt(var + self.eps)
        return (y * self.weight + self.bias).to(x.dtype)


class SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=1)
        return a * b


class NAFBlock(nn.Module):
    def __init__(self, channels: int, kernel: int = 3, dilation: int = 1, expand: int = 2):
        super().__init__()
        c, hidden = channels, channels * expand
        pad = dilation * (kernel - 1) // 2
        self.norm1 = LayerNorm2d(c)
        self.conv1 = nn.Conv2d(c, hidden, 1)
        self.dwconv = nn.Conv2d(hidden, hidden, kernel, padding=pad, dilation=dilation, groups=hidden)
        self.gate = SimpleGate()
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(hidden // 2, hidden // 2, 1))
        self.conv3 = nn.Conv2d(hidden // 2, c, 1)
        self.norm2 = LayerNorm2d(c)
        self.conv4 = nn.Conv2d(c, hidden, 1)
        self.conv5 = nn.Conv2d(hidden // 2, c, 1)
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.gate(self.dwconv(self.conv1(self.norm1(x))))
        y = self.conv3(y * self.sca(y))
        x = x + self.beta * y
        y = self.conv5(self.gate(self.conv4(self.norm2(x))))
        return x + self.gamma * y


def dilated_naf_block(channels: int, dilation: int, kernel: int = 5) -> NAFBlock:
    """The bottleneck block: a depthwise ``kernel`` x ``kernel`` conv at the given dilation."""
    return NAFBlock(channels, kernel=kernel, dilation=dilation)
