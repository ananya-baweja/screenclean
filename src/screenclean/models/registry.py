"""Named model variants: the main model, a tiny one for CPU tests and the browser, and two ablations."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from torch import nn

from screenclean.models.scnet import NetSpec, ScreenCleanNet

SPECS: dict[str, NetSpec] = {
    # the model trained and reported (about 4.6 M parameters)
    "scnet_base": NetSpec(),
    # width 16, one block per level, 2 bottleneck blocks: CPU tests and the in-browser demo
    "scnet_tiny": NetSpec(widths=(16, 32, 64, 128), bottleneck=128, enc_blocks=1, mid_dilations=(1, 2)),
    # ablation: NAFNet-style resampling (2x2 stride-2 conv down, PixelShuffle up) and no dilation
    "plain_unet": NetSpec(down="conv", up="shuffle", mid_dilations=(1, 1, 1, 1)),
    # ablation: scnet_base without dilation in the bottleneck
    "scnet_nodil": NetSpec(mid_dilations=(1, 1, 1, 1)),
}


def model_names() -> list[str]:
    return sorted(SPECS)


def build_model(name: str, **overrides: Any) -> nn.Module:
    """Build a registered model; ``overrides`` replace fields of its spec (e.g. ``enc_blocks=1``)."""
    try:
        spec = SPECS[name]
    except KeyError:
        raise KeyError(f"unknown model {name!r}; choose from {model_names()}") from None
    for key in ("widths", "mid_dilations"):
        if key in overrides:
            overrides[key] = tuple(overrides[key])
    return ScreenCleanNet(replace(spec, **overrides))
