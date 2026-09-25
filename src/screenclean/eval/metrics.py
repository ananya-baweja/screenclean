"""Image-quality metrics: PSNR, SSIM and (in Colab) LPIPS.

All metrics compare uint8 RGB images at full resolution, the usual protocol for
demoiréing papers. Float inputs in [0, 1] are rounded to uint8 first, so every
method is scored on the same 8-bit images it would actually save.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from skimage.metrics import structural_similarity

from screenclean.utils.image import to_uint8

PSNR_CAP = 100.0


def _pair_u8(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a, b = to_uint8(a), to_uint8(b)
    if a.shape != b.shape:
        raise ValueError(f"image shapes differ: {a.shape} vs {b.shape}")
    return a, b


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """Peak signal-to-noise ratio in dB on uint8 RGB (identical images give ``PSNR_CAP``)."""
    a, b = _pair_u8(a, b)
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    if mse == 0:
        return PSNR_CAP
    return min(PSNR_CAP, 10.0 * math.log10(255.0**2 / mse))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Structural similarity on uint8 RGB with the standard Gaussian-window settings."""
    a, b = _pair_u8(a, b)
    return float(
        structural_similarity(
            a,
            b,
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False,
            data_range=255,
            channel_axis=-1,
        )
    )


@lru_cache(maxsize=2)
def _lpips_model(net: str, device: str):
    import lpips  # Colab only (downloads AlexNet weights on first use)

    return lpips.LPIPS(net=net, verbose=False).to(device).eval()


def lpips_distance(a: np.ndarray, b: np.ndarray, net: str = "alex", device: str | None = None) -> float:
    """LPIPS perceptual distance (lower is better). Needs the ``lpips`` package and its weights."""
    import torch

    a, b = _pair_u8(a, b)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = _lpips_model(net, device)

    def to_t(x: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))).float().div(127.5).sub(1.0)
        return t[None].to(device)

    with torch.no_grad():
        return float(model(to_t(a), to_t(b)).item())
