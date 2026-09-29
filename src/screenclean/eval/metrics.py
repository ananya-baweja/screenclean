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
    """Structural similarity on uint8 RGB with the standard Gaussian-window settings.

    On a GPU this uses :func:`ssim_torch`: the same numbers as scikit-image (float64, tested to
    1e-9), in well under a second per 4K image instead of several seconds on Colab's 2-core CPU,
    where scikit-image dominated the evaluation time. Without a GPU, scikit-image is used (float64
    convolution on a CPU is slow in PyTorch).
    """
    try:
        import torch
    except ImportError:
        return ssim_skimage(a, b)
    return ssim_torch(a, b, "cuda") if torch.cuda.is_available() else ssim_skimage(a, b)


def ssim_skimage(a: np.ndarray, b: np.ndarray) -> float:
    """The reference SSIM (scikit-image): Gaussian window σ = 1.5 (11 taps), data range 255."""
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


def ssim_torch(a: np.ndarray, b: np.ndarray, device: str | None = None) -> float:
    """scikit-image's SSIM recipe in float64 PyTorch.

    Per channel: local means, variances and covariance under an 11-tap Gaussian (σ = 1.5,
    normalised like ``scipy.ndimage``), population covariance, C1 = (0.01·255)², C2 = (0.03·255)²;
    the SSIM map is averaged over the interior (5 px border dropped, as scikit-image crops it),
    then over channels. Filtering only the interior ("valid") gives exactly that cropped mean.
    """
    import torch
    import torch.nn.functional as F

    a, b = _pair_u8(a, b)
    if a.ndim == 2:
        a, b = a[..., None], b[..., None]
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1))).to(dev, torch.float64)[:, None]
    y = torch.from_numpy(np.ascontiguousarray(b.transpose(2, 0, 1))).to(dev, torch.float64)[:, None]
    r = 5  # int(3.5 * 1.5 + 0.5), scipy's radius for truncate 3.5
    t = torch.arange(-r, r + 1, dtype=torch.float64, device=dev)
    g = torch.exp(-0.5 * (t / 1.5) ** 2)
    g = g / g.sum()
    maps = torch.cat([x, y, x * x, y * y, x * y])  # (5C, 1, H, W)
    maps = F.conv2d(F.conv2d(maps, g.view(1, 1, -1, 1)), g.view(1, 1, 1, -1))
    ux, uy, uxx, uyy, uxy = maps.chunk(5)
    vx, vy, vxy = uxx - ux * ux, uyy - uy * uy, uxy - ux * uy
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    s = ((2 * ux * uy + c1) * (2 * vxy + c2)) / ((ux * ux + uy * uy + c1) * (vx + vy + c2))
    return float(s.mean(dim=(1, 2, 3)).mean())


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
