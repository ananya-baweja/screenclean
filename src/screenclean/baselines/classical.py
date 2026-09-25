"""Classical (non-learned) moiré removal, used as baselines.

Every function has the form ``fn(img, **params) -> img`` on float32 RGB in [0, 1], HWC.

- ``identity``: returns the input, so its score is the "no cleaning" reference.
- ``chroma_lowpass``: blurs only the colour (Cr/Cb) channels. Much moiré is coloured
  banding, while text detail lives mostly in brightness (Y).
- ``fft_notch``: moiré is a near-periodic pattern, so in the 2-D Fourier spectrum it shows
  up as bright isolated peaks. Find peaks that stand out from the local spectrum
  background, and suppress each one (and its mirror twin) with a Gaussian notch.
- ``fft_notch_local``: the same in overlapping 256-px tiles, because in a photo taken at an
  angle the moiré frequency drifts across the image.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import fft as sfft
from scipy.ndimage import maximum_filter

from screenclean.eval.tiling import tile_starts


def identity(img: np.ndarray) -> np.ndarray:
    """No cleaning."""
    return img.astype(np.float32, copy=True)


def _to_ycc(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.ascontiguousarray(img, dtype=np.float32), cv2.COLOR_RGB2YCrCb)


def _to_rgb(ycc: np.ndarray) -> np.ndarray:
    return np.clip(cv2.cvtColor(np.ascontiguousarray(ycc, dtype=np.float32), cv2.COLOR_YCrCb2RGB), 0.0, 1.0)


def chroma_lowpass(img: np.ndarray, sigma: float = 4.0) -> np.ndarray:
    """Gaussian-blur Cr and Cb with ``sigma`` pixels; keep brightness (Y) untouched."""
    ycc = _to_ycc(img)
    for c in (1, 2):
        ycc[..., c] = cv2.GaussianBlur(ycc[..., c], (0, 0), sigmaX=sigma, sigmaY=sigma)
    return _to_rgb(ycc)


def _median_background(logmag: np.ndarray, size: int) -> np.ndarray:
    """Median filter of the log spectrum. Quantised to 8 bits so OpenCV's fast median works."""
    lo, hi = float(logmag.min()), float(logmag.max())
    scale = 255.0 / max(hi - lo, 1e-6)
    q = np.round((logmag - lo) * scale).astype(np.uint8)
    return cv2.medianBlur(q, size).astype(np.float32) / scale + lo


def periodic_smooth(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split ``u`` into a periodic part and a smooth part (Moisan, 2011): ``u = p + s``.

    The FFT treats an image as if it wrapped around, so the jump between opposite edges
    shows up as a bright cross through the spectrum. That cross would look like
    "peaks" to the notch filter. ``p`` has no such jumps; ``s`` is smooth and holds them.
    """
    h, w = u.shape
    v = np.zeros_like(u, dtype=np.float64)
    v[0, :] += u[-1, :] - u[0, :]
    v[-1, :] -= u[-1, :] - u[0, :]
    v[:, 0] += u[:, -1] - u[:, 0]
    v[:, -1] -= u[:, -1] - u[:, 0]
    q = np.arange(h).reshape(-1, 1)
    r = np.arange(w).reshape(1, -1)
    denom = 2 * np.cos(2 * np.pi * q / h) + 2 * np.cos(2 * np.pi * r / w) - 4
    denom[0, 0] = 1.0
    s_hat = sfft.fft2(v, workers=-1) / denom
    s_hat[0, 0] = 0.0
    s = np.real(sfft.ifft2(s_hat, workers=-1)).astype(np.float32)
    return (u - s).astype(np.float32), s


def find_peaks(
    ch: np.ndarray, r0: float = 0.08, k: float = 6.0, bg_size: int = 9, max_peaks: int = 100
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Centred spectrum of one channel and the moiré peak positions in it.

    ``r0`` is the protected radius around zero frequency, as a fraction of the half
    spectrum (the image's own coarse structure lives there). A peak must exceed the
    local median background by ``k`` robust standard deviations (MAD).
    """
    h, w = ch.shape
    spec = sfft.fftshift(sfft.fft2(ch - ch.mean(), workers=-1))
    logmag = np.log1p(np.abs(spec)).astype(np.float32)
    resid = logmag - _median_background(logmag, bg_size)
    med = float(np.median(resid))
    mad = float(np.median(np.abs(resid - med))) * 1.4826 + 1e-6
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    radius = np.sqrt(((yy - cy) / (h / 2)) ** 2 + ((xx - cx) / (w / 2)) ** 2)
    candidates = (resid > med + k * mad) & (radius > r0) & (resid == maximum_filter(resid, size=5))
    ys, xs = np.nonzero(candidates)
    order = np.argsort(-resid[ys, xs])[:max_peaks]
    return spec, [(int(ys[i]), int(xs[i])) for i in order]


def notch_mask(shape: tuple[int, int], peaks: list[tuple[int, int]], sigma: float) -> np.ndarray:
    """Product of Gaussian notches at each peak and its mirror twin (a real image's spectrum is symmetric)."""
    h, w = shape
    mask = np.ones(shape, np.float32)
    rad = int(np.ceil(3 * sigma))
    dy, dx = np.mgrid[-rad : rad + 1, -rad : rad + 1]
    notch = (1.0 - np.exp(-(dy**2 + dx**2) / (2 * sigma**2))).astype(np.float32)
    cy, cx = h // 2, w // 2
    for py, px in peaks:
        for y, x in ((py, px), ((2 * cy - py) % h, (2 * cx - px) % w)):
            y0, y1, x0, x1 = y - rad, y + rad + 1, x - rad, x + rad + 1
            sy0, sx0 = max(0, -y0), max(0, -x0)
            y0c, y1c, x0c, x1c = max(y0, 0), min(y1, h), max(x0, 0), min(x1, w)
            mask[y0c:y1c, x0c:x1c] *= notch[sy0 : sy0 + (y1c - y0c), sx0 : sx0 + (x1c - x0c)]
    return mask


def notch_channel(
    ch: np.ndarray,
    r0: float = 0.08,
    k: float = 6.0,
    sigma: float = 3.0,
    bg_size: int = 9,
    max_peaks: int = 100,
) -> np.ndarray:
    """Remove periodic peaks from one channel (float32 2-D array).

    Only the periodic component is filtered; the smooth edge component is added back unchanged.
    """
    periodic, smooth = periodic_smooth(ch)
    spec, peaks = find_peaks(periodic, r0, k, bg_size, max_peaks)
    if not peaks:
        return ch.astype(np.float32, copy=True)
    filtered = spec * notch_mask(ch.shape, peaks, sigma)
    out = np.real(sfft.ifft2(sfft.ifftshift(filtered), workers=-1)) + periodic.mean() + smooth
    return out.astype(np.float32)


def _channels(channels: str) -> tuple[int, ...]:
    if channels not in ("y", "ycc"):
        raise ValueError("channels must be 'y' or 'ycc'")
    return (0,) if channels == "y" else (0, 1, 2)


def fft_notch(
    img: np.ndarray,
    r0: float = 0.08,
    k: float = 6.0,
    sigma: float = 3.0,
    bg_size: int = 9,
    max_peaks: int = 100,
    channels: str = "y",
) -> np.ndarray:
    """Global FFT notch filter on Y (``channels="y"``) or on Y, Cr and Cb (``"ycc"``)."""
    ycc = _to_ycc(img)
    for c in _channels(channels):
        ycc[..., c] = notch_channel(ycc[..., c], r0, k, sigma, bg_size, max_peaks)
    return _to_rgb(ycc)


def fft_notch_local(
    img: np.ndarray,
    tile: int = 256,
    hop: int = 128,
    r0: float = 0.08,
    k: float = 6.0,
    sigma: float = 2.0,
    bg_size: int = 5,
    max_peaks: int = 20,
    channels: str = "y",
) -> np.ndarray:
    """FFT notch per overlapping tile, blended with a Hann window (overlap-add)."""
    ycc = _to_ycc(img)
    pad = tile // 2
    padded = cv2.copyMakeBorder(ycc, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
    h, w = padded.shape[:2]
    win1d = np.hanning(tile + 2)[1:-1].astype(np.float32)  # strictly positive Hann
    window = np.outer(win1d, win1d)
    for c in _channels(channels):
        acc = np.zeros((h, w), np.float32)
        wsum = np.zeros((h, w), np.float32)
        for y in tile_starts(h, tile, tile - hop):
            for x in tile_starts(w, tile, tile - hop):
                block = padded[y : y + tile, x : x + tile, c]
                wb = window[: block.shape[0], : block.shape[1]]
                acc[y : y + tile, x : x + tile] += notch_channel(block, r0, k, sigma, bg_size, max_peaks) * wb
                wsum[y : y + tile, x : x + tile] += wb
        padded[..., c] = acc / np.maximum(wsum, 1e-8)
    return _to_rgb(padded[pad : pad + ycc.shape[0], pad : pad + ycc.shape[1]])
