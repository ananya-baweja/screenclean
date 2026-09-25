"""How realistic is the simulator? Compare the spectra of simulated and real moiré.

For a pair (moiré photo, clean target), the *residual* is the photo minus the target after a
per-channel colour fit. Colour and brightness shifts are removed, and what's left is mostly
moiré plus noise. Its radially averaged power spectrum shows at which spatial frequencies the
artefacts live, separately for brightness (Y) and colour (Cr/Cb). If simulated and real
residual spectra have a similar shape and level, the simulator makes the right kind of moiré.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from screenclean.utils.image import resize_max_side, to_float01


def color_matched_residual(moire: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """``moire - fit(gt)`` where ``fit`` is a per-channel least-squares ``a * gt + b``."""
    m, g = to_float01(moire), to_float01(gt)
    out = np.empty_like(m)
    for c in range(3):
        x, y = g[..., c].astype(np.float64), m[..., c].astype(np.float64)
        var = x.var()
        a = ((x - x.mean()) * (y - y.mean())).mean() / var if var > 1e-6 else 1.0  # flat target: offset only
        b = y.mean() - a * x.mean()
        out[..., c] = m[..., c] - (a * g[..., c] + b)
    return out


def radial_spectrum(channel: np.ndarray, n_bins: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Radially averaged power spectrum (Hann-windowed). Returns (frequency in cycles/pixel, power)."""
    h, w = channel.shape
    win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    x = (channel - channel.mean()) * win
    power = np.abs(np.fft.fftshift(np.fft.fft2(x))) ** 2 / (win**2).sum()
    fy = np.fft.fftshift(np.fft.fftfreq(h))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(w))[None, :]
    r = np.sqrt(fx**2 + fy**2)
    edges = np.linspace(0, 0.5, n_bins + 1)
    idx = np.digitize(r.ravel(), edges) - 1
    ok = (idx >= 0) & (idx < n_bins)
    sums = np.bincount(idx[ok], weights=power.ravel()[ok], minlength=n_bins)
    counts = np.maximum(np.bincount(idx[ok], minlength=n_bins), 1)
    return (edges[:-1] + edges[1:]) / 2, sums / counts


def residual_spectra(
    pairs: Iterable[tuple[np.ndarray, np.ndarray]], n_bins: int = 64
) -> dict[str, np.ndarray]:
    """Mean residual spectra over pairs, for brightness (Y) and colour (mean of Cr and Cb)."""
    ys, cs, freqs = [], [], None
    for moire, gt in pairs:
        res = color_matched_residual(moire, gt)
        ycc = cv2.cvtColor(np.clip(res + 0.5, 0, 1).astype(np.float32), cv2.COLOR_RGB2YCrCb)
        freqs, py = radial_spectrum(ycc[..., 0], n_bins)
        _, pcr = radial_spectrum(ycc[..., 1], n_bins)
        _, pcb = radial_spectrum(ycc[..., 2], n_bins)
        ys.append(py)
        cs.append((pcr + pcb) / 2)
    if not ys:
        raise ValueError("no pairs given")
    return {"freq": freqs, "luma": np.mean(ys, axis=0), "chroma": np.mean(cs, axis=0), "n": len(ys)}


def band_energy(spec: dict[str, np.ndarray], lo: float = 0.01, hi: float = 0.25) -> dict[str, float]:
    """Total residual power in a frequency band (default: the band where visible moiré bands live)."""
    band = (spec["freq"] >= lo) & (spec["freq"] < hi)
    return {k: float(spec[k][band].sum()) for k in ("luma", "chroma")}


def spectra_figure(groups: dict[str, dict[str, np.ndarray]], out_path: str | Path, title: str = "") -> Path:
    """Log-log plot of residual spectra per group (brightness and colour panels)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, key, name in zip(axes, ("luma", "chroma"), ("brightness (Y)", "colour (Cr, Cb)"), strict=True):
        for label, spec in groups.items():
            f = spec["freq"]
            ax.loglog(f[1:], spec[key][1:], label=f"{label} (n={spec['n']})")
        ax.axvspan(0.01, 0.25, color="0.9", zorder=0)
        ax.set_xlabel("spatial frequency (cycles / pixel)")
        ax.set_title(f"residual power: {name}")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("power")
    axes[0].legend(fontsize=8)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def pair_grid(
    pairs: list[tuple[np.ndarray, np.ndarray]], cell: int = 256, max_side: int = 1600
) -> np.ndarray:
    """Rows of (moiré | target) thumbnails, as one uint8 image."""
    rows = []
    for moire, gt in pairs:
        m = cv2.resize(moire, (cell, cell), interpolation=cv2.INTER_AREA)
        g = cv2.resize(gt, (cell, cell), interpolation=cv2.INTER_AREA)
        rows.append(np.concatenate([m, g], axis=1))
    return resize_max_side(np.concatenate(rows, axis=0), max_side)


def summarize(groups: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out = {name: {"n": spec["n"], **band_energy(spec)} for name, spec in groups.items()}
    names = list(groups)
    if len(names) >= 2:
        a, b = out[names[0]], out[names[1]]
        out["ratio"] = {k: b[k] / max(a[k], 1e-12) for k in ("luma", "chroma")}
    return out
