"""Classical baselines (identity, chroma low-pass, FFT notch filters) and the ESDNet reference wrapper."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from screenclean.baselines.classical import chroma_lowpass, fft_notch, fft_notch_local, identity

BASELINES: dict[str, Callable[..., np.ndarray]] = {
    "identity": identity,
    "chroma_lowpass": chroma_lowpass,
    "fft_notch": fft_notch,
    "fft_notch_local": fft_notch_local,
}


def get_baseline(name: str) -> Callable[..., np.ndarray]:
    try:
        return BASELINES[name]
    except KeyError:
        raise KeyError(f"unknown baseline {name!r}; choose from {sorted(BASELINES)}") from None
