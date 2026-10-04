"""Confidence intervals by the (paired) bootstrap.

When two methods are scored on the same items (the same images or photos), compare them
*per item*: the difference on each item removes how hard that item is, which is most of the
spread. The bootstrap then asks how much the mean difference would move if we had drawn a
different set of items: resample the items with replacement many times (10,000, seed 0),
recompute the mean each time, and report the middle 95% of those means.

    d = paired_bootstrap(cer_model, cer_input)   # per-photo values, same order
    d.mean, d.lo, d.hi                           # e.g. -0.081 [-0.102, -0.061]

If the interval excludes 0, the difference is unlikely to be luck of the draw.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np

N_RESAMPLES = 10_000
SEED = 0


@dataclass
class Interval:
    mean: float
    lo: float
    hi: float
    n: int

    def excludes_zero(self) -> bool:
        return self.lo > 0 or self.hi < 0

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)

    def __str__(self) -> str:
        return f"{self.mean:+.3f} [{self.lo:+.3f}, {self.hi:+.3f}]"


def bootstrap_mean(
    values: Sequence[float] | np.ndarray,
    n_resamples: int = N_RESAMPLES,
    seed: int = SEED,
    level: float = 0.95,
) -> Interval:
    """Percentile bootstrap interval for the mean of ``values``."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or len(x) == 0:
        raise ValueError("need a non-empty 1-D list of values")
    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples)
    for start in range(0, n_resamples, 1000):  # in chunks, so 10k x 500 items stays small
        stop = min(start + 1000, n_resamples)
        means[start:stop] = x[rng.integers(0, len(x), size=(stop - start, len(x)))].mean(axis=1)
    tail = (1 - level) / 2 * 100
    lo, hi = np.percentile(means, [tail, 100 - tail])
    return Interval(float(x.mean()), float(lo), float(hi), len(x))


def paired_bootstrap(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
    n_resamples: int = N_RESAMPLES,
    seed: int = SEED,
    level: float = 0.95,
) -> Interval:
    """Interval for mean(a - b) over items scored by both methods (same order)."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"paired values need the same shape, got {a.shape} and {b.shape}")
    return bootstrap_mean(a - b, n_resamples, seed, level)


def paired_by_key(a: Mapping[str, float], b: Mapping[str, float], **kwargs) -> Interval:
    """:func:`paired_bootstrap` on the items (keys) both mappings have."""
    keys = sorted(set(a) & set(b))
    if not keys:
        raise ValueError("no items in common")
    return paired_bootstrap([a[k] for k in keys], [b[k] for k in keys], **kwargs)
