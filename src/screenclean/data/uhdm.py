"""UHDM (real 4K moiré pairs): pair discovery, deterministic splits and crops.

Files look like ``<folder>/XXXX_gt.jpg`` and ``<folder>/XXXX_moire.jpg``. The 4-digit
prefixes repeat across folders, so a pair's key is its folder plus prefix, e.g.
``train/pair_22/0254``.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from screenclean.data.transforms import check_pair, random_crop_pair
from screenclean.utils.io import decode_rgb

log = logging.getLogger(__name__)

GT_SUFFIX = "_gt.jpg"
MOIRE_SUFFIX = "_moire.jpg"


@dataclass(frozen=True)
class Pair:
    """One image pair. ``moire`` and ``gt`` are file names (archive members or relative paths)."""

    key: str
    moire: str
    gt: str


def discover_pairs(names: Iterable[str], prefix: str = "") -> tuple[list[Pair], list[str]]:
    """Pair ``*_gt.jpg`` with ``*_moire.jpg`` among ``names`` (POSIX paths) under ``prefix``.

    Keys are the path below ``prefix`` without the suffix, e.g. with ``prefix="train/"``,
    ``train/train/pair_22/0254_gt.jpg`` has the key ``train/pair_22/0254``.
    Returns the pairs sorted by key, plus a list of problems (unmatched files).
    """
    gts, moires = {}, {}
    for name in names:
        if not name.startswith(prefix):
            continue
        rel = name[len(prefix) :]
        if rel.endswith(GT_SUFFIX):
            gts[rel[: -len(GT_SUFFIX)]] = name
        elif rel.endswith(MOIRE_SUFFIX):
            moires[rel[: -len(MOIRE_SUFFIX)]] = name
    pairs, problems = [], []
    for key in sorted(set(gts) | set(moires)):
        if key not in moires:
            problems.append(f"{key}: ground truth without a moire image")
        elif key not in gts:
            problems.append(f"{key}: moire image without a ground truth")
        else:
            pairs.append(Pair(key, moires[key], gts[key]))
    return pairs, problems


def list_files(root: str | Path) -> list[str]:
    """All file paths under ``root``, relative and in POSIX form (for :func:`discover_pairs`)."""
    root = Path(root)
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())


def sample_keys(keys: Iterable[str], n: int, seed: int = 0) -> list[str]:
    """Pick ``n`` keys reproducibly: sort them, then sample without replacement with ``seed``."""
    keys = sorted(keys)
    if n > len(keys):
        raise ValueError(f"asked for {n} keys but only {len(keys)} exist")
    idx = np.random.default_rng(seed).choice(len(keys), size=n, replace=False)
    return sorted(keys[i] for i in idx)


def split_train_val(keys: Iterable[str], n_val: int, seed: int = 0) -> tuple[list[str], list[str]]:
    """Hold out ``n_val`` training images as validation. Returns (train, val), both sorted."""
    keys = sorted(keys)
    val = sample_keys(keys, n_val, seed)
    val_set = set(val)
    return [k for k in keys if k not in val_set], val


def key_seed(key: str, salt: str = "") -> int:
    """A stable per-image seed, so crops don't depend on processing order or restarts."""
    return int(hashlib.sha1(f"{salt}:{key}".encode()).hexdigest()[:15], 16)


def decode_pair(moire_bytes: bytes, gt_bytes: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Decode a pair to uint8 RGB (moire, gt), checking both have the same size."""
    moire, gt = decode_rgb(moire_bytes), decode_rgb(gt_bytes)
    check_pair(moire, gt)
    return moire, gt


def half_scale(img: np.ndarray) -> np.ndarray:
    """Downscale by 2 with area averaging (a proper anti-aliased downscale)."""
    h, w = img.shape[:2]
    return cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)


def train_crops(
    moire: np.ndarray,
    gt: np.ndarray,
    key: str,
    size: int = 512,
    n_full: int = 2,
    n_half: int = 1,
    seed: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Random aligned crops: ``n_full`` at full resolution plus ``n_half`` from the image scaled by 0.5.

    The half-scale crops teach the model coarser moiré at the same crop size.
    """
    rng = np.random.default_rng(key_seed(key, f"train{seed}"))
    crops = [random_crop_pair(moire, gt, size, rng) for _ in range(n_full)]
    if n_half:
        m2, g2 = half_scale(moire), half_scale(gt)
        if min(m2.shape[:2]) >= size:
            crops += [random_crop_pair(m2, g2, size, rng) for _ in range(n_half)]
        else:
            log.warning("%s: too small for a half-scale %d crop", key, size)
    return crops


def val_crops(
    moire: np.ndarray, gt: np.ndarray, key: str, size: int = 512, n: int = 2, seed: int = 0
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Fixed validation crops: the same positions every time for a given key."""
    rng = np.random.default_rng(key_seed(key, f"val{seed}"))
    return [random_crop_pair(moire, gt, size, rng) for _ in range(n)]
