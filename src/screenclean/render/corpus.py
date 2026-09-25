"""The text corpus for rendered pages, split into train / val / test by line.

Synthetic training pages only use ``train`` lines, and the real-photo benchmark pages only
use ``test`` lines. So a model can never have seen the exact text it is scored on.
"""

from __future__ import annotations

import re
from functools import lru_cache
from importlib import resources

import numpy as np

SPLITS = ("train", "val", "test")
_CODE_HINTS = re.compile(
    r"[(){};=\[\]]|^\s|^(import|def|class|return|for|while|try|except|SELECT|UPDATE|CREATE)\b"
)


@lru_cache(maxsize=1)
def load_lines() -> tuple[str, ...]:
    """All corpus lines (the file shipped inside the package)."""
    text = resources.files("screenclean.render").joinpath("corpus_en.txt").read_text(encoding="utf-8")
    return tuple(line.rstrip("\n") for line in text.splitlines() if line.strip())


def split_indices(
    n: int, seed: int = 0, fractions: tuple[float, float, float] = (0.8, 0.1, 0.1)
) -> dict[str, list[int]]:
    """Shuffle line indices with ``seed`` and cut them 80 / 10 / 10 into train, val and test."""
    perm = np.random.default_rng(seed).permutation(n)
    n_train = int(round(fractions[0] * n))
    n_val = int(round(fractions[1] * n))
    parts = (perm[:n_train], perm[n_train : n_train + n_val], perm[n_train + n_val :])
    return {name: sorted(int(i) for i in part) for name, part in zip(SPLITS, parts, strict=True)}


def lines_for(split: str, seed: int = 0) -> list[str]:
    """The corpus lines belonging to ``split`` ("train", "val" or "test")."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    lines = load_lines()
    return [lines[i] for i in split_indices(len(lines), seed)[split]]


def is_code(line: str) -> bool:
    """Rough guess whether a line is source code or a shell command (used to pick lines for code pages)."""
    return bool(_CODE_HINTS.search(line)) and not line.endswith(".")
