"""Seeding for reproducible runs."""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int = 0, deterministic: bool = False) -> int:
    """Seed Python, NumPy and (if installed) PyTorch. Returns the seed for logging.

    ``deterministic=True`` also asks cuDNN for deterministic kernels, which is slower.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed


def make_rng(seed: int) -> np.random.Generator:
    """Create an independent NumPy generator (prefer this over the global RNG)."""
    return np.random.default_rng(seed)
