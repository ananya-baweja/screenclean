"""PyTorch dataset over tar shards of (moiré, ground-truth) pairs.

Samples are read straight from the tar files by byte offset, so shards staged to local
disk (``utils.drive.stage_shards``) can be used without extracting them.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from screenclean.data.shards import group_samples, index_tar, read_member, source_key
from screenclean.data.transforms import augment_pair, check_pair
from screenclean.utils.io import decode_rgb


def to_tensor(img: np.ndarray) -> torch.Tensor:
    """uint8 HWC RGB -> float32 CHW in [0, 1]."""
    return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float().div_(255.0)


class PairsDataset(Dataset):
    """(moire, gt) float tensors from shards.

    Args:
        shards: tar shard paths.
        crop: random crop size (training) or None to return whole samples.
        augment: random flips and 90-degree rotations.
        exclude_keys: source image keys to leave out.
        seed: base seed for augmentation (mixed with the DataLoader worker seed).
    """

    def __init__(
        self,
        shards: Iterable[str | Path],
        crop: int | None = None,
        augment: bool = False,
        exclude_keys: Iterable[str] | None = None,
        seed: int = 0,
    ):
        self.shards = [Path(p) for p in shards]
        self.crop, self.augment, self.seed = crop, augment, seed
        exclude = set(exclude_keys or ())
        self.items: list[tuple[int, str, tuple[int, int], tuple[int, int]]] = []
        for i, path in enumerate(self.shards):
            index = index_tar(path)
            for sid, members in sorted(group_samples(index).items()):
                if source_key(sid) in exclude:
                    continue
                self.items.append((i, sid, index[members["moire"]], index[members["gt"]]))
        self._rng: np.random.Generator | None = None
        self._rng_pid: int | None = None

    def __len__(self) -> int:
        return len(self.items)

    def _get_rng(self) -> np.random.Generator:
        # One generator per process: DataLoader workers get different torch seeds each epoch.
        if self._rng is None or self._rng_pid != os.getpid():
            self._rng = np.random.default_rng([self.seed, torch.initial_seed() % (1 << 63)])
            self._rng_pid = os.getpid()
        return self._rng

    def read_pair(self, idx: int) -> tuple[str, np.ndarray, np.ndarray]:
        """Decoded uint8 RGB (sample_id, moire, gt) without augmentation."""
        shard_i, sid, (mo, ms), (go, gs) = self.items[idx]
        path = self.shards[shard_i]
        moire = decode_rgb(read_member(path, mo, ms))
        gt = decode_rgb(read_member(path, go, gs))
        check_pair(moire, gt)
        return sid, moire, gt

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        _, moire, gt = self.read_pair(idx)
        if self.crop is not None or self.augment:
            moire, gt = augment_pair(moire, gt, self._get_rng(), crop=self.crop, flips=self.augment)
        return to_tensor(moire), to_tensor(gt)
