"""Training and validation data for the trainer.

Training samples are addressed by a *global sample number* k = 0, 1, 2, ...: sample k picks its
source (e.g. UHDM or synthetic text, by weight), its pair, its crop and its flips from a random
generator seeded with (seed, k). So:

- the data order doesn't depend on DataLoader workers or timing;
- a resumed run continues with exactly the samples it would have seen (start at k = iteration x batch);
- several sources can be mixed by weight (the text fine-tune in P8 mixes UHDM and synthetic pairs).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from screenclean.data.pairs_dataset import PairsDataset, to_tensor
from screenclean.data.transforms import augment_pair


class MixedCrops(Dataset):
    """Random crops from one or more pair sources, fully determined by the sample number."""

    def __init__(
        self,
        sources: Sequence[PairsDataset],
        crop: int,
        seed: int = 0,
        weights: Sequence[float] | None = None,
        length: int = 10**12,
    ):
        if not sources or any(len(s) == 0 for s in sources):
            raise ValueError("every training source needs at least one pair")
        self.sources, self.crop, self.seed, self.length = list(sources), crop, seed, length
        w = np.asarray(weights if weights is not None else [1.0] * len(self.sources), float)
        if len(w) != len(self.sources) or (w < 0).any() or w.sum() <= 0:
            raise ValueError("weights must be one non-negative number per source")
        self.p = w / w.sum()

    def __len__(self) -> int:
        return self.length

    def pick(self, k: int) -> tuple[int, int]:
        """(source, pair index) of sample k."""
        rng = np.random.default_rng([self.seed, k])
        src = int(rng.choice(len(self.sources), p=self.p)) if len(self.sources) > 1 else 0
        return src, int(rng.integers(len(self.sources[src])))

    def __getitem__(self, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        src, idx = self.pick(k)
        _, moire, gt = self.sources[src].read_pair(idx)
        rng = np.random.default_rng([self.seed, k, 1])
        moire, gt = augment_pair(moire, gt, rng, crop=self.crop, flips=True)
        return to_tensor(moire), to_tensor(gt)


def train_loader(
    dataset: MixedCrops, batch: int, start_iteration: int, iterations: int, workers: int = 2
) -> DataLoader:
    """Batches for iterations ``start_iteration`` .. ``iterations - 1``, in a fixed order."""
    order = range(start_iteration * batch, iterations * batch)  # the sampler: sample numbers in order
    return DataLoader(
        dataset,
        batch_size=batch,
        sampler=order,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        prefetch_factor=4 if workers > 0 else None,
        drop_last=True,
    )


def shards_in(split_dir: str | Path) -> list[Path]:
    """The tar shards of a split folder (local copy), in name order."""
    return sorted(Path(split_dir).glob("*.tar"))
