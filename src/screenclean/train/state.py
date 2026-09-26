"""Training state that must survive a Colab disconnect: EMA weights, random generators, checkpoints."""

from __future__ import annotations

import copy
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


class EMA:
    """Exponential moving average of the weights; evaluation always uses these.

    The decay ramps up as (1 + n) / (10 + n) until it reaches ``decay``, so the average isn't
    dominated by the random starting weights in short runs.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay, self.updates = decay, 0
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))
        for ema_p, p in zip(self.model.parameters(), model.parameters(), strict=True):
            ema_p.lerp_(p.detach(), 1.0 - d)
        for ema_b, b in zip(self.model.buffers(), model.buffers(), strict=True):
            ema_b.copy_(b)

    def state_dict(self) -> dict[str, Any]:
        return {"model": self.model.state_dict(), "updates": self.updates, "decay": self.decay}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.model.load_state_dict(state["model"])
        self.updates, self.decay = state["updates"], state["decay"]


def rng_state() -> dict[str, Any]:
    state = {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict[str, Any]) -> None:
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path: str | Path, obj: dict[str, Any]) -> Path:
    """Write to a temp file, then rename: a half-written checkpoint can never replace a good one."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)
    return path


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load one of our own checkpoints (it holds random-generator states, so not weights-only)."""
    return torch.load(path, map_location=map_location, weights_only=False)
