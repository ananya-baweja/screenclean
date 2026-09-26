"""Run a trained ScreenCleanNet checkpoint on full-resolution images (PyTorch; ONNX comes in P10).

``ModelRunner(path)`` is a callable HWC float image -> HWC float image in [0, 1], like the
classical baselines, so the ``eval`` task and the scan pipeline can use it directly.

- It loads the EMA weights: ``best.pt`` stores them as ``model``, ``last.pt`` under ``ema``.
- ``tile=None`` runs the whole image at once and falls back to overlapping tiles if the GPU runs
  out of memory; ``tile=512`` always uses tiles. (The channel-attention layers average over their
  whole input: training saw 384 px crops, so tiles close to that size may match training better
  than a full 4K image. The evaluation measures both.)
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from screenclean.eval.tiling import tiled_apply
from screenclean.models.scnet import NetSpec, ScreenCleanNet
from screenclean.train.state import load_checkpoint

log = logging.getLogger(__name__)


def load_trained(
    path: str | Path, device: str | torch.device = "cpu"
) -> tuple[ScreenCleanNet, dict[str, Any]]:
    """The EMA model from a checkpoint, in eval mode, plus facts about it for reports."""
    ck = load_checkpoint(path, map_location="cpu")
    spec = dict(ck["spec"])
    for key in ("widths", "mid_dilations"):
        spec[key] = tuple(spec[key])
    model = ScreenCleanNet(NetSpec(**spec))
    model.load_state_dict(ck["ema"]["model"] if "ema" in ck else ck["model"])
    info = {
        "checkpoint": str(path),
        "model_name": ck.get("model_name") or ck.get("config", {}).get("model"),
        "iteration": ck.get("iteration"),
        "val": ck.get("val"),
    }
    return model.to(device).eval(), info


class ModelRunner:
    def __init__(
        self,
        checkpoint: str | Path,
        device: str | None = None,
        fp16: bool = True,
        tile: int | None = None,
        overlap: int = 64,
        fallback_tile: int = 1024,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.info = load_trained(checkpoint, self.device)
        self.fp16 = fp16 and self.device.startswith("cuda")
        self.tile, self.fallback_tile = tile, fallback_tile
        self.overlap = min(overlap, (tile or fallback_tile) // 4)  # small tiles: keep the overlap sensible
        self.modes: Counter[str] = Counter()
        self.params = sum(p.numel() for p in self.model.parameters())

    @torch.no_grad()
    def _forward(self, img: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))[None].float().to(self.device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.fp16):
            out = self.model(x)
        return out[0].float().clamp(0, 1).cpu().numpy().transpose(1, 2, 0)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        img = np.asarray(img, np.float32)
        if self.tile is None:
            try:
                out = self._forward(img)
                self.modes["full"] += 1
                return out
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                log.warning("full-resolution run out of GPU memory; using %d px tiles", self.fallback_tile)
        self.modes["tiled"] += 1
        return tiled_apply(img, self._forward, self.tile or self.fallback_tile, self.overlap)


def build_runner(spec: dict[str, Any], drive_root: Path) -> ModelRunner:
    """A runner from an eval-config method entry (``checkpoint`` is relative to the Drive root)."""
    runner = ModelRunner(
        Path(drive_root) / spec["checkpoint"],
        fp16=bool(spec.get("fp16", True)),
        tile=spec.get("tile"),
        overlap=int(spec.get("overlap", 64)),
    )
    log.info(
        "%s ready on %s: %.2f M parameters, iteration %s, fp16=%s, tile=%s",
        spec.get("label", spec["name"]),
        runner.device,
        runner.params / 1e6,
        runner.info["iteration"],
        runner.fp16,
        runner.tile,
    )
    return runner
