"""Reference model: ESDNet, the UHDM authors' lightweight network (ECCV 2022), pretrained on UHDM.

It is not our model. Results are always labelled "reference". The authors' code is
cloned at a pinned commit and their pretrained weights are downloaded once, then cached
on Drive. Inference follows their test script: pad the input to a multiple of 32 with their
padding colour, run at full resolution, keep the first (full-scale) output. If the GPU
runs out of memory, the image is processed in overlapping tiles instead, and the summary
reports how many images needed tiles.
"""

from __future__ import annotations

import importlib.util
import logging
import math
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from screenclean.data.download import DownloadError
from screenclean.eval.tiling import tiled_apply

log = logging.getLogger(__name__)

REPO_URL = "https://github.com/CVMI-Lab/UHDM.git"
DRIVE_URL = "https://drive.usercontent.google.com/download?id={id}&export=download&confirm=t"
PAD_VALUES = (0.3827, 0.4141, 0.3912)  # the authors' padding colour (mean RGB of the FHDMi training set)
ARCH = {"en_feature_num": 48, "en_inter_num": 32, "de_feature_num": 64, "de_inter_num": 32, "sam_number": 1}


def ensure_repo(repo_dir: str | Path, sha: str) -> Path:
    """Clone the authors' public repo at ``sha`` (skipped if it's already there)."""
    repo_dir = Path(repo_dir)
    if not (repo_dir / "model" / "nets.py").exists():
        subprocess.run(["git", "clone", "--quiet", REPO_URL, str(repo_dir)], check=True)
        subprocess.run(["git", "-C", str(repo_dir), "checkout", "--quiet", sha], check=True)
    return repo_dir


WEIGHTS_VIEW_URL = "https://drive.google.com/file/d/{id}/view"
MANUAL_COPY_NAMES = ("Copy of uhdm_checkpoint.pth", "uhdm_checkpoint.pth")


def manual_copy_help(spec: dict[str, Any]) -> str:
    return (
        "Workaround: open " + WEIGHTS_VIEW_URL.format(id=spec.get("id", "?")) + " while signed in to Google, "
        "click the three dots > Make a copy (or File > Make a copy). The copy appears in My Drive as "
        "'Copy of uhdm_checkpoint.pth'; leave it there and run the notebook again. The job finds it."
    )


def find_manual_copy(drive_root: Path, expected: int) -> Path | None:
    """A hand-made copy of the checkpoint in My Drive or the project folder, with the right size."""
    for folder in (drive_root.parent, drive_root, drive_root / "models"):
        for name in MANUAL_COPY_NAMES:
            p = folder / name
            if p.exists() and p.stat().st_size == expected:
                return p
    return None


def ensure_weights(
    spec: dict[str, Any], cache: Path, attempts: int = 3, wait_s: float = 30.0, drive_root: Path | None = None
) -> Path:
    """Get the checkpoint into ``cache`` (on Drive) once, checking its size.

    Looks, in order: the cache; a copy made by hand in Drive (see :func:`manual_copy_help`);
    ``spec["path"]`` (local file); a download of ``spec["id"]`` from Google Drive. Drive often
    answers with a "quota exceeded" page for popular files, so a few spaced-out attempts are
    made before giving up with a retry-later error that explains the manual workaround.
    """
    expected = int(spec["bytes"])
    if cache.exists() and cache.stat().st_size == expected:
        return cache
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_name(cache.name + ".tmp")
    manual = find_manual_copy(drive_root, expected) if drive_root is not None else None
    if manual is not None:
        log.info("using the checkpoint copy found at %s", manual)
        tmp.write_bytes(manual.read_bytes())
    elif "path" in spec:
        tmp.write_bytes(Path(spec["path"]).read_bytes())
    else:
        import requests

        for attempt in range(1, attempts + 1):
            r = requests.get(DRIVE_URL.format(id=spec["id"]), stream=True, timeout=120)
            if r.status_code == 200 and "text/html" not in r.headers.get("Content-Type", ""):
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
                if tmp.stat().st_size == expected:
                    break
            log.warning("weights download attempt %d/%d refused (Drive quota?)", attempt, attempts)
            if attempt < attempts:
                time.sleep(wait_s)
        else:
            raise DownloadError(
                "ESDNet weights: Google Drive refused the download (quota exceeded). "
                + manual_copy_help(spec),
                retry_later=True,
            )
    if tmp.stat().st_size != expected:
        raise DownloadError(f"ESDNet weights: expected {expected} bytes, got {tmp.stat().st_size}")
    tmp.replace(cache)
    return cache


def load_model(repo_dir: Path, weights: Path, device: str = "cpu", arch: dict[str, int] | None = None):
    """Build the authors' network from their ``model/nets.py`` and load the weights strictly."""
    import torch

    spec = importlib.util.spec_from_file_location("esdnet_nets", repo_dir / "model" / "nets.py")
    nets = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nets)
    model = nets.my_model(**(arch or ARCH))
    state = torch.load(weights, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


class ESDNetRunner:
    """Callable HWC float image -> HWC float image, like the classical baselines."""

    def __init__(
        self,
        model,
        device: str = "cpu",
        fp16: bool = True,
        tile: int = 1024,
        overlap: int = 64,
        force_tiles: bool = False,
    ):
        self.model, self.device = model, device
        self.fp16 = fp16 and device.startswith("cuda")
        self.tile, self.overlap, self.force_tiles = tile, overlap, force_tiles
        self.modes: Counter[str] = Counter()
        self.params = sum(p.numel() for p in model.parameters())

    def _forward(self, img: np.ndarray) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        x = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))[None].float().to(self.device)
        _, _, h, w = x.shape
        ph, pw = math.ceil(h / 32) * 32 - h, math.ceil(w / 32) * 32 - w
        top, left = ph // 2, pw // 2
        x = torch.cat(
            [
                F.pad(x[:, c : c + 1], (left, pw - left, top, ph - top), value=v)
                for c, v in enumerate(PAD_VALUES)
            ],
            dim=1,
        )
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=self.fp16):
            out = self.model(x)
        out = out[0] if isinstance(out, (tuple, list)) else out
        out = out[:, :, top : top + h, left : left + w].float().clamp(0, 1)
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        return out[0].cpu().numpy().transpose(1, 2, 0)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        import torch

        if not self.force_tiles:
            try:
                out = self._forward(img)
                self.modes["full"] += 1
                return out
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                log.warning("full-resolution ESDNet ran out of GPU memory; using %d-px tiles", self.tile)
        self.modes["tiled"] += 1
        return tiled_apply(img, self._forward, self.tile, self.overlap)


def build_esdnet(spec: dict[str, Any], drive_root: Path) -> ESDNetRunner:
    """Clone, fetch weights (cached on Drive) and return a ready :class:`ESDNetRunner`."""
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    repo = ensure_repo(spec.get("repo_dir", "/content/UHDM"), spec["repo_sha"])
    weights = ensure_weights(
        spec["weights"],
        drive_root / spec.get("weights_cache", "models/esdnet_uhdm.pth"),
        attempts=int(spec.get("download_attempts", 3)),
        wait_s=float(spec.get("download_wait_s", 30)),
        drive_root=drive_root,
    )
    model = load_model(repo, weights, device)
    runner = ESDNetRunner(
        model,
        device,
        fp16=bool(spec.get("fp16", True)),
        tile=int(spec.get("tile", 1024)),
        overlap=int(spec.get("overlap", 64)),
        force_tiles=bool(spec.get("force_tiles", False)),
    )
    log.info(
        "ESDNet reference ready on %s: %.2f M parameters, fp16=%s", device, runner.params / 1e6, runner.fp16
    )
    return runner
