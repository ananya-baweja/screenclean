"""Colab task ``efficiency``: how big and how fast each model is (P9).

For every model, at 1280x720 and 1920x1080:

- parameters, and FLOPs of one forward pass (``FlopCounterMode``: convolutions and matrix
  products, one multiply-add = 2 FLOPs);
- latency on the GPU (fp16, as in the evaluation) and its peak memory;
- latency on the CPU (fp32, 2 threads, like a small server).

Latency is for the whole call as the app makes it: image in, image out, including the copies to
and from the GPU and any padding, timed after warm-up runs (median of several runs). ONNX Runtime
timings come with the export in P10.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from screenclean.jobs import JobContext, TaskResult, register

log = logging.getLogger(__name__)

SIZES = {"1280x720": (720, 1280), "1920x1080": (1080, 1920)}


def _test_image(h: int, w: int, seed: int = 0) -> np.ndarray:
    """A screen-like test image: smooth colours plus fine stripes (the content doesn't change the cost)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w] / max(h, w)
    base = np.stack(
        [0.5 + 0.3 * np.sin(2 * np.pi * (f * xx + g * yy)) for f, g in rng.uniform(0.5, 3, (3, 2))], -1
    )
    stripes = 0.05 * np.sin(2 * np.pi * 0.3 * np.arange(w))[None, :, None]
    return np.clip(base + stripes, 0, 1).astype(np.float32)


def time_call(fn, img: np.ndarray, repeats: int, warmup: int, cuda: bool) -> float:
    """Median seconds per call."""
    import torch

    for _ in range(warmup):
        fn(img)
    times = []
    for _ in range(repeats):
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(img)
        if cuda:
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def count_call_flops(fn, img: np.ndarray) -> int:
    """FLOPs of everything ``fn`` runs in PyTorch for one image (padding and tiles included)."""
    from torch.utils.flop_counter import FlopCounterMode

    with FlopCounterMode(display=False) as counter:
        fn(img)
    return int(counter.get_total_flops())


def build(spec: dict[str, Any], drive_root: Path, device: str, work_dir: Path):
    """A runner (image -> image) on ``device`` for a model entry of the config."""
    spec = {**spec, "device": device, "fp16": device == "cuda"}
    if spec["name"] == "esdnet_ref":
        from screenclean.baselines.esdnet_ref import build_esdnet

        return build_esdnet(spec, drive_root)
    from screenclean.models.runner import ModelRunner, build_runner

    if spec.get("untrained"):  # architecture only (e.g. scnet_tiny): a fresh model, saved like a checkpoint
        from screenclean.models.registry import build_model
        from screenclean.train.state import save_checkpoint

        path = work_dir / f"untrained_{spec['untrained']}.pt"
        if not path.exists():
            model = build_model(spec["untrained"])
            save_checkpoint(path, {"model": model.state_dict(), "spec": model.spec.as_dict()})
        return ModelRunner(path, device=device, fp16=device == "cuda")
    return build_runner(spec, drive_root)


@register("efficiency")
def efficiency_task(ctx: JobContext) -> TaskResult:
    import torch

    cfg = ctx.config
    cuda = torch.cuda.is_available()
    sizes = {k: tuple(v) for k, v in cfg.get("sizes", SIZES).items()}
    rows = []
    for m in cfg["models"]:
        row: dict[str, Any] = {"label": m["label"]}
        devices = (["cuda"] if cuda else []) + (["cpu"] if m.get("cpu", True) else [])
        for device in devices:
            runner = build(m, ctx.layout.root, device, ctx.work_dir)
            row["params_m"] = round(runner.params / 1e6, 3)
            if device == "cpu":
                torch.set_num_threads(int(cfg.get("cpu_threads", 2)))
            for name, (h, w) in sizes.items():
                img = _test_image(h, w)
                if device == "cuda":
                    row[f"gflops_{name}"] = round(count_call_flops(runner, img) / 1e9, 1)
                    torch.cuda.reset_peak_memory_stats()
                    row[f"gpu_ms_{name}"] = round(
                        1000 * time_call(runner, img, int(cfg.get("gpu_repeats", 20)), 3, True), 1
                    )
                    row[f"gpu_peak_gb_{name}"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
                else:
                    if f"gflops_{name}" not in row:
                        row[f"gflops_{name}"] = round(count_call_flops(runner, img) / 1e9, 1)
                    row[f"cpu_s_{name}"] = round(
                        time_call(runner, img, int(cfg.get("cpu_repeats", 3)), 1, False), 2
                    )
                ctx.progress(f"{m['label']} on {device} at {name}: done")
                log.info("%s", row)
            if hasattr(runner, "modes"):
                row[f"modes_{device}"] = dict(runner.modes)
            del runner
            if cuda:
                torch.cuda.empty_cache()
        rows.append(row)
    summary = {
        "rows": rows,
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "cpu_threads": int(cfg.get("cpu_threads", 2)),
        "sizes": {k: list(v) for k, v in sizes.items()},
        "notes": "latency = image in -> image out (median); GPU fp16, CPU fp32",
    }
    return TaskResult("done", summary)
