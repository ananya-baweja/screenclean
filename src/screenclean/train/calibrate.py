"""Measure training speed and memory on the actual GPU, and pick the batch size and schedule length.

For every (crop, batch) pair: a fresh model trains for a few iterations on real data; after a
warm-up, it records iterations per second, peak GPU memory and the share of time spent waiting
for data. Out-of-memory is recorded, not fatal. The recommendation is the largest batch at the
largest crop that leaves memory headroom, and the number of iterations that fills the planned
training time. The loss terms of the untrained model (which returns its input) are measured too,
to calibrate the FFT weight so its term is about 20% of the loss at the start.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import torch

from screenclean.data.pairs_dataset import PairsDataset
from screenclean.models.registry import build_model
from screenclean.train.data import MixedCrops, train_loader
from screenclean.train.losses import TrainLoss, suggest_fft_weight

log = logging.getLogger(__name__)


def measure(
    model_name: str,
    sources: list[PairsDataset],
    crop: int,
    batch: int,
    iters: int,
    warmup: int,
    device: torch.device,
    workers: int = 2,
    amp: bool = True,
) -> dict[str, Any]:
    row: dict[str, Any] = {"crop": crop, "batch": batch}
    amp = amp and device.type == "cuda"
    model = opt = None
    try:
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        model = build_model(model_name).to(device).train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        loss_fn = TrainLoss().to(device)
        loader = train_loader(MixedCrops(sources, crop, seed=123), batch, 0, iters, workers)
        wait, t_start = 0.0, None
        it = iter(loader)
        for i in range(iters):
            t0 = time.monotonic()
            moire, gt = next(it)
            if t_start is not None:
                wait += time.monotonic() - t0
            moire, gt = moire.to(device, non_blocking=True), gt.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                pred = model(moire)
            loss, _ = loss_fn(pred.float(), gt)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if i == warmup - 1:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                t_start = time.monotonic()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seconds = time.monotonic() - t_start
        n = iters - warmup
        row.update(
            it_per_s=round(n / seconds, 3),
            samples_per_s=round(n * batch / seconds, 1),
            data_wait=round(wait / seconds, 3),
            peak_mem_gb=round(torch.cuda.max_memory_allocated(device) / 1e9, 2)
            if device.type == "cuda"
            else 0.0,
            oom=False,
        )
    except torch.cuda.OutOfMemoryError:
        row.update(oom=True)
    finally:
        del model, opt
        if device.type == "cuda":
            torch.cuda.empty_cache()
    log.info("calibration %s", row)
    return row


@torch.no_grad()
def loss_at_start(
    sources: list[PairsDataset], crop: int, batches: int = 8, batch: int = 6
) -> dict[str, float]:
    """Loss terms of the untrained model (the identity) on real training crops."""
    loss_fn = TrainLoss(fft_weight=1.0)
    data = MixedCrops(sources, crop, seed=321)
    sums: dict[str, float] = {"charbonnier": 0.0, "fft": 0.0}
    for b in range(batches):
        moire, gt = zip(*(data[b * batch + i] for i in range(batch)), strict=True)
        _, parts = loss_fn(torch.stack(moire), torch.stack(gt))
        for k in sums:
            sums[k] += float(parts[k]) / batches
    return sums


def recommend(
    rows: list[dict[str, Any]],
    total_mem_gb: float,
    plan_minutes: float,
    headroom: float = 0.75,
    overhead: float = 0.1,
) -> dict[str, Any]:
    """Largest batch at the largest crop that fits in ``headroom`` of the GPU memory; iterations
    for ``plan_minutes`` of training, minus ``overhead`` for validation and checkpoints."""
    ok = [r for r in rows if not r.get("oom") and r.get("peak_mem_gb", 0) <= headroom * total_mem_gb]
    if not ok:
        return {"error": "no setting fits in memory"}
    crop = max(r["crop"] for r in ok)
    best = max((r for r in ok if r["crop"] == crop), key=lambda r: r["batch"])
    iters = int(best["it_per_s"] * plan_minutes * 60 * (1 - overhead) / 100) * 100
    return {
        "crop": crop,
        "batch": best["batch"],
        "it_per_s": best["it_per_s"],
        "peak_mem_gb": best["peak_mem_gb"],
        "iters_for_plan": iters,
        "plan_minutes": plan_minutes,
        "rule": f"largest batch at crop {crop} using at most {headroom:.0%} of {total_mem_gb:.1f} GB",
    }


def calibrate(
    model_name: str,
    sources: list[PairsDataset],
    crops: list[int],
    batches: list[int],
    iters: int = 30,
    warmup: int = 8,
    plan_minutes: float = 300,
    workers: int = 2,
    device: torch.device | None = None,
) -> dict[str, Any]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = [
        measure(model_name, sources, c, b, iters, warmup, device, workers) for c in crops for b in batches
    ]
    total = torch.cuda.get_device_properties(device).total_memory / 1e9 if device.type == "cuda" else 0.0
    start = loss_at_start(sources, max(crops))
    return {
        "model": model_name,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "gpu_mem_gb": round(total, 2),
        "measurements": rows,
        "recommended": recommend(rows, total, plan_minutes) if device.type == "cuda" else None,
        "loss_at_start": {k: round(v, 5) for k, v in start.items()},
        "fft_share_at_0.05": round(0.05 * start["fft"] / (start["charbonnier"] + 0.05 * start["fft"]), 3),
        "fft_weight_for_20pct": round(suggest_fft_weight(start["charbonnier"], start["fft"], 0.2), 4),
    }
