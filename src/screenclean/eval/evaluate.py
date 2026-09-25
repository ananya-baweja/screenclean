"""Colab task ``eval``: run methods on a dataset split and score every image.

For each (image, method): PSNR, SSIM (and LPIPS if enabled), plus the time the method took.
Rows go to ``per_image.csv`` on Drive after every image, so an interrupted run continues
where it stopped. The results zip gets that CSV, a ``summary.json`` with the means per
method, and side-by-side sample crops (input | output | ground truth) of a few images.

Methods come from the config, e.g.::

    methods:
      - {name: identity}
      - {name: fft_notch, params_from: params/baselines.yaml}   # tuned settings from job 0003
      - {name: esdnet_ref, label: "ESDNet (reference)", ...}   # GPU; see baselines/esdnet_ref.py
"""

from __future__ import annotations

import csv
import logging
import multiprocessing as mp
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from screenclean.baselines import BASELINES, get_baseline
from screenclean.data.download import DownloadError
from screenclean.data.shards import group_samples, index_tar, read_member, source_key
from screenclean.data.uhdm import decode_pair
from screenclean.eval.metrics import lpips_distance, psnr, ssim
from screenclean.jobs import JobContext, TaskResult, register
from screenclean.utils.drive import copy_atomic, read_json, stage_shards
from screenclean.utils.image import to_float01
from screenclean.utils.io import write_image

log = logging.getLogger(__name__)

FIELDS = ["key", "method", "psnr", "ssim", "lpips", "seconds", "height", "width"]
SAMPLE_CROP = 512

Item = tuple[
    str, str, tuple[int, int], tuple[int, int]
]  # (shard path, sample id, moire offset/size, gt offset/size)


# --------------------------------------------------------------------------- methods


def resolve_methods(specs: list[dict[str, Any]], drive_root: Path) -> list[dict[str, Any]]:
    """Fill in each method's label and parameters (tuned values from Drive when ``params_from`` is set)."""
    out = []
    for spec in specs:
        name = spec["name"]
        params, source = dict(spec.get("params", {})), "config"
        if "params_from" in spec:
            path = drive_root / spec["params_from"]
            tuned = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else None
            if tuned and name in tuned:
                params, source = {**params, **tuned[name]}, f"{spec['params_from']} (job {tuned.get('job')})"
            else:
                source = "defaults (no tuned settings found)"
                log.warning("%s: no tuned settings in %s; using defaults", name, spec["params_from"])
        if name not in BASELINES and name != "esdnet_ref":
            raise ValueError(f"unknown method {name!r}")
        out.append({**spec, "label": spec.get("label", name), "params": params, "params_source": source})
    return out


def build_fn(method: dict[str, Any], drive_root: Path) -> Callable[[np.ndarray], np.ndarray]:
    if method["name"] == "esdnet_ref":
        from screenclean.baselines.esdnet_ref import build_esdnet

        return build_esdnet(method, drive_root)
    fn = get_baseline(method["name"])
    params = method["params"]
    return lambda img: fn(img, **params)


# --------------------------------------------------------------------------- per image


def _center_crop(img: np.ndarray, size: int = SAMPLE_CROP) -> np.ndarray:
    h, w = img.shape[:2]
    y, x = max(0, (h - size) // 2), max(0, (w - size) // 2)
    return img[y : y + size, x : x + size]


class _Worker:
    """Scores one image with all methods. Created once per process (methods can be expensive to build)."""

    def __init__(self, methods: list[dict[str, Any]], drive_root: Path, use_lpips: bool):
        self.methods = methods
        self.fns = {m["label"]: build_fn(m, drive_root) for m in methods}
        self.use_lpips = use_lpips

    def __call__(
        self, job: tuple[Item, set[str], bool]
    ) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
        (shard, sid, mo, go), pending, want_sample = job
        moire, gt = decode_pair(read_member(shard, *mo), read_member(shard, *go))
        moire, gt = to_float01(moire), to_float01(gt)
        h, w = gt.shape[:2]
        rows, samples = [], {}
        for label in pending:
            t0 = time.perf_counter()
            out = self.fns[label](moire)
            secs = time.perf_counter() - t0
            rows.append(
                {
                    "key": source_key(sid),
                    "method": label,
                    "psnr": psnr(out, gt),
                    "ssim": ssim(out, gt),
                    "lpips": lpips_distance(out, gt) if self.use_lpips else "",
                    "seconds": secs,
                    "height": h,
                    "width": w,
                }
            )
            if want_sample:
                samples[label] = np.concatenate(
                    [_center_crop(moire), _center_crop(out), _center_crop(gt)], axis=1
                )
        return rows, samples


_WORKER: _Worker | None = None


def _init_worker(methods, drive_root, use_lpips) -> None:
    global _WORKER
    _WORKER = _Worker(methods, drive_root, use_lpips)


def _run_in_worker(job):
    return _WORKER(job)


# --------------------------------------------------------------------------- task


def list_items(shards: list[Path]) -> list[Item]:
    items = []
    for shard in shards:
        index = index_tar(shard)
        for sid, m in group_samples(index).items():
            items.append((str(shard), sid, index[m["moire"]], index[m["gt"]]))
    return sorted(items, key=lambda it: it[1])


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        if new:
            w.writeheader()
        w.writerows(rows)


def summarize(rows: list[dict[str, Any]], methods: list[dict[str, Any]]) -> dict[str, Any]:
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_method[r["method"]].append(r)
    input_labels = [m["label"] for m in methods if m["name"] == "identity"]
    base_rows = by_method.get(input_labels[0], []) if input_labels else []
    base = {r["key"]: float(r["psnr"]) for r in base_rows}
    out = {}
    for m in methods:
        rs = by_method.get(m["label"], [])
        if not rs:
            continue
        entry = {
            "name": m["name"],
            "n": len(rs),
            "psnr": float(np.mean([float(r["psnr"]) for r in rs])),
            "ssim": float(np.mean([float(r["ssim"]) for r in rs])),
            "seconds": float(np.mean([float(r["seconds"]) for r in rs])),
            "params": m["params"],
            "params_source": m["params_source"],
            "reference": m["name"] == "esdnet_ref",
        }
        lp = [float(r["lpips"]) for r in rs if r.get("lpips") not in ("", None)]
        if lp:
            entry["lpips"] = float(np.mean(lp))
        paired = [float(r["psnr"]) - base[r["key"]] for r in rs if r["key"] in base]
        if paired and m["name"] != "identity":
            entry["psnr_gain_vs_input"] = float(np.mean(paired))
        out[m["label"]] = entry
    return out


@register("eval")
def eval_task(ctx: JobContext) -> TaskResult:
    cfg = ctx.config
    split = cfg["split"]
    methods = resolve_methods(cfg["methods"], ctx.layout.root)
    labels = [m["label"] for m in methods]
    use_lpips = bool(cfg.get("lpips", False))
    n_samples = int(cfg.get("sample_images", 4))
    gpu = any(m["name"] == "esdnet_ref" for m in methods)
    workers = 1 if gpu else int(cfg.get("workers", 2))

    # Build in-process methods first: a missing model file should stop the job before any data is copied.
    worker = None
    if workers == 1:
        try:
            worker = _Worker(methods, ctx.layout.root, use_lpips)
        except DownloadError as e:
            if not e.retry_later:
                raise
            return TaskResult("partial", {"split": split, "stopped": str(e)}, f"Download stopped: {e}")

    local = Path(cfg.get("local_dir", "/content/eval_data")) / Path(split).name
    shards = stage_shards(ctx.layout.root / split, local)
    items = list_items(shards)[: cfg.get("limit")]

    rows_path = ctx.work_dir / "per_image.csv"
    done: dict[str, set[str]] = defaultdict(set)
    for r in read_rows(rows_path):
        done[r["key"]].add(r["method"])
    # Sample images re-run every method (their crops aren't kept between runs); already-scored rows
    # are not written twice.
    jobs = []
    for i, it in enumerate(items):
        want_sample = i < n_samples
        run = labels if want_sample else [lb for lb in labels if lb not in done[source_key(it[1])]]
        if run:
            jobs.append((it, run, want_sample))
    log.info(
        "%d images, methods %s; %d images still to score", len(items), labels, sum(1 for _, p, _ in jobs if p)
    )

    state, message, scored = "done", "", 0
    pool = None
    if worker is None:
        pool = mp.get_context("spawn").Pool(workers, _init_worker, (methods, ctx.layout.root, use_lpips))
        results = pool.imap(_run_in_worker, jobs)
    else:
        results = map(worker, jobs)
    try:
        for (it, _, _), (rows, samples) in zip(jobs, results, strict=False):
            key = source_key(it[1])
            new_rows = [r for r in rows if r["method"] not in done[key]]
            append_rows(rows_path, new_rows)
            for r in new_rows:
                done[key].add(r["method"])
            for label, strip in samples.items():
                safe = label.replace(" ", "_").replace("(", "").replace(")", "")
                write_image(ctx.results_dir / "samples" / f"{safe}__{it[1]}.jpg", strip, quality=85)
            scored += 1
            ctx.progress(f"image {scored}/{len(jobs)} ({key})")
            if scored < len(jobs) and ctx.time_up(float(cfg.get("stop_margin_min", 10)) * 60):
                state, message = "partial", "Stopped to stay within the time budget; the next run continues."
                break
    finally:
        if pool:
            pool.terminate()

    rows = read_rows(rows_path)
    summary: dict[str, Any] = {
        "split": split,
        "n_images": len(items),
        "methods": summarize(rows, methods),
        "complete": all(len(done[source_key(it[1])]) >= len(labels) for it in items),
    }
    if worker is not None:
        for m in methods:
            fn = worker.fns[m["label"]]
            if hasattr(fn, "modes"):
                summary["methods"].get(m["label"], {}).update(
                    {"inference_modes": dict(fn.modes), "parameters": fn.params, "fp16": fn.fp16}
                )
    manifest = read_json(ctx.layout.root / split / "manifest.json", default={}) or {}
    summary["dataset_params"] = manifest.get("params")
    if rows_path.exists():
        copy_atomic(rows_path, ctx.results_dir / "per_image.csv")
    return TaskResult(state, summary, message)
