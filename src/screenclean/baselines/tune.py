"""Colab task ``tune_notch``: choose classical-baseline settings on validation images.

Validation images come from the held-out list (``data/splits/uhdm_val.txt``), fetched at
full resolution from the dataset zip. The test set is never used here. Each setting in the
grid is scored by mean PSNR. The best settings are written to
``<drive root>/params/baselines.yaml``, where the evaluation job picks them up.
"""

from __future__ import annotations

import csv
import itertools
import json
import logging
import multiprocessing as mp
from collections import defaultdict
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from screenclean.baselines.classical import chroma_lowpass, fft_notch_grid, fft_notch_local
from screenclean.data.download import open_zip_source
from screenclean.data.uhdm import decode_pair, discover_pairs, sample_keys
from screenclean.eval.metrics import psnr
from screenclean.jobs import JobContext, TaskResult, register
from screenclean.utils.drive import atomic_write_text, copy_atomic
from screenclean.utils.image import to_float01

log = logging.getLogger(__name__)

PARAMS_FILE = "params/baselines.yaml"
FIELDS = ["key", "method", "params", "psnr"]


def expand_grid(grid: dict[str, list]) -> list[dict[str, Any]]:
    """``{"a": [1, 2], "b": ["x"]}`` -> ``[{"a": 1, "b": "x"}, {"a": 2, "b": "x"}]``."""
    names = sorted(grid)
    return [dict(zip(names, values, strict=True)) for values in itertools.product(*(grid[n] for n in names))]


def _pkey(params: dict[str, Any]) -> str:
    return json.dumps(params, sort_keys=True)


def score_image(job: tuple[str, bytes, bytes, dict[str, list[dict]]]) -> list[dict[str, Any]]:
    """PSNR of every grid setting on one image pair (runs in a worker process)."""
    key, mb, gb, grids = job
    moire, gt = decode_pair(mb, gb)
    moire, gt = to_float01(moire), to_float01(gt)
    rows = [{"key": key, "method": "identity", "params": "{}", "psnr": psnr(moire, gt)}]
    for params in grids.get("chroma_lowpass", []):
        rows.append(
            {
                "key": key,
                "method": "chroma_lowpass",
                "params": _pkey(params),
                "psnr": psnr(chroma_lowpass(moire, **params), gt),
            }
        )
    for params, out in fft_notch_grid(moire, grids.get("fft_notch", [])):
        rows.append({"key": key, "method": "fft_notch", "params": _pkey(params), "psnr": psnr(out, gt)})
    for params in grids.get("fft_notch_local", []):
        rows.append(
            {
                "key": key,
                "method": "fft_notch_local",
                "params": _pkey(params),
                "psnr": psnr(fft_notch_local(moire, **params), gt),
            }
        )
    return rows


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return [{**r, "psnr": float(r["psnr"])} for r in csv.DictReader(f)]


def _append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        if new:
            w.writeheader()
        w.writerows(rows)


def aggregate(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Per method: settings sorted by mean PSNR (best first), with the number of images."""
    scores: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        scores[(r["method"], r["params"])].append(r["psnr"])
    table: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (method, params), vals in scores.items():
        table[method].append(
            {"params": json.loads(params), "psnr_mean": float(np.mean(vals)), "n": len(vals)}
        )
    for method in table:
        table[method].sort(key=lambda e: -e["psnr_mean"])
    return dict(table)


@register("tune_notch")
def tune_notch_task(ctx: JobContext) -> TaskResult:
    cfg = ctx.config
    data_cfg = yaml.safe_load((ctx.repo_root / cfg["data_config"]).read_text(encoding="utf-8"))
    val_keys = (ctx.repo_root / cfg.get("val_list", "data/splits/uhdm_val.txt")).read_text().split()
    keys = sample_keys(val_keys, int(cfg.get("n_images", 20)), int(cfg.get("seed", 0)))
    grids = {m: expand_grid(g) for m, g in cfg["grids"].items()}
    rows_path = ctx.work_dir / "tune_rows.csv"
    done = {r["key"] for r in _read_rows(rows_path)}
    todo = [k for k in keys if k not in done]
    log.info(
        "tuning on %d validation images (%d left); grid sizes %s",
        len(keys),
        len(todo),
        {m: len(g) for m, g in grids.items()},
    )

    state, message = "done", ""
    if todo:
        src = open_zip_source(data_cfg["source"], ctx.repo_root)
        pairs = {p.key: p for p in discover_pairs(src.names, data_cfg["source"]["train_prefix"])[0]}

        def jobs() -> Iterator[tuple[str, bytes, bytes, dict]]:
            for k in todo:
                yield k, src.read(pairs[k].moire), src.read(pairs[k].gt), grids

        workers = int(cfg.get("workers", 2))
        pool = mp.get_context("spawn").Pool(workers) if workers > 1 else None
        try:
            results = pool.imap(score_image, jobs()) if pool else map(score_image, jobs())
            for i, rows in enumerate(results, start=len(done) + 1):
                _append_rows(rows_path, rows)
                ctx.progress(f"image {i}/{len(keys)} scored")
                if i < len(keys) and ctx.time_up(float(cfg.get("stop_margin_min", 10)) * 60):
                    state, message = (
                        "partial",
                        "Stopped to stay within the time budget; the next run continues.",
                    )
                    break
        finally:
            if pool:
                pool.terminate()

    rows = _read_rows(rows_path)
    table = aggregate(rows)
    summary: dict[str, Any] = {"n_images": len({r["key"] for r in rows}), "val_keys": keys}
    if state == "done":
        best = {m: {"params": e[0]["params"], "psnr_mean": e[0]["psnr_mean"]} for m, e in table.items()}
        summary["best"] = best
        summary["identity_psnr"] = table["identity"][0]["psnr_mean"]
        params = {
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "job": ctx.spec.id,
            "n_images": summary["n_images"],
            "selection": "mean PSNR on validation images",
            **{m: b["params"] for m, b in best.items() if m != "identity"},
        }
        atomic_write_text(ctx.layout.root / PARAMS_FILE, yaml.safe_dump(params, sort_keys=False))
        atomic_write_text(ctx.results_dir / "baselines_params.yaml", yaml.safe_dump(params, sort_keys=False))
        log.info("best settings: %s", best)
    with open(ctx.results_dir / "tune_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["method", "params", "psnr_mean", "n"])
        for m, entries in sorted(table.items()):
            for e in entries:
                w.writerow([m, _pkey(e["params"]), f"{e['psnr_mean']:.4f}", e["n"]])
    if rows_path.exists():
        copy_atomic(rows_path, ctx.results_dir / "tune_rows.csv")
    return TaskResult(state, summary, message)
