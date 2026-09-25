"""Synthetic text pairs: rendered pages photographed by the moiré simulator.

Colab tasks:

- ``gen_synth``: 6,000 / 300 / 300 crops (train / val / test) as shards on Drive under
  ``data/synth_v1/<split>/``. Each split renders pages from its own corpus lines only. Every
  sample stores the simulated photo (the simulator's own JPEG bytes), the clean target, and a
  JSON record with all simulator settings and the text lines fully inside the crop (text +
  corner positions), for OCR scoring later.
- ``sim_realism``: residual spectra of the synthetic set vs real UHDM crops, plus sample grids.

Both resume where they stopped.
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import shutil
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from screenclean.data.shards import (
    build_split,
    group_samples,
    index_tar,
    load_manifest,
    read_member,
    read_meta,
)
from screenclean.data.uhdm import key_seed
from screenclean.jobs import JobContext, TaskResult, register
from screenclean.render import corpus
from screenclean.render.fonts import available_fonts, load
from screenclean.render.pages import TEMPLATES, Renderer
from screenclean.simulate import realism
from screenclean.simulate.moire_sim import load_config, simulate
from screenclean.utils.drive import stage_shards
from screenclean.utils.io import decode_rgb, encode_jpeg, write_image

log = logging.getLogger(__name__)
GB = 1024**3
SPLITS = ("train", "val", "test")


class StopEarly(Exception):
    pass


# --------------------------------------------------------------------------- one sample


@lru_cache(maxsize=4)
def _renderer(split: str) -> Renderer:
    return Renderer(corpus.lines_for(split))


@lru_cache(maxsize=2)
def _page(split: str, page_idx: int, seed: int, lo: int, hi: int):
    """The page for a group of crops (cached: consecutive crops share a page)."""
    rng = np.random.default_rng(key_seed(f"{split}/page{page_idx}", f"synth{seed}"))
    template = TEMPLATES[page_idx % len(TEMPLATES)]
    base = int(rng.integers(lo, hi + 1))
    return _renderer(split).render(template, seed=int(rng.integers(2**31)), base_size=base), base


def _quad(H: np.ndarray, x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    pts = np.array([[x0, y0, 1], [x1, y0, 1], [x1, y1, 1], [x0, y1, 1]], float) @ H.T
    return pts[:, :2] / pts[:, 2:3]


@lru_cache(maxsize=1)
def _font_paths() -> dict[str, str]:
    return {f.name: f.path for f in available_fonts()}


def word_spans(line) -> list[tuple[str, float, float]]:
    """``(word, x_start, x_end)`` in page pixels for each word of a rendered line."""
    font = load(_font_paths()[line.font], line.size)
    left = line.box[0] - font.getbbox(line.text)[0]  # the x the text was drawn at
    spans, pos = [], 0
    for word in line.text.split():
        pos = line.text.index(word, pos)
        start = left + font.getlength(line.text[:pos])
        spans.append((word, start, start + font.getlength(word)))
        pos += len(word)
    return spans


def lines_in_crop(page_lines, H: np.ndarray, crop: int, scale: float) -> list[dict[str, Any]]:
    """For each page line, the run of whole words that is fully visible in the crop.

    Crops usually cut lines at the edges, so partly visible words are dropped. Each entry has the
    visible ``text``, its ``quad`` (corners in crop pixels), ``size_px`` (font size as
    photographed) and ``partial`` (True if words were dropped).
    """
    out = []
    for ln in page_lines:
        x0, y0, x1, y1 = ln.box
        whole = _quad(H, x0, y0, x1, y1)
        if (whole >= 0).all() and (whole <= crop - 1).all():
            out.append(
                {
                    "text": ln.text,
                    "quad": np.round(whole, 1).tolist(),
                    "size_px": round(ln.size * scale, 1),
                    "partial": False,
                }
            )
            continue
        visible = []
        for word, ws, we in word_spans(ln):
            q = _quad(H, ws, y0, we, y1)
            if (q >= 0).all() and (q <= crop - 1).all():
                visible.append((word, ws, we))
            elif visible:  # words are left to right: a run ends at the first cut word
                break
        if visible:
            q = _quad(H, visible[0][1], y0, visible[-1][2], y1)
            out.append(
                {
                    "text": " ".join(w for w, _, _ in visible),
                    "quad": np.round(q, 1).tolist(),
                    "size_px": round(ln.size * scale, 1),
                    "partial": True,
                }
            )
    return out


def make_sample(key: str, cfg: dict[str, Any], sim_cfg: dict[str, Any]) -> tuple[str, bytes, bytes, bytes]:
    """Build sample ``key`` ("<split>/<index>"): (sample id, photo JPEG, target JPEG, JSON record)."""
    split, idx_s = key.split("/")
    index = int(idx_s)
    seed = int(cfg.get("seed", 0))
    lo, hi = cfg.get("base_size", [12, 40])
    page_idx = index // int(cfg.get("crops_per_page", 4))
    page, base = _page(split, page_idx, seed, int(lo), int(hi))
    rng = np.random.default_rng(key_seed(key, f"synth{seed}"))
    if page.lines:  # centre the crop near a random line of text so crops have text
        x0, y0, x1, y1 = page.lines[int(rng.integers(len(page.lines)))].box
        centre = ((x0 + x1) / 2 + rng.uniform(-150, 150), (y0 + y1) / 2 + rng.uniform(-60, 60))
    else:
        centre = None
    moire, gt, meta = simulate(page.image, rng, sim_cfg, center_uv=centre)
    crop = int(sim_cfg["crop"])
    record = {
        "key": key,
        "template": page.template,
        "base_size": base,
        "page_seed": page.seed,
        "params": meta["params"],
        "pose": meta["pose"],
        "center_uv": meta["center_uv"],
        "H_page_to_crop": meta["H_page_to_crop"],
        "lines": lines_in_crop(page.lines, np.array(meta["H_page_to_crop"]), crop, meta["pose"]["scale"]),
    }
    sid = key.replace("/", "_")
    gt_bytes = encode_jpeg(gt, int(cfg.get("gt_jpeg_quality", 95)))
    return sid, meta["moire_jpeg"], gt_bytes, json.dumps(record).encode("utf-8")


_WORKER_ARGS: tuple | None = None


def _init(cfg, sim_cfg) -> None:
    global _WORKER_ARGS
    _WORKER_ARGS = (cfg, sim_cfg)


def _work(key: str):
    return make_sample(key, *_WORKER_ARGS)


# --------------------------------------------------------------------------- tasks


def _load_cfgs(ctx: JobContext) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = ctx.config
    sim_cfg = load_config(ctx.repo_root / cfg.get("sim_config", "configs/sim/default.yaml"))
    return cfg, sim_cfg


@register("gen_synth")
def gen_synth_task(ctx: JobContext) -> TaskResult:
    cfg, sim_cfg = _load_cfgs(ctx)
    out = ctx.layout.root / cfg.get("drive_out", "data/synth_v1")
    tmp = Path(cfg.get("local_dir", "/content/synth_work")) / "tmp_shards"
    out.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(out).free / GB
    if free < float(cfg.get("drive_gb", 2.5)):
        raise RuntimeError(
            f"Google Drive has {free:.1f} GB free; the synthetic set needs about {cfg['drive_gb']} GB"
        )
    counts = {s: int(cfg.get("counts", {}).get(s, 0)) for s in SPLITS}
    per_shard = int(cfg.get("samples_per_shard", 500))
    params = {k: cfg.get(k) for k in ("seed", "crops_per_page", "base_size", "gt_jpeg_quality")} | {
        "sim": sim_cfg
    }
    margin = float(cfg.get("stop_margin_min", 10)) * 60

    def check_time() -> None:
        if ctx.time_up(margin):
            raise StopEarly("stopped to stay within the job's time budget")

    workers = int(cfg.get("workers", 2))
    pool = mp.get_context("spawn").Pool(workers, _init, (cfg, sim_cfg)) if workers > 1 else None
    if pool is None:
        _init(cfg, sim_cfg)
    state, message, t0 = "done", "", time.monotonic()
    try:
        for split in SPLITS:
            keys = [f"{split}/{i:06d}" for i in range(counts[split])]
            groups = [keys[i : i + per_shard] for i in range(0, len(keys), per_shard)]
            build_split(
                out / split,
                split,
                groups,
                lambda g: pool.imap(_work, g, chunksize=4) if pool else map(_work, g),
                params,
                tmp,
                before_shard=check_time,
                after_shard=lambda i, n, e, s=split: ctx.progress(f"{s} shard {i + 1}/{n} done"),
            )
    except StopEarly as e:
        state, message = "partial", f"{e}. Finished shards are saved; the next run continues."
    finally:
        if pool:
            pool.terminate()

    summary: dict[str, Any] = {"splits": {}, "minutes": round((time.monotonic() - t0) / 60, 1)}
    for split in SPLITS:
        m = load_manifest(out / split)
        summary["splits"][split] = {
            "complete": bool(m.get("complete")),
            "samples": m.get("total_samples", 0),
            "gb": round(m.get("total_bytes", 0) / GB, 2),
        }
    first = load_manifest(out / "train").get("shards", [])
    if state == "done" and first:
        shard = out / "train" / first[0]["name"]
        index = index_tar(shard)
        groups_ = sorted(group_samples(index).items())
        pairs, records = [], []
        for sid, members in groups_:
            records.append(read_meta(shard, index, sid))
            if len(pairs) < 8:
                pairs.append(
                    (
                        decode_rgb(read_member(shard, *index[members["moire"]])),
                        decode_rgb(read_member(shard, *index[members["gt"]])),
                    )
                )
        write_image(ctx.results_dir / "sample_grid.jpg", realism.pair_grid(pairs, 256), quality=85)
        summary["first_shard_stats"] = describe(records)
    return TaskResult(state, summary, message)


def describe(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Distribution of key simulator settings and text content over sample records."""
    scale = np.array([r["pose"]["scale"] for r in records])
    n_lines = np.array([len(r["lines"]) for r in records])
    layouts: dict[str, int] = {}
    templates: dict[str, int] = {}
    for r in records:
        layouts[r["params"]["layout"]] = layouts.get(r["params"]["layout"], 0) + 1
        templates[r["template"]] = templates.get(r["template"], 0) + 1
    return {
        "n": len(records),
        "scale_quantiles": np.round(np.quantile(scale, [0.05, 0.25, 0.5, 0.75, 0.95]), 2).tolist(),
        "layouts": layouts,
        "templates": templates,
        "lines_per_crop_mean": round(float(n_lines.mean()), 2),
        "crops_without_full_lines": int((n_lines == 0).sum()),
    }


@register("sim_realism")
def sim_realism_task(ctx: JobContext) -> TaskResult:
    """Compare synthetic pairs with real UHDM validation crops (residual spectra + grids)."""
    cfg = ctx.config
    n = int(cfg.get("n_pairs", 64))
    local = Path(cfg.get("local_dir", "/content/realism_work"))
    groups: dict[str, list] = {}
    for label, split_dir in (("UHDM val (real)", cfg["real_split"]), ("synthetic", cfg["sim_split"])):
        shards = stage_shards(ctx.layout.root / split_dir, local / label.split()[0])
        pairs = []
        for shard in shards:
            index = index_tar(shard)
            for _, m in sorted(group_samples(index).items()):
                pairs.append(
                    (
                        decode_rgb(read_member(shard, *index[m["moire"]])),
                        decode_rgb(read_member(shard, *index[m["gt"]])),
                    )
                )
                if len(pairs) >= n:
                    break
            if len(pairs) >= n:
                break
        groups[label] = pairs
        write_image(
            ctx.results_dir / f"grid_{label.split()[0].lower()}.jpg",
            realism.pair_grid(pairs[:8], 256),
            quality=85,
        )
        ctx.progress(f"loaded {len(pairs)} {label} pairs")
    spectra = {label: realism.residual_spectra(p) for label, p in groups.items()}
    realism.spectra_figure(spectra, ctx.results_dir / "spectra.png", "Moire + noise residual spectra")
    summary = realism.summarize(spectra)
    return TaskResult("done", summary)
