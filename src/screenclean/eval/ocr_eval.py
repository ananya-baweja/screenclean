"""OCR benchmark (P9): how much each moiré-removal method helps text recognition.

The shared machinery: open the OCR engines (``eval/ocr.py``), OCR an image with each one, score
it against the ground-truth text (CER, WER, word F1; ``eval/text_metrics.py``), append the rows
to a CSV so an interrupted job resumes, and summarise with bootstrap confidence intervals
(``eval/stats.py``).

Colab task ``eval_synth_ocr``: every method on the synthetic text test crops (P4), whose ground
truth is the text of the lines fully inside each crop. It runs first in P9: it checks that the
OCR engines install and read correctly, and whether the text fine-tune also reads better than
the main model (the final-model check of P9.0). The real-photo benchmark is in ``eval/real_ocr.py``.
"""

from __future__ import annotations

import csv
import logging
import time
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from screenclean.eval.evaluate import build_fn, resolve_methods
from screenclean.eval.ocr import WorkerOCR, lines_text, open_engine
from screenclean.eval.stats import bootstrap_mean, paired_by_key
from screenclean.eval.text_metrics import ordered_text, quad_box, score
from screenclean.jobs import JobContext, TaskResult, register
from screenclean.utils.drive import copy_atomic, stage_shards
from screenclean.utils.image import to_float01
from screenclean.utils.io import write_image

log = logging.getLogger(__name__)

SCORE_FIELDS = ["cer", "wer", "word_f1", "ref_chars", "ocr_seconds", "method_seconds"]
TARGET = "target"  # pseudo-method: OCR of the clean ground-truth image (the ceiling)


# --------------------------------------------------------------------------- engines and methods


def open_engines(cfg: dict[str, Any], log_dir: Path) -> tuple[dict[str, WorkerOCR], dict[str, Any]]:
    """Start the configured engines (``engines: [paddle, tesseract]``); returns them and a record.

    An engine that fails (and whose fallbacks fail too) is left out and recorded; at least one
    must start.
    """
    engines, records = {}, {}
    for name in cfg.get("engines", ["tesseract"]):
        ocr, record = open_engine(
            name,
            packages_root=cfg.get("ocr_packages"),
            options=cfg.get("engine_options"),
            min_word_f1=float(cfg.get("self_check_min_word_f1", 0.9)),
            log_dir=log_dir,
        )
        records[name] = record
        if ocr is not None:
            engines[record["engine"]] = ocr
    if not engines:
        raise RuntimeError(f"no OCR engine could start: {records}")
    return engines, records


def build_methods(specs: list[dict[str, Any]], drive_root: Path) -> list[dict[str, Any]]:
    """Resolved method entries, each with ``fn`` (image -> image); ``target`` returns the clean image."""
    methods = []
    for spec in specs:
        if spec["name"] == TARGET:
            methods.append({**spec, "label": spec.get("label", "Clean target (upper bound)"), "fn": None})
            continue
        (m,) = resolve_methods([spec], drive_root)
        methods.append({**m, "fn": build_fn(m, drive_root)})
    return methods


# --------------------------------------------------------------------------- rows


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def append_rows(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    if not rows:
        return
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(rows)


class OcrScorer:
    """OCRs images with every engine and scores them; skips (key, method, engine) rows already done."""

    def __init__(self, engines: dict[str, WorkerOCR], tmp_dir: Path, done: set[tuple[str, str, str]]):
        self.engines, self.tmp_dir, self.done = engines, tmp_dir, done
        tmp_dir.mkdir(parents=True, exist_ok=True)

    def pending(self, key: str, method: str) -> list[str]:
        return [e for e in self.engines if (key, method, e) not in self.done]

    def score(self, img: np.ndarray, truth: str, row: dict[str, Any]) -> list[dict[str, Any]]:
        """One row per pending engine: ``row`` (needs ``key`` and ``method``) plus the scores."""
        engines = self.pending(row["key"], row["method"])
        if not engines:
            return []
        path = self.tmp_dir / "page.png"
        write_image(path, img)
        rows = []
        for name in engines:
            t0 = time.perf_counter()
            text = lines_text(self.engines[name].read(path))
            rows.append(
                {**row, "engine": name, **score(truth, text), "ocr_seconds": time.perf_counter() - t0}
            )
            self.done.add((row["key"], row["method"], name))
        return rows


# --------------------------------------------------------------------------- summary


def summarize(
    rows: Iterable[dict[str, Any]],
    baseline: str | dict[str, str] | None,
    pairs: Iterable[tuple[str, str]] = (),
    groups: Iterable[str] = (),
) -> dict[str, Any]:
    """Means with 95% bootstrap intervals per (engine, method), paired differences vs ``baseline``.

    ``baseline`` is a method, or ``{method prefix: method}`` when rows of different kinds have
    their own baseline (``{"oracle:": "oracle:Input", "product:": "product:none"}``). ``pairs``
    adds paired comparisons (method a minus method b) per engine. ``groups`` (row fields) add
    mean CER per value of each field, for breakdowns by screen, phone or template.
    """

    def baseline_of(method: str) -> str | None:
        if isinstance(baseline, dict):
            return next((b for prefix, b in baseline.items() if method.startswith(prefix)), None)
        return baseline

    table: dict[str, dict[str, dict[str, dict[str, float]]]] = defaultdict(lambda: defaultdict(dict))
    groups = tuple(groups)
    by_group: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for r in rows:
        vals = {k: float(r[k]) for k in ("cer", "wer", "word_f1")}
        table[r["engine"]][r["method"]][r["key"]] = vals
        for g in groups:
            by_group[(r["engine"], g, r["method"], str(r.get(g, "?")))].append(vals["cer"])

    out: dict[str, Any] = {}
    for engine, methods in table.items():
        per: dict[str, Any] = {}
        for method, items in methods.items():
            base_name = baseline_of(method)
            base = methods.get(base_name, {}) if base_name else {}
            entry: dict[str, Any] = {"n": len(items)}
            if base and method != base_name:
                entry["baseline"] = base_name
            for metric in ("cer", "wer", "word_f1"):
                values = {k: v[metric] for k, v in items.items()}
                entry[metric] = bootstrap_mean(list(values.values())).as_dict()
                if base and method != base_name and set(values) & set(base):
                    entry[f"{metric}_vs_input"] = paired_by_key(
                        values, {k: v[metric] for k, v in base.items()}
                    ).as_dict()
            per[method] = entry
        out[engine] = {"methods": per}
        comparisons = []
        for a, b in pairs:
            if a in methods and b in methods and set(methods[a]) & set(methods[b]):
                ci = {
                    m: paired_by_key(
                        {k: v[m] for k, v in methods[a].items()}, {k: v[m] for k, v in methods[b].items()}
                    )
                    for m in ("cer", "word_f1")
                }
                comparisons.append(
                    {"a": a, "b": b, **{f"{m}_a_minus_b": ci[m].as_dict() for m in ci}}
                    | {"a_reads_worse": ci["cer"].lo > 0}
                )
        if comparisons:
            out[engine]["comparisons"] = comparisons
        if groups:
            cer_by: dict[str, dict[str, dict[str, Any]]] = {g: defaultdict(dict) for g in groups}
            for (e, g, m, value), v in sorted(by_group.items()):
                if e == engine:
                    cer_by[g][m][value] = {"mean": float(np.mean(v)), "n": len(v)}
            out[engine]["cer_by"] = {g: dict(d) for g, d in cer_by.items()}
    return out


# --------------------------------------------------------------------------- synthetic text task


def synth_items(shards: list[Path]) -> list[tuple[str, str, dict[str, tuple[int, int]]]]:
    """(shard, sample id, tar index) for every complete sample, sorted by id."""
    from screenclean.data.shards import group_samples, index_tar

    items = []
    for shard in shards:
        index = index_tar(shard)
        for sid in group_samples(index):
            if f"{sid}.json" in index:
                items.append((str(shard), sid, index))
    return sorted(items, key=lambda it: it[1])


def synth_truth(record: dict[str, Any]) -> str:
    """Ground-truth text of a synthetic crop: its visible lines in reading order."""
    return ordered_text((ln["text"], quad_box(ln["quad"])) for ln in record.get("lines", []))


def _run_method(method: dict[str, Any], moire: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    out = gt if method["fn"] is None else method["fn"](moire)
    return np.clip(to_float01(out), 0.0, 1.0), time.perf_counter() - t0


@register("eval_synth_ocr")
def eval_synth_ocr_task(ctx: JobContext) -> TaskResult:
    from screenclean.data.shards import read_member, read_meta
    from screenclean.data.uhdm import decode_pair

    cfg = ctx.config
    local = Path(cfg.get("local_dir", "/content/ocr_eval"))
    methods = build_methods(cfg["methods"], ctx.layout.root)  # a missing model stops the job early
    engines, engine_records = open_engines(cfg, ctx.work_dir)
    try:
        shards = stage_shards(ctx.layout.root / cfg["split"], local / "shards")
        items = synth_items(shards)[: cfg.get("limit")]
        rows_path = ctx.work_dir / "per_image.csv"
        fields = ["key", "method", "engine", *SCORE_FIELDS]
        done = {(r["key"], r["method"], r["engine"]) for r in read_rows(rows_path)}
        scorer = OcrScorer(engines, local / "tmp", done)
        state, message, no_text = "done", "", set()
        for i, (shard, sid, index) in enumerate(items):
            if not any(scorer.pending(sid, m["label"]) for m in methods):
                continue
            record = read_meta(shard, index, sid) or {}
            truth = synth_truth(record)
            if not truth:
                no_text.add(sid)
                continue
            moire, gt = decode_pair(
                read_member(shard, *index[f"{sid}.moire.jpg"]), read_member(shard, *index[f"{sid}.gt.jpg"])
            )
            new_rows = []
            for m in methods:
                if scorer.pending(sid, m["label"]):
                    out, secs = _run_method(m, to_float01(moire), to_float01(gt))
                    new_rows += scorer.score(
                        out, truth, {"key": sid, "method": m["label"], "method_seconds": secs}
                    )
            append_rows(rows_path, new_rows, fields)
            ctx.progress(f"crop {i + 1}/{len(items)} ({sid})")
            if i + 1 < len(items) and ctx.time_up(float(cfg.get("stop_margin_min", 10)) * 60):
                state, message = "partial", "Stopped to stay within the time budget; the next run continues."
                break
    finally:
        for ocr in engines.values():
            ocr.close()

    rows = read_rows(rows_path)
    baseline = next((m["label"] for m in methods if m["name"] == "identity"), None)
    summary = {
        "split": cfg["split"],
        "n_items": len(items),
        "skipped_without_text": len(no_text),
        "engines": engine_records,
        "methods": [m["label"] for m in methods],
        "complete": state == "done"
        and all(
            not scorer.pending(sid, m["label"]) for _, sid, _ in items if sid not in no_text for m in methods
        ),
        "ocr": summarize(rows, baseline, [tuple(p) for p in cfg.get("compare", [])]),
    }
    if rows_path.exists():
        copy_atomic(rows_path, ctx.results_dir / "per_image.csv")
    return TaskResult(state, summary, message)
