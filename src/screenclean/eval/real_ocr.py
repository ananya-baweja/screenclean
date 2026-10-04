"""The real-photo benchmark (P9): phone photos of capture-kit pages on real screens.

Colab task ``ingest_real``: find each photo's page from its corner markers and fit the
homography H (photo -> page) (``eval/real_captures.py``). Writes ``manifest.csv`` and
``captures.json`` to ``real_captures/processed/`` on Drive; the results zip gets only the
manifest and counts, never the photos or images made from them.

Colab task ``eval_real_ocr``, two measurements per photo:

1. **Methods, same geometry** (plan C4): every method runs on the original photo (the page's
   area plus a margin, at full resolution); every output is straightened with the *same* H from
   the markers, evened out like a scanned document (the scan pipeline's clean-up, unless
   ``oracle_cleanup: false``), the markers are painted over, and each OCR engine reads the page.
   Differences between methods are then due to the method alone.
2. **The product, end to end** (plan C11): ``scan()`` finds the screen itself, cleans with each
   cleaner (none / classical / the final model), straightens and evens out the page; each engine
   reads it. Screen detection is scored against the page corners known from the markers: corner
   error in pixels and the share of photos whose corners are all within 3% of the photo diagonal
   (the P5 success rule).

Ground truth is the page's text from ``capture_kit/pages.json``. Rows are appended to CSVs, so an
interrupted job continues where it stopped.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from screenclean.eval import real_captures
from screenclean.eval.ocr_eval import (
    SCORE_FIELDS,
    OcrScorer,
    append_rows,
    build_methods,
    open_engines,
    read_rows,
    summarize,
)
from screenclean.eval.text_metrics import ordered_text
from screenclean.jobs import JobContext, TaskResult, register
from screenclean.product.rectify import clean_up, page_homography, rectify
from screenclean.render import aruco
from screenclean.utils.drive import atomic_write_json, copy_atomic
from screenclean.utils.io import read_image

log = logging.getLogger(__name__)

PAGE_SIZE = (1920, 1080)
SUCCESS = 0.03  # detection success: every corner within 3% of the photo diagonal (as in P5)
ROW_FIELDS = [
    "key",
    "kind",
    "method",
    "engine",
    "screen",
    "phone",
    "page_id",
    "template",
    "size",
    *SCORE_FIELDS,
]
DETECT_FIELDS = [
    "key",
    "screen",
    "phone",
    "detected",
    "mean_err_px",
    "max_err_px",
    "max_err_pct_diag",
    "success",
]
TIME_FIELDS = ["key", "cleaner", "detect", "clean", "rectify", "cleanup", "total"]


# --------------------------------------------------------------------------- ground truth and geometry


def load_pages(kit_dir: str | Path) -> dict[int, dict[str, Any]]:
    """Capture-kit pages by id, each with ``truth`` (text in reading order) and ``size`` (body size)."""
    meta = json.loads((Path(kit_dir) / "pages.json").read_text(encoding="utf-8"))
    pages = {}
    for p in meta["pages"]:
        sizes = [ln["size"] for ln in p["lines"]]
        body = int(np.median(sizes)) if sizes else 0
        pages[p["page_id"]] = {
            "truth": ordered_text((ln["text"], tuple(ln["box"])) for ln in p["lines"]),
            "template": p["template"],
            "size": "<16 px" if body < 16 else "16-24 px" if body <= 24 else ">24 px",
        }
    return pages


def marker_zones(page_size: tuple[int, int] = PAGE_SIZE) -> list[np.ndarray]:
    """Each corner marker with its white margin, as a 4-point polygon in page pixels."""
    zones = []
    for corner in range(4):
        x, y = aruco.marker_origin(corner, page_size)
        q, s = aruco.QUIET_PX, aruco.MARKER_PX
        x0, y0, x1, y1 = x - q - 0.5, y - q - 0.5, x + s + q - 0.5, y + s + q - 0.5
        zones.append(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32))
    return zones


def paint_over(page: np.ndarray, zones: list[np.ndarray]) -> np.ndarray:
    """Fill the marker zones with the page's median colour (markers are not text)."""
    out = page.copy()
    fill = tuple(float(v) for v in np.median(page.reshape(-1, page.shape[-1]), axis=0))
    for z in zones:
        cv2.fillPoly(out, [np.round(z).astype(np.int32)], fill)
    return out


def map_points(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(np.asarray(pts, np.float32)[None], H)[0]


def page_crop(corners: np.ndarray, photo_wh: tuple[int, int], margin: int) -> tuple[int, int, int, int]:
    """Bounding box (x0, y0, x1, y1) of the page in the photo, plus a margin, inside the photo."""
    w, h = photo_wh
    x0, y0 = np.floor(corners.min(axis=0)).astype(int) - margin
    x1, y1 = np.ceil(corners.max(axis=0)).astype(int) + margin + 1
    return max(int(x0), 0), max(int(y0), 0), min(int(x1), w), min(int(y1), h)


def detection_scores(found: np.ndarray, detected: bool, truth: np.ndarray, photo_wh: tuple[int, int]) -> dict:
    """Corner errors of a detected screen against the true page corners."""
    diag = float(np.hypot(*photo_wh))
    err = np.linalg.norm(np.asarray(found, np.float64) - np.asarray(truth, np.float64), axis=1)
    return {
        "detected": bool(detected),
        "mean_err_px": float(err.mean()),
        "max_err_px": float(err.max()),
        "max_err_pct_diag": float(100 * err.max() / diag),
        "success": bool(detected and err.max() < SUCCESS * diag),
    }


# --------------------------------------------------------------------------- ingest task


@register("ingest_real")
def ingest_real_task(ctx: JobContext) -> TaskResult:
    cfg = ctx.config
    raw = ctx.layout.root / cfg.get("raw_dir", "real_captures/raw")
    out = ctx.layout.root / cfg.get("out_dir", "real_captures/processed")
    ctx.progress(f"reading photos in {raw}")
    results = real_captures.ingest(raw, out)
    copy_atomic(out / "manifest.csv", ctx.results_dir / "manifest.csv")
    ok = [r for r in results if r.ok]
    errs = [r.reproj_err_px for r in ok if r.reproj_err_px is not None]
    summary = {
        "n_photos": len(results),
        "n_ok": len(ok),
        "by_screen_phone": dict(Counter(f"{r.screen} / {r.phone}" for r in results)),
        "ok_by_screen_phone": dict(Counter(f"{r.screen} / {r.phone}" for r in ok)),
        "pages_covered": len({r.page_id for r in ok}),
        "reproj_err_px": {"median": float(np.median(errs)), "p90": float(np.percentile(errs, 90))}
        if errs
        else None,
        "not_ok": [r.file for r in results if not r.ok],
    }
    if not ok:
        raise RuntimeError(f"none of the {len(results)} photos in {raw} could be matched to a page")
    return TaskResult("done", summary)


# --------------------------------------------------------------------------- evaluation task


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@register("eval_real_ocr")
def eval_real_ocr_task(ctx: JobContext) -> TaskResult:
    from screenclean.product.pipeline import scan
    from screenclean.product.screen_detect import detect_screen

    cfg = ctx.config
    root = ctx.layout.root
    captures = json.loads((root / cfg.get("captures", "real_captures/processed/captures.json")).read_text())
    photos = [c for c in captures if c.get("ok")][: cfg.get("limit")]
    pages = load_pages(ctx.repo_root / cfg.get("kit_dir", "capture_kit"))
    raw = root / cfg.get("raw_dir", "real_captures/raw")
    local = Path(cfg.get("local_dir", "/content/real_ocr"))
    margin = int(cfg.get("crop_margin", 64))
    mode = cfg.get("product_mode", "document")
    # Without the clean-up, OCR on some photographed pages fails on lighting alone (a simulated slide:
    # CER 0.92 -> 0.09 with it), which would drown the differences between methods.
    oracle_cleanup = bool(cfg.get("oracle_cleanup", True))

    methods = build_methods(cfg["methods"], root)
    by_label = {m["label"]: m for m in methods}
    cleaners = []
    for c in cfg.get("product_cleaners", []):  # {label, cleaner: none|fft_notch_local} or {label, method}
        fn = by_label[c["method"]]["fn"] if "method" in c else c.get("cleaner", "none")
        cleaners.append((c["label"], fn))
    engines, engine_records = open_engines(cfg, ctx.work_dir)

    rows_path, det_path, time_path = (
        ctx.work_dir / n for n in ("per_photo.csv", "detection.csv", "timing.csv")
    )
    done = {(r["key"], r["method"], r["engine"]) for r in read_rows(rows_path)}
    detected_keys = {r["key"] for r in _read_csv(det_path)}
    timed = {(r["key"], r["cleaner"]) for r in _read_csv(time_path)}
    scorer = OcrScorer(engines, local / "tmp", done)
    zones = marker_zones()
    state, message = "done", ""
    try:
        for i, cap in enumerate(photos):
            key = cap["file"]
            todo_methods = [m for m in methods if scorer.pending(key, f"oracle:{m['label']}")]
            todo_cleaners = [
                c for c in cleaners if scorer.pending(key, f"product:{c[0]}") or (key, c[0]) not in timed
            ]
            if not todo_methods and not todo_cleaners and key in detected_keys:
                continue
            page = pages[cap["page_id"]]
            info = {"screen": cap["screen"], "phone": cap["phone"], "page_id": cap["page_id"],
                    "template": page["template"], "size": page["size"]}  # fmt: skip
            photo = read_image(raw / key)
            h, w = photo.shape[:2]
            truth_corners = np.asarray(cap["page_corners_px"], np.float32)
            new_rows: list[dict[str, Any]] = []

            # 1. every method on the page's area of the original photo, straightened with the same H
            x0, y0, x1, y1 = page_crop(truth_corners, (w, h), margin)
            crop, corners_in_crop = photo[y0:y1, x0:x1], truth_corners - np.array([x0, y0], np.float32)
            for m in todo_methods:
                t0 = time.perf_counter()
                out = crop if m["fn"] is None else np.clip(m["fn"](crop), 0.0, 1.0)
                secs = time.perf_counter() - t0
                rect, _ = rectify(out, corners_in_crop, size=PAGE_SIZE)
                rect = clean_up(rect) if oracle_cleanup else rect
                new_rows += scorer.score(
                    paint_over(rect, zones),
                    page["truth"],
                    {
                        "key": key,
                        "kind": "oracle",
                        "method": f"oracle:{m['label']}",
                        "method_seconds": secs,
                        **info,
                    },
                )

            # 2. the product: find the screen, clean, straighten, even out; markers painted over
            det = None
            if key not in detected_keys or todo_cleaners:
                t0 = time.perf_counter()
                det = detect_screen(photo)
                detect_s = time.perf_counter() - t0
                found = (
                    det.corners if det.detected else np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
                )
                if key not in detected_keys:
                    scores = detection_scores(found, det.detected, truth_corners, (w, h))
                    append_rows(det_path, [{"key": key, **info, **scores}], DETECT_FIELDS)
                    detected_keys.add(key)
            for label, fn in todo_cleaners:
                res = scan(
                    photo, cleaner=fn, ocr=None, mode=mode, corners=det.corners if det.detected else None
                )
                H_prod = page_homography(res.corners, (res.page.shape[1], res.page.shape[0]))
                H_true_inv = np.linalg.inv(np.asarray(cap["H"], np.float64))
                prod_zones = [map_points(H_prod, map_points(H_true_inv, z)) for z in zones]
                new_rows += scorer.score(
                    paint_over(res.page, prod_zones),
                    page["truth"],
                    {
                        "key": key,
                        "kind": "product",
                        "method": f"product:{label}",
                        "method_seconds": res.timings.get("total", 0.0),
                        **info,
                    },  # fmt: skip
                )
                if (key, label) not in timed:
                    t = {k: round(v, 3) for k, v in res.timings.items()} | {"detect": round(detect_s, 3)}
                    append_rows(time_path, [{"key": key, "cleaner": label, **t}], TIME_FIELDS)
                    timed.add((key, label))
            append_rows(rows_path, new_rows, ROW_FIELDS)
            ctx.progress(f"photo {i + 1}/{len(photos)} ({key})")
            if i + 1 < len(photos) and ctx.time_up(float(cfg.get("stop_margin_min", 10)) * 60):
                state, message = "partial", "Stopped to stay within the time budget; the next run continues."
                break
    finally:
        for ocr in engines.values():
            ocr.close()

    rows = read_rows(rows_path)
    det_rows = _read_csv(det_path)
    baseline = {
        "oracle:": next((f"oracle:{m['label']}" for m in methods if m["name"] == "identity"), ""),
        "product:": next((f"product:{label}" for label, fn in cleaners if fn == "none"), ""),
    }
    pairs = [(a, b) for a, b in cfg.get("compare", [])]
    summary: dict[str, Any] = {
        "n_photos": len(photos),
        "engines": engine_records,
        "methods": [m["label"] for m in methods],
        "product_cleaners": [c[0] for c in cleaners],
        "ocr": summarize(rows, baseline, pairs, groups=("screen", "phone", "template", "size")),
        "complete": state == "done"
        and all(not scorer.pending(c["file"], f"oracle:{m['label']}") for c in photos for m in methods),
    }
    if det_rows:
        errs = np.array([float(r["max_err_pct_diag"]) for r in det_rows])
        summary["detection"] = {
            "n": len(det_rows),
            "detected_rate": float(np.mean([r["detected"] == "True" for r in det_rows])),
            "success_rate": float(np.mean([r["success"] == "True" for r in det_rows])),
            "success_rule": f"every corner within {SUCCESS:.0%} of the photo diagonal",
            "median_mean_err_px": float(np.median([float(r["mean_err_px"]) for r in det_rows])),
            "median_max_err_pct_diag": float(np.median(errs)),
        }
    time_rows = _read_csv(time_path)
    if time_rows:
        summary["seconds_per_photo"] = {
            label: {
                stage: float(
                    np.median([float(r[stage]) for r in time_rows if r["cleaner"] == label and r.get(stage)])
                )
                for stage in ("detect", "clean", "rectify", "cleanup", "total")
                if any(r["cleaner"] == label and r.get(stage) for r in time_rows)
            }
            for label in {r["cleaner"] for r in time_rows}
        }
    for p in (rows_path, det_path, time_path):
        if p.exists():
            copy_atomic(p, ctx.results_dir / p.name)
    atomic_write_json(
        ctx.results_dir / "photos.json", {"n": len(photos), "pages": sorted({c["page_id"] for c in photos})}
    )
    return TaskResult(state, summary, message)
