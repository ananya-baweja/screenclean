"""Scan demo: capture-kit pages "photographed" by the simulator, scanned with and without moiré removal.

For each page: simulate a whole phone photo of it on a screen (``simulate/scene.py``), run the
scan pipeline with cleaner "none" and "fft_notch_local", and score the text against the page's
ground truth (character error rate and word F1). Writes a JSON table and one figure for the docs
(everything in it is synthetic, so nothing personal).

    python tools/make_scan_demo.py [--n 14] [--out results/checks/scan_demo.json]
                                   [--figure docs/figures/scan_demo.jpg] [--show-page 2]

Needs Tesseract 5 (set TESSERACT_CMD if it is not on PATH).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from screenclean.eval import tesseract  # noqa: E402
from screenclean.eval.text_metrics import cer, word_f1  # noqa: E402
from screenclean.product.pipeline import scan  # noqa: E402
from screenclean.render.pages import Line, reading_order  # noqa: E402
from screenclean.simulate.scene import make_scene  # noqa: E402
from screenclean.utils.io import read_image  # noqa: E402

CLEANERS = ("none", "fft_notch_local")
SCENE = {"area_frac": [0.45, 0.75]}  # people fill the frame when they photograph a slide to read it


def ground_truth(page: dict) -> str:
    lines = [Line(ln["text"], tuple(ln["box"]), ln["font"], ln["size"]) for ln in page["lines"]]
    return "\n".join(ln.text for ln in reading_order(lines))


def figure(path: Path, photo, results: dict, row: dict) -> None:
    fig = plt.figure(figsize=(14, 9.2), dpi=100)
    grid = fig.add_gridspec(2, 3, height_ratios=[1.15, 1])
    ax = fig.add_subplot(grid[0, :2])
    ax.imshow(photo)
    quad = np.vstack([results["fft_notch_local"].corners, results["fft_notch_local"].corners[:1]])
    ax.plot(quad[:, 0], quad[:, 1], color="#ff2da0", lw=1.5)
    ax.set_title("1. Simulated phone photo; the detected screen is outlined")
    ax2 = fig.add_subplot(grid[0, 2])
    ax2.imshow(results["fft_notch_local"].page)
    ax2.set_title("2. Found, cleaned and straightened")
    for i, name in enumerate(CLEANERS):
        page = results[name].page
        h, w = page.shape[:2]
        crop = page[int(0.16 * h) : int(0.42 * h), int(0.08 * w) : int(0.5 * w)]
        a = fig.add_subplot(grid[1, i])
        a.imshow(crop)
        label = "no moiré removal" if name == "none" else "classical moiré removal"
        a.set_title(f"3{'ab'[i]}. Detail, {label}\nOCR character error rate {row[name]['cer']:.1%}")
    t = fig.add_subplot(grid[1, 2])
    t.axis("off")
    text = results["fft_notch_local"].text.strip().splitlines()
    snippet = "\n".join(line[:52] + ("…" if len(line) > 52 else "") for line in text[:14])
    t.text(0, 1, snippet, va="top", ha="left", family="monospace", fontsize=8.5)
    t.set_title("4. Recognised text (start)")
    for a in fig.axes[:5]:
        a.set_xticks([])
        a.set_yticks([])
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100, pil_kwargs={"quality": 85})
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--kit", type=Path, default=Path("capture_kit"))
    ap.add_argument("--n", type=int, default=14, help="first n pages of the kit")
    ap.add_argument("--out", type=Path, default=Path("results/checks/scan_demo.json"))
    ap.add_argument("--figure", type=Path, default=Path("docs/figures/scan_demo.jpg"))
    ap.add_argument("--show-page", type=int, default=None, help="page id for the figure (default: best gain)")
    args = ap.parse_args()
    if not tesseract.available():
        raise SystemExit("Tesseract not found: install it or set TESSERACT_CMD")

    meta = json.loads((args.kit / "pages.json").read_text(encoding="utf-8"))
    rows, shown, best_gain = [], None, -np.inf
    for page in meta["pages"][: args.n]:
        pid = page["page_id"]
        img = read_image(args.kit / "pages" / f"page_{pid:03d}.png")
        scene = make_scene(img, np.random.default_rng(7000 + pid), out_size=(3200, 2400), ss=1, cfg=SCENE)
        gt = ground_truth(page)
        row = {"page_id": pid, "template": page["template"], "base_size": page["base_size"]}
        results = {}
        for name in CLEANERS:
            t = time.perf_counter()
            res = scan(scene.photo, cleaner=name)
            results[name] = res
            row[name] = {
                "detected": res.detected,
                "cer": round(cer(gt, res.text), 4),
                "word_f1": round(word_f1(gt, res.text), 4),
                "seconds": round(time.perf_counter() - t, 2),
            }
        rows.append(row)
        scores = [
            f"{n}: CER {row[n]['cer']:.3f} F1 {row[n]['word_f1']:.3f} ({row[n]['seconds']:.1f} s)"
            for n in CLEANERS
        ]
        print(f"page {pid:2d} {page['template']:10s} " + "  ".join(scores))
        gain = row["none"]["cer"] - row["fft_notch_local"]["cer"]
        if (args.show_page == pid) or (args.show_page is None and gain > best_gain):
            shown, best_gain = (scene.photo, results, row), gain  # the figure's page (one kept in memory)

    def mean(name: str, key: str) -> float:
        return round(float(np.mean([r[name][key] for r in rows])), 4)

    report = {
        "what": "capture-kit pages on simulated screens, photographed by the scene simulator, then scanned",
        "scene": {"out_size": [3200, 2400], **SCENE},
        "pages": len(rows),
        "mean": {n: {k: mean(n, k) for k in ("cer", "word_f1", "seconds")} for n in CLEANERS},
        "detected": {n: int(sum(r[n]["detected"] for r in rows)) for n in CLEANERS},
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report["mean"], indent=2))
    photo, results, row = shown
    figure(args.figure, photo, results, row)
    print(f"wrote {args.out} and {args.figure} (page {row['page_id']})")


if __name__ == "__main__":
    main()
