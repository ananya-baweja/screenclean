"""Check that the capture-kit pages are fairly readable when clean (before any photo is taken).

Every ground-truth line is cropped from the clean page and read with Tesseract as a single
line. The character error rate (CER) is reported by body font size, template and font.
The benchmark is only fair if clean text at normal sizes (16 px and up) reads almost
perfectly (target: CER below 3%). Smaller sizes form the labelled "hard" bucket.

    python tools/check_capture_kit_ocr.py [--kit capture_kit] [--out capture_kit/ocr_check.json]

Needs Tesseract 5. Set TESSERACT_CMD if the executable is not on PATH.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from screenclean.eval import tesseract
from screenclean.eval.text_metrics import cer, normalize


def bucket(size: int) -> str:
    return "hard (<16 px)" if size < 16 else "normal (16-24 px)" if size <= 24 else "large (>24 px)"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--kit", type=Path, default=Path("capture_kit"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if not tesseract.available():
        raise SystemExit("Tesseract not found: install it or set TESSERACT_CMD")

    meta = json.loads((args.kit / "pages.json").read_text(encoding="utf-8"))
    groups: dict[str, dict[str, list[tuple[int, int]]]] = {
        k: defaultdict(list) for k in ("bucket", "template", "font")
    }
    worst = []
    for page in meta["pages"]:
        img = Image.open(args.kit / "pages" / f"page_{page['page_id']:03d}.png").convert("RGB")
        for ln in page["lines"]:
            x0, y0, x1, y1 = ln["box"]
            pad = max(4, int(0.35 * ln["size"]))
            crop = img.crop((x0 - pad, y0 - pad, x1 + pad, y1 + pad))
            text = tesseract.ocr(crop, psm=7)
            ref = normalize(ln["text"])
            edits = round(cer(ln["text"], text) * len(ref))
            for key, name in (
                ("bucket", bucket(ln["size"])),
                ("template", page["template"]),
                ("font", ln["font"]),
            ):
                groups[key][name].append((edits, len(ref)))
            if edits:
                worst.append((edits / max(1, len(ref)), page["page_id"], ln["size"], ln["text"], text))

    def summarize(rows: list[tuple[int, int]]) -> dict[str, float]:
        e, n = np.sum([r[0] for r in rows]), np.sum([r[1] for r in rows])
        return {"cer": round(float(e / max(n, 1)), 4), "lines": len(rows), "chars": int(n)}

    report = {k: {name: summarize(rows) for name, rows in sorted(v.items())} for k, v in groups.items()}
    report["worst_lines"] = [
        {"cer": round(c, 3), "page": p, "size": s, "gt": gt, "ocr": o}
        for c, p, s, gt, o in sorted(worst, reverse=True)[:25]
    ]
    for key in ("bucket", "template", "font"):
        print(f"\n{key}:")
        for name, r in report[key].items():
            print(f"  {name:22} CER {100 * r['cer']:5.2f}%  ({r['lines']} lines)")
    print("\nworst lines:")
    for w in report["worst_lines"][:10]:
        print(f"  p{w['page']:02d} {w['size']}px  GT: {w['gt']!r}\n{'':14}OCR: {w['ocr']!r}")
    if args.out:
        args.out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
