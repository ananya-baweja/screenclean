"""Measure screen detection on synthetic photos of screens (the P5.1 acceptance check).

Each photo is a rendered page shown on a simulated screen in a random room
(``simulate/scene.py``), so the true corners are known exactly. A detection *succeeds* when every
corner is within 3% of the image diagonal. Reported: success rate (overall, by template, and for
light vs dark-mode pages), corner error, time per photo, and the failures.

    python tools/check_screen_detect.py [--n 200] [--out results/checks/screen_detect.json]
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from screenclean.product.screen_detect import detect_screen
from screenclean.render import corpus
from screenclean.render.pages import TEMPLATES, Renderer
from screenclean.simulate.scene import make_scene

SUCCESS = 0.03  # worst corner error, as a share of the image diagonal


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--out", type=Path, default=Path("results/checks/screen_detect.json"))
    args = ap.parse_args()

    renderer = Renderer(corpus.lines_for("test"))
    rows = []
    for i in range(args.n):
        template = TEMPLATES[i % len(TEMPLATES)]
        page = renderer.render(template, seed=args.seed + i).image
        scene = make_scene(page, np.random.default_rng(args.seed + i))
        t = time.perf_counter()
        det = detect_screen(scene.photo)
        seconds = time.perf_counter() - t
        diag = float(np.hypot(*scene.photo.shape[:2]))
        err = np.linalg.norm(det.corners - scene.corners, axis=1) / diag
        rows.append(
            {
                "i": i,
                "template": template,
                "theme": "dark" if np.median(page) < 128 else "light",
                "detected": det.detected,
                "method": det.method,
                "success": bool(det.detected and err.max() < SUCCESS),
                "mean_err": float(err.mean()),
                "max_err": float(err.max()),
                "seconds": seconds,
            }
        )
        print(f"{i:4d} {template:10s} {rows[-1]['theme']:5s} success={rows[-1]['success']} {det.method}")

    def rate(sel: list[dict]) -> dict:
        return {"success": round(float(np.mean([r["success"] for r in sel])), 3), "n": len(sel)}

    groups: dict[str, dict[str, list[dict]]] = {"template": defaultdict(list), "theme": defaultdict(list)}
    for r in rows:
        groups["template"][r["template"]].append(r)
        groups["theme"][r["theme"]].append(r)
    report = {
        "n": len(rows),
        "success_rule": f"every corner within {SUCCESS:.0%} of the image diagonal",
        "overall": rate(rows),
        "not_found": sum(not r["detected"] for r in rows),
        "found_but_wrong": sum(r["detected"] and not r["success"] for r in rows),
        "median_mean_corner_err_pct_diag": round(100 * float(np.median([r["mean_err"] for r in rows])), 4),
        "median_err_when_successful_pct_diag": round(
            100 * float(np.median([r["mean_err"] for r in rows if r["success"]])), 4
        ),
        "seconds_mean": round(float(np.mean([r["seconds"] for r in rows])), 3),
        "seconds_max": round(float(np.max([r["seconds"] for r in rows])), 3),
        **{k: {name: rate(sel) for name, sel in sorted(v.items())} for k, v in groups.items()},
        "failures": [
            {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
            for r in rows
            if not r["success"]
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: report[k] for k in ("overall", "theme", "not_found", "found_but_wrong")}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
