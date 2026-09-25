"""Build the capture kit: numbered pages with corner markers, their ground truth, and a slideshow.

The pages are shown full-screen on a monitor, laptop or TV and photographed with a phone.
The markers tell which page each photo shows, and ``pages.json`` holds every line of text on
every page, so OCR on the photos can be scored automatically. Only ``test`` corpus lines
are used.

    python -m screenclean capture-kit        # writes capture_kit/ in the repo
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from screenclean.render import corpus
from screenclean.render.pages import TEMPLATES, Renderer
from screenclean.utils.io import write_image

N_PAGES = 40
SEED = 0
# Body font sizes (px), one per page: 8 small "hard" pages (12-15), 24 normal (16-23), 8 large (26-40).
SIZES = [12, 13, 14, 15] * 2 + list(range(16, 24)) * 3 + [26, 28, 30, 32, 34, 36, 38, 40]  # 8 + 24 + 8 = 40


def page_plan(n_pages: int = N_PAGES, seed: int = SEED) -> list[dict[str, Any]]:
    """Template and body font size for each page: every template appears, sizes spread over all templates."""
    rng = np.random.default_rng(seed)
    templates = [TEMPLATES[i % len(TEMPLATES)] for i in range(n_pages)]
    sizes = [SIZES[i % len(SIZES)] for i in range(n_pages)]
    order = rng.permutation(n_pages)
    return [
        {"page_id": i, "template": templates[i], "base_size": sizes[int(order[i])], "seed": 1000 + i}
        for i in range(n_pages)
    ]


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>ScreenClean capture kit</title>
<style>
  html, body { margin: 0; height: 100%; background: #000; overflow: hidden; }
  img { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: contain; }
  #hint { position: absolute; bottom: 12px; left: 50%; transform: translateX(-50%); color: #bbb;
          font: 16px sans-serif; background: rgba(0,0,0,.6); padding: 6px 12px; border-radius: 6px;
          transition: opacity .5s; }
</style>
</head>
<body>
<img id="page" alt="capture page">
<div id="hint"></div>
<script>
  const N = __N_PAGES__;
  let i = 0;
  const img = document.getElementById("page"), hint = document.getElementById("hint");
  let timer = null;
  const name = k => "pages/page_" + String(k).padStart(3, "0") + ".png";
  function show() {
    img.src = name(i);
    new Image().src = name((i + 1) % N);  // preload the next page
    hint.textContent = "Page " + (i + 1) + " / " + N + "   \\u2190 \\u2192 to move, F for full-screen";
    hint.style.opacity = 1;
    clearTimeout(timer);
    timer = setTimeout(() => hint.style.opacity = 0, 1500);  // the hint must not be in the photos
  }
  document.addEventListener("keydown", e => {
    if (e.key === "ArrowRight" || e.key === " " || e.key === "PageDown") { i = (i + 1) % N; show(); }
    else if (e.key === "ArrowLeft" || e.key === "PageUp") { i = (i + N - 1) % N; show(); }
    else if (e.key === "f" || e.key === "F") {
      if (document.fullscreenElement) document.exitFullscreen();
      else document.documentElement.requestFullscreen();
    }
  });
  img.addEventListener("click", () => { i = (i + 1) % N; show(); });
  show();
</script>
</body>
</html>
"""


def build_kit(out_dir: str | Path, n_pages: int = N_PAGES, seed: int = SEED) -> dict[str, Any]:
    """Write ``pages/page_NNN.png``, ``pages.json`` and ``index.html`` into ``out_dir``."""
    out_dir = Path(out_dir)
    renderer = Renderer(corpus.lines_for("test"))
    pages = []
    for spec in page_plan(n_pages, seed):
        page = renderer.render(
            spec["template"], spec["seed"], base_size=spec["base_size"], page_id=spec["page_id"]
        )
        write_image(out_dir / "pages" / f"page_{spec['page_id']:03d}.png", page.image)
        pages.append({**page.gt(), "base_size": spec["base_size"]})
    meta = {
        "version": 1,
        "corpus_split": "test",
        "page_size": [1920, 1080],
        "marker_dictionary": "DICT_5X5_1000",
        "marker_id_rule": "4 * page_id + corner (0 TL, 1 TR, 2 BR, 3 BL)",
        "pages": pages,
    }
    (out_dir / "pages.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8", newline="\n")
    (out_dir / "index.html").write_text(
        INDEX_HTML.replace("__N_PAGES__", str(n_pages)), encoding="utf-8", newline="\n"
    )
    return meta
