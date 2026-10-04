"""OCR worker: a separate process that reads image files and answers with text lines.

:mod:`screenclean.eval.ocr` starts it as ``python ocr_worker.py --engine paddle``, with
PaddleOCR's packages in their own folder first on the path: PaddleOCR brings its own numpy
and OpenCV versions, which must not clash with PyTorch's in the main process. Only the
standard library is imported at the top, so the file also runs as a plain script.

Protocol, one JSON object per line:

- the worker's first line: ``{"ready": true, "engine": ..., "versions": {...}}``, or
  ``{"ready": false, "error": ...}`` if the engine could not start;
- request ``{"id": ..., "path": "/abs/page.png"}`` -> reply
  ``{"id": ..., "lines": [{"text", "box": [x0, y0, x1, y1], "conf"}], "seconds": ...}``
  (``"error"`` instead of ``"lines"`` if that image failed);
- closing the input stops the worker.

Engines: ``paddle`` (PaddleOCR 3.x), ``rapidocr`` (the same PP-OCR models on ONNX Runtime;
the fallback) and ``tesseract`` (the project's Tesseract setup, as in the scan pipeline).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from importlib import metadata
from typing import Any


def _version(dist: str) -> str | None:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def _box(points: Any) -> list[float]:
    """[x0, y0, x1, y1] around a polygon (4 x 2 points) or from a box (x0, y0, x1, y1)."""
    seq = points.tolist() if hasattr(points, "tolist") else list(points)
    if len(seq) == 4 and all(isinstance(v, (int, float)) for v in seq):  # already a box
        return [float(v) for v in seq]
    xs = [float(p[0]) for p in seq]
    ys = [float(p[1]) for p in seq]
    return [min(xs), min(ys), max(xs), max(ys)]


def _field(result: Any, key: str) -> Any:
    """``result[key]``, also for results that keep their fields under ``.json["res"]``."""
    try:
        return result[key]
    except (KeyError, TypeError, IndexError):
        pass
    data = getattr(result, "json", None)
    if isinstance(data, dict):
        return data.get("res", data).get(key)
    return None


def parse_paddle(result: Any) -> list[dict[str, Any]]:
    """Lines from one PaddleOCR 3.x result (``rec_texts``, ``rec_scores``, ``rec_polys``)."""
    texts = _field(result, "rec_texts")
    if texts is None:
        keys = sorted(result.keys()) if hasattr(result, "keys") else type(result).__name__
        raise KeyError(f"PaddleOCR result has no 'rec_texts' (fields: {keys})")
    scores = _field(result, "rec_scores")
    polys = _field(result, "rec_polys")
    if polys is None:
        polys = _field(result, "rec_boxes")
    if polys is None or len(polys) != len(texts):
        raise KeyError("PaddleOCR result has no 'rec_polys' or 'rec_boxes' matching 'rec_texts'")
    lines = []
    for i, text in enumerate(texts):
        conf = float(scores[i]) if scores is not None and len(scores) > i else 1.0
        if str(text).strip():
            lines.append({"text": str(text), "box": _box(polys[i]), "conf": conf})
    return lines


def parse_rapid(output: Any) -> list[dict[str, Any]]:
    """Lines from RapidOCR: ``.txts/.boxes/.scores`` (2.x and later) or ``(result, elapse)`` (1.x)."""
    if isinstance(output, tuple):
        items = output[0] or []
        return [{"text": str(t), "box": _box(b), "conf": float(s)} for b, t, s in items if str(t).strip()]
    texts = getattr(output, "txts", None)
    if texts is None:
        return []
    boxes, scores = output.boxes, output.scores
    return [
        {"text": str(t), "box": _box(boxes[i]), "conf": float(scores[i])}
        for i, t in enumerate(texts)
        if str(t).strip()
    ]


class PaddleEngine:
    def __init__(self, **options: Any):
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        from paddleocr import PaddleOCR

        kwargs = {
            "lang": "en",
            # all off, for a fair comparison: every method's page is already upright and flat
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
            **options,
        }
        self.ocr = PaddleOCR(**kwargs)
        self.options = kwargs

    def read(self, path: str) -> list[dict[str, Any]]:
        results = list(self.ocr.predict(path))
        return parse_paddle(results[0]) if results else []

    def versions(self) -> dict[str, Any]:
        return {
            "paddleocr": _version("paddleocr"),
            "paddlepaddle": _version("paddlepaddle"),
            "options": self.options,
        }


class PaddleSafeEngine(PaddleEngine):
    """PaddleOCR without oneDNN (CPU acceleration that some Paddle releases break): slower, sturdier."""

    def __init__(self, **options: Any):
        super().__init__(**{"enable_mkldnn": False, **options})


class RapidEngine:
    def __init__(self, **options: Any):
        try:
            from rapidocr import RapidOCR
        except ImportError:
            from rapidocr_onnxruntime import RapidOCR
        self.engine = RapidOCR(**options)

    def read(self, path: str) -> list[dict[str, Any]]:
        return parse_rapid(self.engine(path))

    def versions(self) -> dict[str, Any]:
        return {
            "rapidocr": _version("rapidocr") or _version("rapidocr-onnxruntime"),
            "onnxruntime": _version("onnxruntime"),
        }


class TesseractEngine:
    def __init__(self, **options: Any):
        from screenclean.eval import tesseract
        from screenclean.product.ocr import TesseractOCR
        from screenclean.utils.io import read_image

        if not tesseract.available():
            raise RuntimeError("Tesseract not found: install it or set TESSERACT_CMD")
        self.ocr, self.read_image, self.cmd = TesseractOCR(**options), read_image, tesseract.tesseract_cmd()

    def read(self, path: str) -> list[dict[str, Any]]:
        lines = self.ocr.read(self.read_image(path))
        return [{"text": ln.text, "box": list(ln.box), "conf": ln.conf} for ln in lines]

    def versions(self) -> dict[str, Any]:
        import subprocess

        out = subprocess.run([self.cmd, "--version"], capture_output=True, text=True, check=False)
        first = (out.stdout or out.stderr).splitlines()[:1]
        return {
            "tesseract": first[0] if first else None,
            "psm": self.ocr.psm,
            "thresholding_method": self.ocr.thresholding,
        }


ENGINES = {
    "paddle": PaddleEngine,
    "paddle_safe": PaddleSafeEngine,
    "rapidocr": RapidEngine,
    "tesseract": TesseractEngine,
}


def serve(engine: Any, requests, replies) -> None:
    """Answer requests until the input ends."""
    for raw in requests:
        raw = raw.strip()
        if not raw:
            continue
        req = json.loads(raw)
        t0 = time.perf_counter()
        try:
            reply: dict[str, Any] = {"id": req["id"], "lines": engine.read(req["path"])}
        except Exception as e:  # noqa: BLE001 - report per image and keep serving
            reply = {"id": req["id"], "error": f"{type(e).__name__}: {e}"}
        reply["seconds"] = round(time.perf_counter() - t0, 4)
        replies.write(json.dumps(reply) + "\n")
        replies.flush()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="OCR worker (JSON lines on stdin/stdout)")
    ap.add_argument("--engine", required=True)
    ap.add_argument("--options", default="{}", help="JSON keyword arguments for the engine")
    args = ap.parse_args(argv)

    # Engines print progress, some from C code. Keep the real stdout for replies only and send
    # everything else (file descriptor 1 included) to stderr.
    replies = os.fdopen(os.dup(1), "w", encoding="utf-8")
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    try:
        engine = ENGINES[args.engine](**json.loads(args.options))
        hello = {"ready": True, "engine": args.engine, "versions": engine.versions()}
    except Exception as e:  # noqa: BLE001 - the client reports why the engine could not start
        hello = {"ready": False, "engine": args.engine, "error": f"{type(e).__name__}: {e}"}
    replies.write(json.dumps(hello) + "\n")
    replies.flush()
    if hello["ready"]:
        serve(engine, sys.stdin, replies)
    return 0 if hello["ready"] else 1


if __name__ == "__main__":
    sys.exit(main())
