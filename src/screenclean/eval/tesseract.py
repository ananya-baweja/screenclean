"""Minimal Tesseract runner: sends an image to the ``tesseract`` executable and reads the text back.

Calls the program directly (``tesseract stdin stdout``) instead of going through pytesseract,
which imports pandas and isn't needed for plain text output.
"""

from __future__ import annotations

import io
import os
import shutil
import subprocess
from functools import lru_cache

import numpy as np
from PIL import Image


@lru_cache(maxsize=1)
def tesseract_cmd() -> str | None:
    """Path of the tesseract executable (``TESSERACT_CMD`` env var, else PATH), or None."""
    return os.environ.get("TESSERACT_CMD") or shutil.which("tesseract")


def available() -> bool:
    return tesseract_cmd() is not None


def _run(image: np.ndarray | Image.Image, args: list[str], timeout_s: float) -> str:
    cmd = tesseract_cmd()
    if cmd is None:
        raise RuntimeError("Tesseract not found: install it or set TESSERACT_CMD")
    pil = image if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image, dtype=np.uint8))
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    res = subprocess.run(
        [cmd, "stdin", "stdout", *args],
        input=buf.getvalue(),
        capture_output=True,
        timeout=timeout_s,
        check=True,
    )
    return res.stdout.decode("utf-8", errors="replace")


def ocr(image: np.ndarray | Image.Image, psm: int = 3, lang: str = "eng", timeout_s: float = 120) -> str:
    """Recognise the text in ``image`` (uint8 RGB/grey array or PIL image)."""
    return _run(image, ["--psm", str(psm), "-l", lang], timeout_s).strip()


def ocr_words(
    image: np.ndarray | Image.Image, psm: int = 3, lang: str = "eng", timeout_s: float = 120
) -> list[dict]:
    """Words with boxes: dicts with ``text``, ``box`` (x0, y0, x1, y1 pixels), ``conf`` (0-100) and
    ``line`` (block, paragraph, line numbers), in Tesseract's reading order."""
    tsv = _run(image, ["--psm", str(psm), "-l", lang, "tsv"], timeout_s)
    words = []
    for row in tsv.splitlines()[1:]:
        parts = row.split("\t")
        if len(parts) < 12 or parts[0] != "5" or not parts[11].strip():
            continue  # level 5 = word
        left, top, width, height = (int(v) for v in parts[6:10])
        words.append(
            {
                "text": parts[11],
                "box": (left, top, left + width, top + height),
                "conf": float(parts[10]),
                "line": (int(parts[2]), int(parts[3]), int(parts[4])),
            }
        )
    return words
