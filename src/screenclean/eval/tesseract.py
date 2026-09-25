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


def ocr(image: np.ndarray | Image.Image, psm: int = 3, lang: str = "eng", timeout_s: float = 120) -> str:
    """Recognise the text in ``image`` (uint8 RGB/grey array or PIL image)."""
    cmd = tesseract_cmd()
    if cmd is None:
        raise RuntimeError("Tesseract not found: install it or set TESSERACT_CMD")
    pil = image if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image, dtype=np.uint8))
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    res = subprocess.run(
        [cmd, "stdin", "stdout", "--psm", str(psm), "-l", lang],
        input=buf.getvalue(),
        capture_output=True,
        timeout=timeout_s,
        check=True,
    )
    return res.stdout.decode("utf-8", errors="replace").strip()
