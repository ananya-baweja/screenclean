"""The scan pipeline: phone photo of a screen -> upright, clean page -> text.

::

    photo -> find the screen -> remove moiré (on the photo) -> straighten -> light clean-up -> OCR

**Clean before you warp.** Moiré is removed on the original photo, before the perspective warp:
resampling a moiré photo can create new aliasing, and the cleaners (like the model later) were
built for photos as cameras take them. Only the screen's bounding box is cleaned, to save time.

``scan()`` handles one photo, ``scan_batch()`` several, and also returns a searchable PDF and
Markdown. Cleaners and OCR engines are pluggable: a name, or any callable / object with ``read``.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from screenclean.product.ocr import OcrLine, get_engine, lines_to_text
from screenclean.product.rectify import clean_up, rectify
from screenclean.product.screen_detect import ScreenDetection, detect_screen, full_image_corners
from screenclean.utils.image import to_float01
from screenclean.utils.io import read_image

TUNED_BASELINES = Path(__file__).resolve().parents[3] / "configs" / "baselines" / "tuned.yaml"
MODES = ("document", "photo")


@lru_cache(maxsize=1)
def _tuned(name: str) -> dict[str, Any]:
    """Settings chosen by job 0003_tune_notch (empty if the config isn't available)."""
    try:
        return dict(yaml.safe_load(TUNED_BASELINES.read_text(encoding="utf-8")).get(name, {}))
    except OSError:
        return {}


def get_cleaner(cleaner: str | Callable[[np.ndarray], np.ndarray]) -> Callable[[np.ndarray], np.ndarray]:
    """``"none"``, ``"fft_notch_local"`` (classical, tuned on UHDM), or any image -> image callable."""
    if callable(cleaner):
        return cleaner
    if cleaner == "none":
        return lambda img: img
    if cleaner == "fft_notch_local":
        from screenclean.baselines.classical import fft_notch_local

        params = _tuned("fft_notch_local")
        return lambda img: fft_notch_local(img, **params)
    if cleaner == "scnet":
        raise NotImplementedError(
            "the trained ScreenCleanNet cleaner arrives with the model export (P10-P11)"
        )
    raise KeyError(f"unknown cleaner {cleaner!r}; choose 'none', 'fft_notch_local' or pass a function")


@dataclass
class ScanResult:
    corners: np.ndarray  # (4, 2) screen corners in the photo (TL, TR, BR, BL)
    detected: bool  # False: the whole photo was used
    clean: np.ndarray  # the photo after moiré removal (float32 RGB; only the screen area is cleaned)
    page: np.ndarray  # the upright page (float32 RGB)
    lines: list[OcrLine] = field(default_factory=list)
    text: str = ""
    timings: dict[str, float] = field(default_factory=dict)  # seconds per stage
    detection: dict[str, Any] = field(default_factory=dict)
    name: str = ""

    def summary(self) -> dict[str, Any]:
        """Everything except the images (for JSON)."""
        return {
            "name": self.name,
            "detected": self.detected,
            "corners": np.asarray(self.corners).round(2).tolist(),
            "detection": self.detection,
            "page_size": [int(self.page.shape[1]), int(self.page.shape[0])],
            "lines": [line.as_dict() for line in self.lines],
            "text": self.text,
            "timings": {k: round(v, 3) for k, v in self.timings.items()},
        }


def _clean_region(img: np.ndarray, corners: np.ndarray, cleaner, margin: int = 32) -> np.ndarray:
    """Run ``cleaner`` on the screen's bounding box (plus a margin) and paste the result back."""
    h, w = img.shape[:2]
    x0, y0 = np.floor(corners.min(axis=0)).astype(int) - margin
    x1, y1 = np.ceil(corners.max(axis=0)).astype(int) + margin + 1
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    out = img.copy()
    out[y0:y1, x0:x1] = np.clip(cleaner(img[y0:y1, x0:x1]), 0.0, 1.0)
    return out


def scan(
    image: np.ndarray | str | Path,
    cleaner: str | Callable[[np.ndarray], np.ndarray] = "none",
    ocr: Any = "tesseract",
    mode: str = "document",
    corners: np.ndarray | None = None,
    name: str = "",
    max_side: int | None = 4000,
) -> ScanResult:
    """Scan one photo (an RGB array, or a file path: EXIF orientation is applied).

    The default cleaner is "none" until the trained model arrives: the classical notch filter
    lowered OCR accuracy on average (it mistakes regular text layouts, like code, for moiré; see
    docs/DECISIONS.md). ``cleaner="fft_notch_local"`` still runs it.

    ``corners`` skips detection (e.g. corners the user dragged in the app). ``mode="document"``
    flattens the lighting and stretches contrast; ``"photo"`` keeps the page as photographed.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    t_start = time.perf_counter()
    timings: dict[str, float] = {}
    if isinstance(image, (str, Path)):
        name = name or Path(image).name
        image = read_image(image)
        timings["load"] = time.perf_counter() - t_start
    img = to_float01(np.asarray(image))
    h, w = img.shape[:2]

    t = time.perf_counter()
    if corners is None:
        det = detect_screen(img)
    else:
        det = ScreenDetection(np.asarray(corners, np.float32), True, 1.0, "given", 0)
    timings["detect"] = time.perf_counter() - t
    quad = det.corners if det.detected else full_image_corners(w, h)

    t = time.perf_counter()
    clean = _clean_region(img, quad, get_cleaner(cleaner))
    timings["clean"] = time.perf_counter() - t

    t = time.perf_counter()
    page, _ = rectify(clean, quad, max_side=max_side) if det.detected else (clean, None)
    timings["rectify"] = time.perf_counter() - t
    if mode == "document":
        t = time.perf_counter()
        page = clean_up(page)
        timings["cleanup"] = time.perf_counter() - t

    lines: list[OcrLine] = []
    engine = get_engine(ocr)
    if engine is not None:
        t = time.perf_counter()
        lines = engine.read(page)
        timings["ocr"] = time.perf_counter() - t
    timings["total"] = time.perf_counter() - t_start
    return ScanResult(
        corners=quad,
        detected=det.detected,
        clean=clean,
        page=page,
        lines=lines,
        text=lines_to_text(lines),
        timings=timings,
        detection=det.as_dict(),
        name=name,
    )


def scan_batch(
    images: Sequence[np.ndarray | str | Path],
    names: Sequence[str] | None = None,
    title: str = "Scanned screens",
    **kwargs: Any,
) -> tuple[list[ScanResult], bytes, str]:
    """Scan several photos; return the results, one searchable PDF (a page per photo) and Markdown."""
    from screenclean.product.export import searchable_pdf, to_markdown

    results = []
    for i, img in enumerate(images):
        name = names[i] if names else (Path(img).name if isinstance(img, (str, Path)) else f"photo {i + 1}")
        results.append(scan(img, name=name, **kwargs))
    return results, searchable_pdf(results), to_markdown(results, title)
