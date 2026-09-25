"""Real phone photos of capture-kit pages: find the page, fit the homography, record quality.

Photos live in folders named ``screen-<name>__phone-<name>``. For every photo:

1. apply the EXIF orientation;
2. detect the ArUco corner markers (at a few scales, since phone photos are large);
3. from the markers, get the page id and the homography H (photo -> page);
4. record quality flags: markers found, reprojection error, ``ok``.

The manifest (CSV) has ``file, screen, phone, page_id, n_markers, reproj_err_px, ok``, and a
JSON file keeps each H plus where the page corners are in the photo. The evaluation runs
every method on the *original* photo and then rectifies all outputs with the same H, so
the comparison is fair.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from screenclean.render import aruco
from screenclean.utils.image import to_uint8
from screenclean.utils.io import read_image

log = logging.getLogger(__name__)

PHOTO_EXTS = {".jpg", ".jpeg", ".png", ".heic"}
FIELDS = ["file", "screen", "phone", "page_id", "n_markers", "reproj_err_px", "ok"]


def parse_folder(name: str) -> tuple[str, str]:
    """``screen-hp-laptop__phone-redmi12`` -> ``("hp-laptop", "redmi12")`` (unknown parts become "?")."""
    screen, phone = "?", "?"
    for part in name.split("__"):
        if part.startswith("screen-"):
            screen = part[len("screen-") :]
        elif part.startswith("phone-"):
            phone = part[len("phone-") :]
    return screen, phone


def detect_multiscale(
    img: np.ndarray, scales: tuple[float, ...] = (1.0, 0.5, 0.25)
) -> list[tuple[int, np.ndarray]]:
    """Detect markers at a few scales; keep the scale that finds the most. Corners are in full-res pixels."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    best: list[tuple[int, np.ndarray]] = []
    for s in scales:
        g = gray if s == 1.0 else cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        found = [(i, c / s) for i, c in aruco.detect(g)]
        if len(found) > len(best):
            best = found
        if len(best) >= 4:
            break
    return best


@dataclass
class CaptureResult:
    file: str
    screen: str
    phone: str
    page_id: int | None
    n_markers: int
    reproj_err_px: float | None
    ok: bool
    H: list[list[float]] | None = None  # photo -> page
    page_corners_px: list[list[float]] | None = None  # page TL, TR, BR, BL in photo pixels
    photo_size: list[int] | None = None  # w, h after EXIF orientation

    def row(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in FIELDS}


def process_photo(
    path: str | Path, screen: str = "?", phone: str = "?", page_size: tuple[int, int] = (1920, 1080)
) -> CaptureResult:
    """Match one photo to its capture page."""
    path = Path(path)
    img = to_uint8(read_image(path))
    h, w = img.shape[:2]
    fit = aruco.fit_page(detect_multiscale(img), page_size)
    if fit is None:
        return CaptureResult(path.name, screen, phone, None, 0, None, False, photo_size=[w, h])
    corners = aruco.page_corners_in_photo(fit.H, page_size)
    return CaptureResult(
        file=path.name,
        screen=screen,
        phone=phone,
        page_id=fit.page_id,
        n_markers=fit.n_markers,
        reproj_err_px=round(fit.reproj_err_px, 3),
        ok=fit.ok,
        H=fit.H.tolist(),
        page_corners_px=corners.round(2).tolist(),
        photo_size=[w, h],
    )


def ingest(raw_root: str | Path, out_dir: str | Path) -> list[CaptureResult]:
    """Process all photos in ``raw_root/<screen>__<phone>/``; write manifest.csv and captures.json."""
    raw_root, out_dir = Path(raw_root), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for folder in sorted(p for p in raw_root.iterdir() if p.is_dir()):
        screen, phone = parse_folder(folder.name)
        for photo in sorted(p for p in folder.iterdir() if p.suffix.lower() in PHOTO_EXTS):
            try:
                res = process_photo(photo, screen, phone)
            except Exception as e:  # noqa: BLE001 - one unreadable photo shouldn't stop the batch
                log.warning("could not process %s: %s", photo, e)
                res = CaptureResult(photo.name, screen, phone, None, 0, None, False)
            res.file = f"{folder.name}/{photo.name}"
            results.append(res)
            log.info(
                "%s: page %s, %d markers, err %s, ok=%s",
                res.file,
                res.page_id,
                res.n_markers,
                res.reproj_err_px,
                res.ok,
            )
    with open(out_dir / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(r.row() for r in results)
    (out_dir / "captures.json").write_text(
        json.dumps([r.__dict__ for r in results], indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    return results
