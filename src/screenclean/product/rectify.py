"""Perspective correction and light clean-up: the detected screen becomes a flat, upright page.

- ``estimate_aspect``: the real width/height of the screen from its perspective quadrilateral
  (Zhang & He, "Whiteboard scanning and image enhancement", 2007: the four corners of a
  rectangle fix the camera's focal length, and then its true shape). Snapped to 16:9, 16:10
  or 4:3 when close.
- ``rectify``: warp the photo so the quadrilateral fills the output. The output keeps at least
  the photo's resolution along every side; if it must shrink (``max_side`` or a given size), the
  photo is first blurred just enough that the warp does not alias.
- ``flatten_illumination`` + ``stretch_contrast``: remove uneven lighting and glare, then use the
  full brightness range ("document" mode; off for "photo" mode).

Corners use pixel-centre coordinates: the outer corners of a W x H image are (-0.5, -0.5) and
(W - 0.5, H - 0.5).
"""

from __future__ import annotations

import cv2
import numpy as np

from screenclean.utils.image import to_float01

COMMON_ASPECTS = {"16:9": 16 / 9, "16:10": 16 / 10, "4:3": 4 / 3}


def _side_lengths(c: np.ndarray) -> tuple[float, float, float, float]:
    top, right = np.linalg.norm(c[1] - c[0]), np.linalg.norm(c[2] - c[1])
    bottom, left = np.linalg.norm(c[2] - c[3]), np.linalg.norm(c[3] - c[0])
    return float(top), float(right), float(bottom), float(left)


def estimate_aspect(corners: np.ndarray, image_size: tuple[int, int]) -> float:
    """Width / height of the real rectangle whose photo has these corners (TL, TR, BR, BL).

    ``image_size`` = (w, h); the principal point is assumed at the image centre. When the view is
    nearly straight on, the focal length can't be recovered and the average side lengths are used.
    """
    c = np.asarray(corners, np.float64)
    top, right, bottom, left = _side_lengths(c)
    fallback = (top + bottom) / max(left + right, 1e-9)
    u0, v0 = (image_size[0] - 1) / 2, (image_size[1] - 1) / 2
    m1, m2, m4, m3 = (np.array([x - u0, y - v0, 1.0]) for x, y in c)  # Zhang's order: TL, TR, BL, BR
    k2 = np.dot(np.cross(m1, m4), m3) / np.dot(np.cross(m2, m4), m3)
    k3 = np.dot(np.cross(m1, m4), m2) / np.dot(np.cross(m3, m4), m2)
    n2, n3 = k2 * m2 - m1, k3 * m3 - m1
    if abs(n2[2]) < 1e-9 or abs(n3[2]) < 1e-9:  # no vanishing point: an affine view
        return float(np.hypot(n2[0], n2[1]) / max(np.hypot(n3[0], n3[1]), 1e-9))
    f2 = -(n2[0] * n3[0] + n2[1] * n3[1]) / (n2[2] * n3[2])
    diag2 = float(image_size[0] ** 2 + image_size[1] ** 2)
    if not 0.05 * diag2 < f2 < 25 * diag2:  # focal length implausible: nearly fronto-parallel
        return float(fallback)
    a = np.diag([1 / f2, 1 / f2, 1.0])  # (K K^T)^-1 with the principal point at the origin
    return float(np.sqrt((n2 @ a @ n2) / (n3 @ a @ n3)))


def snap_aspect(aspect: float, tolerance: float = 0.04) -> tuple[float, str | None]:
    """Snap to a common screen shape (either orientation) when within ``tolerance`` (relative)."""
    options = []
    for label, r in COMMON_ASPECTS.items():
        wide, tall = label.split(":")
        options += [(r, label), (1 / r, f"{tall}:{wide}")]
    value, name = min(options, key=lambda o: abs(np.log(aspect / o[0])))
    return (value, name) if abs(aspect / value - 1) < tolerance else (aspect, None)


def output_size(corners: np.ndarray, aspect: float, max_side: int | None = None) -> tuple[int, int]:
    """(w, h) with the given aspect, big enough that no side of the quadrilateral is shrunk."""
    top, right, bottom, left = _side_lengths(np.asarray(corners, np.float64))
    w, h = max(top, bottom), max(left, right)
    if w / h > aspect:
        h = w / aspect
    else:
        w = h * aspect
    if max_side and max(w, h) > max_side:
        k = max_side / max(w, h)
        w, h = w * k, h * k
    return max(1, round(w)), max(1, round(h))


def page_homography(corners: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Photo -> page homography mapping the quadrilateral onto a (w, h) page."""
    w, h = size
    dst = np.array([[-0.5, -0.5], [w - 0.5, -0.5], [w - 0.5, h - 0.5], [-0.5, h - 0.5]], np.float32)
    return cv2.getPerspectiveTransform(np.asarray(corners, np.float32), dst)


def _min_scale(H: np.ndarray, size: tuple[int, int]) -> float:
    """Smallest photo -> page scale over the page (below 1: the page is smaller than the photo there)."""
    w, h = size
    Hinv = np.linalg.inv(H)
    worst = np.inf
    for x, y in ((0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1), ((w - 1) / 2, (h - 1) / 2)):
        p = np.array([[x, y], [x + 1, y], [x, y + 1]], np.float32)[None]
        a, b, c = cv2.perspectiveTransform(p, Hinv)[0].astype(np.float64)
        (ux, uy), (vx, vy) = b - a, c - a
        area = abs(ux * vy - uy * vx)  # photo pixels covered by one page pixel
        worst = min(worst, 1.0 / np.sqrt(max(area, 1e-12)))
    return float(worst)


def rectify(
    img: np.ndarray,
    corners: np.ndarray,
    size: tuple[int, int] | None = None,
    aspect: float | None = None,
    snap: bool = True,
    max_side: int | None = 4000,
) -> tuple[np.ndarray, np.ndarray]:
    """Warp the quadrilateral ``corners`` (TL, TR, BR, BL) of ``img`` into an upright page.

    ``size`` = (w, h) fixes the output; otherwise it comes from ``aspect`` (estimated from the
    corners, and snapped to a common screen shape if ``snap``). Returns (page, H photo -> page),
    with the page as float32 in [0, 1].
    """
    src = to_float01(img)
    corners = np.asarray(corners, np.float64)
    if size is None:
        if aspect is None:
            aspect = estimate_aspect(corners, (src.shape[1], src.shape[0]))
            if snap:
                aspect, _ = snap_aspect(aspect)
        size = output_size(corners, aspect, max_side)
    H = page_homography(corners, size)
    shrink = _min_scale(H, size)
    if shrink < 0.8:  # the warp would skip photo pixels: blur away detail the page can't hold
        src = cv2.GaussianBlur(src, (0, 0), 0.5 * float(np.sqrt(1.0 / shrink**2 - 1.0)))
    page = cv2.warpPerspective(src, H, size, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return np.clip(page, 0.0, 1.0), H


def flatten_illumination(img: np.ndarray, size_frac: float = 0.04) -> np.ndarray:
    """Even out lighting: estimate the page background with a morphological filter and a wide
    blur, then divide it out (light pages) or subtract it (dark pages, where glare adds light)."""
    x = to_float01(img)
    h, w = x.shape[:2]
    k = max(3, int(size_frac * min(h, w)) | 1)
    luma = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) if x.ndim == 3 else x
    light = float(np.median(luma)) > 0.5
    op = cv2.MORPH_CLOSE if light else cv2.MORPH_OPEN  # remove the text, keep the background
    bg = cv2.morphologyEx(x, op, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    bg = cv2.GaussianBlur(bg, (0, 0), k)
    ref = np.median(bg.reshape(-1, bg.shape[-1] if bg.ndim == 3 else 1), axis=0)
    if light:
        out = x / np.maximum(bg, 1e-3) * ref
    else:
        out = x - bg + ref
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def stretch_contrast(img: np.ndarray, low_pct: float = 1.0, high_pct: float = 99.0) -> np.ndarray:
    """Map the ``low_pct`` / ``high_pct`` brightness percentiles to 0 and 1 (same for all channels)."""
    x = to_float01(img)
    luma = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) if x.ndim == 3 else x
    lo, hi = np.percentile(luma, [low_pct, high_pct])
    if hi - lo < 1e-3:
        return x
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def clean_up(img: np.ndarray) -> np.ndarray:
    """ "Document" mode: flatten the lighting, then stretch the contrast."""
    return stretch_contrast(flatten_illumination(img))
