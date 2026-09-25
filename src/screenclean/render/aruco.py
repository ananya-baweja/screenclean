"""ArUco corner markers: tell which page a photo shows and map the photo back onto the page.

Every capture page gets four markers from ``DICT_5X5_1000``, one per corner, with
``marker id = 4 * page_id + corner`` (corner 0 = top-left, 1 = top-right, 2 = bottom-right,
3 = bottom-left). In a photo, the detected markers give both the page id and 4 exact point
matches each, which is enough to fit the homography (the 3x3 perspective map) between photo
and page. With the homography, the photo is "rectified" into page coordinates and compared
with the page's ground-truth text.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

MARKER_PX = 96  # marker side, including its 1-module black border
QUIET_PX = 24  # white margin around each marker (detection needs contrast around it)
INSET_PX = 16  # distance of the white margin from the page edge
MAX_PAGES = 250  # DICT_5X5_1000 has 1000 ids = 250 pages x 4 corners
MIN_MARKERS = 3


def dictionary():
    return cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)


def marker_origin(corner: int, page_size: tuple[int, int], size: int = MARKER_PX) -> tuple[int, int]:
    """Top-left pixel of the marker for ``corner`` on a page of ``page_size`` (w, h)."""
    w, h = page_size
    off = INSET_PX + QUIET_PX
    x = off if corner in (0, 3) else w - off - size
    y = off if corner in (0, 1) else h - off - size
    return x, y


def marker_corners(corner: int, page_size: tuple[int, int], size: int = MARKER_PX) -> np.ndarray:
    """The marker's 4 outer corners in page pixels, in OpenCV's order (TL, TR, BR, BL of the marker).

    OpenCV measures positions from pixel centres, so the outer edge of a marker whose first
    pixel is ``x`` lies at ``x - 0.5`` (and its far edge at ``x + size - 0.5``).
    """
    x, y = marker_origin(corner, page_size, size)
    x0, y0, x1, y1 = x - 0.5, y - 0.5, x + size - 0.5, y + size - 0.5
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)


def _marker_image(marker_id: int, size: int) -> np.ndarray:
    d = dictionary()
    if hasattr(cv2.aruco, "generateImageMarker"):  # OpenCV >= 4.7
        return cv2.aruco.generateImageMarker(d, marker_id, size, borderBits=1)
    return cv2.aruco.drawMarker(d, marker_id, size, borderBits=1)


def draw_markers(img: np.ndarray, page_id: int, size: int = MARKER_PX) -> np.ndarray:
    """Return a copy of ``img`` (uint8 RGB) with the four corner markers of ``page_id``."""
    if not 0 <= page_id < MAX_PAGES:
        raise ValueError(f"page_id must be in [0, {MAX_PAGES})")
    out = img.copy()
    h, w = out.shape[:2]
    for corner in range(4):
        x, y = marker_origin(corner, (w, h), size)
        out[y - QUIET_PX : y + size + QUIET_PX, x - QUIET_PX : x + size + QUIET_PX] = 255
        out[y : y + size, x : x + size] = _marker_image(4 * page_id + corner, size)[..., None]
    return out


def detect(image: np.ndarray) -> list[tuple[int, np.ndarray]]:
    """Find markers in an RGB or grey image. Returns ``[(marker_id, corners (4, 2) float32)]``."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if hasattr(cv2.aruco, "ArucoDetector"):  # OpenCV >= 4.7
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        corners, ids, _ = cv2.aruco.ArucoDetector(dictionary(), params).detectMarkers(gray)
    else:
        params = cv2.aruco.DetectorParameters_create()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary(), parameters=params)
    if ids is None:
        return []
    return [(int(i), c.reshape(4, 2).astype(np.float32)) for i, c in zip(ids.ravel(), corners, strict=True)]


@dataclass
class PageFit:
    """Result of matching a photo to a page."""

    page_id: int
    H: np.ndarray  # 3x3, photo pixels -> page pixels
    n_markers: int
    reproj_err_px: float  # mean distance (page pixels) between mapped marker corners and their true positions

    @property
    def ok(self) -> bool:
        return self.n_markers >= MIN_MARKERS and self.reproj_err_px < 3.0


def fit_page(
    detections: list[tuple[int, np.ndarray]],
    page_size: tuple[int, int] = (1920, 1080),
    min_markers: int = MIN_MARKERS,
) -> PageFit | None:
    """Pick the page with the most detected markers and fit photo -> page with RANSAC."""
    by_page: dict[int, list[tuple[int, np.ndarray]]] = {}
    for mid, c in detections:
        if mid < 4 * MAX_PAGES:
            by_page.setdefault(mid // 4, []).append((mid % 4, c))
    if not by_page:
        return None
    page_id, found = max(by_page.items(), key=lambda kv: (len({c for c, _ in kv[1]}), -kv[0]))
    corners_seen = {}
    for corner, c in found:  # if a corner id is seen twice (reflection?), keep the first
        corners_seen.setdefault(corner, c)
    if len(corners_seen) < min_markers:
        return None
    src = np.concatenate([corners_seen[k] for k in sorted(corners_seen)])
    dst = np.concatenate([marker_corners(k, page_size) for k in sorted(corners_seen)])
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if H is None:
        return None
    mapped = cv2.perspectiveTransform(src[None], H)[0]
    err = float(np.mean(np.linalg.norm(mapped - dst, axis=1)))
    return PageFit(page_id, H, len(corners_seen), err)


def rectify(photo: np.ndarray, H: np.ndarray, page_size: tuple[int, int] = (1920, 1080)) -> np.ndarray:
    """Warp the photo into page coordinates (``page_size`` = (w, h))."""
    return cv2.warpPerspective(photo, H, page_size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def page_corners_in_photo(H: np.ndarray, page_size: tuple[int, int] = (1920, 1080)) -> np.ndarray:
    """Where the page's 4 outer corners land in the photo (TL, TR, BR, BL), given H (photo -> page)."""
    w, h = page_size
    pts = np.array([[[-0.5, -0.5], [w - 0.5, -0.5], [w - 0.5, h - 0.5], [-0.5, h - 0.5]]], np.float32)
    return cv2.perspectiveTransform(pts, np.linalg.inv(H))[0]
