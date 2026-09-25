"""Find the screen in a photo: the four corners of the displayed content area (not the bezel).

Classical pipeline, on a grey copy downscaled to about 1000 px:

1. **Candidates.** Convex quadrilaterals covering at least ``min_area`` of the photo, with the
   real shape of a screen (from a tall phone to an ultrawide monitor), from
   - contours of edge maps (bilateral filter + Canny at three sensitivities, dilated),
   - contours of bright regions (Otsu threshold) and of "busy" regions (lots of text),
   - long straight lines (Hough), paired up into quadrilaterals.
2. **Snap.** Each side moves to the strongest straight edge nearby, measured by the median
   brightness step across the whole side. A long straight edge is found even when it is faint
   (a dark-mode page next to a black bezel), because many samples vote for it.
3. **Keep** quadrilaterals whose four sides are real edges (``passes``): a clear step along at
   least 70% of each third of every side, with the same sign all round. The sign is *polarity*:
   +1 when the inside is brighter. Light pages are brighter than the bezel (+1); dark-mode pages
   are often darker than a bezel that reflects the room (-1).
   Almost-quadrilaterals become **frames**: three good sides (a monitor stand breaks the fourth),
   or four good sides of mixed polarity (the whole device, against a room that is brighter on
   one side and darker on another).
4. **Peel.** Inside a frame, or inside the chosen quadrilateral, look for a display behind a
   bezel: each side moves inwards to the first edge, across a thin ring without text.
5. **Choose.** Prefer lit-inside (+1) quadrilaterals and, among those, the largest one with text
   inside ("busy"); panels inside the page are smaller. A faint side (median step under 10 grey
   levels) counts against a quadrilateral in proportion. A frame is used only if nothing else is
   found (it is then off by the bezel width).
6. **Refine** the winner on a sharper copy (up to 2000 px) for sub-pixel corners.
7. **Nothing found:** return the whole image with ``detected=False``; the app lets users drag corners.

On 200 synthetic photos (``simulate/scene.py``): 95% found on light pages, 59% on dark-mode pages,
where the page edge is often truly invisible against the bezel; median corner error 0.03% of the
image diagonal when found.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import cv2
import numpy as np

from screenclean.utils.image import to_uint8

WORK_SIDE = 1000
REFINE_SIDE = 2000


@dataclass
class ScreenDetection:
    corners: np.ndarray  # (4, 2) float32: TL, TR, BR, BL in full-resolution pixels
    detected: bool
    confidence: float  # 0..1: how clearly the weakest side stands out
    method: str  # candidate source: "edges", "bright", "busy", "lines", or "none"
    polarity: int = 0  # +1 screen brighter than its surroundings, -1 darker, 0 not detected

    def as_dict(self) -> dict:
        return {
            "corners": np.asarray(self.corners).round(2).tolist(),
            "detected": self.detected,
            "confidence": round(float(self.confidence), 3),
            "method": self.method,
            "polarity": self.polarity,
        }


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left (clockwise on screen)."""
    pts = np.asarray(pts, np.float64).reshape(4, 2)
    c = pts.mean(axis=0)
    pts = pts[np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))]  # y points down: clockwise
    return np.roll(pts, -int(np.argmin(pts.sum(axis=1))), axis=0)


def full_image_corners(w: int, h: int) -> np.ndarray:
    return np.array([[-0.5, -0.5], [w - 0.5, -0.5], [w - 0.5, h - 0.5], [-0.5, h - 0.5]], np.float32)


def _area(q: np.ndarray) -> float:
    x, y = q[:, 0], q[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _angles_ok(q: np.ndarray, lo: float = 30.0, hi: float = 150.0) -> bool:
    """Convex, with every interior angle in [lo, hi] degrees."""
    u = np.roll(q, 1, axis=0) - q
    v = np.roll(q, -1, axis=0) - q
    nu, nv = np.linalg.norm(u, axis=1), np.linalg.norm(v, axis=1)
    if (nu < 1e-6).any() or (nv < 1e-6).any():
        return False
    ang = np.degrees(np.arccos(np.clip((u * v).sum(1) / (nu * nv), -1, 1)))
    cross = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    return bool(((ang >= lo) & (ang <= hi)).all() and ((cross > 0).all() or (cross < 0).all()))


def _shape_ok(q: np.ndarray, image_wh: tuple[int, int], lo: float = 0.4, hi: float = 2.6) -> bool:
    """Convex with sensible angles, and the real shape of a screen: from a tall phone (9:19.5)
    to an ultrawide monitor (21:9)."""
    if not _angles_ok(q):
        return False
    from screenclean.product.rectify import estimate_aspect

    return lo <= estimate_aspect(q, image_wh) <= hi


def _max_area_quad(poly: np.ndarray) -> np.ndarray | None:
    """The largest quadrilateral with corners on the polygon's vertices (polygon simplified to <= 12)."""
    poly = poly.reshape(-1, 2).astype(np.float32)
    if len(poly) < 4:
        return None
    peri = cv2.arcLength(poly, True)
    simple = poly
    for eps in (0.005, 0.01, 0.02, 0.03, 0.05, 0.08):
        simple = cv2.approxPolyDP(poly, eps * peri, True).reshape(-1, 2)
        if len(simple) <= 12:
            break
    if not 4 <= len(simple) <= 12:
        return None
    combos = np.array(list(itertools.combinations(range(len(simple)), 4)))
    q = simple[combos].astype(np.float64)  # (n, 4, 2)
    x, y = q[..., 0], q[..., 1]
    areas = 0.5 * np.abs((x * np.roll(y, -1, axis=1)).sum(1) - (y * np.roll(x, -1, axis=1)).sum(1))
    return order_corners(q[int(np.argmax(areas))])


class _Sampler:
    """Brightness steps across straight lines of a grey image (bilinear sampling)."""

    def __init__(self, gray: np.ndarray):
        self.gray = gray
        self.h, self.w = gray.shape

    def values(self, pts: np.ndarray) -> np.ndarray:
        """Sample at pts (..., 2); points outside the image give NaN."""
        shp = pts.shape[:-1]
        flat = pts.reshape(-1, 2).astype(np.float32)
        n = len(flat)
        cols = min(n, 4096)  # remap needs maps narrower than 32767
        if n % cols:
            flat = np.concatenate([flat, np.zeros((cols - n % cols, 2), np.float32)])
        flat = flat.reshape(-1, cols, 2)
        v = cv2.remap(self.gray, flat[..., 0], flat[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                      borderValue=np.nan)  # fmt: skip
        return v.reshape(-1)[:n].reshape(shp)

    def steps(self, p0: np.ndarray, p1: np.ndarray, n_in: np.ndarray, n: int = 64, delta: float = 2.0):
        """Inside-minus-outside brightness at n points along each line p0[i] -> p1[i]: (L, n)."""
        t = np.linspace(0.08, 0.92, n)[None, :, None]
        pts = p0[:, None, :] + t * (p1 - p0)[:, None, :]
        return self.values(pts + delta * n_in) - self.values(pts - delta * n_in)


def _inward_normals(q: np.ndarray) -> np.ndarray:
    """Unit normals of the 4 sides (i -> i+1) pointing into a clockwise-ordered quadrilateral."""
    d = np.roll(q, -1, axis=0) - q
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-12
    return np.stack([-d[:, 1], d[:, 0]], axis=1)


def side_evidence(s: _Sampler, q: np.ndarray, polarity, n: int = 96) -> tuple[np.ndarray, np.ndarray]:
    """Per side: median step times ``polarity``, and the share of samples with that sign in the
    side's *worst third* (an edge stitched from two objects has a gap somewhere).

    A side that leaves the image gets (-inf, 0).
    """
    pols = np.broadcast_to(np.asarray(polarity), (4,))[:, None]
    st = s.steps(q, np.roll(q, -1, axis=0), _inward_normals(q)[:, None, :], n) * pols
    off = np.isnan(st).any(axis=1)
    st = np.nan_to_num(st, nan=0.0)
    share = np.stack([(part > 0).mean(axis=1) for part in np.array_split(st, 3, axis=1)]).min(axis=0)
    return np.where(off, -np.inf, np.median(st, axis=1)), np.where(off, 0.0, share)


def passes(med: np.ndarray, share: np.ndarray, min_step: float, min_share: float) -> np.ndarray:
    """Which sides are real edges: a clear step along most of each third, or a faint one (half
    the step) that is consistent almost everywhere (at least 90%)."""
    return ((med >= min_step) & (share >= min_share)) | ((med >= min_step / 2) & (share >= 0.9))


def _intersect(p0, p1, q0, q1) -> np.ndarray | None:
    d1, d2 = p1 - p0, q1 - q0
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < 1e-9:
        return None
    t = ((q0[0] - p0[0]) * d2[1] - (q0[1] - p0[1]) * d2[0]) / den
    return p0 + t * d1


def _lines_to_quad(lines: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray | None:
    """Corners TL, TR, BR, BL from 4 side lines given in order (top, right, bottom, left)."""
    pts = []
    for i in range(4):
        a, b = lines[i - 1], lines[i]
        p = _intersect(a[0], a[1], b[0], b[1])
        if p is None:
            return None
        pts.append(p)
    return np.array(pts)


def snap(
    s: _Sampler, q: np.ndarray, polarity, radius: float, step: float, delta: float = 2.0
) -> np.ndarray | None:
    """Move each side (both ends independently, along its normal) onto the strongest edge of this
    polarity (one sign, or one per side).

    ``delta`` is how far each side of the line brightness is sampled. 2 px tolerates a rough start;
    1 px peaks exactly at a sharp edge (with 2 px the step is flat over a band, and the first
    position on the plateau would win).
    """
    nrm = _inward_normals(q)
    pols = np.broadcast_to(np.asarray(polarity), (4,))
    offs = np.arange(-radius, radius + 1e-9, step)
    a, b = (g.ravel() for g in np.meshgrid(offs, offs, indexing="ij"))
    lines = []
    for i in range(4):
        p0, p1, n_in = q[i], q[(i + 1) % 4], nrm[i]
        q0, q1 = p0 + a[:, None] * n_in, p1 + b[:, None] * n_in
        st = s.steps(q0, q1, n_in[None, None, :], n=48, delta=delta) * pols[i]
        med = np.median(np.where(np.isnan(st).any(1, keepdims=True), -np.inf, st), axis=1)  # off-image: never
        k = int(np.argmax(med))
        lines.append((q0[k], q1[k]))
    return _lines_to_quad(lines)  # sides are top, right, bottom, left


def _hough_quads(
    edges: np.ndarray, min_len: float, max_lines: int = 8, max_gap: float = 4.0
) -> list[np.ndarray]:
    """Quadrilaterals from pairs of long straight segments. A small ``max_gap`` keeps rows of text
    (edges broken between letters) from passing as lines; a screen's border is continuous."""
    segs = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=60, minLineLength=min_len, maxLineGap=max_gap)
    if segs is None:
        return []
    segs = segs.reshape(-1, 4).astype(np.float64)  # OpenCV 4: (N, 1, 4); OpenCV 5: (N, 4)
    length = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    groups: dict[str, list[tuple[np.ndarray, np.ndarray, float, float]]] = {"h": [], "v": []}
    for i in np.argsort(-length):
        p0, p1 = segs[i, :2], segs[i, 2:]
        ang = np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0])) % 180
        key = "h" if ang < 45 or ang > 135 else "v"
        mid = (p0 + p1) / 2
        pos = mid[1] if key == "h" else mid[0]
        if any(abs(pos - o[3]) < 10 and min(abs(ang - o[2]), 180 - abs(ang - o[2])) < 4 for o in groups[key]):
            continue
        if len(groups[key]) < max_lines:
            groups[key].append((p0, p1, ang, pos))
    quads = []
    for t, b in itertools.combinations(sorted(groups["h"], key=lambda g: g[3]), 2):
        for lft, r in itertools.combinations(sorted(groups["v"], key=lambda g: g[3]), 2):
            q = _lines_to_quad([(t[0], t[1]), (r[0], r[1]), (b[0], b[1]), (lft[0], lft[1])])
            if q is not None:
                quads.append(order_corners(q))
    return quads


def _candidates(gray8: np.ndarray, bil: np.ndarray, edges: np.ndarray, min_area: float):
    h, w = gray8.shape
    img_area = float(h * w)
    out: list[tuple[str, np.ndarray]] = []

    def add_contours(binary: np.ndarray, method: str, mode=cv2.RETR_LIST) -> None:
        contours, _ = cv2.findContours(binary, mode, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if len(c) < 4:
                continue
            hull = cv2.convexHull(c)
            if cv2.contourArea(hull) < min_area * img_area:
                continue
            q = _max_area_quad(hull)
            if q is not None:
                out.append((method, q))

    k3 = np.ones((3, 3), np.uint8)
    for lo, hi in ((10, 30), (30, 90), (60, 180)):
        e = edges if (lo, hi) == (30, 90) else cv2.Canny(bil, lo, hi)
        add_contours(cv2.dilate(e, k3), "edges")
        if lo <= 30:
            out.extend(("lines", q) for q in _hough_quads(e, 0.15 * min(h, w)))
    blur = cv2.GaussianBlur(gray8, (5, 5), 0)
    _, bright = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    add_contours(cv2.morphologyEx(bright, cv2.MORPH_CLOSE, close), "bright", cv2.RETR_EXTERNAL)
    busy = cv2.morphologyEx(gray8, cv2.MORPH_GRADIENT, k3)
    _, busy = cv2.threshold(busy, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    big = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31))
    add_contours(cv2.morphologyEx(busy, cv2.MORPH_CLOSE, big), "busy", cv2.RETR_EXTERNAL)
    return out


def _dedupe(cands: list[tuple[str, np.ndarray]], tol: float) -> list[tuple[str, np.ndarray]]:
    """Drop quadrilaterals whose 4 corners are all within ``tol`` of an earlier one."""
    if not cands:
        return []
    q = np.stack([c for _, c in cands])
    dist = np.linalg.norm(q[:, None] - q[None], axis=3).max(axis=2)  # (n, n): worst corner distance
    kept: list[int] = []
    for i in range(len(cands)):
        if not kept or dist[i, kept].min() >= tol:
            kept.append(i)
    return [cands[i] for i in kept]


def _gray(img: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    """Grey uint8 copy with the long side at most ``max_side``, and its scale (work px per image px)."""
    u8 = to_uint8(img)
    g = cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY) if u8.ndim == 3 else u8
    h, w = g.shape
    s = min(1.0, max_side / max(h, w))
    if s < 1.0:
        g = cv2.resize(g, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    return g, s


def _to_scale(q: np.ndarray, s_from: float, s_to: float) -> np.ndarray:
    """Convert pixel-centre coordinates between two resolutions of the same image."""
    return (q + 0.5) * (s_to / s_from) - 0.5


@dataclass
class _Kept:
    method: str
    quad: np.ndarray
    polarity: int
    strength: float  # weakest side's median step (grey levels)
    area: float
    busy: float  # share of edge pixels inside
    side_pol: np.ndarray | None = None  # per-side polarity when the sides disagree (a "mixed" frame)

    def sides(self) -> np.ndarray:
        return np.full(4, self.polarity) if self.side_pol is None else self.side_pol


def _inside_ring(outer: np.ndarray, inner: np.ndarray, max_gap: float, min_area_ratio: float = 0.6) -> bool:
    """``inner`` lies just inside ``outer``: every corner moved inwards by at most ``max_gap``."""
    if _area(inner) < min_area_ratio * _area(outer) or _area(inner) >= _area(outer) * 0.995:
        return False
    if np.linalg.norm(inner - outer, axis=1).max() > max_gap:
        return False
    poly = outer.astype(np.float32)
    return all(cv2.pointPolygonTest(poly, (float(x), float(y)), True) >= -2.0 for x, y in inner)  # 2 px slack


def _ring_busy(outer: np.ndarray, inner: np.ndarray, edges: np.ndarray) -> float:
    """Share of edge pixels in the ring between two nested quadrilaterals (its borders excluded)."""
    ring = _fill(outer, edges.shape) & (1 - _fill(inner, edges.shape))
    ring = cv2.erode(ring, np.ones((5, 5), np.uint8))
    return float(edges[ring > 0].mean() / 255.0) if ring.any() else 0.0


def _choose(
    kept: list[_Kept], edges: np.ndarray, max_ring: float, min_busy: float, full_strength: float = 10.0
) -> _Kept:
    """Largest lit-inside, busy quadrilateral; a quadrilateral with a faint side counts as smaller.
    If another one sits just inside it across a ring without text (a bezel), take that one."""
    pos = [k for k in kept if k.polarity > 0]
    pool = pos or kept
    busy = [k for k in pool if k.busy >= min_busy]
    best = max(busy or pool, key=lambda k: k.area * min(1.0, k.strength / full_strength))
    rings = [
        k for k in kept
        if k.busy >= min_busy and _inside_ring(best.quad, k.quad, max_ring)
        and _ring_busy(best.quad, k.quad, edges) < min_busy / 2
    ]  # fmt: skip
    if rings:  # ``best`` is the device and the ring is its bezel: take the display inside
        best = max(rings, key=lambda k: k.area)
    return best


def _fill(q: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(mask, q.round().astype(np.int32), 1)
    return mask


def _line_grid(s: _Sampler, p0, p1, n_in, polarity: int, offs: np.ndarray, n: int = 48):
    """Evidence for lines whose two ends move by ``offs[i]`` and ``offs[j]`` along ``n_in``.

    Returns (median step, worst-third share, line ends (.., 2, 2), shift of end 0, shift of end 1),
    each on the (i, j) grid; lines far from parallel to the side get -inf.
    """
    ia, ib = np.meshgrid(np.arange(len(offs)), np.arange(len(offs)), indexing="ij")
    sa, sb = offs[ia], offs[ib]
    sel = (np.abs(sa - sb) <= np.maximum(6.0, 0.5 * np.maximum(np.abs(sa), np.abs(sb)))).ravel()
    q0 = p0 + sa.ravel()[sel][:, None] * n_in
    q1 = p1 + sb.ravel()[sel][:, None] * n_in
    st = s.steps(q0, q1, n_in[None, None, :], n=n) * polarity
    bad = np.isnan(st).any(axis=1)
    st = np.nan_to_num(st, nan=0.0)
    med = np.full(ia.shape, -np.inf)
    med.ravel()[sel] = np.where(bad, -np.inf, np.median(st, axis=1))
    share = np.zeros(ia.shape)
    share.ravel()[sel] = np.stack([(p > 0).mean(axis=1) for p in np.array_split(st, 3, axis=1)]).min(0)
    ends = np.zeros(ia.shape + (2, 2))
    ends.reshape(-1, 2, 2)[sel] = np.stack([q0, q1], axis=1)
    return med, share, ends, sa, sb


def peel(
    s: _Sampler,
    q: np.ndarray,
    edges: np.ndarray,
    max_ring: float,
    min_step: float,
    min_share: float,
    min_busy: float,
    min_ratio: float = 0.75,
) -> tuple[np.ndarray, int, float] | None:
    """Look just inside ``q`` for a display behind a bezel: a quadrilateral with text inside, across
    a thin ring without text.

    A side of ``q`` that is a real edge moves inwards to the *first* strong straight edge. A side
    that is not (the device edge can vanish against something equally dark behind it, so that
    side of ``q`` is only a guess) takes the strongest edge nearby, on either side of the guess.
    Returns (inner quad, polarity, weakest side step), or None when there is no such ring.
    """
    nrm = _inward_normals(q)
    step = 2.0
    inward = np.arange(0.0, max_ring + 1e-9, step)
    around = np.arange(-max_ring / 2, max_ring + 1e-9, step)
    real = np.zeros(4, bool)
    for pol in (1, -1):
        real |= passes(*side_evidence(s, q, pol), min_step, min_share)
    found = []
    for pol in (1, -1):
        lines = []
        for i in range(4):
            p0, p1, n_in = q[i], q[(i + 1) % 4], nrm[i]
            if real[i]:
                med, share, ends, sa, sb = _line_grid(s, p0, p1, n_in, pol, inward)
                # an edge: a local maximum when the line shifts inwards or outwards (not q's own edge)
                peak = np.zeros_like(med, bool)
                peak[1:-1, 1:-1] = (med[1:-1, 1:-1] >= med[:-2, :-2]) & (med[1:-1, 1:-1] >= med[2:, 2:])
                ok = passes(med, share, min_step, min_share) & peak & (np.maximum(sa, sb) >= 2 * step)
                if not ok.any():
                    break
                depth = np.where(ok, sa + sb, np.inf)
                k = int(np.argmax(np.where(ok & (depth <= depth.min() + 2 * step), med, -np.inf)))
            else:
                med, share, ends, sa, sb = _line_grid(s, p0, p1, n_in, pol, around)
                ok = passes(med, share, min_step, min_share)
                if not ok.any():
                    break
                k = int(np.argmax(np.where(ok, med, -np.inf)))
            lines.append(tuple(ends.reshape(-1, 2, 2)[k]))
        if len(lines) < 4:
            continue
        inner = _lines_to_quad(lines)
        if inner is None:
            continue
        inner = snap(s, order_corners(inner), pol, radius=2.0, step=0.25, delta=1.0)
        if inner is None:
            continue
        inner = order_corners(inner)
        if not _shape_ok(inner, edges.shape[::-1]) or not min_ratio * _area(q) <= _area(
            inner
        ) < 0.995 * _area(q):
            continue
        med, share = side_evidence(s, inner, pol)
        if not passes(med, share, min_step, min_share).all():
            continue
        inner_busy = float(edges[_fill(inner, edges.shape) > 0].mean() / 255.0)
        if _ring_busy(q, inner, edges) < min_busy / 2 and inner_busy >= min_busy:
            found.append((inner, pol, float(med.min())))
    return max(found, key=lambda f: _area(f[0])) if found else None


def detect_screen(
    img: np.ndarray,
    min_area: float = 0.15,
    min_step: float = 4.0,
    min_share: float = 0.7,
    max_ring: float = 0.08,
    min_busy: float = 0.01,
) -> ScreenDetection:
    """Find the screen's content area in an RGB photo (uint8 or float in [0, 1]).

    - ``min_step``: a side's median brightness step (grey levels, 0-255) must reach this;
    - ``min_share``: share of each third of every side that must show the step (rules out half-edges);
    - ``max_ring``: widest bezel, as a share of the image diagonal;
    - ``min_busy``: share of edge pixels (text) inside for a quadrilateral to count as "busy".
    """
    h, w = img.shape[:2]
    gray8, s = _gray(img, WORK_SIDE)
    gh, gw = gray8.shape
    diag = float(np.hypot(gw, gh))
    bil = cv2.bilateralFilter(gray8, 9, 40, 7)
    edges = cv2.Canny(bil, 30, 90)
    sampler = _Sampler(cv2.GaussianBlur(gray8.astype(np.float32), (0, 0), 1.0))
    cands = [(m, q) for m, q in _candidates(gray8, bil, edges, min_area) if _shape_ok(q, (gw, gh))]
    cands = _dedupe(cands, 0.01 * diag)

    # Quick check of all candidates at once: is every side near an edge of either polarity?
    if cands:
        quads = np.stack([q for _, q in cands])  # (m, 4, 2)
        nrm = np.stack([_inward_normals(q) for q in quads]).reshape(-1, 1, 2)
        st = sampler.steps(quads.reshape(-1, 2), np.roll(quads, -1, axis=1).reshape(-1, 2), nrm, n=32)
        quick = np.median(np.nan_to_num(st, nan=0.0), axis=1).reshape(-1, 4)  # (m, 4)
    kept: list[_Kept] = []
    frames: list[_Kept] = []
    for j, (method, q) in enumerate(cands):
        for pol in (1, -1):
            if (quick[j] * pol).min() < min_step / 2:  # nowhere near an edge of this polarity
                continue
            r = snap(sampler, q, pol, radius=2.5, step=0.5)
            r = None if r is None else snap(sampler, order_corners(r), pol, radius=2.0, step=0.25, delta=1.0)
            if r is None:
                continue
            r = order_corners(r)
            if not _shape_ok(r, (gw, gh)) or _area(r) < min_area * gw * gh:
                continue
            med, share = side_evidence(sampler, r, pol)
            ok = passes(med, share, min_step, min_share)
            if ok.all():
                busy = float(edges[_fill(r, edges.shape) > 0].mean() / 255.0)
                kept.append(_Kept(method, r, pol, float(med.min()), _area(r), busy))
            elif ok.sum() == 3 and med.min() >= min_step / 2:  # e.g. a stand interrupts the bottom edge
                busy = float(edges[_fill(r, edges.shape) > 0].mean() / 255.0)
                frames.append(_Kept("frame", r, pol, float(med.min()) / 2, _area(r), busy))
        # The whole device, when the room is brighter than the bezel on some sides and darker on others.
        signs = np.sign(quick[j])
        if np.abs(quick[j]).min() >= min_step / 2 and abs(signs.sum()) < 4:
            r = snap(sampler, q, signs, radius=2.5, step=0.5)
            r = (
                None
                if r is None
                else snap(sampler, order_corners(r), signs, radius=2.0, step=0.25, delta=1.0)
            )
            if r is not None and np.allclose(order_corners(r), r) and _shape_ok(r, (gw, gh)):
                med, share = side_evidence(sampler, r, signs)
                if passes(med, share, min_step, min_share).all() and _area(r) >= min_area * gw * gh:
                    busy = float(edges[_fill(r, edges.shape) > 0].mean() / 255.0)
                    frames.append(_Kept("frame", r, 0, float(med.min()) / 2, _area(r), busy, signs))

    # The display may be found inside a frame; a frame itself is only a last resort.
    frames.sort(key=lambda f: -f.area)
    for f in frames[:4]:
        peeled = peel(sampler, f.quad, edges, max_ring * diag, min_step, min_share, min_busy)
        if peeled is not None:
            inner, pol, strength = peeled
            busy = float(edges[_fill(inner, edges.shape) > 0].mean() / 255.0)
            kept.append(_Kept("frame+peel", inner, pol, strength, _area(inner), busy))
    if not kept:
        busy_frames = [f for f in frames if f.busy >= min_busy]
        if not busy_frames:
            return ScreenDetection(full_image_corners(w, h), False, 0.0, "none", 0)
        kept = busy_frames[:1]  # the largest: most likely the whole device, off by the bezel width
    kept = _dedupe_kept(kept, 0.005 * diag)
    best = _choose(kept, edges, max_ring * diag, min_busy)
    q, pol, strength, method = best.quad, best.polarity, best.strength, best.method
    side_pol = best.sides()
    if method == "frame":
        pass  # already tried
    elif not method.endswith("peel"):
        peeled = peel(sampler, q, edges, max_ring * diag, min_step, min_share, min_busy)
        if peeled is not None:  # ``best`` was the whole device: keep the display inside its bezel
            q, pol, strength = peeled
            side_pol = np.full(4, pol)
            method += "+peel"

    # Refine on a sharper copy.
    gray_r, sr = _gray(img, REFINE_SIDE)
    if sr > s * 1.2:
        fine = _Sampler(cv2.GaussianBlur(gray_r.astype(np.float32), (0, 0), 0.7))
        qr = _to_scale(q, s, sr)
        r = snap(fine, qr, side_pol, radius=1.5 * sr / s, step=0.5)
        r = None if r is None else snap(fine, order_corners(r), side_pol, radius=0.5, step=0.125, delta=1.0)
        if r is not None:
            r = order_corners(r)
            if np.linalg.norm(r - qr, axis=1).max() < 3 * sr / s:
                qr = r
        q, s = qr, sr
    corners = _to_scale(q, s, 1.0).astype(np.float32)
    conf = float(np.clip(strength / 40.0, 0, 1))
    return ScreenDetection(corners, True, conf, method, pol)


def _dedupe_kept(kept: list[_Kept], tol: float) -> list[_Kept]:
    order = sorted(range(len(kept)), key=lambda i: -kept[i].strength)
    out: list[_Kept] = []
    for i in order:
        k = kept[i]
        if all(o.polarity != k.polarity or np.linalg.norm(o.quad - k.quad, axis=1).max() >= tol for o in out):
            out.append(k)
    return out
