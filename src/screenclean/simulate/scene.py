"""Whole-photo scenes: a screen showing a page, somewhere in a room, photographed by a phone.

Used to test screen detection and the scan pipeline without real photos. ``make_scene(page, rng)``
returns the photo and where the page's corners really are in it.

A simplified version of ``moire_sim`` (which models 512 px crops in detail), for whole photos:

1. the room: smooth lighting, clutter (boxes, discs, a desk, sometimes a bright window), fine texture;
2. the device: page -> sensor homography from a random camera pose, a bezel around the page
   (usually dark, sometimes silver), sometimes a stand, glare on the glass;
3. the page and the screen's subpixel pattern are blurred by the lens, then point-sampled at
   sensor resolution (``ss`` times the output size): that is where moiré appears;
4. sensor + ISP: vignetting, exposure, Bayer mosaic, noise, demosaic, white balance, tone curve,
   sometimes defocus or shake, downscale to the output size, JPEG.

All light levels are linear and relative to the screen's white (1.0).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from screenclean.simulate import isp, sensor
from screenclean.simulate.camera import rotation
from screenclean.simulate.display import subpixel_mask
from screenclean.utils.image import to_float01

DEFAULTS: dict[str, Any] = {
    "area_frac": [0.2, 0.75],  # content area / photo area
    "yaw_deg": 35.0,
    "pitch_deg": 30.0,
    "roll_deg": 12.0,
    "focal_rel": [0.75, 1.3],  # focal length / sensor width (phone main cameras: ~0.8)
    "bezel_rel": [0.005, 0.05],  # bezel width / page width
    "portrait_prob": 0.2,
    "room_level": [0.03, 0.6],  # room brightness vs screen white (log-uniform)
    "screen_brightness": [0.6, 1.0],
    "glare": [0.0, 0.08],
    "psf_sigma": [0.3, 0.6],  # lens blur, sensor pixels (the pixel area is added on top)
    "blur_prob": 0.4,
    "blur_sigma": [0.5, 2.0],  # sensor pixels
    "photons": [300, 3000],
    "jpeg_quality": [75, 95],
}


@dataclass
class Scene:
    photo: np.ndarray  # uint8 RGB, (h, w, 3)
    corners: np.ndarray  # (4, 2) float32: page corners TL, TR, BR, BL in photo pixels
    H: np.ndarray  # page pixels -> photo pixels
    meta: dict[str, Any] = field(default_factory=dict)


def _log_u(rng: np.random.Generator, lo_hi) -> float:
    return float(np.exp(rng.uniform(np.log(lo_hi[0]), np.log(lo_hi[1]))))


def _tint(rng: np.random.Generator, level: float, sat: float = 0.25) -> np.ndarray:
    return (level * (1.0 + rng.uniform(-sat, sat, 3))).astype(np.float32)


def room(rng: np.random.Generator, w: int, h: int, cfg: dict[str, Any] | None = None) -> np.ndarray:
    """A plausible indoor background in linear light: (h, w, 3) float32."""
    cfg = {**DEFAULTS, **(cfg or {})}
    level = _log_u(rng, cfg["room_level"])
    xx = np.linspace(-1, 1, w, dtype=np.float32)[None, :]
    yy = np.linspace(-1, 1, h, dtype=np.float32)[:, None]
    theta = rng.uniform(0, 2 * np.pi)
    shade = 1.0 + rng.uniform(-0.5, 0.5) * (xx * np.cos(theta) + yy * np.sin(theta)) / 1.5
    lowfreq = cv2.resize(rng.normal(0, 1, (6, 8)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    img = _tint(rng, level)[None, None] * (shade * (1.0 + 0.12 * lowfreq))[..., None]

    if rng.random() < 0.6:  # a desk across the lower part, with a faint wood grain
        y0 = rng.uniform(0.55, 0.95) * h
        tilt = rng.uniform(-0.08, 0.08) * h
        edge = y0 + tilt * (xx[0] + 1) / 2
        desk = np.arange(h)[:, None] > edge[None, :]
        grain = 1.0 + 0.1 * np.sin(2 * np.pi * (xx * rng.uniform(3, 12) + 0.3 * lowfreq))
        img = np.where(
            desk[..., None], _tint(rng, level * rng.uniform(0.4, 1.5), 0.4) * grain[..., None], img
        )

    for _ in range(int(rng.integers(2, 9))):  # boxes, posters, lamps, a window
        c = _tint(rng, _log_u(rng, (0.01, 1.5)), 0.4)
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        size = rng.uniform(0.05, 0.4) * min(w, h)
        if rng.random() < 0.7:
            ang = rng.uniform(-0.3, 0.3)
            half = np.array([size * rng.uniform(0.5, 1.5), size * rng.uniform(0.5, 1.5)]) / 2
            box = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * half
            rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
            pts = (box @ rot.T + [cx, cy]).round().astype(np.int32)
            cv2.fillConvexPoly(img, pts, c.tolist(), lineType=cv2.LINE_AA)
        else:
            cv2.circle(img, (int(cx), int(cy)), int(size / 2), c.tolist(), -1, lineType=cv2.LINE_AA)
    img *= 1.0 + 0.02 * rng.standard_normal(img.shape[:2], dtype=np.float32)[..., None]  # surface texture
    return np.clip(img, 0, None).astype(np.float32)


def _project(Hm: np.ndarray, pts: np.ndarray) -> np.ndarray:
    p = np.c_[pts, np.ones(len(pts))] @ Hm.T
    return p[:, :2] / p[:, 2:]


def _poly_area(q: np.ndarray) -> float:
    x, y = q[:, 0], q[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def page_corners(page_wh: tuple[int, int], pad: float = 0.0) -> np.ndarray:
    """The page's outer corners (TL, TR, BR, BL) in page pixel coordinates (pixel centres at integers)."""
    pw, ph = page_wh
    return np.array(
        [[-0.5 - pad, -0.5 - pad], [pw - 0.5 + pad, -0.5 - pad], [pw - 0.5 + pad, ph - 0.5 + pad],
         [-0.5 - pad, ph - 0.5 + pad]], np.float64,
    )  # fmt: skip


def sample_homography(
    rng: np.random.Generator,
    page_wh: tuple[int, int],
    sensor_wh: tuple[int, int],
    bezel_px: float,
    cfg: dict[str, Any] | None = None,
) -> np.ndarray:
    """Page -> sensor homography for a random pose; the whole device stays inside the frame.

    The camera's optical axis goes through the sensor centre, as in real phones; the screen is
    placed off-centre by moving it in 3D (which also changes the perspective, as it should).
    """
    cfg = {**DEFAULTS, **(cfg or {})}
    sw, sh = sensor_wh
    pw, ph = page_wh
    f = sw * rng.uniform(*cfg["focal_rel"])
    K = np.array([[f, 0, (sw - 1) / 2], [0, f, (sh - 1) / 2], [0, 0, 1.0]])
    center = np.array([[1, 0, -(pw - 1) / 2], [0, 1, -(ph - 1) / 2], [0, 0, 1.0]])
    margin = 0.02 * min(sw, sh)
    shrink = 1.0  # after each failed placement, aim a little smaller
    for _ in range(100):
        R = rotation(*(rng.uniform(-cfg[k], cfg[k]) for k in ("yaw_deg", "pitch_deg", "roll_deg")))

        def hom(d: float, tx: float = 0.0, ty: float = 0.0, R=R) -> np.ndarray:
            # plane point (X, Y, 0) -> camera R [X, Y, 0] + (tx, ty, d): the screen moves in 3D
            return K @ np.column_stack([R[:, 0], R[:, 1], [tx, ty, d]]) @ center

        target = rng.uniform(*cfg["area_frac"]) * sw * sh * shrink
        shrink *= 0.95
        d = f
        for _ in range(3):  # area ~ 1 / d^2
            d *= np.sqrt(_poly_area(_project(hom(d), page_corners(page_wh))) / target)
        outer = _project(hom(d), page_corners(page_wh, pad=bezel_px * 1.2))
        lo, hi = outer.min(0), outer.max(0)
        room_x = (margin - lo[0], sw - 1 - margin - hi[0])
        room_y = (margin - lo[1], sh - 1 - margin - hi[1])
        if room_x[0] > room_x[1] or room_y[0] > room_y[1]:
            continue
        # sideways move that shifts the screen centre by about (dx, dy) pixels
        dx, dy = rng.uniform(*room_x), rng.uniform(*room_y)
        Hm = hom(d, dx * d / f, dy * d / f)
        outer = _project(Hm, page_corners(page_wh, pad=bezel_px * 1.2))
        if (outer.min(0) < margin).any() or (outer.max(0) > [sw - 1 - margin, sh - 1 - margin]).any():
            continue
        return Hm / Hm[2, 2]
    raise RuntimeError("could not place the screen inside the frame")


def _wrap_blur(tile: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur of a periodic tile (wraps around its edges)."""
    if sigma < 0.05:
        return tile
    pad = int(np.ceil(3 * sigma))
    reps = -(-pad // tile.shape[0])
    big = np.tile(tile, (2 * reps + 1, 2 * reps + 1, 1))
    big = cv2.GaussianBlur(big, (0, 0), sigma)
    n0, n1 = reps * tile.shape[0], reps * tile.shape[1]
    return np.ascontiguousarray(big[n0 : n0 + tile.shape[0], n1 : n1 + tile.shape[1]])


class _PageMaps:
    """Page coordinates (u, v) seen by every sensor pixel, optionally shifted by a sub-pixel offset."""

    def __init__(self, Hinv: np.ndarray, w: int, h: int):
        self.Hinv = Hinv.astype(np.float32)
        xs = np.arange(w, dtype=np.float32)[None, :]
        ys = np.arange(h, dtype=np.float32)[:, None]
        a = self.Hinv
        self.nu = a[0, 0] * xs + a[0, 1] * ys + a[0, 2]
        self.nv = a[1, 0] * xs + a[1, 1] * ys + a[1, 2]
        self.den = a[2, 0] * xs + a[2, 1] * ys + a[2, 2]

    def __call__(self, dx: float = 0.0, dy: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        a = self.Hinv
        den = self.den + (a[2, 0] * dx + a[2, 1] * dy)
        u = (self.nu + (a[0, 0] * dx + a[0, 1] * dy)) / den
        v = (self.nv + (a[1, 0] * dx + a[1, 1] * dy)) / den
        return u, v


def make_scene(
    page_rgb: np.ndarray,
    rng: np.random.Generator,
    out_size: tuple[int, int] = (1600, 1200),
    ss: int = 2,
    cfg: dict[str, Any] | None = None,
) -> Scene:
    """Photograph ``page_rgb`` (uint8/float RGB) on a screen in a random room.

    ``out_size`` is (w, h) for a landscape photo (swapped for portrait shots); the sensor works at
    ``ss`` times that resolution and the result is downscaled, like a phone's resized photo.
    """
    cfg = {**DEFAULTS, **(cfg or {})}
    ow, oh = out_size
    portrait = rng.random() < cfg["portrait_prob"]
    if portrait:
        ow, oh = oh, ow
    sw, sh = ow * ss, oh * ss
    page = to_float01(page_rgb)
    ph, pw = page.shape[:2]
    bezel = float(rng.uniform(*cfg["bezel_rel"]) * pw)
    chin = bezel * (rng.uniform(1.0, 3.0) if rng.random() < 0.4 else 1.0)  # laptops: thicker bottom
    Hm = sample_homography(rng, (pw, ph), (sw, sh), max(bezel, chin), cfg)
    Hinv = np.linalg.inv(Hm)

    img = cv2.resize(
        room(rng, ow, oh, cfg), (sw, sh), interpolation=cv2.INTER_LINEAR
    )  # smooth: low res is fine
    dark_bezel = rng.random() < 0.85
    bezel_c = _tint(rng, _log_u(rng, (0.004, 0.04)) if dark_bezel else rng.uniform(0.15, 0.5), 0.1)
    if rng.random() < 0.4:  # monitor stand
        neck = pw * rng.uniform(0.05, 0.15)
        stand = np.array([[pw / 2 - neck, ph], [pw / 2 + neck, ph], [pw / 2 + neck * 1.3, ph * 1.5],
                          [pw / 2 - neck * 1.3, ph * 1.5]])  # fmt: skip
        cv2.fillConvexPoly(img, _project(Hm, stand).round().astype(np.int32), (bezel_c * 1.5).tolist())
    body = np.array(page_corners((pw, ph)), np.float64)
    body[:, 0] += np.array([-bezel, bezel, bezel, -bezel])
    body[:, 1] += np.array([-bezel, -bezel, chin, chin])
    cv2.fillConvexPoly(img, (_project(Hm, body) * 16).round().astype(np.int32), bezel_c.tolist(),
                       lineType=cv2.LINE_AA, shift=4)  # fmt: skip

    # The screen: page colour times the subpixel pattern, both blurred by the lens (and the pixel's
    # light-collecting area) *before* the sensor samples them. Blur widths are in page pixels, using
    # the scale at the screen centre.
    brightness = rng.uniform(*cfg["screen_brightness"])
    layout = ("rgb", "bgr", "pentile")[int(rng.choice(3, p=[0.6, 0.2, 0.2]))]
    psf = float(np.hypot(rng.uniform(*cfg["psf_sigma"]), 0.9 / np.sqrt(12)))  # sensor pixels
    centre = np.array(
        [[(pw - 1) / 2, (ph - 1) / 2], [(pw + 1) / 2, (ph - 1) / 2], [(pw - 1) / 2, (ph + 1) / 2]]
    )
    c0, cx, cy = _project(Hm, centre)
    (ax, ay), (bx, by) = cx - c0, cy - c0
    scale = float(np.sqrt(abs(ax * by - ay * bx)))  # sensor pixels per page pixel
    sigma_page = psf / scale
    up = 8
    mask = subpixel_mask(layout, up, round(rng.uniform(0.6, 0.9), 3), round(rng.uniform(0.75, 0.95), 3))
    mask = _wrap_blur(mask, sigma_page * up)
    lin = cv2.GaussianBlur((page**2.2 * brightness).astype(np.float32), (0, 0), sigma_page)
    maps = _PageMaps(Hinv, sw, sh)
    u, v = maps()
    screen = cv2.remap(lin, u, v, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    screen *= cv2.remap(  # the subpixel pattern repeats every 2 page pixels: wrap around the 2x2 tile
        mask, (u + 0.5) * up - 0.5, (v + 0.5) * up - 0.5, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP
    )
    inside = ((u >= -0.5) & (u < pw - 0.5) & (v >= -0.5) & (v < ph - 0.5)).astype(np.float32)
    inside = cv2.GaussianBlur(inside, (0, 0), 0.5)  # anti-aliased page edge
    img = cv2.blendLinear(img, screen, 1.0 - inside, inside)
    del screen, inside

    # Camera: lens blur, then the sensor (vignetting and glare are grey, so they can act on the raw).
    blur = 0.0
    if rng.random() < cfg["blur_prob"]:  # defocus or shake
        blur = rng.uniform(*cfg["blur_sigma"])
        img = cv2.GaussianBlur(img, (0, 0), blur)
    cfa = ("RGGB", "BGGR")[int(rng.integers(2))]
    raw = sensor.mosaic(img, cfa)
    del img
    xx = np.linspace(-1, 1, sw, dtype=np.float32)[None, :] ** 2
    yy = np.linspace(-1, 1, sh, dtype=np.float32)[:, None] ** 2
    k = rng.uniform(0, 0.35) / 2
    raw *= (1.0 - k * xx) - k * yy
    glare_amp = rng.uniform(*cfg["glare"])
    if glare_amp > 0:  # soft reflection of a lamp or window on the glass
        gx, gy = rng.uniform(0, pw), rng.uniform(0, ph)
        r2 = 2 * (rng.uniform(0.2, 0.8) * pw) ** 2
        glass = (u > -0.5 - bezel) & (u < pw - 0.5 + bezel) & (v > -0.5 - bezel) & (v < ph - 0.5 + chin)
        raw += glare_amp * np.exp(-((u - gx) ** 2 + (v - gy) ** 2) / r2) * glass
    del u, v
    raw = sensor.expose(raw, 99.5, rng.uniform(0.8, 0.95), floor=0.3 * brightness)
    photons, read = _log_u(rng, cfg["photons"]), rng.uniform(0.001, 0.005)
    sd = np.sqrt(raw / photons + read**2)  # shot + read noise, Gaussian approximation (fast)
    raw = np.clip(raw + sd * rng.standard_normal(raw.shape, dtype=np.float32), 0, 1)
    rgb = isp.demosaic(raw, cfa, ("bilinear", "ea")[int(rng.integers(2))])
    if ss > 1:
        rgb = cv2.resize(rgb, (ow, oh), interpolation=cv2.INTER_AREA)
    rgb = isp.white_balance(rgb, 1.0 + rng.uniform(-0.06, 0.06, 3))
    rgb = isp.tone(rgb, rng.uniform(2.0, 2.4), rng.uniform(0.95, 1.15))
    quality = int(rng.integers(cfg["jpeg_quality"][0], cfg["jpeg_quality"][1] + 1))
    _, photo = isp.jpeg(rgb, quality)

    down = np.array([[1 / ss, 0, 0.5 / ss - 0.5], [0, 1 / ss, 0.5 / ss - 0.5], [0, 0, 1.0]])
    H_photo = down @ Hm
    H_photo /= H_photo[2, 2]
    corners = _project(H_photo, page_corners((pw, ph))).astype(np.float32)
    meta = {
        "portrait": bool(portrait),
        "layout": layout,
        "bezel_px": bezel,
        "chin_px": chin,
        "dark_bezel": bool(dark_bezel),
        "brightness": float(brightness),
        "glare": float(glare_amp),
        "blur_sigma": float(blur),
        "jpeg_quality": quality,
        "area_frac": _poly_area(corners.astype(np.float64)) / (ow * oh),
    }
    return Scene(photo=photo, corners=corners, H=H_photo, meta=meta)
