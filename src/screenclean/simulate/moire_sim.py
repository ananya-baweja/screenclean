"""End-to-end moiré simulator: clean page -> (simulated phone photo crop, aligned clean target).

``simulate(clean_rgb, rng, cfg)`` returns ``(moire_rgb, gt_rgb, meta)``. Both images are uint8
RGB crops of ``cfg["crop"]`` pixels, and ``meta`` holds every sampled parameter plus the
page->crop homography. Steps (see configs/sim/default.yaml for the ranges):

1. display: subpixel layout, fill factors, gamma, brightness (``display.py``)
2. geometry: camera pose -> homography, lens distortion (``camera.py``)
3. optics + sensor: Gaussian PSF, Bayer point sampling, auto-exposure, noise (``sensor.py``)
4. ISP: demosaic, white balance, tone curve, sharpening, JPEG (``isp.py``)
5. ground truth: the clean page under the same geometry, anti-aliased, with nothing else applied.

Everything random comes from ``rng``, so the same seed gives the same pair.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from screenclean.simulate import isp, sensor
from screenclean.simulate.camera import Pose, crop_to_page_maps, page_to_crop, sample_pose
from screenclean.simulate.display import render_display
from screenclean.utils.image import to_float01, to_uint8

DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "sim" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    return yaml.safe_load(Path(path or DEFAULT_CONFIG).read_text(encoding="utf-8"))


@dataclass
class SimParams:
    layout: str
    supersample: int
    fill_x: float
    fill_y: float
    brightness: float
    psf_sigma: float
    cfa: str
    exposure_target: float
    photons: float
    read_noise: float
    demosaic: str
    wb_gains: list[float]
    gamma: float
    contrast: float
    sharpen: float
    sharpen_sigma: float
    jpeg_quality: int


def _u(rng: np.random.Generator, lo_hi) -> float:
    return float(rng.uniform(lo_hi[0], lo_hi[1]))


def _choice(rng: np.random.Generator, options):
    if isinstance(options, dict):
        names, probs = list(options), np.array(list(options.values()), float)
        return names[int(rng.choice(len(names), p=probs / probs.sum()))]
    return options[int(rng.integers(len(options)))]


def sample_params(rng: np.random.Generator, cfg: dict[str, Any]) -> tuple[SimParams, Pose]:
    d, o, s, i = cfg["display"], cfg["optics"], cfg["sensor"], cfg["isp"]
    pose = sample_pose(rng, cfg["geometry"], int(cfg["crop"]))
    params = SimParams(
        layout=_choice(rng, d["layouts"]),
        supersample=int(d.get("supersample", 4)),
        fill_x=_u(rng, d["fill_x"]),
        fill_y=_u(rng, d["fill_y"]),
        brightness=_u(rng, d["brightness"]),
        psf_sigma=_u(rng, o["psf_sigma"]),
        cfa=_choice(rng, s["cfa"]),
        exposure_target=_u(rng, s["exposure_target"]),
        photons=float(np.exp(_u(rng, np.log(s["photons"])))),  # log-uniform
        read_noise=_u(rng, s["read_noise"]),
        demosaic=_choice(rng, i["demosaic"]),
        wb_gains=[float(g) for g in 1.0 + rng.uniform(-i["wb_jitter"], i["wb_jitter"], 3)],
        gamma=_u(rng, i["gamma"]),
        contrast=_u(rng, i["contrast"]),
        sharpen=_u(rng, i["sharpen"]),
        sharpen_sigma=_u(rng, i["sharpen_sigma"]),
        jpeg_quality=int(round(_u(rng, i["jpeg_quality"]))),
    )
    return params, pose


def _page_region(page: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    """``page[y0:y1, x0:x1]``, padding with black (the screen bezel) where it leaves the page."""
    h, w = page.shape[:2]
    cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    region = page[cy0:cy1, cx0:cx1]
    return cv2.copyMakeBorder(region, cy0 - y0, y1 - cy1, cx0 - x0, x1 - cx1, cv2.BORDER_CONSTANT, value=0)


def simulate(
    clean_rgb: np.ndarray,
    rng: np.random.Generator,
    cfg: dict[str, Any],
    center_uv: tuple[float, float] | None = None,
    overrides: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Simulate one photographed crop of ``clean_rgb`` (a page, uint8 or float RGB).

    ``center_uv`` picks the page point at the crop centre (random if None). ``overrides``
    replaces sampled parameters by name (tests and figures), e.g. ``{"psf_sigma": 3.0}``.
    """
    page = to_float01(clean_rgb)
    crop = int(cfg["crop"])
    params, pose = sample_params(rng, cfg)
    for k, v in (overrides or {}).items():
        if hasattr(params, k):
            setattr(params, k, v)
        elif hasattr(pose, k):
            setattr(pose, k, v)
        else:
            raise KeyError(f"unknown simulator parameter {k!r}")
    h, w = page.shape[:2]
    if center_uv is None:
        center_uv = (float(rng.uniform(0, w - 1)), float(rng.uniform(0, h - 1)))

    H = page_to_crop(pose, center_uv, crop)
    map_u, map_v = crop_to_page_maps(pose, H, crop)

    # Blur width: lens PSF plus the pixel's light-collecting area (a box, approximated as a Gaussian).
    aperture = float(cfg["optics"].get("pixel_aperture", 0.9))
    sigma_sensor = float(np.hypot(params.psf_sigma, aperture / np.sqrt(12)))
    up = params.supersample
    sigma_page = sigma_sensor / pose.scale  # in display pixels
    margin = int(np.ceil(4 * sigma_page)) + 3
    x0, y0 = int(np.floor(map_u.min())) - margin, int(np.floor(map_v.min())) - margin
    x1, y1 = int(np.ceil(map_u.max())) + margin + 1, int(np.ceil(map_v.max())) + margin + 1
    region = _page_region(page, x0, y0, x1, y1)

    # ---- simulated photo
    gamma_disp = float(cfg["display"].get("gamma", 2.2))
    light = (region**gamma_disp) * params.brightness
    raster = render_display(light, params.layout, up, params.fill_x, params.fill_y, origin=(x0, y0))
    raster = sensor.blur(raster, sigma_page * up)
    rx = (map_u - x0 + 0.5) * up - 0.5
    ry = (map_v - y0 + 0.5) * up - 0.5
    rgb_lin = sensor.sample(raster, rx, ry)
    raw = sensor.mosaic(rgb_lin, params.cfa)
    s_cfg = cfg["sensor"]
    floor = float(s_cfg.get("exposure_floor", 0.35)) * params.brightness  # vs the screen's white level
    raw = sensor.expose(raw, float(s_cfg.get("exposure_percentile", 98)), params.exposure_target, floor)
    raw = sensor.add_noise(raw, rng, params.photons, params.read_noise)
    rgb = isp.demosaic(raw, params.cfa, params.demosaic)
    rgb = isp.white_balance(rgb, np.array(params.wb_gains))
    rgb = isp.tone(rgb, params.gamma, params.contrast)
    rgb = isp.sharpen(rgb, params.sharpen, params.sharpen_sigma)
    jpeg_bytes, moire = isp.jpeg(rgb, params.jpeg_quality)

    # ---- ground truth: same geometry, anti-aliased resampling of the clean page
    gt_src = region
    if pose.scale < 1.0:  # the page is shrunk: remove detail finer than the sensor can hold
        gt_src = cv2.GaussianBlur(region, (0, 0), 0.5 * float(np.sqrt(1.0 / pose.scale**2 - 1.0)) + 1e-3)
    gt = cv2.remap(
        gt_src, map_u - x0, map_v - y0, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT
    )
    gt = to_uint8(np.clip(gt, 0, 1))

    meta = {
        "params": asdict(params),
        "pose": pose.as_dict(),
        "center_uv": [float(c) for c in center_uv],
        "H_page_to_crop": H.tolist(),
        "moire_jpeg": jpeg_bytes,
    }
    return moire, gt, meta
