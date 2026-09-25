"""Perspective correction, aspect ratio and light clean-up (product/rectify.py)."""

import cv2
import numpy as np
import pytest

from screenclean.eval.metrics import psnr
from screenclean.product.rectify import (
    estimate_aspect,
    flatten_illumination,
    output_size,
    rectify,
    snap_aspect,
    stretch_contrast,
)
from screenclean.render import corpus
from screenclean.render.pages import Renderer
from screenclean.simulate.scene import _project, page_corners, sample_homography
from screenclean.utils.image import to_float01


def _page(template: str, seed: int, size=(640, 360)) -> np.ndarray:
    page = Renderer(corpus.lines_for("test")).render(template, seed=seed).image
    page = to_float01(cv2.resize(page, size, interpolation=cv2.INTER_AREA))
    return cv2.GaussianBlur(page, (0, 0), 0.7)  # as a lens delivers it


@pytest.mark.parametrize("aspect", [16 / 9, 4 / 3, 2.1, 0.5625])
def test_estimate_aspect_is_exact_for_a_projected_rectangle(aspect):
    rng = np.random.default_rng(1)
    pw, ph = (1000, round(1000 / aspect)) if aspect >= 1 else (round(1000 * aspect), 1000)
    for _ in range(20):
        H = sample_homography(rng, (pw, ph), (4000, 3000), 20)
        corners = _project(H, page_corners((pw, ph)))
        assert estimate_aspect(corners, (4000, 3000)) == pytest.approx(pw / ph, rel=1e-4)


def test_estimate_aspect_straight_on_uses_side_lengths():
    corners = np.array([[100, 100], [900, 100], [900, 550], [100, 550]], float) - 0.5
    assert estimate_aspect(corners, (1000, 700)) == pytest.approx(800 / 450)


def test_snap_aspect():
    assert snap_aspect(1.76) == (pytest.approx(16 / 9), "16:9")
    assert snap_aspect(1.61)[1] == "16:10" and snap_aspect(1.34)[1] == "4:3"
    assert snap_aspect(0.57) == (pytest.approx(9 / 16), "9:16")
    assert snap_aspect(1.5) == (1.5, None)


def test_output_size_never_shrinks_a_side():
    corners = np.array([[10, 20], [810, 60], [790, 520], [30, 470]], float)
    w, h = output_size(corners, 16 / 9)
    assert w >= 800 and h >= 450 and w / h == pytest.approx(16 / 9, rel=0.01)
    assert max(output_size(corners, 16 / 9, max_side=400)) == 400


@pytest.mark.parametrize("template,seed", [("doc", 0), ("code_light", 3), ("slide", 5)])
def test_warp_then_unwarp_round_trip(template, seed):
    page = _page(template, seed)
    rng = np.random.default_rng(seed)
    cfg = {"area_frac": [0.6, 0.75], "yaw_deg": 25, "pitch_deg": 20, "roll_deg": 10}
    H = sample_homography(rng, (640, 360), (1600, 1200), 0, cfg)
    photo = cv2.warpPerspective(page, H, (1600, 1200), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    corners = _project(H, page_corners((640, 360)))

    back, H_back = rectify(photo, corners, size=(640, 360))
    assert back.shape == (360, 640, 3) and back.dtype == np.float32
    good = psnr(back, page)
    assert good > 30.0
    shifted, _ = rectify(photo, corners + 0.5, size=(640, 360))  # a half-pixel convention slip shows
    assert psnr(shifted, page) < good - 1.0
    # H_back maps the photo corners onto the page corners
    mapped = cv2.perspectiveTransform(corners[None].astype(np.float32), H_back)[0]
    np.testing.assert_allclose(mapped, page_corners((640, 360)), atol=1e-3)


def test_rectify_estimates_and_snaps_the_shape():
    page = _page("table", 2)
    H = sample_homography(np.random.default_rng(4), (640, 360), (1600, 1200), 0, {"area_frac": [0.5, 0.6]})
    photo = cv2.warpPerspective(page, H, (1600, 1200), flags=cv2.INTER_LINEAR)
    corners = _project(H, page_corners((640, 360))) + np.random.default_rng(0).normal(0, 1.0, (4, 2))
    out, _ = rectify(photo, corners)
    assert out.shape[1] / out.shape[0] == pytest.approx(16 / 9, rel=0.005)
    assert out.shape[1] >= 640  # never smaller than the screen appears in the photo


def test_flatten_illumination_light_and_dark_pages():
    yy, xx = np.mgrid[0:360, 0:640].astype(np.float32)
    text = np.zeros((360, 640), bool)
    text[40:300:20, 40:600] = True  # thin "text" lines
    shade = 0.55 + 0.4 * xx / 640  # light falls off across the page
    light = np.where(text, 0.1, 0.95)[..., None] * shade[..., None] * np.ones(3, np.float32)
    flat = flatten_illumination(light)
    bg = ~cv2.dilate(text.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    assert flat[bg].std() < 0.25 * light[bg].std()
    assert flat[text].mean() < flat[bg].mean() - 0.4  # text stays dark

    glare = 0.25 * np.exp(-((xx - 500) ** 2 + (yy - 100) ** 2) / (2 * 120.0**2))
    dark = (np.where(text, 0.85, 0.08) + glare)[..., None] * np.ones(3, np.float32)
    flat = flatten_illumination(np.clip(dark, 0, 1))
    assert flat[bg].std() < 0.3 * dark[bg].std() and flat[text].mean() > flat[bg].mean() + 0.5


def test_stretch_contrast():
    x = np.linspace(0.3, 0.6, 1000, dtype=np.float32).reshape(10, 100, 1) * np.ones(3, np.float32)
    y = stretch_contrast(x)
    assert y.min() == 0.0 and y.max() == 1.0
    np.testing.assert_allclose(stretch_contrast(np.full((4, 4, 3), 0.5, np.float32)), 0.5)
