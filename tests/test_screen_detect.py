"""Screen detection (product/screen_detect.py) on simple shapes and on synthetic photos of screens."""

import cv2
import numpy as np
import pytest

from screenclean.product.screen_detect import detect_screen, full_image_corners, order_corners
from screenclean.render import corpus
from screenclean.render.pages import TEMPLATES, Renderer
from screenclean.simulate.scene import make_scene


@pytest.fixture(scope="module")
def renderer():
    return Renderer(corpus.lines_for("test"))


def _max_err(found, true, shape) -> float:
    """Worst corner error as a share of the image diagonal."""
    return float(np.linalg.norm(found - true, axis=1).max() / np.hypot(*shape[:2]))


def test_order_corners():
    pts = np.array([[10, 10], [110, 12], [105, 80], [8, 75]], float)
    for perm in ([2, 0, 3, 1], [3, 2, 1, 0], [1, 3, 0, 2]):
        np.testing.assert_array_equal(order_corners(pts[perm]), pts)


@pytest.mark.parametrize("template,dark", [("code_light", False), ("doc", True)])  # doc seed 3: dark theme
def test_finds_a_plain_page(renderer, template, dark):
    page = cv2.resize(renderer.render(template, seed=3).image, (640, 360), interpolation=cv2.INTER_AREA)
    assert (np.median(page) < 128) == dark
    img = np.full((600, 800, 3), 150, np.uint8)
    corners = np.array([[120, 90], [700, 130], [680, 480], [100, 440]], np.float32)
    cv2.fillConvexPoly(
        img, (corners + [[-12, -12], [12, -12], [12, 12], [-12, 12]]).astype(np.int32), (20, 20, 20)
    )
    src = np.array([[-0.5, -0.5], [639.5, -0.5], [639.5, 359.5], [-0.5, 359.5]], np.float32)
    H = cv2.getPerspectiveTransform(src, corners)
    warped = cv2.warpPerspective(page, H, (800, 600), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.ones((360, 640), np.uint8), H, (800, 600), flags=cv2.INTER_NEAREST)
    img[mask > 0] = warped[mask > 0]
    det = detect_screen(img)
    assert det.detected and np.abs(det.corners - corners).max() < 1.5
    if dark:  # a dark page next to a black bezel: found inside the device outline, faint edge
        assert det.method.endswith("peel") and det.confidence < 0.5
    else:
        assert det.polarity == 1 and det.confidence > 0.9


def test_nothing_to_find():
    img = np.random.default_rng(0).integers(100, 140, (480, 640, 3)).astype(np.uint8)
    det = detect_screen(img)
    assert not det.detected and det.method == "none"
    np.testing.assert_array_equal(det.corners, full_image_corners(640, 480))


@pytest.mark.parametrize("i", [0, 1, 7, 12, 21])  # edges, peel inside a frame (light, dark), dark page
def test_finds_screens_in_synthetic_photos(renderer, i):
    page = renderer.render(TEMPLATES[i % 7], seed=500 + i).image
    scene = make_scene(page, np.random.default_rng(500 + i), out_size=(1000, 750))
    det = detect_screen(scene.photo)
    assert det.detected and _max_err(det.corners, scene.corners, scene.photo.shape) < 0.01


@pytest.mark.slow
def test_detection_rate_on_synthetic_photos(renderer):
    """Regression guard for the P5.1 acceptance run (tools/check_screen_detect.py measures 200)."""
    light, dark, errors = [], [], []
    for i in range(42):
        page = renderer.render(TEMPLATES[i % 7], seed=1000 + i).image
        scene = make_scene(page, np.random.default_rng(1000 + i))
        det = detect_screen(scene.photo)
        err = _max_err(det.corners, scene.corners, scene.photo.shape)
        errors.append(err)
        (dark if np.median(page) < 128 else light).append(det.detected and err < 0.03)
    assert np.mean(light) >= 0.9 and np.mean(light + dark) >= 0.75
    assert np.median(errors) < 0.015
