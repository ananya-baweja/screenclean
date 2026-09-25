"""Whole-photo scenes (simulate/scene.py): sizes, determinism, and true corners checked by ArUco."""

import numpy as np
import pytest

from screenclean.eval.real_captures import detect_multiscale
from screenclean.render import aruco, corpus
from screenclean.render.pages import Renderer
from screenclean.simulate.scene import make_scene, room


@pytest.fixture(scope="module")
def renderer():
    return Renderer(corpus.lines_for("test"))


def test_scene_shapes_and_determinism(renderer):
    page = renderer.render("doc", seed=1).image
    a = make_scene(page, np.random.default_rng(3), out_size=(640, 480))
    b = make_scene(page, np.random.default_rng(3), out_size=(640, 480))
    assert np.array_equal(a.photo, b.photo) and np.array_equal(a.corners, b.corners)
    h, w = a.photo.shape[:2]
    assert (w, h) in ((640, 480), (480, 640)) and a.photo.dtype == np.uint8
    assert (a.corners >= 0).all() and (a.corners[:, 0] < w).all() and (a.corners[:, 1] < h).all()
    assert 0.15 < a.meta["area_frac"] < 0.8
    portrait = [
        make_scene(page, np.random.default_rng(s), out_size=(320, 240)).meta["portrait"] for s in range(12)
    ]
    assert any(portrait) and not all(portrait)


def test_room_is_linear_light_float32():
    img = room(np.random.default_rng(0), 200, 150)
    assert img.shape == (150, 200, 3) and img.dtype == np.float32 and img.min() >= 0


@pytest.mark.parametrize("seed", [0, 1])
def test_true_corners_match_aruco_markers(renderer, seed):
    """The corners the scene reports agree with the page position found from its corner markers."""
    page = renderer.render("slide", seed=seed, page_id=seed)
    cfg = {"area_frac": [0.45, 0.6], "blur_prob": 0.0, "glare": [0.0, 0.0], "portrait_prob": 0.0}
    scene = make_scene(page.image, np.random.default_rng(seed), out_size=(1600, 1200), cfg=cfg)
    fit = aruco.fit_page(detect_multiscale(scene.photo, scales=(1.0,)))
    assert fit is not None and fit.page_id == seed and fit.n_markers >= 3
    found = aruco.page_corners_in_photo(fit.H)
    assert np.abs(found - scene.corners).max() < 1.5
