"""Tests for the moiré simulator (display, camera, sensor, ISP and the end-to-end pipeline)."""

import time

import cv2
import numpy as np
import pytest

from screenclean.render import corpus
from screenclean.render.pages import Renderer
from screenclean.simulate import display, isp, sensor
from screenclean.simulate.camera import Pose, crop_to_page_maps, page_to_crop
from screenclean.simulate.moire_sim import load_config, simulate


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def page():
    return Renderer(corpus.lines_for("train")).render("doc", seed=4, base_size=22).image


WHITE = np.full((1080, 1920, 3), 235, np.uint8)


def moire_contrast(img: np.ndarray) -> float:
    """Structure left after light smoothing on a flat screen = moiré (noise is mostly smoothed away)."""
    f = img.astype(np.float32) / 255
    lp = cv2.GaussianBlur(f, (0, 0), 2.0)[32:-32, 32:-32]
    return float((lp.std(axis=(0, 1)) / lp.mean(axis=(0, 1))).mean())


# --------------------------------------------------------------------------- parts


@pytest.mark.parametrize("layout", display.LAYOUTS)
def test_subpixel_mask_keeps_average_brightness(layout):
    mask = display.subpixel_mask(layout, 4, 0.8, 0.9)
    assert mask.shape == (8, 8, 3)
    np.testing.assert_allclose(mask.mean(axis=(0, 1)), 1.0, atol=1e-5)
    assert (mask == 0).any()  # black matrix between subpixels


def test_rgb_and_bgr_are_mirrored():
    rgb = display.subpixel_mask("rgb", 6, 0.8, 0.9)
    bgr = display.subpixel_mask("bgr", 6, 0.8, 0.9)
    np.testing.assert_allclose(rgb[..., 0], bgr[..., 2], atol=1e-6)


def test_render_display_shape_and_mean():
    page = np.full((10, 12, 3), 0.5, np.float32)
    out = display.render_display(page, "rgb", 4, 0.8, 0.9)
    assert out.shape == (40, 48, 3)
    np.testing.assert_allclose(out.mean(axis=(0, 1)), 0.5, atol=1e-5)


def test_homography_maps_page_center_to_crop_center():
    pose = Pose(0, 0, 0, scale=2.0, focal_px=3000, k1=0.0, crop_offset=(0, 0), sensor_half_diag=2500)
    H = page_to_crop(pose, (700.0, 400.0), 512)
    p = H @ [700, 400, 1]
    np.testing.assert_allclose(p[:2] / p[2], [255.5, 255.5], atol=1e-6)
    q = H @ [701, 400, 1]
    assert q[0] / q[2] - 255.5 == pytest.approx(2.0, rel=1e-3)  # one display pixel = ``scale`` sensor pixels
    u, v = crop_to_page_maps(pose, H, 512)
    assert u[256, 256] == pytest.approx(700 + 0.5 / 2, abs=1e-3)


@pytest.mark.parametrize("pattern", sorted(sensor.CFA))
@pytest.mark.parametrize("method", ["bilinear", "ea", "vng"])
def test_demosaic_recovers_smooth_images(pattern, method):
    rng = np.random.default_rng(0)
    rgb = cv2.GaussianBlur(rng.random((64, 64, 3)).astype(np.float32), (0, 0), 4)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
    out = isp.demosaic(sensor.mosaic(rgb, pattern), pattern, method)
    assert np.abs(out[4:-4, 4:-4] - rgb[4:-4, 4:-4]).mean() < 0.02


def test_noise_statistics():
    rng = np.random.default_rng(1)
    flat = np.full((200, 200), 0.5, np.float32)
    noisy = sensor.add_noise(flat, rng, photons=1000, read_noise=0.0)
    assert noisy.mean() == pytest.approx(0.5, abs=0.005)
    assert noisy.std() == pytest.approx(np.sqrt(0.5 / 1000), rel=0.1)  # shot noise


def test_exposure_floor_keeps_dark_pages_dark():
    dark = np.full((50, 50), 0.01, np.float32)
    assert sensor.expose(dark, 98, 0.8).mean() == pytest.approx(0.8)
    assert sensor.expose(dark, 98, 0.8, floor=0.35).mean() == pytest.approx(0.01 * 0.8 / 0.35)


# --------------------------------------------------------------------------- end to end


def test_simulate_outputs(cfg, page):
    moire, gt, meta = simulate(page, np.random.default_rng(0), cfg)
    for img in (moire, gt):
        assert img.shape == (512, 512, 3) and img.dtype == np.uint8
    assert isinstance(meta["moire_jpeg"], bytes) and meta["moire_jpeg"][:2] == b"\xff\xd8"
    decoded = cv2.cvtColor(
        cv2.imdecode(np.frombuffer(meta["moire_jpeg"], np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB
    )
    np.testing.assert_array_equal(decoded, moire)
    assert set(meta["params"]) >= {"layout", "psf_sigma", "cfa", "demosaic", "jpeg_quality"}


def test_simulate_is_deterministic(cfg, page):
    a = simulate(page, np.random.default_rng(5), cfg)
    b = simulate(page, np.random.default_rng(5), cfg)
    assert a[2]["moire_jpeg"] == b[2]["moire_jpeg"]
    np.testing.assert_array_equal(a[1], b[1])
    c = simulate(page, np.random.default_rng(6), cfg)
    assert c[2]["moire_jpeg"] != a[2]["moire_jpeg"]


def test_input_and_target_stay_aligned(cfg, page):
    """After heavy blurring (removing moiré and noise), input and target align to a fraction of a pixel."""
    shifts = []
    for seed in range(6):
        rng = np.random.default_rng(seed)
        moire, gt, _ = simulate(page, rng, cfg, center_uv=(960, 500))
        a = cv2.GaussianBlur(cv2.cvtColor(moire, cv2.COLOR_RGB2GRAY).astype(np.float32), (0, 0), 4)
        b = cv2.GaussianBlur(cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY).astype(np.float32), (0, 0), 4)
        (dx, dy), _ = cv2.phaseCorrelate(a, b)
        shifts.append(np.hypot(dx, dy))
    assert max(shifts) < 0.5, shifts


def test_default_parameters_make_moire(cfg):
    vals = [
        moire_contrast(simulate(WHITE, np.random.default_rng(100 + k), cfg, center_uv=(960, 540))[0])
        for k in range(16)
    ]
    blurred = [
        moire_contrast(
            simulate(
                WHITE, np.random.default_rng(100 + k), cfg, center_uv=(960, 540), overrides={"psf_sigma": 3.0}
            )[0]
        )
        for k in range(4)
    ]
    assert np.mean(vals) > 0.02, vals  # clearly visible on average
    assert max(blurred) < 0.01 < np.mean(vals)  # a very blurry lens removes it (control)


def test_moire_peaks_near_two_sensor_pixels_per_display_pixel(cfg):
    kw = {"psf_sigma": 0.3, "sharpen": 0.0, "photons": 3000.0, "yaw_deg": 3.0, "pitch_deg": 3.0}
    at = {
        s: moire_contrast(
            simulate(
                WHITE, np.random.default_rng(1), cfg, center_uv=(960, 540), overrides={**kw, "scale": s}
            )[0]
        )
        for s in (1.3, 2.0)
    }
    assert at[2.0] > 3 * at[1.3], at


def test_overrides_validate_names(cfg, page):
    with pytest.raises(KeyError):
        simulate(page, np.random.default_rng(0), cfg, overrides={"nonsense": 1})


def test_speed_per_512_crop(cfg, page):
    rng = np.random.default_rng(3)
    simulate(page, rng, cfg)  # warm-up (OpenCV and mask caches)
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        simulate(page, rng, cfg)
        times.append(time.perf_counter() - t0)
    assert np.median(times) < 1.0, times
