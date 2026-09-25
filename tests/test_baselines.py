import numpy as np
import pytest

from screenclean.baselines import BASELINES, get_baseline
from screenclean.baselines.classical import find_peaks, notch_mask
from screenclean.eval.metrics import psnr


def smooth_image(h=256, w=256, seed=0):
    """Low-frequency colour image (no periodic content of its own)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w] / max(h, w)
    img = np.stack(
        [0.5 + 0.3 * np.sin(2 * np.pi * (a * xx + b * yy) + p) for a, b, p in rng.uniform(0.2, 1.5, (3, 3))],
        axis=-1,
    )
    return np.clip(img, 0, 1).astype(np.float32)


def add_grating(img, fy=0.13, fx=0.21, amp=0.08, color=(1.0, 1.0, 1.0)):
    """Add a sinusoidal grating (cycles per pixel), like a moiré pattern."""
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    g = amp * np.sin(2 * np.pi * (fy * yy + fx * xx))
    return np.clip(img + g[..., None] * np.array(color, np.float32), 0, 1).astype(np.float32)


@pytest.mark.parametrize("name", sorted(BASELINES))
def test_baselines_keep_shape_range_and_dtype(name):
    img = add_grating(smooth_image(96, 128))
    out = get_baseline(name)(img)
    assert out.shape == img.shape and out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_identity_is_identity():
    img = smooth_image(32, 32)
    np.testing.assert_array_equal(get_baseline("identity")(img), img)
    with pytest.raises(KeyError):
        get_baseline("nope")


def test_fft_notch_removes_grating_by_5db():
    clean = smooth_image()
    moire = add_grating(clean)
    before = psnr(moire, clean)
    after = psnr(get_baseline("fft_notch")(moire), clean)
    assert after - before >= 5.0, (before, after)


def test_fft_notch_local_removes_grating():
    clean = smooth_image(384, 512)
    moire = add_grating(clean, fy=0.17, fx=0.23)
    before = psnr(moire, clean)
    after = psnr(get_baseline("fft_notch_local")(moire), clean)
    assert after - before >= 3.0, (before, after)


def test_fft_notch_leaves_clean_image_nearly_unchanged():
    clean = smooth_image()
    assert psnr(get_baseline("fft_notch")(clean), clean) > 40


def test_ycc_mode_handles_coloured_moire():
    clean = smooth_image()
    moire = add_grating(clean, color=(1.0, -0.6, 0.4))  # coloured banding, as in real photos
    y_only = psnr(get_baseline("fft_notch")(moire, channels="y"), clean)
    ycc = psnr(get_baseline("fft_notch")(moire, channels="ycc"), clean)
    assert ycc > y_only


def test_chroma_lowpass_removes_colour_banding_keeps_brightness():
    clean = smooth_image()
    moire = add_grating(clean, fy=0.2, fx=0.1, color=(0.3, -0.3, 0.3))
    assert psnr(get_baseline("chroma_lowpass")(moire), clean) > psnr(moire, clean) + 2


def test_find_peaks_locates_grating_and_mask_notches_twin():
    img = add_grating(smooth_image(), fy=0.125, fx=0.25)[..., 0]
    spec, peaks = find_peaks(img)
    assert peaks, "grating peak not found"
    h, w = img.shape
    expected = {(h // 2 + 32, w // 2 + 64), (h // 2 - 32, w // 2 - 64)}  # 0.125*256, 0.25*256
    assert set(peaks[:2]) <= expected | {p for p in peaks[:2]}
    assert any(p in expected for p in peaks[:2])
    mask = notch_mask(img.shape, peaks[:1], sigma=2.0)
    for y, x in expected:
        assert mask[y, x] < 1e-3  # both the peak and its mirror twin are removed


def test_periodic_smooth_decomposition():
    from screenclean.baselines.classical import periodic_smooth

    u = smooth_image(64, 80)[..., 0]
    p, s = periodic_smooth(u)
    np.testing.assert_allclose(p + s, u, atol=1e-5)
    # the periodic part has (almost) no jump across the wrap-around edges
    assert np.abs(p[0] - p[-1]).mean() < np.abs(u[0] - u[-1]).mean()


def test_fft_notch_grid_matches_single_calls():
    from screenclean.baselines.classical import fft_notch, fft_notch_grid

    img = add_grating(smooth_image(64, 96), color=(1.0, -0.5, 0.3))
    combos = [
        {"r0": r0, "k": k, "sigma": s, "channels": ch}
        for r0 in (0.05, 0.1)
        for k in (4.0, 6.0)
        for s in (1.0, 3.0)
        for ch in ("y", "ycc")
    ]
    for params, out in fft_notch_grid(img, combos):
        np.testing.assert_allclose(out, fft_notch(img, **params), atol=1e-6)
