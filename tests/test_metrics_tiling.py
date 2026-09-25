import math

import numpy as np
import pytest

from screenclean.eval import metrics, tiling


def _img(seed=0, h=64, w=96):
    return np.random.default_rng(seed).integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def test_psnr_identical_is_capped():
    a = _img()
    assert metrics.psnr(a, a) == metrics.PSNR_CAP == 100.0


def test_psnr_matches_known_noise():
    # Constant mid-grey + Gaussian noise (sigma 5, no clipping) -> PSNR ~ 20*log10(255/5).
    rng = np.random.default_rng(1)
    clean = np.full((512, 512, 3), 128, np.uint8)
    noisy = np.clip(np.round(128 + rng.normal(0, 5, clean.shape)), 0, 255).astype(np.uint8)
    rounded_var = 25 + 1 / 12  # rounding to integers adds ~1/12 variance
    expected = 10 * math.log10(255**2 / rounded_var)
    assert metrics.psnr(noisy, clean) == pytest.approx(expected, abs=0.1)


def test_psnr_accepts_float_images():
    a = _img()
    assert metrics.psnr(a / 255.0, a) == 100.0


def test_ssim_range_and_identity():
    a, b = _img(0), _img(1)
    assert metrics.ssim(a, a) == pytest.approx(1.0)
    assert -1.0 <= metrics.ssim(a, b) < 0.2


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        metrics.psnr(_img(h=64), _img(h=32))


@pytest.mark.parametrize(
    ("length", "tile", "overlap"), [(100, 32, 8), (64, 64, 8), (50, 64, 8), (1000, 256, 64)]
)
def test_tile_starts_cover_everything(length, tile, overlap):
    starts = tiling.tile_starts(length, tile, overlap)
    covered = np.zeros(length, bool)
    for s in starts:
        covered[s : s + tile] = True
    assert covered.all() and starts[0] == 0
    assert all(b - a <= tile - overlap for a, b in zip(starts, starts[1:], strict=False))


def test_tiled_identity_equals_untiled():
    img = np.random.default_rng(2).random((300, 420, 3), dtype=np.float32)
    out = tiling.tiled_apply(img, lambda t: t, tile=128, overlap=16)
    np.testing.assert_allclose(out, img, atol=1e-6)


def test_tiled_pointwise_fn_equals_untiled():
    img = np.random.default_rng(3).random((200, 260, 3), dtype=np.float32)
    fn = lambda t: np.sqrt(t) * 0.5  # noqa: E731 - per-pixel, so tiling must not change it
    np.testing.assert_allclose(tiling.tiled_apply(img, fn, tile=64, overlap=16), fn(img), atol=1e-6)


def test_tiled_rejects_size_changes():
    img = np.zeros((100, 100, 3), np.float32)
    with pytest.raises(ValueError):
        tiling.tiled_apply(img, lambda t: t[:-1], tile=64, overlap=8)
