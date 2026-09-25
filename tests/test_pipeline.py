"""The scan pipeline (product/pipeline.py) and OCR engines (product/ocr.py)."""

import numpy as np
import pytest
from PIL import Image

from screenclean.eval import tesseract
from screenclean.eval.text_metrics import cer
from screenclean.product.ocr import OcrLine, TesseractOCR, get_engine, lines_to_text, prepare_for_ocr
from screenclean.product.pipeline import get_cleaner, scan, scan_batch
from screenclean.render import corpus
from screenclean.render.pages import Renderer
from screenclean.simulate.scene import make_scene

needs_tesseract = pytest.mark.skipif(not tesseract.available(), reason="Tesseract not installed")


class FakeOCR:
    """Returns one line per call, and remembers what it was given."""

    def __init__(self):
        self.pages = []

    def read(self, page):
        self.pages.append(page)
        return [
            OcrLine("hello world", (1, 2, 30, 12), 0.9, 0),
            OcrLine("second para", (1, 20, 40, 30), 0.8, 1),
        ]


@pytest.fixture(scope="module")
def photo():
    page = Renderer(corpus.lines_for("test")).render("slide", seed=500).image
    return make_scene(page, np.random.default_rng(500), out_size=(1000, 750))  # found in test_screen_detect


def test_get_cleaner():
    img = np.random.default_rng(0).random((64, 64, 3)).astype(np.float32)
    assert get_cleaner("none")(img) is img
    assert get_cleaner("fft_notch_local")(img).shape == img.shape
    f = lambda x: x * 0  # noqa: E731
    assert get_cleaner(f) is f
    with pytest.raises(KeyError):
        get_cleaner("magic")
    with pytest.raises(NotImplementedError):
        get_cleaner("scnet")


def test_lines_to_text_and_engines():
    lines = [
        OcrLine("a b", (0, 0, 1, 1), 1, 0),
        OcrLine("c", (0, 0, 1, 1), 1, 0),
        OcrLine("d", (0, 0, 1, 1), 1, 1),
    ]
    assert lines_to_text(lines) == "a b\nc\n\nd"
    assert get_engine("none") is None and isinstance(get_engine("tesseract"), TesseractOCR)
    with pytest.raises(KeyError):
        get_engine("nope")
    dark = np.full((10, 10, 3), 0.1, np.float32)
    assert prepare_for_ocr(dark).dtype == np.uint8 and prepare_for_ocr(dark).mean() > 200  # inverted


def test_scan_finds_straightens_and_reads(photo):
    seen = []

    def spy_cleaner(img):
        seen.append(img.shape)
        return img

    ocr = FakeOCR()
    res = scan(photo.photo, cleaner=spy_cleaner, ocr=ocr, name="p1")
    assert res.detected and np.abs(res.corners - photo.corners).max() < 5
    h, w = res.page.shape[:2]
    assert w / h == pytest.approx(16 / 9, rel=0.01) and res.page.dtype == np.float32
    assert (
        seen and seen[0][0] * seen[0][1] < photo.photo.shape[0] * photo.photo.shape[1]
    )  # only the screen's box
    assert ocr.pages[0] is res.page and res.text == "hello world\n\nsecond para"
    assert {"detect", "clean", "rectify", "cleanup", "ocr", "total"} <= set(res.timings)
    s = res.summary()
    assert s["name"] == "p1" and s["detected"] and len(s["lines"]) == 2 and "page" not in s


def test_scan_with_given_corners_and_photo_mode(photo):
    res = scan(photo.photo, cleaner="none", ocr=None, corners=photo.corners, mode="photo")
    assert res.detection["method"] == "given" and "detect" in res.timings and "cleanup" not in res.timings
    assert res.lines == [] and res.text == ""
    with pytest.raises(ValueError):
        scan(photo.photo, mode="poster")


def test_scan_from_file_applies_exif_orientation(tmp_path, photo):
    path = tmp_path / "shot.jpg"
    rotated = np.ascontiguousarray(np.rot90(photo.photo, 1))  # stored sideways, EXIF says "rotate back"
    exif = Image.Exif()
    exif[0x0112] = 6  # orientation: rotate 90 degrees clockwise to display
    Image.fromarray(rotated).save(path, quality=95, exif=exif)
    res = scan(path, cleaner="none", ocr=None)
    assert res.name == "shot.jpg" and res.clean.shape[:2] == photo.photo.shape[:2] and res.detected
    assert "load" in res.timings


def test_scan_nothing_found_uses_whole_photo():
    img = np.random.default_rng(1).integers(100, 140, (300, 400, 3)).astype(np.uint8)
    res = scan(img, cleaner="none", ocr=None)
    assert not res.detected and res.page.shape == (300, 400, 3)


def test_scan_batch(photo):
    results, pdf, md = scan_batch([photo.photo, photo.photo], names=["a", "b"], cleaner="none", ocr=FakeOCR())
    assert [r.name for r in results] == ["a", "b"] and pdf.startswith(b"%PDF")
    assert "## 1. a" in md and "## 2. b" in md and "hello world" in md


@needs_tesseract
def test_scan_reads_a_simulated_photo_with_tesseract():
    page = Renderer(corpus.lines_for("test")).render("doc", seed=10, base_size=28)
    assert np.median(page.image) > 128  # a light page
    cfg = {"area_frac": [0.6, 0.7], "blur_prob": 0.0, "portrait_prob": 0.0}
    scene = make_scene(page.image, np.random.default_rng(3), out_size=(2000, 1500), ss=1, cfg=cfg)
    res = scan(scene.photo)  # classical moiré cleaner (without it, CER was 14% on this photo)
    assert res.detected and res.lines and all(0 <= ln.conf <= 1 for ln in res.lines)
    assert cer(page.text(), res.text) < 0.05
