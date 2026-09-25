"""Tests for the corpus, fonts, page renderer, ArUco markers, capture kit and real-capture ingest."""

import csv
import json

import numpy as np
import pytest
from conftest import synthetic_photo

from screenclean.eval import real_captures, text_metrics
from screenclean.render import aruco, corpus, fonts
from screenclean.render.capture_kit import SIZES, build_kit, page_plan
from screenclean.render.pages import MARKER_MARGIN, TEMPLATES, Renderer, reading_order
from screenclean.utils.io import write_image

# --------------------------------------------------------------------------- corpus and fonts


def test_corpus_splits_are_disjoint_and_complete():
    lines = corpus.load_lines()
    assert len(lines) == 1500
    parts = corpus.split_indices(len(lines))
    assert [len(parts[s]) for s in corpus.SPLITS] == [1200, 150, 150]
    all_idx = sorted(i for s in corpus.SPLITS for i in parts[s])
    assert all_idx == list(range(len(lines)))
    assert corpus.split_indices(len(lines)) == parts  # deterministic


def test_corpus_has_no_real_addresses():
    for line in corpus.load_lines():
        if "@" in line or "://" in line:
            assert "example.com" in line or "example.org" in line or "example.net" in line, line


def test_is_code():
    assert corpus.is_code("for i in range(10):") and corpus.is_code("    x = f(y)")
    assert not corpus.is_code("The model improves the error rate by twelve percent.")


def test_bundled_fonts_cover_all_families():
    fs = fonts.available_fonts()
    assert {f.family for f in fs} == set(fonts.FAMILIES)
    assert all(fonts.covers_ascii(f.path) for f in fs)
    assert fonts.fonts_of("mono", "italic")  # falls back to other mono styles


# --------------------------------------------------------------------------- pages


@pytest.fixture(scope="module")
def renderer():
    return Renderer(corpus.lines_for("test"))


@pytest.mark.parametrize("template", TEMPLATES)
def test_pages_have_valid_ground_truth(renderer, template):
    page = renderer.render(template, seed=7, base_size=16, page_id=3)
    h, w = page.image.shape[:2]
    assert (w, h) == (1920, 1080) and page.image.dtype == np.uint8
    assert len(page.lines) >= 3
    for ln in page.lines:
        x0, y0, x1, y1 = ln.box
        assert ln.text.strip()
        assert 0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h
        # content stays clear of the corner markers
        near_corner_x = x0 < MARKER_MARGIN - 20 or x1 > w - MARKER_MARGIN + 20
        near_corner_y = y0 < MARKER_MARGIN - 20 or y1 > h - MARKER_MARGIN + 20
        assert not (near_corner_x and near_corner_y), (template, ln)
    gt = page.gt()
    json.dumps(gt)
    assert gt["page_id"] == 3 and gt["template"] == template


@pytest.mark.parametrize("template", ["slide", "table", "code_dark"])
def test_pages_are_deterministic(renderer, template):
    a = renderer.render(template, seed=11)
    b = renderer.render(template, seed=11)
    np.testing.assert_array_equal(a.image, b.image)
    assert a.gt() == b.gt()
    assert not np.array_equal(a.image, renderer.render(template, seed=12).image)


def test_page_text_only_uses_the_given_lines():
    lines = corpus.lines_for("test")
    words = {w for ln in lines for w in ln.split()} | {
        "Item",
        "Region",
        "Owner",
        "Date",
        "Status",
        "Amount",
        "Qty",
        "Score",
        "Notes",
        "Code",
    }
    page = Renderer(lines).render("doc", seed=3)
    for ln in page.lines:
        for word in ln.text.split():
            assert any(word in w or w in word for w in words), word  # wrapped / cut words are fragments


def test_reading_order_rows_then_columns(renderer):
    page = renderer.render("table", seed=5, base_size=18)
    ordered = reading_order(page.lines)
    ys = [(ln.box[1] + ln.box[3]) / 2 for ln in ordered]
    assert ys == sorted(ys) or all(b >= a - 20 for a, b in zip(ys, ys[1:], strict=False))


# --------------------------------------------------------------------------- aruco


def test_markers_on_clean_page(renderer):
    page = renderer.render("slide", seed=1, page_id=42)
    fit = aruco.fit_page(aruco.detect(page.image))
    assert fit.page_id == 42 and fit.n_markers == 4 and fit.reproj_err_px < 0.5 and fit.ok
    np.testing.assert_allclose(
        aruco.page_corners_in_photo(fit.H), [[0, 0], [1920, 0], [1920, 1080], [0, 1080]], atol=1.0
    )


def test_page_id_range():
    with pytest.raises(ValueError):
        aruco.draw_markers(np.zeros((1080, 1920, 3), np.uint8), aruco.MAX_PAGES)


def _corner_error(fit, H_true, size=(1920, 1080)):
    """Mean distance (page px) between true page corners and where the estimated H maps them."""
    import cv2

    w, h = size
    page_pts = np.array([[[0, 0], [w, 0], [w, h], [0, h]]], np.float32)
    photo_pts = cv2.perspectiveTransform(page_pts, np.linalg.inv(H_true))
    est = cv2.perspectiveTransform(photo_pts, fit.H)
    return float(np.mean(np.linalg.norm(est[0] - page_pts[0], axis=1)))


def _warp_trials(renderer, n, seed):
    rng = np.random.default_rng(seed)
    ok, errs = 0, []
    for t in range(n):
        pid = int(rng.integers(0, aruco.MAX_PAGES))
        page = renderer.render(TEMPLATES[t % len(TEMPLATES)], seed=t, page_id=pid)
        photo, H_true = synthetic_photo(page.image, rng)
        fit = aruco.fit_page(real_captures.detect_multiscale(photo))
        if fit is not None and fit.page_id == pid:
            err = _corner_error(fit, H_true)
            errs.append(err)
            ok += err < 1.5
    return ok, errs


def test_synthetic_photos_recover_page_and_homography(renderer):
    ok, errs = _warp_trials(renderer, 20, seed=0)
    assert ok >= 19, (ok, errs)  # >= 95%


@pytest.mark.slow
def test_synthetic_photos_many(renderer):
    ok, errs = _warp_trials(renderer, 200, seed=1)
    assert ok >= 190, (ok, np.percentile(errs, [50, 95, 100]))


def test_rectify_restores_page(renderer):
    page = renderer.render("doc", seed=2, base_size=24, page_id=9)
    photo, _ = synthetic_photo(
        page.image, np.random.default_rng(3), blur_max=0.0, noise_max=0.0, jpeg_quality=95
    )
    fit = aruco.fit_page(aruco.detect(photo))
    rect = aruco.rectify(photo, fit.H)
    inner = (slice(200, 880), slice(200, 1720))  # away from the page border
    diff = np.abs(rect[inner].astype(int) - page.image[inner].astype(int)).mean()
    assert diff < 12, diff


# --------------------------------------------------------------------------- capture kit


def test_page_plan_covers_templates_and_sizes():
    plan = page_plan()
    assert len(plan) == 40 and [p["page_id"] for p in plan] == list(range(40))
    assert {p["template"] for p in plan} == set(TEMPLATES)
    sizes = [p["base_size"] for p in plan]
    assert min(sizes) == 12 and max(sizes) == 40 and sorted(sizes) == sorted(SIZES)
    assert sum(s < 16 for s in sizes) == 8


def test_build_small_kit(tmp_path):
    meta = build_kit(tmp_path / "kit", n_pages=3)
    assert (tmp_path / "kit" / "index.html").read_text().count("const N = 3;") == 1
    assert [p["page_id"] for p in meta["pages"]] == [0, 1, 2]
    saved = json.loads((tmp_path / "kit" / "pages.json").read_text())
    assert saved == json.loads(json.dumps(meta))
    for i in range(3):
        assert (tmp_path / "kit" / "pages" / f"page_{i:03d}.png").exists()


def test_committed_kit_matches_generator(tmp_path):
    """The committed kit must match the generator exactly (rebuild: `python -m screenclean capture-kit`)."""
    from pathlib import Path

    kit = Path(__file__).resolve().parents[1] / "capture_kit"
    committed = json.loads((kit / "pages.json").read_text(encoding="utf-8"))
    fresh = build_kit(tmp_path / "kit")
    assert committed == json.loads(json.dumps(fresh))
    from screenclean.utils.io import read_image

    for name in ("page_000.png", "page_017.png", "page_039.png"):  # pixels, not PNG bytes (encoders differ)
        a, b = read_image(kit / "pages" / name), read_image(tmp_path / "kit" / "pages" / name)
        assert a.shape == b.shape and np.abs(a - b).mean() < 0.002


# --------------------------------------------------------------------------- text metrics


def test_text_metrics():
    assert text_metrics.cer("Hello   World", "hello world") == 0.0
    assert text_metrics.cer("abcd", "abce") == pytest.approx(0.25)
    assert text_metrics.normalize("“Quote” — x") == '"quote" - x'
    assert text_metrics.wer("a b c d", "a b x d") == pytest.approx(0.25)
    assert text_metrics.word_f1("b a c", "a b c") == 1.0
    assert text_metrics.word_f1("a b", "c d") == 0.0
    assert text_metrics.cer("", "") == 0.0 and text_metrics.cer("", "x") == 1.0


# --------------------------------------------------------------------------- real captures


def test_parse_folder():
    assert real_captures.parse_folder("screen-hp-laptop__phone-redmi12") == ("hp-laptop", "redmi12")
    assert real_captures.parse_folder("misc") == ("?", "?")


def test_ingest_synthetic_captures(tmp_path, renderer):
    rng = np.random.default_rng(5)
    raw = tmp_path / "raw"
    for folder, pids in {"screen-monitor__phone-a": [4, 7], "screen-tv__phone-b": [11]}.items():
        for pid in pids:
            page = renderer.render("slide", seed=pid, page_id=pid)
            photo, _ = synthetic_photo(page.image, rng)
            write_image(raw / folder / f"IMG_{pid:04d}.jpg", photo, quality=92)
    write_image(raw / "screen-tv__phone-b" / "IMG_blank.jpg", np.full((600, 800, 3), 128, np.uint8))
    results = real_captures.ingest(raw, tmp_path / "out")
    by_file = {r.file: r for r in results}
    assert by_file["screen-monitor__phone-a/IMG_0004.jpg"].page_id == 4
    assert by_file["screen-tv__phone-b/IMG_0011.jpg"].phone == "b"
    assert all(r.ok for f, r in by_file.items() if "blank" not in f)
    assert not by_file["screen-tv__phone-b/IMG_blank.jpg"].ok
    rows = list(csv.DictReader((tmp_path / "out" / "manifest.csv").open()))
    assert len(rows) == 4 and set(rows[0]) == set(real_captures.FIELDS)
    caps = json.loads((tmp_path / "out" / "captures.json").read_text())
    assert len(caps[0]["page_corners_px"]) == 4 and len(caps[0]["H"]) == 3
