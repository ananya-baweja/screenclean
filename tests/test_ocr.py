import io
import json

import numpy as np
import pytest

from screenclean.eval import ocr_worker, tesseract
from screenclean.eval.ocr import OcrEngineError, WorkerOCR, lines_text, open_engine, self_check
from screenclean.eval.text_metrics import normalize, order_lines, ordered_text, quad_box, score
from screenclean.render import corpus
from screenclean.render.pages import Renderer, reading_order

needs_tesseract = pytest.mark.skipif(not tesseract.available(), reason="Tesseract not installed")


def test_bullets_are_not_text():
    assert normalize("• First point\n* second  � third ¢ fourth") == "first point second third fourth"
    assert normalize("a - b * c*d") == "a - b c*d"  # only stand-alone symbols


def test_order_lines_matches_the_ground_truth_rule():
    page = Renderer(corpus.lines_for("test")).render("table", seed=3)
    expected = [ln.text for ln in reading_order(page.lines)]
    shuffled = list(page.lines)
    np.random.default_rng(0).shuffle(shuffled)
    assert order_lines((ln.text, ln.box) for ln in shuffled) == expected
    assert quad_box([[3, 9], [7, 2], [8, 5], [1, 4]]) == (1.0, 2.0, 8.0, 9.0)


def test_score_counts_order_only_in_cer():
    ref = ordered_text([("left", (0, 0, 10, 10)), ("right", (20, 1, 40, 9)), ("below", (0, 20, 10, 30))])
    assert ref == "left\nright\nbelow"
    s = score(ref, "below left right")
    assert s["word_f1"] == 1.0 and s["cer"] > 0 and s["ref_chars"] == len("left right below")


def test_parse_paddle_result_fields():
    poly = np.array([[10, 20], [110, 20], [110, 40], [10, 40]], dtype=np.int16)
    result = {"rec_texts": ["Hello", " "], "rec_scores": np.array([0.9, 0.1]), "rec_polys": [poly, poly]}
    expected = [{"text": "Hello", "box": [10.0, 20.0, 110.0, 40.0], "conf": 0.9}]
    assert ocr_worker.parse_paddle(result) == expected

    class JsonResult:  # results that keep their fields under .json["res"]
        json = {"res": {"rec_texts": ["Hi"], "rec_scores": [0.5], "rec_boxes": np.array([[1, 2, 3, 4]])}}

    assert ocr_worker.parse_paddle(JsonResult())[0]["box"] == [1.0, 2.0, 3.0, 4.0]
    with pytest.raises(KeyError, match="rec_texts"):
        ocr_worker.parse_paddle({"texts": []})


def test_parse_rapid_both_output_styles():
    box = [[0, 0], [5, 0], [5, 2], [0, 2]]

    class Output:
        txts, boxes, scores = ("abc",), np.array([box], float), (0.7,)

    assert ocr_worker.parse_rapid(Output()) == [{"text": "abc", "box": [0.0, 0.0, 5.0, 2.0], "conf": 0.7}]
    assert ocr_worker.parse_rapid(([[box, "abc", 0.7]], [0.1]))[0]["text"] == "abc"
    assert ocr_worker.parse_rapid((None, None)) == []


def test_serve_reports_errors_per_image_and_keeps_going():
    class Engine:
        def read(self, path):
            if path == "bad":
                raise ValueError("unreadable")
            return [{"text": path, "box": [0, 0, 1, 1], "conf": 1.0}]

    requests = io.StringIO('{"id": "1", "path": "bad"}\n\n{"id": "2", "path": "ok"}\n')
    replies = io.StringIO()
    ocr_worker.serve(Engine(), requests, replies)
    first, second = (json.loads(line) for line in replies.getvalue().splitlines())
    assert first["id"] == "1" and "unreadable" in first["error"]
    assert second["lines"][0]["text"] == "ok" and "seconds" in second


def test_unknown_engine_reports_why():
    with pytest.raises(OcrEngineError, match="did not start"):
        WorkerOCR("no_such_engine", start_timeout_s=120)


@needs_tesseract
def test_tesseract_worker_reads_a_clean_capture_page():
    # P3.4: clean pages at normal sizes must read almost perfectly (CER below 3%)
    with WorkerOCR("tesseract") as ocr:
        check = self_check(ocr)
        assert ocr.info["versions"]["tesseract"].startswith("tesseract")
    assert check["cer"] < 0.03 and check["word_f1"] > 0.95


@needs_tesseract
def test_open_engine_falls_back(tmp_path):
    ocr, record = open_engine("paddle_missing", fallbacks={"paddle_missing": ["tesseract"]}, log_dir=tmp_path)
    try:
        assert record["engine"] == "tesseract" and record["self_check"]["word_f1"] > 0.95
        assert "did not start" in record["attempts"][0]["error"]
        with pytest.raises(OcrEngineError, match="missing.png"):
            ocr.read(tmp_path / "missing.png")
        assert lines_text([]) == ""  # the worker is still serving after a bad image:
        assert self_check(ocr)["word_f1"] > 0.95
    finally:
        ocr.close()
