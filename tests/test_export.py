"""Scan outputs (product/export.py): Markdown, plain text, and a searchable PDF."""

import io

import numpy as np
import pytest
from pypdf import PdfReader

from screenclean.product.export import searchable_pdf, to_markdown, to_text
from screenclean.product.ocr import OcrLine
from screenclean.product.pipeline import ScanResult


def _result(lines, name="shot.jpg", detected=True, size=(600, 1000)) -> ScanResult:
    page = np.full((*size, 3), 0.95, np.float32)
    text = "\n".join(line.text for line in lines)
    return ScanResult(np.zeros((4, 2)), detected, page, page, lines, text, {}, {}, name)


LINES = [
    OcrLine("Quarterly revenue grew 12%", (50, 40, 700, 80), 0.95, 0),
    OcrLine("Café naïve façade — über", (50, 120, 620, 160), 0.90, 0),
    OcrLine("SELECT name FROM users;", (50, 220, 560, 250), 0.88, 1),
]


def test_searchable_pdf_text_layer_is_extractable():
    pdf = searchable_pdf([_result(LINES), _result(LINES[:1], size=(900, 1600))])
    reader = PdfReader(io.BytesIO(pdf))
    assert len(reader.pages) == 2
    text = reader.pages[0].extract_text()
    for word in ("Quarterly", "revenue", "12%", "Café", "naïve", "façade", "über", "SELECT", "users;"):
        assert word in text
    # page size follows the image (150 dpi): 1000 x 600 px -> 480 x 288 pt
    box = reader.pages[0].mediabox
    assert float(box.width) == pytest.approx(480) and float(box.height) == pytest.approx(288)
    assert float(reader.pages[1].mediabox.width) == pytest.approx(768)


def test_pdf_text_is_invisible_but_the_image_is_there():
    pdf = searchable_pdf([_result(LINES)])
    content = PdfReader(io.BytesIO(pdf)).pages[0].get_contents().get_data()
    assert b"3 Tr" in content  # text render mode 3: neither filled nor stroked
    assert b"Do" in content  # the page image is drawn


def test_markdown_and_text():
    results = [_result(LINES), _result([], name="", detected=False)]
    md = to_markdown(results, title="Lecture 4")
    assert md.startswith("# Lecture 4\n") and "## 1. shot.jpg" in md and "## 2. Photo 2" in md
    assert "Café naïve" in md and "_Screen not found" in md and "_No text found._" in md
    txt = to_text(results)
    assert "===== 1. shot.jpg =====" in txt and "SELECT name FROM users;" in txt
