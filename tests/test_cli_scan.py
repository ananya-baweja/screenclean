"""The ``scan`` command: inputs, outputs, and a run on simulated photos (OCR off, so no Tesseract needed)."""

import json

import numpy as np
from PIL import Image
from pypdf import PdfReader

from screenclean.cli import expand_inputs, main
from screenclean.render import corpus
from screenclean.render.pages import Renderer
from screenclean.simulate.scene import make_scene


def test_expand_inputs(tmp_path):
    for name in ("b.jpg", "a.JPG", "c.png", "notes.txt"):
        (tmp_path / name).write_bytes(b"x")
    found = expand_inputs([str(tmp_path), str(tmp_path / "*.jpg"), str(tmp_path / "c.png")])
    assert [p.name for p in found] == ["a.JPG", "b.jpg", "c.png"]  # folder: sorted images only, no repeats
    assert [p.name for p in expand_inputs([str(tmp_path / "*.png")])] == ["c.png"]


def test_scan_command_writes_pdf_markdown_json_and_pages(tmp_path, capsys):
    page = Renderer(corpus.lines_for("test")).render("slide", seed=500).image
    photos = tmp_path / "photos"
    photos.mkdir()
    scene = make_scene(page, np.random.default_rng(500), out_size=(1000, 750))  # found in test_screen_detect
    Image.fromarray(scene.photo).save(photos / "slide1.jpg", quality=92)
    Image.fromarray(np.full((300, 400, 3), 120, np.uint8)).save(photos / "blank.jpg")

    out = tmp_path / "out"
    outputs = [
        "--pdf",
        out / "n.pdf",
        "--md",
        out / "n.md",
        "--json",
        out / "n.json",
        "--pages",
        out / "pages",
    ]
    code = main(["scan", str(photos), "--ocr", "none", "--cleaner", "none", *map(str, outputs)])
    assert code == 0
    printed = capsys.readouterr().out
    assert "[1/2] blank.jpg: screen NOT found" in printed and "[2/2] slide1.jpg: screen found" in printed
    assert len(PdfReader(out / "n.pdf").pages) == 2
    md = (out / "n.md").read_text(encoding="utf-8")
    assert "## 1. blank.jpg" in md and "_Screen not found" in md and "## 2. slide1.jpg" in md
    details = json.loads((out / "n.json").read_text(encoding="utf-8"))
    assert [d["detected"] for d in details] == [False, True] and "timings" in details[1]
    assert sorted(p.name for p in (out / "pages").iterdir()) == ["blank_page.jpg", "slide1_page.jpg"]


def test_scan_command_missing_photo(tmp_path, capsys):
    assert main(["scan", str(tmp_path / "nope.jpg"), "--ocr", "none", "--md", str(tmp_path / "x.md")]) == 2
    assert "no such photo" in capsys.readouterr().err
