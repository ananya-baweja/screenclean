import csv
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image, ImageDraw, ImageFont

from screenclean import jobs
from screenclean.data import shards
from screenclean.eval import tesseract
from screenclean.eval.ocr_eval import summarize, synth_truth
from screenclean.render.fonts import matplotlib_font_dir
from screenclean.utils.io import encode_jpeg

needs_tesseract = pytest.mark.skipif(not tesseract.available(), reason="Tesseract not installed")

LINES = [
    ["Quarterly review of the budget", "Server load times fell by half"],
    ["Contact the research group today", "Release notes for every course"],
    ["Each student exports the results", "The mobile app detects images"],
]


def _text_crop(lines: list[str]) -> tuple[np.ndarray, list[dict]]:
    img = Image.new("RGB", (512, 512), (250, 250, 248))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(str(matplotlib_font_dir() / "DejaVuSans.ttf"), 26)
    records = []
    for i, text in enumerate(lines):
        x, y = 30, 120 + 90 * i
        x0, y0, x1, y1 = draw.textbbox((x, y), text, font=font)
        draw.text((x, y), text, font=font, fill=(20, 20, 30))
        records.append({"text": text, "quad": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], "partial": False})
    return np.asarray(img, np.float32) / 255.0, records


def _make_synth_split(drive: Path) -> None:
    split_dir = drive / "data" / "synth_v1" / "test"
    m = shards.load_manifest(split_dir)
    items = []
    yy, xx = np.mgrid[0:512, 0:512]
    grating = 0.12 * np.sin(2 * np.pi * (0.21 * xx + 0.13 * yy))[..., None] * np.array([1.0, -0.6, 0.4])
    for i, lines in enumerate(LINES):
        gt, records = _text_crop(lines)
        moire = np.clip(gt + grating, 0, 1)
        meta = json.dumps({"key": f"test/{i:06d}", "lines": records}).encode()
        items.append((f"test_{i:06d}", encode_jpeg(moire), encode_jpeg(gt), meta))
    # a crop without visible text is skipped
    blank = np.full((512, 512, 3), 0.9, np.float32)
    items.append(("test_000099", encode_jpeg(blank), encode_jpeg(blank), json.dumps({"lines": []}).encode()))
    shards.add_shard(split_dir, m, shards.write_shard(split_dir / "test-00000.tar", items))
    m["complete"] = True
    shards.save_manifest(split_dir, m)


def _repo(tmp_path, cfg, job_id="0019_eval_synth_ocr"):
    repo = tmp_path / "repo"
    (repo / "jobs" / "queue").mkdir(parents=True, exist_ok=True)
    (repo / "configs").mkdir(exist_ok=True)
    (repo / "configs" / f"{job_id}.yaml").write_text(yaml.safe_dump(cfg))
    spec = {"id": job_id, "task": "eval_synth_ocr", "runtime": "cpu", "config": f"configs/{job_id}.yaml"}
    (repo / "jobs" / "queue" / f"{job_id}.yaml").write_text(yaml.safe_dump(spec))
    return repo


def test_synth_truth_reads_lines_in_order():
    record = {
        "lines": [
            {"text": "second", "quad": [[0, 50], [40, 50], [40, 60], [0, 60]]},
            {"text": "first", "quad": [[0, 10], [40, 10], [40, 20], [0, 20]]},
        ]
    }
    assert synth_truth(record) == "first\nsecond"
    assert synth_truth({}) == ""


def test_summarize_pairs_and_intervals():
    rows = []
    for k in range(30):
        noise = 0.01 * (k % 3)
        for method, cer in (("Input", 0.5 + noise), ("A", 0.2 + noise), ("B", 0.1 + noise)):
            rows.append(
                {
                    "key": f"k{k}",
                    "method": method,
                    "engine": "tess",
                    "cer": cer,
                    "wer": cer,
                    "word_f1": 1 - cer,
                }
            )
    out = summarize(rows, "Input", [("A", "B")], group=None)["tess"]
    a = out["methods"]["A"]
    assert a["n"] == 30 and a["cer_vs_input"]["mean"] == pytest.approx(-0.3)
    assert a["cer_vs_input"]["hi"] < 0  # clearly better than the input
    (cmp,) = out["comparisons"]
    assert cmp["cer_a_minus_b"]["mean"] == pytest.approx(0.1) and cmp["a_reads_worse"] is True


@needs_tesseract
def test_eval_synth_ocr_end_to_end_and_resume(tmp_path, capsys):
    drive = tmp_path / "drive"
    _make_synth_split(drive)
    cfg = {
        "split": "data/synth_v1/test",
        "local_dir": str(tmp_path / "local"),
        "engines": ["tesseract"],
        "stop_margin_min": 0,
        "limit": 2,
        "methods": [
            {"name": "identity", "label": "Input (no cleaning)"},
            {"name": "chroma_lowpass", "params": {"sigma": 4}},
            {"name": "target"},
        ],
        "compare": [["chroma_lowpass", "Input (no cleaning)"]],
    }
    repo = _repo(tmp_path, cfg)
    marker = tmp_path / "m.txt"
    assert jobs.run("auto", drive, repo, marker, runtime="cpu") == 0
    assert "JOB FINISHED" in capsys.readouterr().out

    # run the rest (no limit): earlier rows are kept, none is written twice
    cfg.pop("limit")
    (repo / "configs" / "0019_eval_synth_ocr.yaml").write_text(yaml.safe_dump(cfg))
    assert jobs.run("0019_eval_synth_ocr", drive, repo, marker, runtime="cpu") == 0
    with zipfile.ZipFile(Path(marker.read_text())) as zf:
        summary = json.loads(zf.read("summary.json"))
        rows = list(csv.DictReader(zf.read("per_image.csv").decode().splitlines()))
    assert len(rows) == 3 * 3 and len({(r["key"], r["method"], r["engine"]) for r in rows}) == 9
    assert summary["complete"] and summary["skipped_without_text"] == 1
    assert summary["engines"]["tesseract"]["engine"] == "tesseract"
    tess = summary["ocr"]["tesseract"]["methods"]
    assert tess["Clean target (upper bound)"]["cer"]["mean"] < 0.05
    assert tess["Clean target (upper bound)"]["word_f1"]["mean"] > 0.95
    assert "cer_vs_input" in tess["chroma_lowpass"]
    assert summary["ocr"]["tesseract"]["comparisons"][0]["a"] == "chroma_lowpass"
