import csv
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import yaml
from conftest import synthetic_photo

from screenclean import jobs
from screenclean.eval import tesseract
from screenclean.eval.real_ocr import detection_scores, load_pages, marker_zones, paint_over
from screenclean.utils.io import read_image, write_image

needs_tesseract = pytest.mark.skipif(not tesseract.available(), reason="Tesseract not installed")
REPO = Path(__file__).resolve().parents[1]
KIT = REPO / "capture_kit"


def _repo(tmp_path, jobs_cfg: dict[str, tuple[str, dict]]) -> Path:
    repo = tmp_path / "repo"
    (repo / "jobs" / "queue").mkdir(parents=True, exist_ok=True)
    (repo / "configs").mkdir(exist_ok=True)
    for job_id, (task, cfg) in jobs_cfg.items():
        (repo / "configs" / f"{job_id}.yaml").write_text(yaml.safe_dump(cfg))
        spec = {"id": job_id, "task": task, "runtime": "cpu", "config": f"configs/{job_id}.yaml"}
        (repo / "jobs" / "queue" / f"{job_id}.yaml").write_text(yaml.safe_dump(spec))
    return repo


def test_pages_truth_and_marker_zones():
    pages = load_pages(KIT)
    assert len(pages) == 40 and pages[0]["truth"].startswith("The research group")
    assert {p["size"] for p in pages.values()} <= {"<16 px", "16-24 px", ">24 px"}
    page = read_image(KIT / "pages" / "page_000.png")
    painted = paint_over(page, marker_zones())
    assert np.abs(painted[:150, :150] - painted[75, 75]).max() == 0  # top-left marker gone
    assert np.array_equal(painted[300:700, 300:1500], page[300:700, 300:1500])  # text untouched


def test_detection_scores():
    truth = np.array([[100, 100], [900, 100], [900, 600], [100, 600]], float)
    s = detection_scores(truth + [10, 0], True, truth, (1000, 750))
    assert s["success"] and s["mean_err_px"] == pytest.approx(10)
    s = detection_scores(truth + [80, 0], True, truth, (1000, 750))
    assert not s["success"] and s["max_err_pct_diag"] == pytest.approx(6.4)
    assert not detection_scores(truth, False, truth, (1000, 750))["success"]


@needs_tesseract
def test_ingest_and_eval_real_ocr_end_to_end(tmp_path, capsys):
    drive = tmp_path / "drive"
    rng = np.random.default_rng(3)
    folder = drive / "real_captures" / "raw" / "screen-sim__phone-sim"
    for pid in (0, 2):
        page = (read_image(KIT / "pages" / f"page_{pid:03d}.png") * 255).astype(np.uint8)
        photo, _ = synthetic_photo(
            page, rng, photo_size=(2000, 1400), blur_max=0.6, noise_max=2, jpeg_quality=90
        )
        write_image(folder / f"IMG_{pid:04d}.jpg", photo, quality=92)
    eval_cfg = {
        "kit_dir": str(KIT),
        "local_dir": str(tmp_path / "local"),
        "engines": ["tesseract"],
        "stop_margin_min": 0,
        "methods": [
            {"name": "identity", "label": "Input (no cleaning)"},
            {"name": "chroma_lowpass", "params": {"sigma": 4}},
        ],
        "product_cleaners": [
            {"label": "none", "cleaner": "none"},
            {"label": "chroma", "method": "chroma_lowpass"},
        ],
        "compare": [["oracle:chroma_lowpass", "oracle:Input (no cleaning)"]],
    }
    repo = _repo(
        tmp_path,
        {"0024_ingest_real": ("ingest_real", {}), "0025_eval_real_ocr": ("eval_real_ocr", eval_cfg)},
    )
    marker = tmp_path / "m.txt"
    assert jobs.run("0024_ingest_real", drive, repo, marker, runtime="cpu") == 0
    with zipfile.ZipFile(Path(marker.read_text())) as zf:
        assert set(zf.namelist()) >= {"manifest.csv", "summary.json"}
        assert not [n for n in zf.namelist() if n.endswith((".jpg", ".png"))]  # no images of the photos
        assert json.loads(zf.read("summary.json"))["n_ok"] == 2

    assert jobs.run("0025_eval_real_ocr", drive, repo, marker, runtime="cpu") == 0
    assert "JOB FINISHED" in capsys.readouterr().out
    with zipfile.ZipFile(Path(marker.read_text())) as zf:
        summary = json.loads(zf.read("summary.json"))
        rows = list(csv.DictReader(zf.read("per_photo.csv").decode().splitlines()))
        det = list(csv.DictReader(zf.read("detection.csv").decode().splitlines()))
    assert len(rows) == 2 * (2 + 2) and {r["kind"] for r in rows} == {"oracle", "product"}
    assert len(det) == 2 and summary["detection"]["n"] == 2
    tess = summary["ocr"]["tesseract"]
    input_f1 = tess["methods"]["oracle:Input (no cleaning)"]["word_f1"]["mean"]
    assert input_f1 > 0.6  # the same H straightens every output: a wrong H would read nothing
    assert tess["methods"]["product:chroma"]["baseline"] == "product:none"  # each kind has its own baseline
    assert (
        "cer_vs_input" in tess["methods"]["product:chroma"]
        and "baseline" not in tess["methods"]["product:none"]
    )
    assert set(tess["cer_by"]) == {"screen", "phone", "template", "size"}
    assert set(summary["seconds_per_photo"]) == {"none", "chroma"}
    assert summary["complete"]

    # a second run finds nothing left to do and changes no row
    assert jobs.run("0025_eval_real_ocr", drive, repo, marker, runtime="cpu") == 0
    with zipfile.ZipFile(Path(marker.read_text())) as zf:
        assert len(list(csv.DictReader(zf.read("per_photo.csv").decode().splitlines()))) == len(rows)
