"""Dry run of the real-photo benchmark (P9.2) on simulated phone photos, on the laptop CPU.

Capture-kit pages are "photographed" on a simulated screen in a simulated room (moiré included,
``simulate/scene.py``), stored like Ananya's photos (``real_captures/raw/screen-..__phone-..``)
in a temporary Drive folder, and run through the two Colab tasks: ``ingest_real`` (markers ->
page id and homography) and ``eval_real_ocr`` (methods with the same geometry, and the scan
pipeline end to end), with Tesseract only and the classical cleaners. Writes a JSON summary.

    python tools/dry_run_real_ocr.py [--n 10] [--out results/checks/real_ocr_dryrun.json]

Needs Tesseract 5 (set TESSERACT_CMD if it is not on PATH).
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import yaml

from screenclean import jobs
from screenclean.eval import tesseract
from screenclean.simulate.scene import make_scene
from screenclean.utils.io import read_image, write_image

ROOT = Path(__file__).resolve().parents[1]
SCENE = {"area_frac": [0.45, 0.75]}
METHODS = [
    {"name": "identity", "label": "Input (no cleaning)"},
    {"name": "chroma_lowpass", "params_from": "params/baselines.yaml"},
    {"name": "fft_notch_local", "params_from": "params/baselines.yaml"},
]
CLEANERS = [{"label": "none", "cleaner": "none"}, {"label": "fft_notch_local", "cleaner": "fft_notch_local"}]


def _job(repo: Path, job_id: str, task: str, cfg: dict) -> None:
    (repo / "configs" / f"{job_id}.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    spec = {
        "id": job_id,
        "task": task,
        "runtime": "cpu",
        "config": f"configs/{job_id}.yaml",
        "max_minutes": 120,
    }
    (repo / "jobs" / "queue" / f"{job_id}.yaml").write_text(yaml.safe_dump(spec), encoding="utf-8")


def _summary(marker: Path) -> dict:
    with zipfile.ZipFile(Path(marker.read_text(encoding="utf-8"))) as zf:
        return json.loads(zf.read("summary.json"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=10, help="number of photos (pages spread over the kit)")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "checks" / "real_ocr_dryrun.json")
    args = ap.parse_args()
    if not tesseract.available():
        raise SystemExit("Tesseract not found: install it or set TESSERACT_CMD")

    with tempfile.TemporaryDirectory(prefix="real_ocr_dryrun_") as tmp:
        tmp = Path(tmp)
        drive, repo = tmp / "drive", tmp / "repo"
        (repo / "jobs" / "queue").mkdir(parents=True)
        (repo / "configs").mkdir()
        params = drive / "params" / "baselines.yaml"
        params.parent.mkdir(parents=True)
        params.write_text((ROOT / "configs" / "baselines" / "tuned.yaml").read_text(encoding="utf-8"))

        t0 = time.perf_counter()
        page_ids = np.linspace(0, 39, args.n).round().astype(int)
        for pid in page_ids:
            page = read_image(ROOT / "capture_kit" / "pages" / f"page_{pid:03d}.png")
            scene = make_scene(
                page, np.random.default_rng(9000 + int(pid)), out_size=(3200, 2400), ss=1, cfg=SCENE
            )
            write_image(
                drive / "real_captures/raw/screen-sim__phone-sim" / f"IMG_{pid:04d}.jpg", scene.photo, 92
            )
        print(f"{len(page_ids)} simulated photos in {time.perf_counter() - t0:.0f} s")

        _job(repo, "0001_ingest_real", "ingest_real", {})
        _job(
            repo,
            "0002_eval_real_ocr",
            "eval_real_ocr",
            {
                "kit_dir": str(ROOT / "capture_kit"),
                "local_dir": str(tmp / "local"),
                "engines": ["tesseract"],
                "methods": METHODS,
                "product_cleaners": CLEANERS,
                "compare": [["oracle:fft_notch_local", "oracle:Input (no cleaning)"]],
            },
        )
        marker = tmp / "marker.txt"
        jobs.run("0001_ingest_real", drive, repo, marker, runtime="cpu")
        ingest = _summary(marker)
        t0 = time.perf_counter()
        jobs.run("0002_eval_real_ocr", drive, repo, marker, runtime="cpu")
        evaluation = _summary(marker)
        minutes = (time.perf_counter() - t0) / 60

    tess = evaluation["ocr"]["tesseract"]["methods"]

    def short(entry: dict) -> dict:
        out = {k: round(entry[k]["mean"], 4) for k in ("cer", "word_f1")}
        if "cer_vs_input" in entry:
            ci = entry["cer_vs_input"]
            out["cer_vs_input"] = [round(ci[k], 4) for k in ("mean", "lo", "hi")]
        return out

    report = {
        "what": "jobs ingest_real + eval_real_ocr on simulated photos of capture-kit pages (laptop CPU)",
        "photos": len(page_ids),
        "pages": page_ids.tolist(),
        "ingest": {k: ingest[k] for k in ("n_photos", "n_ok", "reproj_err_px", "not_ok")},
        "ocr_tesseract": {m: short(e) for m, e in tess.items()},
        "detection": evaluation.get("detection"),
        "seconds_per_photo_median": evaluation.get("seconds_per_photo"),
        "eval_minutes": round(minutes, 1),
        "complete": evaluation["complete"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
