import json
from pathlib import Path

import pytest

from screenclean.eval.ocr_eval import summarize
from screenclean.eval.report import build_report

pytest.importorskip("matplotlib")


def _write(repo: Path, job: str, summary: dict) -> None:
    d = repo / "results" / "jobs" / job
    d.mkdir(parents=True, exist_ok=True)
    (d / "summary.json").write_text(json.dumps({"id": job, **summary}))


def _m(name, psnr, gain=None, **kw):
    out = {
        "name": name,
        "n": 100,
        "psnr": psnr,
        "ssim": 0.7,
        "seconds": 1.0,
        "params": {},
        "params_source": "x",
    }
    if gain is not None:
        out["psnr_gain_vs_input"] = gain
    return {**out, **kw}


def test_report_fills_what_exists_and_names_what_is_pending(tmp_path):
    repo = tmp_path
    _write(repo, "0017_eval", {"task": "eval", "split": "data/uhdm_v1/dev100", "methods": {
        "Input (no cleaning)": _m("identity", 17.1),
        "ScreenCleanNet (final)": _m("scnet", 19.8, 2.7, parameters=4_650_000, lpips=0.31),
        "Ablation: full model, short schedule": _m("scnet", 19.3, 2.2, lpips=0.33),
        "Ablation: no FFT loss": _m("scnet", 19.3, 2.2, lpips=0.36),
    }})  # fmt: skip
    rows = [
        {
            "key": f"k{i}",
            "method": m,
            "engine": "tesseract",
            "cer": c + 0.01 * (i % 2),
            "wer": c,
            "word_f1": 1 - c,
        }
        for i in range(12)
        for m, c in (("Input (no cleaning)", 0.4), ("ScreenCleanNet (final)", 0.1))
    ]
    _write(repo, "0019_eval_synth_ocr", {
        "task": "eval_synth_ocr",
        "engines": {"paddle": {"engine": "rapidocr", "versions": {"rapidocr": "3.9.2"}}},
        "ocr": summarize(rows, "Input (no cleaning)", [("ScreenCleanNet (final)", "Input (no cleaning)")]),
    })  # fmt: skip
    _write(repo, "0023_efficiency", {
        "task": "efficiency", "gpu": "Tesla T4", "cpu_threads": 2, "notes": "median",
        "sizes": {"1280x720": [720, 1280]},
        "rows": [{"label": "ScreenCleanNet (final)", "params_m": 4.65, "gflops_1280x720": 180.0,
                  "gpu_ms_1280x720": 40.0, "cpu_s_1280x720": 3.1}],
    })  # fmt: skip

    written = build_report(repo)
    text = (repo / "docs" / "RESULTS.md").read_text(encoding="utf-8")
    assert "| ScreenCleanNet (final) | 19.80 | +2.70 |" in text
    assert "*Pending: job `0021/0022_eval_test500`" in text  # no 500-pair results yet
    assert "*Pending: job `0025_eval_real_ocr`" in text
    assert "rapidocr 3.9.2 (fallback for paddle)" in text
    assert "ScreenCleanNet (final) minus Input (no cleaning): CER -0.300" in text
    assert "| ScreenCleanNet (final) | 4.65 M | 180 | 40 | 3.10 |" in text
    assert "| no FFT loss | 19.30 | +0.00 | 0.360 |" in text
    names = {p.name for p in written}
    assert {"RESULTS.md", "ablations.png", "ocr_synth.png", "psnr_vs_params.png"} <= names
