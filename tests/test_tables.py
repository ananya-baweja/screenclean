import json

from screenclean import cli
from screenclean.eval.tables import build_tables, collect


def _summary(root, job_id, methods, split="data/uhdm_v1/dev100", complete=True, task="eval"):
    d = root / "results" / "jobs" / job_id
    d.mkdir(parents=True)
    s = {"id": job_id, "task": task, "split": split, "complete": complete, "methods": methods}
    (d / "summary.json").write_text(json.dumps(s))


def _m(name, psnr, n=100, **kw):
    return {
        "name": name,
        "n": n,
        "psnr": psnr,
        "ssim": 0.7,
        "seconds": 1.0,
        "params": {},
        "params_source": "config",
        **kw,
    }


def test_tables_work_with_no_results(tmp_path):
    (path,) = build_tables(tmp_path)
    assert "no results yet" in path.read_text(encoding="utf-8")


def test_tables_merge_jobs_and_mark_reference(tmp_path, capsys):
    _summary(
        tmp_path,
        "0004_eval_baselines_dev100",
        {
            "Input (no cleaning)": _m("identity", 17.1),
            "fft_notch": _m(
                "fft_notch", 17.3, psnr_gain_vs_input=0.2, params={"k": 4}, params_source="params/x"
            ),
        },
    )
    _summary(
        tmp_path,
        "0005_esdnet_ref_dev100",
        {
            "Input (no cleaning)": _m("identity", 17.1, lpips=0.5),
            "ESDNet (reference)": _m(
                "esdnet_ref",
                22.1,
                n=40,
                reference=True,
                lpips=0.2,
                psnr_gain_vs_input=5.0,
                inference_modes={"full": 40},
            ),
        },
        complete=False,
    )
    _summary(tmp_path, "0002_prepare_uhdm", {}, task="prepare_uhdm")
    rows = collect(tmp_path / "results" / "jobs", "dev100")
    assert [r["label"] for r in rows] == ["Input (no cleaning)", "fft_notch", "ESDNet (reference)"]

    assert cli.main(["tables", "--repo-root", str(tmp_path)]) == 0
    text = (tmp_path / "results" / "tables" / "uhdm_dev100.md").read_text(encoding="utf-8")
    assert "| ESDNet (reference) | 22.10 | +5.00 |" in text
    assert "40 (partial)" in text and "`0005_esdnet_ref_dev100`" in text
    assert "| fft_notch | 17.30 | +0.20 |" in text
    assert "**fft_notch**: k=4 (from params/x)" in text
    assert "**ESDNet (reference)**: authors' pretrained weights" in text
    assert "40 images at full resolution" in text
