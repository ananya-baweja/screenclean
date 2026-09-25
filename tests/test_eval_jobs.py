"""Tests for the tune_notch and eval tasks and the ESDNet wrapper (tiny fake data, CPU only)."""

import csv
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import yaml

from screenclean import jobs
from screenclean.data import shards
from screenclean.utils.io import encode_jpeg

torch = pytest.importorskip("torch")

FAKE_NETS = """
import torch
import torch.nn as nn


class my_model(nn.Module):
    def __init__(self, en_feature_num, en_inter_num, de_feature_num, de_inter_num, sam_number=1):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        assert x.shape[-1] % 32 == 0 and x.shape[-2] % 32 == 0, "input must be padded to a multiple of 32"
        y = x * self.scale
        return y, y[..., ::2, ::2], y[..., ::4, ::4]
"""


def _grating_pair(rng, h, w):
    yy, xx = np.mgrid[0:h, 0:w] / max(h, w)
    gt = np.stack([0.5 + 0.3 * np.sin(2 * np.pi * (1.3 * xx + c * yy)) for c in (0.4, 0.8, 1.2)], -1)
    moire = (
        gt
        + 0.08 * np.sin(2 * np.pi * (0.17 * np.arange(h)[:, None] + 0.23 * np.arange(w)[None, :]))[..., None]
    )
    noise = rng.normal(0, 0.005, gt.shape)
    return np.clip(moire + noise, 0, 1), np.clip(gt, 0, 1)


def _make_dev_split(drive: Path, n=5, h=150, w=230) -> None:
    rng = np.random.default_rng(0)
    split_dir = drive / "data" / "uhdm_v1" / "dev100"
    m = shards.load_manifest(split_dir)
    items = []
    for i in range(n):
        moire, gt = _grating_pair(rng, h, w)
        items.append((shards.sample_id(f"test/{i:04d}"), encode_jpeg(moire), encode_jpeg(gt)))
    for si, chunk in enumerate([items[:3], items[3:]]):
        shards.add_shard(split_dir, m, shards.write_shard(split_dir / f"dev100-{si:05d}.tar", chunk))
    m["complete"] = True
    shards.save_manifest(split_dir, m)


def _repo(tmp_path, task, cfg, job_id="0004_eval"):
    repo = tmp_path / "repo"
    (repo / "jobs" / "queue").mkdir(parents=True, exist_ok=True)
    (repo / "configs").mkdir(exist_ok=True)
    (repo / "configs" / f"{job_id}.yaml").write_text(yaml.safe_dump(cfg))
    spec = {
        "id": job_id,
        "task": task,
        "runtime": "cpu",
        "config": f"configs/{job_id}.yaml",
        "max_minutes": 30,
    }
    (repo / "jobs" / "queue" / f"{job_id}.yaml").write_text(yaml.safe_dump(spec))
    return repo


def _eval_cfg(tmp_path, methods, **kw):
    return {
        "split": "data/uhdm_v1/dev100",
        "local_dir": str(tmp_path / "local"),
        "methods": methods,
        "workers": 1,
        "sample_images": 2,
        "stop_margin_min": 0,
        **kw,
    }


# --------------------------------------------------------------------------- eval


def test_eval_scores_every_image_and_method(tmp_path, capsys):
    drive = tmp_path / "drive"
    _make_dev_split(drive)
    (drive / "params").mkdir(parents=True)
    (drive / "params" / "baselines.yaml").write_text(
        yaml.safe_dump(
            {"job": "0003_tune_notch", "fft_notch": {"r0": 0.05, "k": 5.0, "sigma": 2.0, "channels": "y"}}
        )
    )
    methods = [
        {"name": "identity"},
        {"name": "chroma_lowpass", "params": {"sigma": 3.0}},
        {"name": "fft_notch", "params_from": "params/baselines.yaml"},
        {"name": "fft_notch_local", "params_from": "params/baselines.yaml"},
    ]
    repo = _repo(tmp_path, "eval", _eval_cfg(tmp_path, methods))
    marker = tmp_path / "m.txt"
    assert jobs.run("auto", drive, repo, marker, runtime="cpu") == 0
    assert "JOB FINISHED: 0004_eval" in capsys.readouterr().out

    with zipfile.ZipFile(Path(marker.read_text())) as zf:
        names = set(zf.namelist())
        summary = json.loads(zf.read("summary.json"))
        rows = list(csv.DictReader(zf.read("per_image.csv").decode().splitlines()))
    assert len(rows) == 5 * 4 and summary["complete"] and summary["n_images"] == 5
    ms = summary["methods"]
    assert set(ms) == {"identity", "chroma_lowpass", "fft_notch", "fft_notch_local"}
    assert "0003_tune_notch" in ms["fft_notch"]["params_source"] and ms["fft_notch"]["params"]["k"] == 5.0
    assert ms["fft_notch_local"]["params_source"].startswith("defaults")
    assert ms["fft_notch"]["psnr_gain_vs_input"] > 3  # the grating is removed
    assert "psnr_gain_vs_input" not in ms["identity"]
    assert sum(n.startswith("samples/") for n in names) == 2 * 4


def test_eval_resumes_without_duplicates(tmp_path, capsys, monkeypatch):
    drive = tmp_path / "drive"
    _make_dev_split(drive)
    repo = _repo(tmp_path, "eval", _eval_cfg(tmp_path, [{"name": "identity"}, {"name": "chroma_lowpass"}]))
    monkeypatch.setattr(jobs.JobContext, "time_up", lambda self, margin_s=0: True)
    assert jobs.run("auto", drive, repo, tmp_path / "m.txt", runtime="cpu") == 0
    assert "TIME BUDGET REACHED" in capsys.readouterr().out
    rows_file = drive / "jobs" / "0004_eval" / "per_image.csv"
    assert len(list(csv.DictReader(rows_file.open()))) == 2  # one image, two methods

    monkeypatch.undo()
    assert jobs.run("auto", drive, repo, tmp_path / "m.txt", runtime="cpu") == 0
    rows = list(csv.DictReader(rows_file.open()))
    assert len(rows) == 10 and len({(r["key"], r["method"]) for r in rows}) == 10
    assert b"\r" not in rows_file.read_bytes()  # LF line endings


def test_eval_with_two_worker_processes(tmp_path, capsys):
    drive = tmp_path / "drive"
    _make_dev_split(drive, n=3)
    repo = _repo(
        tmp_path, "eval", _eval_cfg(tmp_path, [{"name": "identity"}, {"name": "fft_notch"}], workers=2)
    )
    assert jobs.run("auto", drive, repo, tmp_path / "m.txt", runtime="cpu") == 0
    assert "JOB FINISHED" in capsys.readouterr().out


# --------------------------------------------------------------------------- esdnet


@pytest.fixture
def fake_esdnet(tmp_path):
    from screenclean.baselines.esdnet_ref import ARCH

    repo = tmp_path / "UHDM"
    (repo / "model").mkdir(parents=True)
    (repo / "model" / "nets.py").write_text(FAKE_NETS)
    import importlib.util

    spec = importlib.util.spec_from_file_location("fake_nets", repo / "model" / "nets.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    weights = tmp_path / "w.pth"
    torch.save(mod.my_model(**ARCH).state_dict(), weights)
    return {
        "repo_dir": str(repo),
        "repo_sha": "unused",
        "weights": {"path": str(weights), "bytes": weights.stat().st_size},
    }


@pytest.mark.parametrize("shape", [(37, 53), (64, 96), (100, 33)])
def test_esdnet_runner_pads_and_crops(tmp_path, fake_esdnet, shape):
    from screenclean.baselines.esdnet_ref import build_esdnet

    runner = build_esdnet(fake_esdnet, tmp_path / "drive")
    img = np.random.default_rng(0).random((*shape, 3), dtype=np.float32)
    np.testing.assert_allclose(runner(img), img, atol=1e-6)  # identity model: output == input
    assert runner.modes["full"] == 1 and runner.params == 1
    assert (tmp_path / "drive" / "models" / "esdnet_uhdm.pth").exists()  # cached on "Drive"


def test_esdnet_tiled_mode_matches(tmp_path, fake_esdnet):
    from screenclean.baselines.esdnet_ref import build_esdnet

    runner = build_esdnet({**fake_esdnet, "force_tiles": True, "tile": 64, "overlap": 16}, tmp_path / "drive")
    img = np.random.default_rng(1).random((150, 170, 3), dtype=np.float32)
    np.testing.assert_allclose(runner(img), img, atol=1e-6)
    assert runner.modes["tiled"] == 1


def test_esdnet_weights_size_is_checked(tmp_path, fake_esdnet):
    from screenclean.baselines.esdnet_ref import ensure_weights
    from screenclean.data.download import DownloadError

    bad = {**fake_esdnet["weights"], "bytes": 12345}
    with pytest.raises(DownloadError, match="expected 12345 bytes"):
        ensure_weights(bad, tmp_path / "cache.pth")


def test_eval_with_esdnet_reference(tmp_path, fake_esdnet, capsys):
    drive = tmp_path / "drive"
    _make_dev_split(drive, n=2)
    methods = [
        {"name": "identity", "label": "Input (no cleaning)"},
        {"name": "esdnet_ref", "label": "ESDNet (reference)", **fake_esdnet},
    ]
    repo = _repo(tmp_path, "eval", _eval_cfg(tmp_path, methods, workers=2), job_id="0005_esdnet")
    marker = tmp_path / "m.txt"
    assert jobs.run("auto", drive, repo, marker, runtime="cpu") == 0
    assert "JOB FINISHED" in capsys.readouterr().out
    with zipfile.ZipFile(Path(marker.read_text())) as zf:
        ref = json.loads(zf.read("summary.json"))["methods"]["ESDNet (reference)"]
    assert ref["reference"] and ref["parameters"] == 1 and ref["inference_modes"] == {"full": 2}
    assert ref["psnr_gain_vs_input"] == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------- tune


def test_tune_notch_picks_settings_and_writes_params(tmp_path, fake_uhdm, capsys):
    drive = tmp_path / "drive"
    data_cfg = tmp_path / "repo_data.yaml"
    data_cfg.write_text(yaml.safe_dump({"source": {"path": str(fake_uhdm["zip"]), "train_prefix": "train/"}}))
    val_list = tmp_path / "val.txt"
    val_list.write_text("\n".join(["train/" + k for k in fake_uhdm["train_keys"][:3]]) + "\n")
    cfg = {
        "data_config": str(data_cfg),
        "val_list": str(val_list),
        "n_images": 2,
        "workers": 1,
        "stop_margin_min": 0,
        "grids": {
            "chroma_lowpass": {"sigma": [2, 8]},
            "fft_notch": {"r0": [0.05, 0.1], "k": [4, 8], "sigma": [2], "channels": ["y", "ycc"]},
            "fft_notch_local": {"r0": [0.1], "k": [6], "sigma": [2], "channels": ["y"]},
        },
    }
    repo = _repo(tmp_path, "tune_notch", cfg, job_id="0003_tune_notch")
    marker = tmp_path / "m.txt"
    assert jobs.run("auto", drive, repo, marker, runtime="cpu") == 0
    assert "JOB FINISHED" in capsys.readouterr().out

    params = yaml.safe_load((drive / "params" / "baselines.yaml").read_text())
    assert params["job"] == "0003_tune_notch" and params["n_images"] == 2
    assert set(params["fft_notch"]) == {"r0", "k", "sigma", "channels"}
    assert params["chroma_lowpass"]["sigma"] in (2, 8)
    with zipfile.ZipFile(Path(marker.read_text())) as zf:
        summary = json.loads(zf.read("summary.json"))
        table = zf.read("tune_table.csv").decode().splitlines()
    assert summary["n_images"] == 2 and set(summary["best"]) == {
        "identity",
        "chroma_lowpass",
        "fft_notch",
        "fft_notch_local",
    }
    assert len(table) == 1 + 1 + 2 + 8 + 1  # header + identity + chroma + notch grid + local grid


def test_cli_subprocess_with_worker_processes(tmp_path):
    """Run the real ``python -m screenclean jobs run`` command, as Colab does, with 2 spawned workers."""
    import subprocess
    import sys

    drive = tmp_path / "drive"
    _make_dev_split(drive, n=3)
    repo = _repo(
        tmp_path, "eval", _eval_cfg(tmp_path, [{"name": "identity"}, {"name": "chroma_lowpass"}], workers=2)
    )
    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "screenclean",
            "jobs",
            "run",
            "--job",
            "auto",
            "--drive-root",
            str(drive),
            "--repo-root",
            str(repo),
            "--marker",
            str(tmp_path / "m.txt"),
            "--runtime",
            "cpu",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert res.returncode == 0, res.stdout + res.stderr
    assert res.stdout.count("JOB FINISHED: 0004_eval") == 1
    rows = list(csv.DictReader((drive / "jobs" / "0004_eval" / "per_image.csv").open()))
    assert len(rows) == 6


def test_esdnet_quota_gives_partial_before_copying_data(tmp_path, fake_esdnet, capsys, monkeypatch):
    from screenclean.baselines import esdnet_ref
    from screenclean.data.download import DownloadError

    def refuse(*a, **k):
        raise DownloadError(
            "ESDNet weights: quota exceeded. " + esdnet_ref.manual_copy_help({"id": "X"}), True
        )

    monkeypatch.setattr(esdnet_ref, "ensure_weights", refuse)
    drive = tmp_path / "drive"
    _make_dev_split(drive, n=2)
    methods = [{"name": "identity"}, {"name": "esdnet_ref", **fake_esdnet}]
    repo = _repo(tmp_path, "eval", _eval_cfg(tmp_path, methods), job_id="0005_esdnet")
    assert jobs.run("auto", drive, repo, tmp_path / "m.txt", runtime="cpu") == 0
    out = capsys.readouterr().out
    assert "Download stopped" in out and "Make a copy" in out and "TIME BUDGET REACHED" in out
    assert jobs.StatusStore(jobs.DriveLayout(drive)).state("0005_esdnet") == "partial"
    assert not (tmp_path / "local").exists()  # the test images were never staged


def test_esdnet_uses_manual_drive_copy(tmp_path, fake_esdnet):
    from screenclean.baselines.esdnet_ref import ensure_weights

    drive = tmp_path / "MyDrive" / "screenclean"
    drive.mkdir(parents=True)
    src = Path(fake_esdnet["weights"]["path"])
    (drive.parent / "Copy of uhdm_checkpoint.pth").write_bytes(src.read_bytes())
    spec = {"id": "never-downloaded", "bytes": src.stat().st_size}  # no path: a download would be tried
    cache = ensure_weights(spec, drive / "models" / "esdnet_uhdm.pth", attempts=1, wait_s=0, drive_root=drive)
    assert cache.read_bytes() == src.read_bytes()
