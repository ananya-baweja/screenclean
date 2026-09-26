"""Training (train/): data order, EMA, checkpoints, the trainer, calibration and the ``train`` job task."""

import csv
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from screenclean import jobs
from screenclean.data import shards
from screenclean.data.pairs_dataset import PairsDataset
from screenclean.train.calibrate import recommend
from screenclean.train.data import MixedCrops, train_loader
from screenclean.train.state import EMA, load_checkpoint, rng_state, save_checkpoint, set_rng_state
from screenclean.train.trainer import Trainer, lr_factor, merge
from screenclean.utils.io import encode_jpeg


def make_split(split_dir: Path, n: int, size: int = 64, seed: int = 0, prefix: str = "s") -> Path:
    """A split folder like the real ones: smooth "pages" (gt) and the same plus stripes (moiré)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    items = []
    for i in range(n):
        base = rng.uniform(0.3, 0.9, (size // 8, size // 8, 3))
        gt = np.kron(base, np.ones((8, 8, 1)))
        period = rng.uniform(3, 6)
        stripes = 0.12 * np.sin(2 * np.pi * (xx * np.cos(i) + yy * np.sin(i)) / period)[..., None]
        moire = np.clip(gt + stripes * np.array([1.0, 0.6, -0.8]), 0, 1)
        items.append((f"{prefix}{i:03d}", encode_jpeg(moire, 98), encode_jpeg(gt, 98)))
    m = shards.load_manifest(split_dir)
    shards.add_shard(split_dir, m, shards.write_shard(split_dir / f"{prefix}-00000.tar", items))
    m["complete"] = True
    shards.save_manifest(split_dir, m)
    return split_dir


def tiny_cfg(**kw):
    cfg = {
        "model": "scnet_tiny",
        "crop": 32,
        "batch": 2,
        "workers": 0,
        "iters": 6,
        "val_every": 3,
        "val_max": 4,
        "val_batch": 2,
        "log_every": 2,
        "sample_grid": 2,
        "optim": {"warmup": 2},
    }
    return merge(cfg, kw)


@pytest.fixture
def data(tmp_path):
    train = make_split(tmp_path / "train", 6, seed=1)
    val = make_split(tmp_path / "val", 4, seed=2, prefix="v")
    return PairsDataset(sorted(train.glob("*.tar"))), PairsDataset(sorted(val.glob("*.tar")))


def test_lr_schedule_and_merge():
    assert lr_factor(0, 10, 100, 0.01) == pytest.approx(0.1)
    assert lr_factor(9, 10, 100, 0.01) == pytest.approx(1.0)
    assert lr_factor(10, 10, 100, 0.01) == pytest.approx(1.0)
    assert lr_factor(100, 10, 100, 0.01) == pytest.approx(0.01)
    assert lr_factor(55, 10, 100, 0.01) == pytest.approx(0.505)
    assert merge({"a": {"b": 1, "c": 2}, "d": 3}, {"a": {"b": 5}}) == {"a": {"b": 5, "c": 2}, "d": 3}


def test_samples_depend_only_on_their_number(data):
    train, _ = data
    ds = MixedCrops([train], crop=32, seed=7)
    a, b = ds[5], ds[5]
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) and a[0].shape == (3, 32, 32)
    assert not torch.equal(ds[6][0], a[0])
    batches = list(train_loader(ds, batch=2, start_iteration=1, iterations=3, workers=0))
    assert len(batches) == 2 and torch.equal(batches[0][0][0], ds[2][0])  # iteration 1 starts at sample 2


def test_sources_are_mixed_by_weight(data):
    train, val = data
    ds = MixedCrops([train, val], crop=32, weights=[3, 1])
    picks = np.array([ds.pick(k)[0] for k in range(2000)])
    assert 0.7 < (picks == 0).mean() < 0.8
    with pytest.raises(ValueError):
        MixedCrops([train], crop=32, weights=[1, 1])


def test_ema_and_checkpoint_round_trip(tmp_path):
    model = torch.nn.Linear(3, 2)
    ema = EMA(model, decay=0.9)
    with torch.no_grad():
        model.weight.fill_(1.0)
    for _ in range(200):
        ema.update(model)
    assert torch.allclose(ema.model.weight, torch.ones(2, 3), atol=1e-6) and ema.updates == 200
    state = rng_state()
    first = torch.rand(3)
    save_checkpoint(tmp_path / "ck" / "last.pt", {"ema": ema.state_dict(), "rng": state})
    assert not (tmp_path / "ck" / "last.pt.tmp").exists()
    ck = load_checkpoint(tmp_path / "ck" / "last.pt")
    set_rng_state(ck["rng"])
    assert torch.equal(torch.rand(3), first)
    ema2 = EMA(torch.nn.Linear(3, 2))
    ema2.load_state_dict(ck["ema"])
    assert torch.equal(ema2.model.weight, ema.model.weight) and ema2.updates == 200


def test_trainer_runs_validates_and_checkpoints(tmp_path, data):
    train, val = data
    summary = Trainer(tiny_cfg(), tmp_path / "run", [train], val).fit()
    run = tmp_path / "run"
    assert summary["state"] == "done" and summary["iteration"] == 6 and not summary["resumed"]
    assert (run / "last.pt").exists() and (run / "best.pt").exists()
    assert sorted(p.name for p in (run / "samples").iterdir()) == ["val_000003.jpg", "val_000006.jpg"]
    with open(run / "train_log.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [int(r["iter"]) for r in rows] == [2, 4, 6] and all(float(r["loss"]) > 0 for r in rows)
    assert 0 < float(rows[0]["fft_share"]) < 1
    with open(run / "val_log.csv", newline="") as f:
        assert [int(r["iter"]) for r in csv.DictReader(f)] == [3, 6]
    assert summary["final_val"]["psnr"] > 0 and summary["input_psnr"] > 0
    best = load_checkpoint(run / "best.pt")
    assert best["model_name"] == "scnet_tiny" and "ending.weight" in best["model"]


def test_trainer_stops_on_time_and_resumes(tmp_path, data):
    train, val = data
    calls = {"n": 0}

    def stop_after_3():
        calls["n"] += 1
        return calls["n"] > 3

    part = Trainer(tiny_cfg(), tmp_path / "run", [train], val, should_stop=stop_after_3).fit()
    assert part["state"] == "partial" and part["iteration"] == 3
    rest = Trainer(tiny_cfg(), tmp_path / "run", [train], val).fit()
    assert (
        rest["state"] == "done" and rest["resumed"] and rest["iteration"] == 6 and rest["iters_this_run"] == 3
    )


def test_recommend():
    rows = [
        {"crop": 256, "batch": 8, "it_per_s": 9.0, "peak_mem_gb": 5.0},
        {"crop": 384, "batch": 4, "it_per_s": 6.0, "peak_mem_gb": 6.0},
        {"crop": 384, "batch": 6, "it_per_s": 4.5, "peak_mem_gb": 9.0},
        {"crop": 384, "batch": 8, "it_per_s": 3.5, "peak_mem_gb": 12.5},  # over 75% of 15 GB
        {"crop": 512, "batch": 8, "oom": True},
    ]
    r = recommend(rows, total_mem_gb=15.0, plan_minutes=300)
    assert (r["crop"], r["batch"]) == (384, 6) and r["iters_for_plan"] == 72900
    assert "error" in recommend([{"crop": 384, "batch": 4, "oom": True}], 15.0, 300)


def test_train_job_end_to_end(tmp_path, capsys):
    drive = tmp_path / "drive"
    make_split(drive / "data" / "t" / "train", 6, seed=1)
    make_split(drive / "data" / "t" / "val", 4, seed=2, prefix="v")
    repo = tmp_path / "repo"
    (repo / "jobs" / "queue").mkdir(parents=True)
    (repo / "configs").mkdir()
    cfg = {
        "data": {"train": ["data/t/train"], "val": "data/t/val", "local_dir": str(tmp_path / "local")},
        "calibrate": {"crops": [32], "batches": [2], "iters": 4, "warmup": 2, "plan_minutes": 10},
        "train": tiny_cfg(),
    }
    (repo / "configs" / "t.yaml").write_text(yaml.safe_dump(cfg))
    spec = {
        "id": "0100_train_t",
        "task": "train",
        "runtime": "cpu",
        "config": "configs/t.yaml",
        "max_minutes": 30,
    }
    (repo / "jobs" / "queue" / "0100_train_t.yaml").write_text(yaml.safe_dump(spec))
    marker = tmp_path / "m.txt"
    assert jobs.run("auto", drive, repo, marker, runtime="cpu") == 0
    out = capsys.readouterr().out
    assert "JOB FINISHED: 0100_train_t" in out and "best val PSNR" in out
    with zipfile.ZipFile(Path(marker.read_text())) as zf:
        names = set(zf.namelist())
        summary = json.loads(zf.read("summary.json"))
        cal = json.loads(zf.read("calibration.json"))
    assert {"train_log.csv", "val_log.csv", "val_samples_latest.jpg", "calibration.json"} <= names
    assert summary["iteration"] == 6 and summary["run_dir"] == "runs/0100_train_t"
    assert cal["measurements"][0]["crop"] == 32 and "fft_weight_for_20pct" in cal
    assert (drive / "runs" / "0100_train_t" / "last.pt").exists()


@pytest.mark.slow
def test_tiny_model_overfits_four_pairs(tmp_path):
    """P6.5: scnet_tiny learns to remove the stripes from 4 pairs within 200 iterations."""
    split = make_split(tmp_path / "four", 4, seed=3)
    pairs = PairsDataset(sorted(split.glob("*.tar")))
    cfg = tiny_cfg(crop=64, batch=4, iters=200, val_every=200, log_every=20, optim={"lr": 2e-3, "warmup": 20})
    summary = Trainer(cfg, tmp_path / "run", [pairs], pairs).fit()
    assert summary["final_val"]["psnr"] >= summary["input_psnr"] + 3.0


@pytest.mark.slow
def test_resumed_run_matches_an_uninterrupted_one(tmp_path, data):
    """P6.5: stop at 20, resume to 50: same iteration count and the same loss curve as one run."""
    train, val = data
    cfg = tiny_cfg(iters=50, val_every=25, log_every=5)
    whole = Trainer(cfg, tmp_path / "whole", [train], val).fit()
    calls = {"n": 0}

    def stop_after_20():
        calls["n"] += 1
        return calls["n"] > 20

    Trainer(cfg, tmp_path / "split", [train], val, should_stop=stop_after_20).fit()
    resumed = Trainer(cfg, tmp_path / "split", [train], val).fit()
    assert resumed["iteration"] == whole["iteration"] == 50

    def losses(run):
        with open(tmp_path / run / "train_log.csv", newline="") as f:
            return [(int(r["iter"]), float(r["loss"])) for r in csv.DictReader(f)]

    a, b = losses("whole"), losses("split")
    assert [i for i, _ in a] == [i for i, _ in b] == list(range(5, 55, 5))
    np.testing.assert_allclose([v for _, v in a], [v for _, v in b], rtol=1e-4)
    assert resumed["final_val"]["psnr"] == pytest.approx(whole["final_val"]["psnr"], abs=0.01)
