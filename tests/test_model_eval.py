"""Trained checkpoints at evaluation time (models/runner.py and the ``eval`` task's ``scnet`` method)."""

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch
from test_eval_jobs import _eval_cfg, _make_dev_split, _repo
from test_train import make_split, tiny_cfg

from screenclean import jobs
from screenclean.data.pairs_dataset import PairsDataset
from screenclean.eval.tables import collect
from screenclean.models.registry import build_model
from screenclean.models.runner import ModelRunner, load_trained
from screenclean.train.state import save_checkpoint
from screenclean.train.trainer import Trainer


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """A run folder with best.pt and last.pt from a few iterations of scnet_tiny."""
    tmp = tmp_path_factory.mktemp("run")
    train = PairsDataset(sorted(make_split(tmp / "train", 6, seed=1).glob("*.tar")))
    val = PairsDataset(sorted(make_split(tmp / "val", 2, seed=2, prefix="v").glob("*.tar")))
    Trainer(tiny_cfg(iters=6, val_every=6), tmp / "run", [train], val).fit()
    return tmp / "run"


def test_load_best_and_last(trained):
    best, info = load_trained(trained / "best.pt")
    last, _ = load_trained(trained / "last.pt")
    assert info["model_name"] == "scnet_tiny" and info["iteration"] == 6 and info["val"]["psnr"] > 0
    for a, b in zip(best.state_dict().values(), last.state_dict().values(), strict=True):
        assert torch.equal(a, b)  # both hold the EMA weights of iteration 6
    assert not best.training


def test_runner_full_and_tiled(trained, tmp_path):
    img = np.random.default_rng(0).random((70, 90, 3)).astype(np.float32)
    full = ModelRunner(trained / "best.pt", device="cpu")
    tiled = ModelRunner(trained / "best.pt", device="cpu", tile=48, overlap=16)
    a, b = full(img), tiled(img)
    assert a.shape == b.shape == img.shape and 0 <= a.min() and a.max() <= 1
    assert full.modes == {"full": 1} and tiled.modes == {"tiled": 1}
    # an untrained model is the identity, whole or in tiles
    model = build_model("scnet_tiny")
    save_checkpoint(tmp_path / "fresh.pt", {"model": model.state_dict(), "spec": model.spec.as_dict()})
    fresh = ModelRunner(tmp_path / "fresh.pt", device="cpu", tile=48, overlap=16)
    np.testing.assert_allclose(fresh(img), img, atol=1e-6)


def test_eval_task_scores_a_trained_checkpoint(tmp_path, trained, capsys):
    drive = tmp_path / "drive"
    _make_dev_split(drive, n=3, h=64, w=96)
    (drive / "runs" / "t").mkdir(parents=True)
    (drive / "runs" / "t" / "best.pt").write_bytes((trained / "best.pt").read_bytes())
    methods = [
        {"name": "identity", "label": "Input (no cleaning)"},
        {"name": "scnet", "label": "ScreenCleanNet", "checkpoint": "runs/t/best.pt"},
        {"name": "scnet", "label": "ScreenCleanNet (tiles)", "checkpoint": "runs/t/best.pt", "tile": 48},
    ]
    repo = _repo(tmp_path, "eval", _eval_cfg(tmp_path, methods, sample_images=1), job_id="0010_eval_models")
    marker = tmp_path / "m.txt"
    assert jobs.run("auto", drive, repo, marker, runtime="cpu") == 0
    assert "JOB FINISHED: 0010_eval_models" in capsys.readouterr().out
    with zipfile.ZipFile(Path(marker.read_text())) as zf:
        summary = json.loads(zf.read("summary.json"))
    m = summary["methods"]["ScreenCleanNet"]
    assert m["n"] == 3 and m["checkpoint"]["iteration"] == 6 and m["inference_modes"] == {"full": 3}
    assert summary["methods"]["ScreenCleanNet (tiles)"]["inference_modes"] == {"tiled": 3}
    assert "psnr_gain_vs_input" in m and not m["reference"]
    assert m["params_source"] == "checkpoint runs/t/best.pt"

    results = tmp_path / "results"
    (results / "0010_eval_models").mkdir(parents=True)
    (results / "0010_eval_models" / "summary.json").write_text(json.dumps(summary))
    labels = [r["label"] for r in collect(results, "dev100")]
    assert labels[0] == "Input (no cleaning)" and {"ScreenCleanNet", "ScreenCleanNet (tiles)"} <= set(labels)
