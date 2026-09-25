"""Tests for the gen_synth and sim_realism Colab tasks (tiny crops, a handful of samples)."""

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import yaml

from screenclean import jobs
from screenclean.data import shards, synthetic
from screenclean.render import corpus
from screenclean.simulate.moire_sim import load_config
from screenclean.utils.io import encode_jpeg


def _repo(tmp_path, task, cfg, job_id):
    repo = tmp_path / "repo"
    (repo / "jobs" / "queue").mkdir(parents=True, exist_ok=True)
    (repo / "configs").mkdir(exist_ok=True)
    sim = load_config()
    sim["crop"] = 128
    (repo / "configs" / "sim.yaml").write_text(yaml.safe_dump(sim))
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


def _synth_cfg(tmp_path, **kw):
    return {
        "sim_config": "configs/sim.yaml",
        "drive_out": "data/synth_v1",
        "local_dir": str(tmp_path / "local"),
        "seed": 0,
        "counts": {"train": 6, "val": 2, "test": 2},
        "crops_per_page": 2,
        "base_size": [14, 30],
        "samples_per_shard": 4,
        "workers": 1,
        "stop_margin_min": 0,
        "drive_gb": 0.001,
        **kw,
    }


def _hashes(drive):
    out = {}
    for split in synthetic.SPLITS:
        m = shards.load_manifest(drive / "data" / "synth_v1" / split)
        out.update({s["name"]: s["sha1"] for s in m["shards"]})
    return out


def _records(drive, split):
    recs = []
    split_dir = drive / "data" / "synth_v1" / split
    for s in shards.load_manifest(split_dir)["shards"]:
        path = split_dir / s["name"]
        index = shards.index_tar(path)
        for sid in shards.group_samples(index):
            recs.append(shards.read_meta(path, index, sid))
    return recs


def test_gen_synth_end_to_end(tmp_path, capsys):
    drive, marker = tmp_path / "drive", tmp_path / "m.txt"
    repo = _repo(tmp_path, "gen_synth", _synth_cfg(tmp_path), "0006_gen_synth")
    assert jobs.run("auto", drive, repo, marker, runtime="cpu") == 0
    assert "JOB FINISHED: 0006_gen_synth" in capsys.readouterr().out

    train = shards.load_manifest(drive / "data" / "synth_v1" / "train")
    assert train["complete"] and train["total_samples"] == 6 and len(train["shards"]) == 2  # 4 + 2
    for split, n in (("val", 2), ("test", 2)):
        assert shards.load_manifest(drive / "data" / "synth_v1" / split)["total_samples"] == n

    recs = _records(drive, "train")
    assert len(recs) == 6 and all(r["params"]["layout"] in ("rgb", "bgr", "pentile") for r in recs)
    assert sum(len(r["lines"]) for r in recs) > 0
    for r in recs:
        for ln in r["lines"]:
            q = np.array(ln["quad"])
            assert q.shape == (4, 2) and (q >= 0).all() and (q <= 127).all() and ln["text"].strip()

    # Each split only shows text from its own corpus lines.
    for split in ("train", "test"):
        words = {w for line in corpus.lines_for(split) for w in line.split()}
        other = {
            w for s in synthetic.SPLITS if s != split for line in corpus.lines_for(s) for w in line.split()
        }
        seen = [w for r in _records(drive, split) for ln in r["lines"] for w in ln["text"].split()]
        only_elsewhere = [w for w in seen if w not in words and w in other and len(w) > 6]
        assert not only_elsewhere, (split, only_elsewhere[:5])

    with zipfile.ZipFile(Path(marker.read_text())) as zf:
        summary = json.loads(zf.read("summary.json"))
        assert "sample_grid.jpg" in zf.namelist()
    assert summary["splits"]["train"]["complete"] and summary["first_shard_stats"]["n"] == 4


def test_gen_synth_resumes_identically(tmp_path, capsys, monkeypatch):
    ref_repo = _repo(tmp_path / "ref", "gen_synth", _synth_cfg(tmp_path / "ref"), "0006_gen_synth")
    assert jobs.run("auto", tmp_path / "ref_drive", ref_repo, tmp_path / "r.txt", runtime="cpu") == 0
    reference = _hashes(tmp_path / "ref_drive")

    repo = _repo(tmp_path, "gen_synth", _synth_cfg(tmp_path), "0006_gen_synth")
    calls = {"n": 0}

    def time_up(self, margin_s=0):
        calls["n"] += 1
        return calls["n"] > 1  # allow the first shard, then "run out of time"

    monkeypatch.setattr(jobs.JobContext, "time_up", time_up)
    assert jobs.run("auto", tmp_path / "drive", repo, tmp_path / "m.txt", runtime="cpu") == 0
    assert "TIME BUDGET REACHED" in capsys.readouterr().out
    assert len(_hashes(tmp_path / "drive")) == 1
    monkeypatch.undo()
    assert jobs.run("auto", tmp_path / "drive", repo, tmp_path / "m.txt", runtime="cpu") == 0
    assert _hashes(tmp_path / "drive") == reference


def test_make_sample_is_deterministic():
    cfg = {"seed": 3, "crops_per_page": 2, "base_size": [16, 20], "gt_jpeg_quality": 90}
    sim = load_config()
    sim["crop"] = 96
    a = synthetic.make_sample("val/000001", cfg, sim)
    synthetic._page.cache_clear()
    b = synthetic.make_sample("val/000001", cfg, sim)
    assert a == b and a[0] == "val_000001"
    assert synthetic.make_sample("val/000002", cfg, sim)[1] != a[1]


@pytest.mark.parametrize("workers", [2])
def test_sim_realism_task(tmp_path, capsys, workers):
    drive = tmp_path / "drive"
    repo = _repo(tmp_path, "gen_synth", _synth_cfg(tmp_path, workers=workers), "0006_gen_synth")
    assert jobs.run("auto", drive, repo, tmp_path / "m.txt", runtime="cpu") == 0
    # a fake "real" validation split
    rng = np.random.default_rng(0)
    val = drive / "data" / "uhdm_v1" / "val"
    m = shards.load_manifest(val)
    items = []
    for i in range(4):
        gt = (rng.random((128, 128, 3)) * 255).astype(np.uint8)
        moire = np.clip(gt.astype(int) + rng.integers(-15, 15, gt.shape), 0, 255).astype(np.uint8)
        items.append((f"k{i}", encode_jpeg(moire), encode_jpeg(gt)))
    shards.add_shard(val, m, shards.write_shard(val / "val-00000.tar", items))
    m["complete"] = True
    shards.save_manifest(val, m)

    cfg = {
        "real_split": "data/uhdm_v1/val",
        "sim_split": "data/synth_v1/train",
        "n_pairs": 4,
        "local_dir": str(tmp_path / "rl"),
    }
    repo2 = _repo(tmp_path / "r2", "sim_realism", cfg, "0007_sim_realism")
    marker = tmp_path / "m2.txt"
    assert jobs.run("auto", drive, repo2, marker, runtime="cpu") == 0
    assert "JOB FINISHED: 0007_sim_realism" in capsys.readouterr().out
    with zipfile.ZipFile(Path(marker.read_text())) as zf:
        names = set(zf.namelist())
        summary = json.loads(zf.read("summary.json"))
    assert {"spectra.png", "grid_uhdm.jpg", "grid_synthetic.jpg"} <= names
    assert set(summary["ratio"]) == {"luma", "chroma"}


def test_word_spans_match_rendered_boxes():
    from screenclean.render.pages import Renderer

    page = Renderer(corpus.lines_for("train")).render("slide", seed=5, base_size=24)
    for ln in page.lines:
        spans = synthetic.word_spans(ln)
        assert [w for w, _, _ in spans] == ln.text.split()
        assert abs(spans[0][1] - ln.box[0]) < 3 and abs(spans[-1][2] - ln.box[2]) < 3
        assert all(a[2] <= b[1] for a, b in zip(spans, spans[1:], strict=False))  # left to right, no overlap


def test_lines_in_crop_keeps_only_whole_visible_words():
    from screenclean.render.pages import Line

    ln = Line("alpha beta gamma delta", (100, 50, 400, 70), "DejaVuSans", 20)
    spans = synthetic.word_spans(ln)
    # a crop showing page x in [spans[1].start - 2, spans[2].end + 2]: exactly "beta gamma"
    x_from, x_to = spans[1][1] - 2, spans[2][2] + 2
    H = np.array([[1.0, 0, -x_from], [0, 1, -40], [0, 0, 1]])
    crop = int(np.ceil(x_to - x_from)) + 1
    (entry,) = synthetic.lines_in_crop([ln], H, crop, scale=1.0)
    assert entry["text"] == "beta gamma" and entry["partial"]
