"""End-to-end test of the prepare_uhdm task on a tiny fake UHDM (split archives, local 'downloads')."""

import json
import zipfile
from pathlib import Path

import pytest
import yaml

from screenclean import jobs
from screenclean.data import prepare_uhdm, shards


def _setup_repo(tmp_path, fake_uhdm, **overrides):
    repo = tmp_path / "repo"
    (repo / "jobs" / "queue").mkdir(parents=True)
    (repo / "configs").mkdir()
    cfg = {
        "drive_out": "data/uhdm_v1",
        "local_dir": str(tmp_path / "local"),
        "seed": 0,
        "crop": 64,
        "jpeg_quality": 90,
        "stop_margin_min": 0,
        "download_backoff_s": 0,
        "download_workers": 3,
        "train": {"n_full": 2, "n_half": 1, "pairs_per_shard": 4, "drive_gb": 0.001},
        "val": {"n_images": 3, "crops_per_image": 2, "pairs_per_shard": 10, "drive_gb": 0.001},
        "dev100": {"n_images": 2, "pairs_per_shard": 1, "drive_gb": 0.001},
        "source": {"path": str(fake_uhdm["zip"]), "train_prefix": "train/", "test_prefix": "test/"},
        **overrides,
    }
    (repo / "configs" / "uhdm.yaml").write_text(yaml.safe_dump(cfg))
    spec = {
        "id": "0002_prepare_uhdm",
        "task": "prepare_uhdm",
        "runtime": "cpu",
        "config": "configs/uhdm.yaml",
        "max_minutes": 30,
        "resume": True,
    }
    (repo / "jobs" / "queue" / "0002_prepare_uhdm.yaml").write_text(yaml.safe_dump(spec))
    return repo


def _run(repo, drive, marker):
    return jobs.run("0002_prepare_uhdm", drive, repo, marker, runtime="cpu")


def _shard_hashes(drive):
    out = {}
    for split in ("train", "val", "dev100"):
        m = shards.load_manifest(drive / "data" / "uhdm_v1" / split)
        out.update({s["name"]: s["sha1"] for s in m["shards"]})
    return out


def test_prepare_end_to_end(tmp_path, fake_uhdm, capsys):
    repo, drive, marker = _setup_repo(tmp_path, fake_uhdm), tmp_path / "drive", tmp_path / "m.txt"
    assert _run(repo, drive, marker) == 0
    assert "JOB FINISHED: 0002_prepare_uhdm" in capsys.readouterr().out
    out = drive / "data" / "uhdm_v1"

    train_m = shards.load_manifest(out / "train")
    val_m = shards.load_manifest(out / "val")
    dev_m = shards.load_manifest(out / "dev100")
    assert train_m["complete"] and val_m["complete"] and dev_m["complete"]
    # 12 train images - 3 val = 9 -> shards of 4 images -> 3 shards; 3 crops each
    assert len(train_m["shards"]) == 3 and train_m["total_samples"] == 27
    assert val_m["total_samples"] == 6
    assert len(dev_m["shards"]) == 2 and dev_m["total_samples"] == 2

    val_keys = (out / "splits" / "uhdm_val.txt").read_text().split()
    train_keys = (out / "splits" / "uhdm_train.txt").read_text().split()
    assert len(val_keys) == 3 and len(train_keys) == 9 and not set(val_keys) & set(train_keys)
    assert all(k.startswith("train/pair_") for k in train_keys)
    train_sources = {k for s in train_m["shards"] for k in s["sources"]}
    assert train_sources == set(train_keys)
    dev_keys = (out / "splits" / "uhdm_dev100.txt").read_text().split()
    assert len(dev_keys) == 2 and all(k.startswith("test/") for k in dev_keys)
    assert len((out / "splits" / "uhdm_test.txt").read_text().split()) == 6

    # dev100 keeps the original JPEG bytes
    dev_shard = out / "dev100" / dev_m["shards"][0]["name"]
    idx = shards.index_tar(dev_shard)
    key = dev_m["shards"][0]["sources"][0]
    name = shards.sample_id(key) + ".gt.jpg"
    original = fake_uhdm["src"] / f"{key}_gt.jpg"  # key test/NNNN -> src/test/NNNN_gt.jpg
    assert shards.read_member(dev_shard, *idx[name]) == original.read_bytes()

    with zipfile.ZipFile(Path(marker.read_text())) as zf:
        names = set(zf.namelist())
        assert {"sample_grid.jpg", "splits/uhdm_val.txt", "splits/uhdm_dev100.txt", "summary.json"} <= names
        summary = json.loads(zf.read("summary.json"))
    assert summary["splits"]["train"]["complete"] and summary["counts"]["uhdm_train"] == 9
    assert summary["source"]["members"] == 40 and summary["source"]["downloaded_mb"] > 0

    # Running again changes nothing and doesn't even open the archive.
    before = _shard_hashes(drive)
    assert _run(repo, drive, marker) == 0
    with zipfile.ZipFile(Path(marker.read_text())) as zf:
        assert json.loads(zf.read("summary.json"))["source"] == {}
    assert _shard_hashes(drive) == before


def test_prepare_stops_early_and_resumes_identically(tmp_path, fake_uhdm, capsys, monkeypatch):
    # Reference: one uninterrupted run.
    ref_repo = _setup_repo(tmp_path / "ref", fake_uhdm)
    assert _run(ref_repo, tmp_path / "ref_drive", tmp_path / "ref.txt") == 0
    reference = _shard_hashes(tmp_path / "ref_drive")

    # Interrupted run: stop after 3 shards have been written.
    repo, drive, marker = _setup_repo(tmp_path, fake_uhdm), tmp_path / "drive", tmp_path / "m.txt"
    calls = {"n": 0}
    real_check = prepare_uhdm.Preparer.check_time

    def stop_after_three(self):
        calls["n"] += 1
        if calls["n"] > 3:
            raise prepare_uhdm.StopEarly("test stop")
        return real_check(self)

    monkeypatch.setattr(prepare_uhdm.Preparer, "check_time", stop_after_three)
    capsys.readouterr()
    assert _run(repo, drive, marker) == 0
    out = capsys.readouterr().out
    assert "Stopped early" in out and "TIME BUDGET REACHED" in out
    assert not marker.exists()
    assert jobs.StatusStore(jobs.DriveLayout(drive)).state("0002_prepare_uhdm") == "partial"
    assert len(_shard_hashes(drive)) == 3

    monkeypatch.setattr(prepare_uhdm.Preparer, "check_time", real_check)
    assert _run(repo, drive, marker) == 0
    assert "JOB FINISHED" in capsys.readouterr().out
    assert _shard_hashes(drive) == reference  # same bytes as the uninterrupted run


def test_prepare_network_loss_is_partial_then_resumes(tmp_path, fake_uhdm, capsys, monkeypatch):
    repo, drive, marker = _setup_repo(tmp_path, fake_uhdm), tmp_path / "drive", tmp_path / "m.txt"

    real_read = prepare_uhdm.ZipSource.read
    calls = {"n": 0}

    def flaky(self, name):
        calls["n"] += 1
        if calls["n"] > 6:  # the network goes away after a few files
            raise prepare_uhdm.DownloadError("server unreachable", retry_later=True)
        return real_read(self, name)

    monkeypatch.setattr(prepare_uhdm.ZipSource, "read", flaky)
    assert _run(repo, drive, marker) == 0
    out = capsys.readouterr().out
    assert "Download stopped: server unreachable" in out and "TIME BUDGET REACHED" in out

    monkeypatch.setattr(prepare_uhdm.ZipSource, "read", real_read)
    assert _run(repo, drive, marker) == 0
    assert "JOB FINISHED" in capsys.readouterr().out


def test_prepare_fails_cleanly_when_drive_is_full(tmp_path, fake_uhdm, capsys):
    repo = _setup_repo(tmp_path, fake_uhdm, dev100={"n_images": 2, "drive_gb": 1e9})
    assert _run(repo, tmp_path / "drive", tmp_path / "m.txt") == jobs.EXIT_FAILED
    assert "Google Drive has" in capsys.readouterr().out


@pytest.mark.parametrize("job_file", ["0002_prepare_uhdm.yaml"])
def test_committed_job_and_config_are_consistent(job_file):
    root = Path(__file__).resolve().parents[1]
    spec = jobs.load_spec(root / "jobs" / "queue" / job_file)
    cfg = yaml.safe_load((root / spec.config).read_text())
    assert spec.runtime == "cpu" and spec.resume
    assert "/resolve/" in cfg["source"]["url"] and "/main/" not in cfg["source"]["url"]  # pinned revision
