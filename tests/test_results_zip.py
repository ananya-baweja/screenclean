import zipfile

import pytest

from screenclean.utils import results_zip as rz


def _make_results(tmp_path):
    d = tmp_path / "results"
    (d / "figs").mkdir(parents=True)
    (d / "summary.json").write_text('{"psnr": 30.1}')
    (d / "figs" / "a.png").write_bytes(b"\x89PNG" + b"0" * 100)
    return d


def test_pack_and_ingest_roundtrip(tmp_path):
    zp = rz.pack_results(_make_results(tmp_path), tmp_path / "zips" / "0001_x.zip")
    with zipfile.ZipFile(zp) as zf:
        assert sorted(zf.namelist()) == ["figs/a.png", "summary.json"]
    repo = tmp_path / "repo"
    dest = rz.ingest_zip(zp, repo)
    assert dest == repo / "results" / "jobs" / "0001_x"
    assert (dest / "summary.json").read_text() == '{"psnr": 30.1}'
    assert (dest / "figs" / "a.png").exists()
    with pytest.raises(rz.IngestError, match="already exists"):
        rz.ingest_zip(zp, repo)
    rz.ingest_zip(zp, repo, overwrite=True)


def test_pack_skips_files_over_limit(tmp_path):
    d = tmp_path / "r"
    d.mkdir()
    (d / "small.txt").write_text("ok")
    (d / "big.bin").write_bytes(b"0" * 5000)
    zp = rz.pack_results(d, tmp_path / "z.zip", max_bytes=1000)
    with zipfile.ZipFile(zp) as zf:
        assert zf.namelist() == ["small.txt"]


def test_ingest_rejects_oversized_zip(tmp_path):
    zp = rz.pack_results(_make_results(tmp_path), tmp_path / "0002_y.zip")
    with pytest.raises(rz.IngestError, match="limit"):
        rz.ingest_zip(zp, tmp_path / "repo", max_bytes=10)


def test_ingest_rejects_path_traversal(tmp_path):
    zp = tmp_path / "0003_evil.zip"
    with zipfile.ZipFile(zp, "w") as zf:
        zf.writestr("../outside.txt", "nope")
    with pytest.raises(rz.IngestError, match="unsafe"):
        rz.ingest_zip(zp, tmp_path / "repo")
    assert not (tmp_path / "repo").exists()


def test_ingest_forbid_pattern(tmp_path):
    zp = tmp_path / "0004_words.zip"
    with zipfile.ZipFile(zp, "w") as zf:
        zf.writestr("ok.json", "{}")
        zf.writestr("log_tail.txt", "line one\nmentions Forbiddenword here\n")
    with pytest.raises(rz.IngestError, match="forbidden"):
        rz.ingest_zip(zp, tmp_path / "repo", forbid_pattern="forbiddenword|other")
    assert not (tmp_path / "repo").exists()
    rz.ingest_zip(zp, tmp_path / "repo", forbid_pattern="notpresent")


def test_marker(tmp_path):
    m = tmp_path / "marker.txt"
    rz.write_marker(tmp_path / "x.zip", m)
    assert m.read_text() == str(tmp_path / "x.zip")
    rz.clear_marker(m)
    rz.clear_marker(m)  # no error when already gone
    assert not m.exists()
