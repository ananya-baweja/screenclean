import json
import zipfile
from pathlib import Path

import pytest

from screenclean import cli, jobs
from screenclean.utils.drive import DriveLayout

REPO_ROOT = Path(__file__).resolve().parents[1]


def write_spec(queue: Path, job_id: str, **fields) -> Path:
    fields = {"id": job_id, "task": "hello", "runtime": "cpu", **fields}
    lines = [f"{k}: {json.dumps(v)}" for k, v in fields.items()]
    path = queue / f"{job_id}.yaml"
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    (root / "jobs" / "queue").mkdir(parents=True)
    return root


@pytest.fixture
def fake_tasks():
    """Register throwaway tasks for a test and remove them afterwards."""
    added = []

    def add(name, fn):
        jobs.register(name)(fn)
        added.append(name)

    yield add
    for name in added:
        jobs.TASKS.pop(name, None)


def test_committed_queue_is_valid():
    specs = jobs.load_queue(REPO_ROOT / "jobs" / "queue")
    assert specs and specs[0].id == "0001_hello"
    for s in specs:
        jobs.get_task(s.task)


@pytest.mark.parametrize(
    ("fields", "match"),
    [
        ({"id": "wrong_name"}, "must match"),
        ({"runtime": "tpu"}, "runtime"),
        ({"task": "nope"}, "unknown task"),
        ({"attempt": 0}, "attempt"),
        ({"max_minutes": -1}, "max_minutes"),
        ({"colour": "red"}, "unknown fields"),
    ],
)
def test_spec_validation(repo, fields, match):
    q = repo / "jobs" / "queue"
    path = write_spec(q, "0001_a", **fields)
    with pytest.raises(jobs.SpecError, match=match):
        jobs.load_spec(path)


def test_queue_rejects_unknown_dependency(repo):
    write_spec(repo / "jobs" / "queue", "0001_a", depends_on=["0000_missing"])
    with pytest.raises(jobs.SpecError, match="unknown jobs"):
        jobs.load_queue(repo / "jobs" / "queue")


def test_select_job_auto(repo, tmp_path):
    q = repo / "jobs" / "queue"
    write_spec(q, "0001_a")
    write_spec(q, "0002_b", depends_on=["0001_a"])
    write_spec(q, "0003_c", attempt=1)
    queue = jobs.load_queue(q)
    store = jobs.StatusStore(DriveLayout(tmp_path / "drive").ensure())

    assert jobs.select_job(queue, store)[0].id == "0001_a"

    store.set("0001_a", state="failed", attempt=1)
    spec, reasons = jobs.select_job(queue, store)
    assert spec.id == "0003_c"
    assert any("0001_a: failed" in r for r in reasons)
    assert any("0002_b: waiting for 0001_a" in r for r in reasons)

    store.set("0001_a", state="done")
    assert jobs.select_job(queue, store)[0].id == "0002_b"

    for j in ("0002_b", "0003_c"):
        store.set(j, state="done")
    assert jobs.select_job(queue, store) == (None, [])

    # an explicit id runs regardless of state
    assert jobs.select_job(queue, store, "0001_a")[0].id == "0001_a"
    with pytest.raises(jobs.SpecError):
        jobs.select_job(queue, store, "9999_nope")


def test_failed_job_runs_again_after_attempt_bump(repo, tmp_path):
    q = repo / "jobs" / "queue"
    write_spec(q, "0001_a", attempt=2)
    store = jobs.StatusStore(DriveLayout(tmp_path / "drive").ensure())
    store.set("0001_a", state="failed", attempt=1)
    assert jobs.select_job(jobs.load_queue(q), store)[0].id == "0001_a"


def test_hello_end_to_end(repo, tmp_path, capsys):
    write_spec(repo / "jobs" / "queue", "0001_hello", max_minutes=5)
    drive_root, marker = tmp_path / "drive", tmp_path / "last_zip.txt"

    code = jobs.run("auto", drive_root, repo, marker, runtime="cpu")
    out = capsys.readouterr().out
    assert code == 0
    assert "JOB FINISHED: 0001_hello" in out

    status = json.loads((drive_root / "job_status" / "0001_hello.json").read_text())
    assert status["state"] == "done" and status["attempt"] == 1
    zip_path = Path(marker.read_text())
    assert zip_path == drive_root / "results_zips" / "0001_hello.zip"
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        assert {
            "status.json",
            "summary.json",
            "log_tail.txt",
            "config_resolved.yaml",
            "env.json",
            "hello.txt",
        } <= names
        summary = json.loads(zf.read("summary.json"))
        assert summary["drive_write_read_ok"] is True and summary["state"] == "done"
        assert "Starting 0001_hello" in zf.read("log_tail.txt").decode()

    # Now everything is done: nothing to run, and no stale marker left behind.
    assert jobs.run("auto", drive_root, repo, marker, runtime="cpu") == 0
    assert "Nothing to run" in capsys.readouterr().out
    assert not marker.exists()


def test_failed_task_writes_zip(repo, tmp_path, capsys, fake_tasks):
    def boom(ctx):
        raise ValueError("kaboom")

    fake_tasks("boom", boom)
    write_spec(repo / "jobs" / "queue", "0001_boom", task="boom")
    marker = tmp_path / "m.txt"
    code = jobs.run("auto", tmp_path / "drive", repo, marker, runtime="cpu")
    out = capsys.readouterr().out
    assert code == jobs.EXIT_FAILED
    assert "JOB FAILED: 0001_boom" in out and "kaboom" in out
    with zipfile.ZipFile(Path(marker.read_text())) as zf:
        assert "ValueError: kaboom" in json.loads(zf.read("status.json"))["error"]
        assert "kaboom" in zf.read("log_tail.txt").decode()
    # auto mode now skips it until the attempt is bumped
    assert jobs.run("auto", tmp_path / "drive", repo, marker, runtime="cpu") == 0
    assert "failed at attempt 1" in capsys.readouterr().out


def test_partial_then_resume(repo, tmp_path, capsys, fake_tasks):
    def slow(ctx):
        counter = ctx.work_dir / "count.txt"
        n = int(counter.read_text()) + 1 if counter.exists() else 1
        counter.write_text(str(n))
        (ctx.results_dir / f"chunk{n}.txt").write_text("x")
        return jobs.TaskResult("done" if n >= 2 else "partial", {"chunks": n})

    fake_tasks("slow", slow)
    write_spec(repo / "jobs" / "queue", "0001_slow", task="slow", resume=True)
    drive_root, marker = tmp_path / "drive", tmp_path / "m.txt"

    assert jobs.run("auto", drive_root, repo, marker, runtime="cpu") == 0
    assert "TIME BUDGET REACHED — run the notebook again to continue 0001_slow" in capsys.readouterr().out
    assert not marker.exists()

    assert jobs.run("auto", drive_root, repo, marker, runtime="cpu") == 0
    assert "JOB FINISHED: 0001_slow" in capsys.readouterr().out
    with zipfile.ZipFile(Path(marker.read_text())) as zf:
        assert {"chunk1.txt", "chunk2.txt"} <= set(zf.namelist())  # results kept across the resume


def test_gpu_job_refused_on_cpu(repo, tmp_path, capsys):
    write_spec(repo / "jobs" / "queue", "0001_gpu", runtime="gpu")
    code = jobs.run("auto", tmp_path / "drive", repo, tmp_path / "m.txt", runtime="cpu")
    assert code == jobs.EXIT_WRONG_RUNTIME
    assert "needs a GPU" in capsys.readouterr().out
    assert not (tmp_path / "drive" / "job_status" / "0001_gpu.json").exists()


def test_cpu_job_on_gpu_warns(repo, tmp_path, capsys):
    write_spec(repo / "jobs" / "queue", "0001_hello")
    assert jobs.run("auto", tmp_path / "drive", repo, tmp_path / "m.txt", runtime="gpu") == 0
    assert "save GPU quota" in capsys.readouterr().out


def test_cli_jobs_run_and_list(repo, tmp_path, capsys):
    write_spec(repo / "jobs" / "queue", "0001_hello")
    drive_root = tmp_path / "drive"
    args = [
        "jobs",
        "run",
        "--drive-root",
        str(drive_root),
        "--repo-root",
        str(repo),
        "--marker",
        str(tmp_path / "m.txt"),
        "--runtime",
        "cpu",
    ]
    assert cli.main(args) == 0
    assert "JOB FINISHED: 0001_hello" in capsys.readouterr().out
    assert cli.main(["jobs", "list", "--drive-root", str(drive_root), "--repo-root", str(repo)]) == 0
    assert "0001_hello" in capsys.readouterr().out
