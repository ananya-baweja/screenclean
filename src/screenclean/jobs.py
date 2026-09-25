"""Job queue runner for Google Colab.

The repo holds job specs in ``jobs/queue/NNNN_<name>.yaml``. The Colab notebook
calls ``python -m screenclean jobs run``, which:

1. picks a job (named explicitly, or the first runnable one in ``auto`` mode),
2. checks the runtime type (CPU / GPU) matches the job,
3. runs the task within its wall-clock budget, writing to Google Drive,
4. packs a small results zip and records its path for the notebook's download cell,
5. prints exactly one final line: ``JOB FINISHED``, ``TIME BUDGET REACHED`` or ``JOB FAILED``.

Job states live on Drive in ``job_status/<id>.json`` (``running``, ``done``,
``partial``, ``failed``), because results only reach the repo later. The runner
never pushes anything to GitHub.
"""

from __future__ import annotations

import dataclasses
import importlib
import logging
import shutil
import subprocess
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from screenclean.utils.drive import DriveLayout, atomic_write_json, atomic_write_text, read_json
from screenclean.utils.env import collect_env, git_sha
from screenclean.utils.results_zip import DEFAULT_MARKER, clear_marker, pack_results, write_marker
from screenclean.utils.timer import Deadline

log = logging.getLogger(__name__)

RUNTIMES = ("cpu", "gpu")
STATES = ("running", "done", "partial", "failed")
LOG_TAIL_LINES = 200

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_WRONG_RUNTIME = 2


# --------------------------------------------------------------------------- specs


class SpecError(ValueError):
    """A job spec is missing, malformed or refers to an unknown task."""


@dataclass
class JobSpec:
    """One job from ``jobs/queue/`` (see ``jobs/README.md`` for the fields)."""

    id: str
    task: str
    runtime: str
    config: str | None = None
    depends_on: list[str] = field(default_factory=list)
    max_minutes: float = 60
    resume: bool = True
    attempt: int = 1
    drive_subdir: str | None = None
    notes: str = ""

    @property
    def work_subdir(self) -> str:
        """Folder under the Drive root where this job keeps checkpoints and results."""
        return self.drive_subdir or f"jobs/{self.id}"

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def load_spec(path: str | Path) -> JobSpec:
    """Load and validate one job spec file."""
    path = Path(path)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as e:
        raise SpecError(f"{path.name}: cannot read spec ({e})") from e
    if not isinstance(data, dict):
        raise SpecError(f"{path.name}: spec must be a YAML mapping")

    known = {f.name for f in dataclasses.fields(JobSpec)}
    unknown = set(data) - known
    if unknown:
        raise SpecError(f"{path.name}: unknown fields {sorted(unknown)}")
    missing = {"id", "task", "runtime"} - set(data)
    if missing:
        raise SpecError(f"{path.name}: missing fields {sorted(missing)}")

    data["depends_on"] = data.get("depends_on") or []
    spec = JobSpec(**data)
    if spec.id != path.stem:
        raise SpecError(f"{path.name}: id {spec.id!r} must match the file name {path.stem!r}")
    if spec.runtime not in RUNTIMES:
        raise SpecError(f"{path.name}: runtime must be one of {RUNTIMES}, got {spec.runtime!r}")
    if not isinstance(spec.attempt, int) or spec.attempt < 1:
        raise SpecError(f"{path.name}: attempt must be an integer >= 1")
    if not isinstance(spec.max_minutes, (int, float)) or spec.max_minutes <= 0:
        raise SpecError(f"{path.name}: max_minutes must be a positive number")
    if not isinstance(spec.depends_on, list) or not all(isinstance(d, str) for d in spec.depends_on):
        raise SpecError(f"{path.name}: depends_on must be a list of job ids")
    if not is_known_task(spec.task):
        raise SpecError(f"{path.name}: unknown task {spec.task!r}")
    return spec


def load_queue(queue_dir: str | Path) -> list[JobSpec]:
    """All specs in ``queue_dir``, sorted by file name, with dependencies checked."""
    specs = [load_spec(p) for p in sorted(Path(queue_dir).glob("*.yaml"))]
    ids = {s.id for s in specs}
    for s in specs:
        unknown = [d for d in s.depends_on if d not in ids]
        if unknown:
            raise SpecError(f"{s.id}: depends on unknown jobs {unknown}")
    return specs


# --------------------------------------------------------------------------- tasks


@dataclass
class TaskResult:
    """What a task returns: ``done`` or ``partial`` (budget reached, resume later)."""

    state: str = "done"
    summary: dict[str, Any] = field(default_factory=dict)


@dataclass
class JobContext:
    """Everything a task needs while it runs."""

    spec: JobSpec
    layout: DriveLayout
    repo_root: Path
    work_dir: Path
    results_dir: Path
    deadline: Deadline
    config: dict[str, Any]

    def time_up(self, margin_s: float = 120.0) -> bool:
        """True when the task should checkpoint and return ``partial``."""
        return self.deadline.expired(margin_s)


TaskFn = Callable[[JobContext], TaskResult]
TASKS: dict[str, TaskFn] = {}

# Tasks implemented in other modules, imported only when a job needs them.
TASK_MODULES: dict[str, str] = {}


def register(name: str) -> Callable[[TaskFn], TaskFn]:
    """Decorator that adds a task function to the registry."""

    def deco(fn: TaskFn) -> TaskFn:
        TASKS[name] = fn
        return fn

    return deco


def is_known_task(name: str) -> bool:
    return name in TASKS or name in TASK_MODULES


def get_task(name: str) -> TaskFn:
    """Look up a task, importing its module on first use."""
    if name not in TASKS and name in TASK_MODULES:
        importlib.import_module(TASK_MODULES[name])
    try:
        return TASKS[name]
    except KeyError:
        raise SpecError(f"unknown task {name!r}") from None


@register("hello")
def hello_task(ctx: JobContext) -> TaskResult:
    """Smoke test: write a file to Drive, read it back, and report the environment."""
    probe = ctx.work_dir / "hello_probe.txt"
    message = f"hello from {ctx.spec.id} at {datetime.now(timezone.utc).isoformat(timespec='seconds')}"
    atomic_write_text(probe, message)
    read_back = probe.read_text(encoding="utf-8")
    if read_back != message:
        raise RuntimeError(f"Drive read-back mismatch: wrote {message!r}, read {read_back!r}")
    log.info("Drive write/read OK: %s", probe)

    env = collect_env(ctx.repo_root, disk_path=ctx.work_dir)
    (ctx.results_dir / "hello.txt").write_text(message + "\n", encoding="utf-8")
    return TaskResult(
        "done",
        {
            "drive_write_read_ok": True,
            "python": env["python"],
            "torch": env["packages"].get("torch"),
            "gpu": env["gpu"].get("name") if env["gpu"].get("available") else None,
            "cpu_count": env["cpu_count"],
            "ram_gb": env["ram_gb"],
            "disk_free_gb": env["disk_free_gb"],
        },
    )


# --------------------------------------------------------------------------- status


class StatusStore:
    """Job states kept on Drive as ``job_status/<id>.json``."""

    def __init__(self, layout: DriveLayout):
        self.dir = layout.job_status

    def path(self, job_id: str) -> Path:
        return self.dir / f"{job_id}.json"

    def get(self, job_id: str) -> dict[str, Any]:
        return read_json(self.path(job_id), default={}) or {}

    def state(self, job_id: str) -> str | None:
        return self.get(job_id).get("state")

    def set(self, job_id: str, **fields: Any) -> dict[str, Any]:
        status = {**self.get(job_id), **fields, "id": job_id, "updated_utc": _now()}
        atomic_write_json(self.path(job_id), status)
        return status


def select_job(
    queue: list[JobSpec], store: StatusStore, requested: str = "auto"
) -> tuple[JobSpec | None, list[str]]:
    """Choose the job to run and explain why other jobs were skipped.

    ``requested`` is a job id (run it regardless of state) or ``"auto"``: the first
    job that isn't done, didn't fail at its current ``attempt``, and whose
    dependencies are all done.
    """
    if requested != "auto":
        for spec in queue:
            if spec.id == requested:
                return spec, []
        raise SpecError(f"no job named {requested!r} in the queue")

    reasons = []
    for spec in queue:
        status = store.get(spec.id)
        state = status.get("state")
        if state == "done":
            continue
        if state == "failed" and status.get("attempt") == spec.attempt:
            reasons.append(f"{spec.id}: failed at attempt {spec.attempt} (needs a fix and an attempt bump)")
            continue
        waiting = [d for d in spec.depends_on if store.state(d) != "done"]
        if waiting:
            reasons.append(f"{spec.id}: waiting for {', '.join(waiting)}")
            continue
        return spec, reasons
    return None, reasons


def detect_runtime() -> str:
    """``"gpu"`` if a CUDA GPU is visible, else ``"cpu"``."""
    try:
        import torch

        return "gpu" if torch.cuda.is_available() else "cpu"
    except ImportError:
        pass
    if shutil.which("nvidia-smi"):
        ok = subprocess.run(["nvidia-smi", "-L"], capture_output=True, check=False).returncode == 0
        return "gpu" if ok else "cpu"
    return "cpu"


# --------------------------------------------------------------------------- running


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _tail(path: Path, n: int = LOG_TAIL_LINES) -> str:
    if not path.exists():
        return ""
    with open(path, encoding="utf-8", errors="replace") as f:
        return "".join(deque(f, maxlen=n))


def _load_config(spec: JobSpec, repo_root: Path) -> dict[str, Any]:
    if not spec.config:
        return {}
    cfg_path = repo_root / spec.config
    if not cfg_path.exists():
        raise SpecError(f"{spec.id}: config file {spec.config} not found")
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}


def run_job(
    spec: JobSpec,
    drive_root: str | Path,
    repo_root: str | Path = ".",
    marker: str | Path = DEFAULT_MARKER,
    runtime: str = "cpu",
) -> int:
    """Run one job end to end and print the final status line. Returns an exit code."""
    layout = DriveLayout(Path(drive_root)).ensure()
    repo_root = Path(repo_root).resolve()
    store = StatusStore(layout)
    work_dir = layout.root / spec.work_subdir
    results_dir = work_dir / "results"

    previous = store.state(spec.id)
    resuming = spec.resume and previous in ("partial", "running")
    if results_dir.exists() and not resuming:
        shutil.rmtree(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    log_path = work_dir / "job.log"
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    if root_logger.level > logging.INFO or root_logger.level == logging.NOTSET:
        root_logger.setLevel(logging.INFO)

    started = time.monotonic()
    state, error, summary = "failed", None, {}
    try:
        log.info(
            "Starting %s (task=%s, runtime=%s, attempt=%d, resuming=%s)",
            spec.id,
            spec.task,
            runtime,
            spec.attempt,
            resuming,
        )
        store.set(
            spec.id,
            state="running",
            attempt=spec.attempt,
            task=spec.task,
            runtime=runtime,
            started_utc=_now(),
        )
        env = collect_env(repo_root, disk_path=layout.root)
        atomic_write_json(results_dir / "env.json", env)
        config = _load_config(spec, repo_root)
        atomic_write_text(
            results_dir / "config_resolved.yaml",
            yaml.safe_dump({"job": spec.to_dict(), "config": config}, sort_keys=False),
        )
        ctx = JobContext(
            spec=spec,
            layout=layout,
            repo_root=repo_root,
            work_dir=work_dir,
            results_dir=results_dir,
            deadline=Deadline(spec.max_minutes * 60),
            config=config,
        )
        result = get_task(spec.task)(ctx)
        if result.state not in ("done", "partial"):
            raise RuntimeError(f"task returned invalid state {result.state!r}")
        state, summary = result.state, result.summary
    except Exception as e:  # noqa: BLE001 - any task error must end as a clean 'failed' status
        log.exception("Job %s failed", spec.id)
        error = f"{type(e).__name__}: {e}"

    elapsed = round(time.monotonic() - started, 1)
    log.info("Job %s ended as %s after %.1f s", spec.id, state, elapsed)
    root_logger.removeHandler(handler)
    handler.close()

    status = store.set(
        spec.id, state=state, attempt=spec.attempt, error=error, elapsed_s=elapsed, finished_utc=_now()
    )
    atomic_write_json(results_dir / "status.json", status)
    atomic_write_json(
        results_dir / "summary.json",
        {
            "id": spec.id,
            "task": spec.task,
            "state": state,
            "attempt": spec.attempt,
            "elapsed_s": elapsed,
            "git_sha": git_sha(repo_root),
            **summary,
        },
    )
    atomic_write_text(results_dir / "log_tail.txt", _tail(log_path))

    if state == "partial":
        print(f"TIME BUDGET REACHED — run the notebook again to continue {spec.id}")
        return EXIT_OK

    zip_path = pack_results(results_dir, layout.results_zips / f"{spec.id}.zip")
    write_marker(zip_path, marker)
    if state == "done":
        print(f"JOB FINISHED: {spec.id} — the next cell downloads {spec.id}.zip; move it into results_inbox")
        return EXIT_OK
    print(f"ERROR: {error}")
    print(
        f"JOB FAILED: {spec.id} — the next cell downloads {spec.id}.zip; "
        "move it into results_inbox so the error can be checked"
    )
    return EXIT_FAILED


def run(
    requested: str,
    drive_root: str | Path,
    repo_root: str | Path = ".",
    marker: str | Path = DEFAULT_MARKER,
    runtime: str | None = None,
) -> int:
    """Entry point used by the CLI: select a job, check the runtime, run it."""
    clear_marker(marker)
    repo_root = Path(repo_root)
    layout = DriveLayout(Path(drive_root)).ensure()
    queue = load_queue(repo_root / "jobs" / "queue")
    spec, reasons = select_job(queue, StatusStore(layout), requested)
    if spec is None:
        print("Nothing to run right now.")
        for r in reasons or ["every job in the queue is done"]:
            print(f"  - {r}")
        return EXIT_OK

    runtime = runtime or detect_runtime()
    if spec.runtime == "gpu" and runtime == "cpu":
        print(
            f"Job {spec.id} needs a GPU, but this runtime has none.\n"
            "Runtime -> Change runtime type -> T4 GPU, then Runtime -> Run all."
        )
        return EXIT_WRONG_RUNTIME
    if spec.runtime == "cpu" and runtime == "gpu":
        print(f"Note: {spec.id} is a CPU job. Switch to a CPU runtime next time to save GPU quota.")
    return run_job(spec, drive_root, repo_root, marker, runtime)


def describe_queue(repo_root: str | Path, drive_root: str | Path | None = None) -> list[str]:
    """One line per queued job with its state on Drive (if a Drive root is given)."""
    store = StatusStore(DriveLayout(Path(drive_root))) if drive_root else None
    lines = []
    for spec in load_queue(Path(repo_root) / "jobs" / "queue"):
        state = (store.state(spec.id) if store else None) or "-"
        lines.append(f"{spec.id:<32} {spec.task:<14} {spec.runtime:<4} {state}")
    return lines
