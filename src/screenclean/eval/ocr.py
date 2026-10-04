"""OCR engines for the evaluation: each runs in a worker process (``eval/ocr_worker.py``).

- **PaddleOCR 3.x** (``paddle``): installed on Colab into its own folder (``pip install --target``)
  that only the worker puts on its path, so its numpy / OpenCV versions never meet PyTorch's.
- **Tesseract 5** (``tesseract``): the scan pipeline's setup (local Sauvola thresholding).
- **RapidOCR** (``rapidocr``): the same PP-OCR models on ONNX Runtime; used if PaddleOCR fails
  to install, start or pass its self-check, also with its CPU acceleration off (``paddle_safe``).

Every engine reads one clean capture-kit page first (the self-check), and the result is
recorded, so a broken install can't silently produce bad numbers.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from screenclean.eval.text_metrics import ordered_text, score

log = logging.getLogger(__name__)

WORKER = Path(__file__).with_name("ocr_worker.py")
REPO_ROOT = Path(__file__).resolve().parents[3]
SELF_CHECK_PAGE = 0  # capture-kit page used for the self-check (a slide: title + bullet lines)

# Pinned after checking the result fields of these versions (see ocr_worker.parse_paddle).
PACKAGES = {
    "paddle": ["paddlepaddle==3.3.1", "paddleocr==3.7.0"],
    "rapidocr": ["rapidocr==3.9.2", "onnxruntime==1.30.0"],
}
PACKAGES["paddle_safe"] = PACKAGES["paddle"]
PACKAGE_DIRS = {"paddle_safe": "paddle"}  # the same install
FALLBACKS = {"paddle": ["paddle_safe", "rapidocr"]}


class OcrEngineError(RuntimeError):
    """An OCR engine could not be installed, started or used."""


def install(engine: str, target: str | Path, timeout_s: float = 1800) -> dict[str, Any]:
    """``pip install --target`` the engine's pinned packages (once per folder)."""
    target = Path(target)
    done, wanted = target / ".installed", " ".join(PACKAGES[engine]) + "\n"
    if done.exists() and done.read_text(encoding="utf-8") == wanted:
        return {"ok": True, "seconds": 0.0, "cached": True}
    t0 = time.perf_counter()
    cmd = [sys.executable, "-m", "pip", "install", "-q", "--target", str(target), *PACKAGES[engine]]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, check=False)
    except subprocess.TimeoutExpired:
        return {"ok": False, "seconds": round(time.perf_counter() - t0, 1), "error": "pip timed out"}
    info: dict[str, Any] = {"ok": res.returncode == 0, "seconds": round(time.perf_counter() - t0, 1)}
    if res.returncode == 0:
        done.write_text(wanted, encoding="utf-8")
    else:
        info["error"] = "\n".join((res.stderr or res.stdout).strip().splitlines()[-15:])
    return info


class WorkerOCR:
    """One OCR worker process: ``read(path)`` returns the page's lines (``text``, ``box``, ``conf``)."""

    def __init__(
        self,
        engine: str,
        packages_dir: str | Path | None = None,
        options: dict[str, Any] | None = None,
        start_timeout_s: float = 900,
        read_timeout_s: float = 300,
        log_path: str | Path | None = None,
    ):
        self.engine, self.read_timeout_s = engine, read_timeout_s
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        if packages_dir:
            env["PYTHONPATH"] = os.pathsep.join(p for p in (str(packages_dir), env.get("PYTHONPATH")) if p)
        if log_path is None:
            fd, log_path = tempfile.mkstemp(prefix=f"ocr_{engine}_", suffix=".log")
            os.close(fd)
        self.log_path = Path(log_path)
        self._log = open(self.log_path, "a", encoding="utf-8")  # noqa: SIM115 - closed in close()
        self.proc = subprocess.Popen(
            [sys.executable, str(WORKER), "--engine", engine, "--options", json.dumps(options or {})],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._log,
            text=True,
            encoding="utf-8",
            env=env,
        )
        self._lines: queue.Queue[str | None] = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()
        try:
            hello = self._next(start_timeout_s)
        except OcrEngineError:
            self.close()
            raise
        if not hello.get("ready"):
            tail = self._log_tail()
            self.close()
            raise OcrEngineError(f"{engine} did not start: {hello.get('error')}{tail}")
        self.info = {k: v for k, v in hello.items() if k != "ready"}
        self._n = 0

    def _pump(self) -> None:
        for line in self.proc.stdout:
            self._lines.put(line)
        self._lines.put(None)

    def _next(self, timeout_s: float) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                line = self._lines.get(timeout=max(0.01, deadline - time.monotonic()))
            except queue.Empty:
                self.close()
                raise OcrEngineError(f"{self.engine}: no answer within {timeout_s:.0f} s") from None
            if line is None:
                code = self.proc.wait()
                raise OcrEngineError(f"{self.engine} worker exited (code {code}){self._log_tail()}")
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue  # stray output; replies are always JSON

    def _log_tail(self, n: int = 12) -> str:
        try:
            self._log.flush()
            tail = self.log_path.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-n:]
        except OSError:
            return ""
        return ("\n  " + "\n  ".join(tail)) if tail else ""

    def read(self, path: str | Path) -> list[dict[str, Any]]:
        self._n += 1
        rid = str(self._n)
        self.proc.stdin.write(json.dumps({"id": rid, "path": str(Path(path).resolve())}) + "\n")
        self.proc.stdin.flush()
        reply = self._next(self.read_timeout_s)
        if reply.get("id") != rid:
            raise OcrEngineError(f"{self.engine}: reply out of order ({reply.get('id')} for {rid})")
        if "error" in reply:
            raise OcrEngineError(f"{self.engine} failed on {Path(path).name}: {reply['error']}")
        return reply["lines"]

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                self.proc.stdin.close()
                self.proc.wait(timeout=30)
            except (OSError, subprocess.TimeoutExpired):
                self.proc.kill()
                self.proc.wait()
        self._log.close()

    def __enter__(self) -> WorkerOCR:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def lines_text(lines: list[dict[str, Any]]) -> str:
    """OCR lines as text in reading order (the same rule as the ground truth)."""
    return ordered_text((ln["text"], tuple(ln["box"])) for ln in lines)


def self_check(ocr: WorkerOCR, kit_dir: str | Path = REPO_ROOT / "capture_kit") -> dict[str, Any]:
    """Read one clean capture-kit page; returns its scores and the time it took."""
    kit_dir = Path(kit_dir)
    pages = json.loads((kit_dir / "pages.json").read_text(encoding="utf-8"))["pages"]
    page = next(p for p in pages if p["page_id"] == SELF_CHECK_PAGE)
    truth = ordered_text((ln["text"], tuple(ln["box"])) for ln in page["lines"])
    t0 = time.perf_counter()
    lines = ocr.read(kit_dir / "pages" / f"page_{SELF_CHECK_PAGE:03d}.png")
    return {**score(truth, lines_text(lines)), "seconds": round(time.perf_counter() - t0, 2)}


def open_engine(
    name: str,
    packages_root: str | Path | None = None,
    options: dict[str, dict[str, Any]] | None = None,
    fallbacks: dict[str, list[str]] | None = None,
    min_word_f1: float = 0.9,
    log_dir: str | Path | None = None,
) -> tuple[WorkerOCR | None, dict[str, Any]]:
    """Start engine ``name`` (installing it into ``packages_root/<engine>`` first if given), or a fallback.

    Returns the worker (None if every candidate failed) and a record of every attempt.
    """
    record: dict[str, Any] = {"requested": name, "attempts": []}
    for candidate in [name, *(FALLBACKS if fallbacks is None else fallbacks).get(name, [])]:
        attempt: dict[str, Any] = {"engine": candidate}
        record["attempts"].append(attempt)
        packages_dir = None
        if packages_root is not None and candidate in PACKAGES:
            packages_dir = Path(packages_root) / PACKAGE_DIRS.get(candidate, candidate)
            attempt["install"] = install(candidate, packages_dir)
            if not attempt["install"]["ok"]:
                continue
        try:
            log_path = Path(log_dir) / f"ocr_{candidate}.log" if log_dir else None
            ocr = WorkerOCR(candidate, packages_dir, (options or {}).get(candidate), log_path=log_path)
            attempt["versions"] = ocr.info.get("versions")
            attempt["self_check"] = check = self_check(ocr)
        except (OcrEngineError, OSError) as e:
            attempt["error"] = str(e)
            log.warning("OCR engine %s failed: %s", candidate, e)
            continue
        if check["word_f1"] < min_word_f1:
            attempt["error"] = f"self-check word F1 {check['word_f1']:.3f} < {min_word_f1}"
            ocr.close()
            continue
        record.update(engine=candidate, versions=attempt["versions"], self_check=check)
        log.info("OCR engine %s ready (self-check %s)", candidate, check)
        return ocr, record
    record["engine"] = None
    return None, record
