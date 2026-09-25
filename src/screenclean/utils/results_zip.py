"""Pack a job's small results into ``<id>.zip`` and unpack returned zips into the repo.

Colab never pushes to GitHub. Instead each finished job leaves a small zip on
Drive (``results_zips/<id>.zip``) and the notebook downloads it; the zip is then
dropped into the local results inbox and ingested into ``results/jobs/<id>/``.
"""

from __future__ import annotations

import logging
import re
import shutil
import zipfile
from pathlib import Path, PurePosixPath

from screenclean.utils.drive import atomic_write_text

log = logging.getLogger(__name__)

MAX_ZIP_BYTES = 10 * 1024 * 1024
DEFAULT_MARKER = Path("/content/last_results_zip.txt")
TEXT_SUFFIXES = {".json", ".txt", ".csv", ".yaml", ".yml", ".md", ".log", ".html", ".svg"}


class IngestError(ValueError):
    """A results zip failed a safety check and was not unpacked."""


def pack_results(results_dir: str | Path, zip_path: str | Path, max_bytes: int = MAX_ZIP_BYTES) -> Path:
    """Zip every file under ``results_dir`` into ``zip_path``, staying under ``max_bytes``.

    Files are added smallest first; any file that would push the archive over the
    limit is skipped with a warning (the essentials - status, summary, logs - are tiny).
    """
    results_dir, zip_path = Path(results_dir), Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    files = sorted((p for p in results_dir.rglob("*") if p.is_file()), key=lambda p: p.stat().st_size)
    tmp = zip_path.with_name(zip_path.name + ".tmp")
    total = 0
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in files:
            size = p.stat().st_size
            if total + size > max_bytes:
                log.warning("Skipping %s (%d bytes): results zip would exceed %d bytes", p, size, max_bytes)
                continue
            zf.write(p, p.relative_to(results_dir).as_posix())
            total += size
    tmp.replace(zip_path)
    return zip_path


def write_marker(zip_path: str | Path, marker: str | Path = DEFAULT_MARKER) -> None:
    """Record the zip's path so the notebook's download cell can find it."""
    atomic_write_text(marker, str(zip_path))


def clear_marker(marker: str | Path = DEFAULT_MARKER) -> None:
    """Remove a marker left by an earlier run."""
    Path(marker).unlink(missing_ok=True)


def _safe_member(name: str) -> PurePosixPath:
    p = PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts or ":" in name:
        raise IngestError(f"unsafe path in zip: {name!r}")
    return p


def ingest_zip(
    zip_path: str | Path,
    repo_root: str | Path,
    max_bytes: int = MAX_ZIP_BYTES,
    forbid_pattern: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Unpack a job's results zip into ``<repo_root>/results/jobs/<id>/``.

    Checks, in order: the zip is at most ``max_bytes``; every member path stays
    inside the target folder; if ``forbid_pattern`` (a regex, case-insensitive) is
    given, no member name or text file matches it. Nothing is written unless all
    checks pass. Returns the target folder.
    """
    zip_path = Path(zip_path)
    size = zip_path.stat().st_size
    if size > max_bytes:
        raise IngestError(f"{zip_path.name} is {size} bytes, over the {max_bytes}-byte limit")
    job_id = zip_path.stem
    dest = Path(repo_root) / "results" / "jobs" / job_id
    replace = dest.exists() and any(dest.iterdir())
    if replace and not overwrite:
        raise IngestError(f"{dest} already exists (pass overwrite=True to replace it)")
    forbid = re.compile(forbid_pattern, re.IGNORECASE) if forbid_pattern else None

    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.infolist() if not m.is_dir()]
        for m in members:
            p = _safe_member(m.filename)
            if forbid and forbid.search(m.filename):
                raise IngestError(f"forbidden text in file name: {m.filename}")
            if forbid and p.suffix.lower() in TEXT_SUFFIXES:
                text = zf.read(m).decode("utf-8", errors="replace")
                hit = forbid.search(text)
                if hit:
                    raise IngestError(f"forbidden text {hit.group(0)!r} in {m.filename}")
        if replace:  # all checks passed: drop the old results so no stale files remain
            shutil.rmtree(dest)
        for m in members:
            target = dest / _safe_member(m.filename)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zf.read(m))
    log.info("Ingested %s into %s (%d files)", zip_path.name, dest, len(members))
    return dest
