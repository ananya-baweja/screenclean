"""Google Drive folder layout and crash-safe file writes.

On Colab, Drive is mounted at ``/content/drive`` and the project root is
``/content/drive/MyDrive/screenclean``. Locally (tests) any folder works.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def atomic_write_text(path: str | Path, text: str) -> Path:
    """Write text to a temp file, then rename it, so readers never see a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    return path


def atomic_write_json(path: str | Path, obj: Any) -> Path:
    """Write ``obj`` as pretty JSON with :func:`atomic_write_text`."""
    return atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")


def sha1_file(path: str | Path, chunk: int = 1 << 20) -> str:
    """SHA-1 hex digest of a file (used to check shard copies)."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def copy_atomic(src: str | Path, dst: str | Path) -> Path:
    """Copy ``src`` to ``dst`` via a temp name, so ``dst`` is either absent or complete."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)
    return dst


def stage_shards(split_dir: str | Path, dest_dir: str | Path, verify_sha1: bool = False) -> list[Path]:
    """Copy a split's shards (listed in its ``manifest.json``) from Drive to fast local disk.

    Shards already present locally with the right size are skipped. With
    ``verify_sha1`` every local copy is also checked against the manifest hash.
    Returns the local shard paths in manifest order.
    """
    split_dir, dest_dir = Path(split_dir), Path(dest_dir)
    manifest = read_json(split_dir / "manifest.json")
    if not manifest or not manifest.get("shards"):
        raise FileNotFoundError(f"no shards listed in {split_dir / 'manifest.json'}")
    if not manifest.get("complete", False):
        log.warning("%s is not marked complete; staging the shards that exist", split_dir)
    out = []
    for s in manifest["shards"]:
        dst = dest_dir / s["name"]
        if not (dst.exists() and dst.stat().st_size == s["bytes"]):
            log.info("staging %s (%.0f MB)", s["name"], s["bytes"] / 1e6)
            copy_atomic(split_dir / s["name"], dst)
        if dst.stat().st_size != s["bytes"]:
            raise OSError(f"size mismatch after copying {s['name']}")
        if verify_sha1 and sha1_file(dst) != s["sha1"]:
            raise OSError(f"sha1 mismatch for {s['name']}")
        out.append(dst)
    return out


def read_json(path: str | Path, default: Any = None) -> Any:
    """Read a JSON file, or return ``default`` if it doesn't exist."""
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class DriveLayout:
    """Standard folders under the Drive project root."""

    root: Path

    @property
    def job_status(self) -> Path:
        return self.root / "job_status"

    @property
    def results_zips(self) -> Path:
        return self.root / "results_zips"

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    @property
    def data(self) -> Path:
        return self.root / "data"

    def ensure(self) -> DriveLayout:
        """Create the standard folders if they are missing."""
        for p in (self.job_status, self.results_zips, self.runs, self.data):
            p.mkdir(parents=True, exist_ok=True)
        return self
