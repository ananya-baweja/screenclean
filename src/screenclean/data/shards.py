"""Tar shards of image pairs, plus a manifest per split.

Copying thousands of small files over the Google Drive mount is very slow, while a
few ~500 MB files copy quickly. So datasets are stored as plain (uncompressed) tar
files. Each sample is two members::

    <sample_id>.moire.jpg
    <sample_id>.gt.jpg

``manifest.json`` next to the shards lists every finished shard with its size and
sha1. Writing is resumable: a shard that's already in the manifest is never rewritten.
"""

from __future__ import annotations

import io
import re
import tarfile
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from screenclean.utils.drive import atomic_write_json, read_json, sha1_file

KINDS = ("moire", "gt")
_CROP_SUFFIX = re.compile(r"--c\d+$")
MANIFEST = "manifest.json"


def sample_id(key: str, index: int | None = None) -> str:
    """File-name-safe sample id from an image key and optional crop index.

    ``("train/pair_22/0254", 1)`` -> ``train__pair_22__0254--c1``
    """
    base = key.replace("/", "__")
    return base if index is None else f"{base}--c{index}"


def source_key(sid: str) -> str:
    """Inverse of :func:`sample_id` (drops the crop index)."""
    return _CROP_SUFFIX.sub("", sid).replace("__", "/")


def write_shard(path: str | Path, samples: Iterable[tuple[str, bytes, bytes]]) -> dict[str, Any]:
    """Write ``(sample_id, moire_bytes, gt_bytes)`` samples to a tar file. Returns its manifest entry."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with tarfile.open(path, "w", format=tarfile.USTAR_FORMAT) as tf:
        for sid, moire, gt in samples:
            for kind, data in zip(KINDS, (moire, gt), strict=True):
                info = tarfile.TarInfo(f"{sid}.{kind}.jpg")
                info.size, info.mtime, info.mode = len(data), 0, 0o644
                tf.addfile(info, io.BytesIO(data))
            n += 1
    return {"name": path.name, "samples": n, "bytes": path.stat().st_size, "sha1": sha1_file(path)}


def index_tar(path: str | Path) -> dict[str, tuple[int, int]]:
    """Map member name -> (data offset, size), for fast random access without extracting."""
    with tarfile.open(path, "r:") as tf:
        return {m.name: (m.offset_data, m.size) for m in tf if m.isfile()}


def read_member(path: str | Path, offset: int, size: int) -> bytes:
    with open(path, "rb") as f:
        f.seek(offset)
        return f.read(size)


def group_samples(names: Iterable[str]) -> dict[str, dict[str, str]]:
    """Group member names into ``{sample_id: {"moire": name, "gt": name}}``, dropping incomplete samples."""
    groups: dict[str, dict[str, str]] = {}
    for name in names:
        stem, _, ext = name.rpartition(".")
        sid, _, kind = stem.rpartition(".")
        if ext == "jpg" and kind in KINDS:
            groups.setdefault(sid, {})[kind] = name
    return {sid: g for sid, g in groups.items() if len(g) == len(KINDS)}


# --------------------------------------------------------------------------- manifest


def load_manifest(split_dir: str | Path) -> dict[str, Any]:
    return read_json(Path(split_dir) / MANIFEST, default=None) or {"shards": [], "complete": False}


def save_manifest(split_dir: str | Path, manifest: dict[str, Any]) -> None:
    manifest["updated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest["total_samples"] = sum(s["samples"] for s in manifest["shards"])
    manifest["total_bytes"] = sum(s["bytes"] for s in manifest["shards"])
    atomic_write_json(Path(split_dir) / MANIFEST, manifest)


def finished_shards(split_dir: str | Path, manifest: dict[str, Any]) -> set[str]:
    """Names of shards recorded in the manifest whose file exists with the recorded size."""
    split_dir = Path(split_dir)
    done = set()
    for s in manifest.get("shards", []):
        p = split_dir / s["name"]
        if p.exists() and p.stat().st_size == s["bytes"]:
            done.add(s["name"])
    return done


def add_shard(split_dir: str | Path, manifest: dict[str, Any], entry: dict[str, Any]) -> None:
    """Record a finished shard (replacing an older entry with the same name) and save the manifest."""
    manifest["shards"] = [s for s in manifest["shards"] if s["name"] != entry["name"]] + [entry]
    manifest["shards"].sort(key=lambda s: s["name"])
    save_manifest(split_dir, manifest)
