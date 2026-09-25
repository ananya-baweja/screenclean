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
import json
import logging
import re
import tarfile
import time
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from screenclean.utils.drive import atomic_write_json, copy_atomic, read_json, sha1_file

log = logging.getLogger(__name__)

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


def _add(tf: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size, info.mtime, info.mode = len(data), 0, 0o644
    tf.addfile(info, io.BytesIO(data))


def write_shard(path: str | Path, samples: Iterable[tuple]) -> dict[str, Any]:
    """Write samples to a tar file and return its manifest entry.

    Each sample is ``(sample_id, moire_bytes, gt_bytes)`` or ``(sample_id, moire_bytes, gt_bytes,
    meta_bytes)``; ``meta_bytes`` (JSON) is stored as ``<sample_id>.json``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with tarfile.open(path, "w", format=tarfile.USTAR_FORMAT) as tf:
        for sid, moire, gt, *meta in samples:
            for kind, data in zip(KINDS, (moire, gt), strict=True):
                _add(tf, f"{sid}.{kind}.jpg", data)
            if meta:
                _add(tf, f"{sid}.json", meta[0])
            n += 1
    return {"name": path.name, "samples": n, "bytes": path.stat().st_size, "sha1": sha1_file(path)}


def read_meta(path: str | Path, index: dict[str, tuple[int, int]], sid: str) -> dict[str, Any] | None:
    """The JSON record stored with a sample, or None if it has none."""
    entry = index.get(f"{sid}.json")
    return None if entry is None else json.loads(read_member(path, *entry))


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


def build_split(
    split_dir: str | Path,
    split: str,
    groups: list[list[str]],
    make_samples: Callable[[list[str]], Iterable[tuple]],
    params: dict[str, Any],
    tmp_dir: str | Path,
    before_shard: Callable[[], None] | None = None,
    after_shard: Callable[[int, int, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Write one shard per group of source keys, skipping shards that are already finished.

    A shard counts as finished only if its file is intact (size as recorded) and it holds exactly
    the planned group. If ``params`` changed since the last run, everything is rebuilt.
    ``before_shard()`` may raise to stop early (time budget); ``after_shard(i, n, entry)`` reports
    progress. The shard is built in ``tmp_dir`` (fast local disk), then copied to ``split_dir``.
    """
    split_dir, tmp_dir = Path(split_dir), Path(tmp_dir)
    manifest = load_manifest(split_dir)
    if manifest.get("params") not in (None, params):
        log.warning("%s: settings changed since the last run; rebuilding all shards", split)
        manifest["shards"] = []
    manifest.update(split=split, planned_shards=len(groups), params=params)
    manifest["complete"] = False
    planned = {f"{split}-{gi:05d}.tar": keys for gi, keys in enumerate(groups)}
    intact = finished_shards(split_dir, manifest)
    manifest["shards"] = [
        s for s in manifest["shards"] if s["name"] in intact and s.get("sources") == planned.get(s["name"])
    ]
    done = {s["name"] for s in manifest["shards"]}
    save_manifest(split_dir, manifest)
    for gi, keys in enumerate(groups):
        name = f"{split}-{gi:05d}.tar"
        if name in done:
            continue
        if before_shard:
            before_shard()
        t0 = time.monotonic()
        local_path = tmp_dir / name
        entry = write_shard(local_path, make_samples(keys))
        entry["sources"] = keys
        copy_atomic(local_path, split_dir / name)
        local_path.unlink()
        add_shard(split_dir, manifest, entry)
        log.info(
            "%s (%d/%d): %d samples, %.0f MB, %.0f s",
            name,
            gi + 1,
            len(groups),
            entry["samples"],
            entry["bytes"] / 1e6,
            time.monotonic() - t0,
        )
        if after_shard:
            after_shard(gi, len(groups), entry)
    manifest["complete"] = True
    save_manifest(split_dir, manifest)
    return manifest


def add_shard(split_dir: str | Path, manifest: dict[str, Any], entry: dict[str, Any]) -> None:
    """Record a finished shard (replacing an older entry with the same name) and save the manifest."""
    manifest["shards"] = [s for s in manifest["shards"] if s["name"] != entry["name"]] + [entry]
    manifest["shards"].sort(key=lambda s: s["name"])
    save_manifest(split_dir, manifest)
