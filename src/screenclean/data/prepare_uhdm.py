"""Colab task ``prepare_uhdm``: build compact UHDM shards on Google Drive.

The source is one large zip (see ``configs/data/uhdm.yaml``). Only its table of
contents is read up front; images are then fetched one by one with HTTP range
requests, so nothing is extracted to disk.

Output under ``<drive root>/<drive_out>/`` (``data/uhdm_v1`` by default):

- ``dev100/``: 100 fixed full-resolution test pairs (original JPEG bytes, not re-encoded)
- ``val/``: 2 fixed 512 crops for each of 200 held-out training images
- ``train/``: 2 random full-scale 512 crops + 1 crop at half scale, for every other training image
- ``splits/``: the image lists (``uhdm_test.txt``, ``uhdm_dev100.txt``, ``uhdm_train.txt``, ``uhdm_val.txt``)

Every split has a ``manifest.json``. The task is idempotent: finished shards are
skipped (their images aren't downloaded again), and it stops cleanly (``partial``)
before the time budget runs out or if the server stays unreachable.
"""

from __future__ import annotations

import logging
import shutil
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from screenclean.data.download import DownloadError, ZipSource, open_zip_source
from screenclean.data.shards import (
    add_shard,
    finished_shards,
    group_samples,
    index_tar,
    load_manifest,
    read_member,
    sample_id,
    save_manifest,
    write_shard,
)
from screenclean.data.uhdm import (
    Pair,
    decode_pair,
    discover_pairs,
    sample_keys,
    split_train_val,
    train_crops,
    val_crops,
)
from screenclean.jobs import JobContext, TaskResult, register
from screenclean.utils.drive import copy_atomic
from screenclean.utils.image import resize_max_side
from screenclean.utils.io import decode_rgb, encode_jpeg, write_image

log = logging.getLogger(__name__)

GB = 1024**3

Sample = tuple[str, bytes, bytes]


class StopEarly(Exception):
    """Raised to leave the task with state ``partial``."""


def _chunks(items: list[str], n: int) -> list[list[str]]:
    return [items[i : i + n] for i in range(0, len(items), n)]


def _write_list(path: Path, items: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{x}\n" for x in items), encoding="utf-8")


class Preparer:
    """Runs the phases of the task; holds the config, paths, source and summary."""

    def __init__(self, ctx: JobContext):
        self.ctx = ctx
        self.cfg: dict[str, Any] = ctx.config
        self.out = ctx.layout.root / self.cfg.get("drive_out", "data/uhdm_v1")
        self.tmp = Path(self.cfg.get("local_dir", "/content/uhdm_work")) / "tmp_shards"
        self.seed = int(self.cfg.get("seed", 0))
        self.crop = int(self.cfg.get("crop", 512))
        self.quality = int(self.cfg.get("jpeg_quality", 95))
        self.workers = int(self.cfg.get("download_workers", 8))
        self.margin_s = float(self.cfg.get("stop_margin_min", 10)) * 60
        self.summary: dict[str, Any] = {"source": {}, "problems": {}, "phase_seconds": {}}
        self._source: ZipSource | None = None
        self._pairs: dict[str, Pair] = {}

    # ------------------------------------------------------------------ helpers

    @property
    def source(self) -> ZipSource:
        """Open the archive lazily: a run with nothing left to do never touches the network."""
        if self._source is None:
            spec = self.cfg["source"]
            t0 = time.monotonic()
            self._source = open_zip_source(
                spec,
                self.ctx.repo_root,
                retries=int(self.cfg.get("download_retries", 6)),
                backoff_s=float(self.cfg.get("download_backoff_s", 5)),
            )
            self.summary["source"] = {
                "location": spec.get("url") or Path(spec["path"]).name,
                "members": len(self._source.infos),
                "archive_gb": round(self._source.reader.size / GB, 2),
                "index_seconds": round(time.monotonic() - t0, 1),
            }
            log.info("source opened: %s", self.summary["source"])
        return self._source

    def pairs(self, prefix: str, label: str) -> dict[str, Pair]:
        pairs, problems = discover_pairs(self.source.names, prefix)
        if not pairs:
            raise RuntimeError(f"no image pairs found under '{prefix}' in the archive")
        self.summary["problems"][label] = problems
        log.info("%s: %d pairs (%d unmatched files)", label, len(pairs), len(problems))
        found = {p.key: p for p in pairs}
        self._pairs.update(found)
        return found

    def check_time(self) -> None:
        if self.ctx.time_up(self.margin_s):
            raise StopEarly("stopped to stay within the job's time budget")

    def check_drive_space(self, need_gb: float) -> None:
        self.out.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(self.out).free / GB
        if free < need_gb:
            raise RuntimeError(
                f"Google Drive has {free:.1f} GB free, but this step needs about {need_gb:.1f} GB. "
                "Free up Drive space (or lower crops/quality in configs/data/uhdm.yaml) and run again."
            )

    def process(self, keys: list[str], make: Callable[[str, bytes, bytes], list[Sample]]) -> Iterator[Sample]:
        """Download each pair and turn it into samples, a few pairs at a time in worker threads.

        JPEG decoding/encoding releases the GIL, so threads use both Colab CPUs. Results come
        back in key order, so shards are identical however fast each download is. A retryable
        error (network) stops the run; a bad pair is skipped and recorded.
        """

        def work(key: str) -> tuple[str, list[Sample] | None, str]:
            p = self._pairs[key]
            try:
                return key, make(key, self.source.read(p.moire), self.source.read(p.gt)), ""
            except DownloadError as e:
                if e.retry_later:
                    raise
                return key, None, str(e)
            except Exception as e:  # noqa: BLE001 - skip one bad image, keep the job going
                return key, None, str(e)

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            # Submit in small windows so only ~2x workers pairs are in memory at once.
            window = max(1, 2 * self.workers)
            for i in range(0, len(keys), window):
                for key, samples, reason in pool.map(work, keys[i : i + window]):
                    if samples is None:
                        self._skip(key, reason)
                    else:
                        yield from samples

    def _skip(self, key: str, reason: str) -> None:
        log.warning("skipping %s: %s", key, reason)
        self.summary["problems"].setdefault("skipped", []).append(f"{key}: {reason}")

    def crop_samples(self, keys: list[str], crop_fn: Callable) -> Iterator[Sample]:
        """Samples of JPEG-encoded crops; ``crop_fn(moire, gt, key)`` returns the crop pairs."""

        def make(key: str, mb: bytes, gb: bytes) -> list[Sample]:
            moire, gt = decode_pair(mb, gb)
            return [
                (sample_id(key, j), encode_jpeg(cm, self.quality), encode_jpeg(cg, self.quality))
                for j, (cm, cg) in enumerate(crop_fn(moire, gt, key))
            ]

        return self.process(keys, make)

    def original_samples(self, keys: list[str]) -> Iterator[Sample]:
        """Samples with the original JPEG bytes (after checking both decode and match in size)."""

        def make(key: str, mb: bytes, gb: bytes) -> list[Sample]:
            decode_pair(mb, gb)
            return [(sample_id(key), mb, gb)]

        return self.process(keys, make)

    def build_split(
        self,
        split: str,
        groups: list[list[str]],
        make_samples: Callable[[list[str]], Iterator[Sample]],
        params: dict,
    ) -> None:
        """Write one shard per group of image keys, skipping shards that are already finished."""
        split_dir = self.out / split
        manifest = load_manifest(split_dir)
        if manifest.get("params") not in (None, params):
            log.warning("%s: settings changed since the last run; rebuilding all shards", split)
            manifest["shards"] = []
        manifest.update(split=split, planned_shards=len(groups), params=params)
        manifest["complete"] = False
        # A shard counts as done only if its file is intact and it holds exactly the planned images.
        planned = {f"{split}-{gi:05d}.tar": keys for gi, keys in enumerate(groups)}
        intact = finished_shards(split_dir, manifest)
        manifest["shards"] = [
            s
            for s in manifest["shards"]
            if s["name"] in intact and s.get("sources") == planned.get(s["name"])
        ]
        done = {s["name"] for s in manifest["shards"]}
        save_manifest(split_dir, manifest)
        for gi, keys in enumerate(groups):
            name = f"{split}-{gi:05d}.tar"
            if name in done:
                continue
            self.check_time()
            t0 = time.monotonic()
            local_path = self.tmp / name
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
        manifest["complete"] = True
        save_manifest(split_dir, manifest)

    # ------------------------------------------------------------------ phases

    def phase_dev100(self) -> None:
        if load_manifest(self.out / "dev100").get("complete"):
            log.info("dev100 already complete")
            return
        d = self.cfg.get("dev100", {})
        self.check_drive_space(float(d.get("drive_gb", 2.5)))
        pairs = self.pairs(self.cfg["source"].get("test_prefix", "test/"), "test")
        keys = sorted(pairs)
        dev = sample_keys(keys, int(d.get("n_images", 100)), self.seed)
        _write_list(self.out / "splits" / "uhdm_test.txt", keys)
        _write_list(self.out / "splits" / "uhdm_dev100.txt", dev)
        self.build_split(
            "dev100",
            _chunks(dev, int(d.get("pairs_per_shard", 25))),
            self.original_samples,
            {"source": "UHDM test", "n_test_pairs": len(keys), "seed": self.seed, "format": "original JPEGs"},
        )

    def phase_train_val(self) -> None:
        if load_manifest(self.out / "val").get("complete") and load_manifest(self.out / "train").get(
            "complete"
        ):
            log.info("train and val already complete")
            return
        t, v = self.cfg.get("train", {}), self.cfg.get("val", {})
        self.check_drive_space(float(t.get("drive_gb", 6.0)) + float(v.get("drive_gb", 0.3)))
        pairs = self.pairs(self.cfg["source"].get("train_prefix", "train/"), "train")
        train_keys, val_keys = split_train_val(pairs, int(v.get("n_images", 200)), self.seed)
        _write_list(self.out / "splits" / "uhdm_train.txt", train_keys)
        _write_list(self.out / "splits" / "uhdm_val.txt", val_keys)

        n_val = int(v.get("crops_per_image", 2))
        self.build_split(
            "val",
            _chunks(val_keys, int(v.get("pairs_per_shard", 200))),
            lambda g: self.crop_samples(g, lambda m, gt, k: val_crops(m, gt, k, self.crop, n_val, self.seed)),
            {"crop": self.crop, "crops_per_image": n_val, "jpeg_quality": self.quality, "seed": self.seed},
        )
        n_full, n_half = int(t.get("n_full", 2)), int(t.get("n_half", 1))
        self.build_split(
            "train",
            _chunks(train_keys, int(t.get("pairs_per_shard", 400))),
            lambda g: self.crop_samples(
                g, lambda m, gt, k: train_crops(m, gt, k, self.crop, n_full, n_half, self.seed)
            ),
            {
                "crop": self.crop,
                "n_full": n_full,
                "n_half": n_half,
                "jpeg_quality": self.quality,
                "seed": self.seed,
            },
        )

    # ------------------------------------------------------------------ outputs

    def sample_grid(self, n: int = 8) -> Path | None:
        """moire | gt rows from the first train shard, saved small for the results zip."""
        manifest = load_manifest(self.out / "train")
        if not manifest["shards"]:
            return None
        shard = self.out / "train" / manifest["shards"][0]["name"]
        index = index_tar(shard)
        rows = []
        for _, members in sorted(group_samples(index).items())[:n]:
            m = decode_rgb(read_member(shard, *index[members["moire"]]))
            g = decode_rgb(read_member(shard, *index[members["gt"]]))
            rows.append(np.concatenate([m, g], axis=1))
        grid = resize_max_side(np.concatenate(rows, axis=0), 1600)
        return write_image(self.ctx.results_dir / "sample_grid.jpg", grid, quality=85)

    def finish_summary(self) -> dict[str, Any]:
        splits = {}
        for split in ("train", "val", "dev100"):
            m = load_manifest(self.out / split)
            splits[split] = {
                "complete": bool(m.get("complete")),
                "shards": len(m["shards"]),
                "planned_shards": m.get("planned_shards"),
                "samples": m.get("total_samples", 0),
                "gb": round(m.get("total_bytes", 0) / GB, 2),
            }
        self.summary["splits"] = splits
        if self._source is not None:
            self.summary["source"]["downloaded_mb"] = round(self._source.bytes_read / 1e6, 1)
        self.summary["drive_free_gb"] = round(shutil.disk_usage(self.out).free / GB, 1)
        self.summary["problems"] = {k: v[:50] for k, v in self.summary["problems"].items()}
        splits_src = self.out / "splits"
        if splits_src.exists():
            files = sorted(splits_src.glob("*.txt"))
            for f in files:
                copy_atomic(f, self.ctx.results_dir / "splits" / f.name)
            self.summary["counts"] = {f.stem: len(f.read_text(encoding="utf-8").splitlines()) for f in files}
        return self.summary


@register("prepare_uhdm")
def prepare_uhdm_task(ctx: JobContext) -> TaskResult:
    """Build the UHDM train/val/dev100 shards on Drive (see module docstring)."""
    prep = Preparer(ctx)
    state, message = "done", ""
    t_start = time.monotonic()
    try:
        for phase in (prep.phase_dev100, prep.phase_train_val):
            t0 = time.monotonic()
            phase()
            prep.summary["phase_seconds"][phase.__name__] = round(time.monotonic() - t0, 1)
    except StopEarly as e:
        state, message = "partial", f"Stopped early: {e}. Finished shards are saved; the next run continues."
    except DownloadError as e:
        if not e.retry_later:
            raise
        state = "partial"
        message = f"Download stopped: {e}\nNothing is lost. Run the notebook again later to continue."
    summary = prep.finish_summary()
    if prep._source is not None:
        secs = time.monotonic() - t_start
        summary["source"]["mb_per_s"] = round(prep._source.bytes_read / 1e6 / max(secs, 1e-6), 1)
    if state == "done":
        prep.sample_grid()
    return TaskResult(state, summary, message)
