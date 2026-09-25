"""Random access to single files inside a large remote (or local) zip archive.

The zip's central directory (its table of contents) sits at the end of the file, so
listing a 48 GB archive costs about 1 MB of HTTP range requests. After that, each
member is fetched with one bounded range request, decompressed and checked against
its CRC-32. Nothing is ever extracted to disk, and resuming a job only fetches the
files it still needs.
"""

from __future__ import annotations

import io
import logging
import struct
import threading
import time
import zipfile
import zlib
from pathlib import Path
from typing import Any, Protocol

log = logging.getLogger(__name__)

LOCAL_HEADER = struct.Struct("<4sHHHHHIIIHH")  # 30 bytes
LOCAL_SIG = b"PK\x03\x04"


class DownloadError(RuntimeError):
    """A download failed for good in this run.

    ``retry_later`` is True when running the job again later should work (server busy,
    repeated network drops), and False for permanent problems (file missing, bad data).
    """

    def __init__(self, message: str, retry_later: bool = False):
        super().__init__(message)
        self.retry_later = retry_later


class RangeReader(Protocol):
    size: int

    def read_range(self, start: int, length: int) -> bytes: ...


class LocalRange:
    """Byte ranges from a local file (tests, or an archive downloaded by hand)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.size = self.path.stat().st_size

    def read_range(self, start: int, length: int) -> bytes:
        with open(self.path, "rb") as f:
            f.seek(start)
            return f.read(max(0, min(length, self.size - start)))


class HTTPRange:
    """Byte ranges over HTTP(S) with retries. No login: for public files only."""

    def __init__(self, url: str, retries: int = 6, backoff_s: float = 5.0, timeout_s: float = 120.0):
        import requests

        self.url, self.retries, self.backoff_s, self.timeout_s = url, retries, backoff_s, timeout_s
        self._local = threading.local()
        self._requests = requests
        self._resolved: str | None = None
        self._lock = threading.Lock()
        self.size = self._resolve()

    def _session(self):
        if not hasattr(self._local, "session"):
            self._local.session = self._requests.Session()
        return self._local.session

    def _resolve(self) -> int:
        """Follow redirects once (e.g. to a CDN) and read the file size."""
        r = self._session().head(self.url, allow_redirects=True, timeout=self.timeout_s)
        if r.status_code == 404:
            raise DownloadError(f"{self.url}: not found (404)")
        r.raise_for_status()
        with self._lock:
            self._resolved = r.url
        return int(r.headers["Content-Length"])

    def read_range(self, start: int, length: int) -> bytes:
        end = min(start + length, self.size) - 1
        if end < start:
            return b""
        want = end - start + 1
        for attempt in range(1, self.retries + 2):
            try:
                r = self._session().get(
                    self._resolved, headers={"Range": f"bytes={start}-{end}"}, timeout=self.timeout_s
                )
                if r.status_code in (401, 403, 410):  # signed CDN link expired: resolve again
                    self._resolve()
                    raise OSError(f"HTTP {r.status_code} (link refreshed)")
                if r.status_code == 404:
                    raise DownloadError(f"{self.url}: not found (404)")
                if r.status_code != 206:
                    raise OSError(f"HTTP {r.status_code} for a range request")
                if "text/html" in r.headers.get("Content-Type", ""):
                    raise OSError("got an HTML page instead of file data")
                if len(r.content) != want:
                    raise OSError(f"short read: {len(r.content)} of {want} bytes")
                return r.content
            except DownloadError:
                raise
            except Exception as e:  # noqa: BLE001 - network hiccups: back off and retry
                if attempt > self.retries:
                    raise DownloadError(
                        f"bytes {start}-{end}: giving up after {self.retries} retries ({e})", retry_later=True
                    ) from e
                wait = self.backoff_s * attempt
                log.warning("bytes %d-%d: %s; retry %d in %.0f s", start, end, e, attempt, wait)
                time.sleep(wait)
        raise AssertionError("unreachable")


class _SeekableRange(io.RawIOBase):
    """File-like view of a RangeReader, so :mod:`zipfile` can read the central directory."""

    def __init__(self, reader: RangeReader):
        self.reader, self.pos = reader, 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = {0: offset, 1: self.pos + offset, 2: self.reader.size + offset}[whence]
        return self.pos

    def readinto(self, b) -> int:  # type: ignore[override]
        data = self.reader.read_range(self.pos, len(b))
        b[: len(data)] = data
        self.pos += len(data)
        return len(data)


class ZipSource:
    """Read individual members of a zip archive through a RangeReader (thread-safe)."""

    def __init__(self, reader: RangeReader):
        self.reader = reader
        with zipfile.ZipFile(io.BufferedReader(_SeekableRange(reader), 1 << 20)) as zf:
            self.infos = {i.filename: i for i in zf.infolist() if not i.is_dir()}
        self.bytes_read = 0
        self._lock = threading.Lock()

    @property
    def names(self) -> list[str]:
        return sorted(self.infos)

    def read(self, name: str) -> bytes:
        """Fetch, decompress and CRC-check one member (one range request in most cases)."""
        info = self.infos[name]
        name_len = len(info.filename.encode("utf-8"))
        guess = LOCAL_HEADER.size + name_len + 512 + info.compress_size
        buf = self.reader.read_range(info.header_offset, guess)
        fields = LOCAL_HEADER.unpack_from(buf)
        if fields[0] != LOCAL_SIG:
            raise DownloadError(f"{name}: bad local header in the archive")
        start = LOCAL_HEADER.size + fields[9] + fields[10]
        need = start + info.compress_size
        if len(buf) < need:
            buf += self.reader.read_range(info.header_offset + len(buf), need - len(buf))
        data = buf[start:need]
        if info.compress_type == zipfile.ZIP_STORED:
            out = data
        elif info.compress_type == zipfile.ZIP_DEFLATED:
            out = zlib.decompress(data, -zlib.MAX_WBITS)
        else:
            raise DownloadError(f"{name}: unsupported compression {info.compress_type}")
        if len(out) != info.file_size or zlib.crc32(out) != info.CRC:
            raise DownloadError(f"{name}: checksum mismatch", retry_later=True)
        with self._lock:
            self.bytes_read += len(buf)
        return out


def open_zip_source(spec: dict[str, Any], base_dir: Path | None = None, **http_kwargs) -> ZipSource:
    """Open a zip from ``{"url": ...}`` (public HTTP) or ``{"path": ...}`` (local file)."""
    if "url" in spec:
        return ZipSource(HTTPRange(spec["url"], **http_kwargs))
    if "path" in spec:
        path = Path(spec["path"])
        return ZipSource(LocalRange(path if path.is_absolute() or base_dir is None else base_dir / path))
    raise ValueError("source needs a 'url' or a 'path'")
