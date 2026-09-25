import zipfile

import pytest

from screenclean.data.download import DownloadError, LocalRange, ZipSource, open_zip_source


class CountingRange(LocalRange):
    """LocalRange that counts requests and bytes, and can corrupt data."""

    def __init__(self, path, corrupt=False):
        super().__init__(path)
        self.requests, self.bytes, self.corrupt = 0, 0, corrupt

    def read_range(self, start, length):
        data = super().read_range(start, length)
        self.requests += 1
        self.bytes += len(data)
        if self.corrupt and length > 1000:
            mid = len(data) // 2  # inside the member's data (the tail is over-read slack)
            data = data[:mid] + bytes([data[mid] ^ 0xFF]) + data[mid + 1 :]
        return data


def _zip(tmp_path, members, compression=zipfile.ZIP_DEFLATED):
    path = tmp_path / "a.zip"
    with zipfile.ZipFile(path, "w", compression=compression) as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return path


def test_zip_source_reads_members_like_zipfile(fake_uhdm):
    src = open_zip_source({"path": str(fake_uhdm["zip"])})
    with zipfile.ZipFile(fake_uhdm["zip"]) as zf:
        assert src.names == sorted(i.filename for i in zf.infolist() if not i.is_dir())
        for name in src.names[:5]:
            assert src.read(name) == zf.read(name)


@pytest.mark.parametrize("compression", [zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED])
def test_one_request_per_member(tmp_path, compression):
    payload = bytes(range(256)) * 200
    path = _zip(tmp_path, {"x/a.bin": payload, "x/b.bin": payload[::-1]}, compression)
    reader = CountingRange(path)
    src = ZipSource(reader)
    reader.requests = 0
    assert src.read("x/b.bin") == payload[::-1]
    assert reader.requests == 1  # header + data in a single range request
    assert src.bytes_read > 0


def test_checksum_mismatch_is_detected(tmp_path):
    path = _zip(tmp_path, {"a.bin": bytes(range(256)) * 100}, zipfile.ZIP_STORED)
    src = ZipSource(LocalRange(path))
    src.reader = CountingRange(path, corrupt=True)
    with pytest.raises(DownloadError, match="checksum") as e:
        src.read("a.bin")
    assert e.value.retry_later


def test_bad_local_header_is_permanent(tmp_path):
    path = _zip(tmp_path, {"a.bin": b"hello" * 100})
    src = ZipSource(LocalRange(path))
    src.infos["a.bin"].header_offset += 1  # point into the middle of the header
    with pytest.raises(DownloadError, match="bad local header") as e:
        src.read("a.bin")
    assert not e.value.retry_later


def test_open_zip_source_needs_url_or_path():
    with pytest.raises(ValueError):
        open_zip_source({})
