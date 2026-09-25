import json
import random

import numpy as np
import pytest
from PIL import Image

from screenclean.utils import drive, env, image, io, seed, timer


def test_float_uint8_roundtrip():
    rng = np.random.default_rng(0)
    u8 = rng.integers(0, 256, size=(8, 9, 3), dtype=np.uint8)
    f = image.to_float01(u8)
    assert f.dtype == np.float32 and f.min() >= 0 and f.max() <= 1
    np.testing.assert_array_equal(image.to_uint8(f), u8)


def test_to_float01_clips_and_handles_uint16():
    assert image.to_float01(np.array([[-0.5, 1.5]])).tolist() == [[0.0, 1.0]]
    assert image.to_float01(np.array([[65535]], dtype=np.uint16))[0, 0] == pytest.approx(1.0)


def test_rgb_bgr_swap():
    img = np.zeros((2, 2, 3), np.float32)
    img[..., 0] = 1.0
    bgr = image.rgb_to_bgr(img)
    assert bgr[0, 0].tolist() == [0.0, 0.0, 1.0]
    np.testing.assert_array_equal(image.bgr_to_rgb(bgr), img)


def test_resize_max_side_never_upscales():
    img = np.zeros((100, 200, 3), np.float32)
    assert image.resize_max_side(img, 50).shape == (25, 50, 3)
    assert image.resize_max_side(img, 400).shape == img.shape


def test_torch_layout_roundtrip():
    pytest.importorskip("torch")
    img = np.random.default_rng(1).random((5, 7, 3), dtype=np.float32)
    t = image.hwc_to_bchw(img)
    assert tuple(t.shape) == (1, 3, 5, 7)
    np.testing.assert_array_equal(image.bchw_to_hwc(t), img)


def test_png_roundtrip(tmp_path):
    img = np.random.default_rng(2).random((16, 24, 3), dtype=np.float32)
    path = io.write_image(tmp_path / "sub" / "a.png", img)
    back = io.read_image(path)
    assert back.shape == img.shape and back.dtype == np.float32
    np.testing.assert_array_equal(image.to_uint8(back), image.to_uint8(img))


def test_read_applies_exif_orientation(tmp_path):
    # Stored 40 wide x 20 tall, tagged "rotate 90 CW" (orientation 6): should read as 20 x 40.
    exif = Image.Exif()
    exif[0x0112] = 6
    Image.new("RGB", (40, 20), (200, 10, 10)).save(tmp_path / "rot.jpg", exif=exif.tobytes())
    assert io.read_image(tmp_path / "rot.jpg").shape == (40, 20, 3)


def test_seed_everything_is_deterministic():
    def draw():
        seed.seed_everything(123)
        return random.random(), float(np.random.rand())

    assert draw() == draw()
    assert seed.make_rng(5).random() == seed.make_rng(5).random()


def test_deadline_and_timer():
    d = timer.Deadline(budget_s=1000)
    assert not d.expired() and d.expired(margin_s=2000)
    assert timer.Deadline(budget_s=0).expired()
    with timer.Timer() as t:
        pass
    assert t.seconds >= 0


def test_atomic_json_and_layout(tmp_path):
    p = drive.atomic_write_json(tmp_path / "a" / "b.json", {"x": 1})
    assert drive.read_json(p) == {"x": 1}
    assert not list(tmp_path.rglob("*.tmp"))
    assert drive.read_json(tmp_path / "missing.json", default={}) == {}
    layout = drive.DriveLayout(tmp_path / "drive").ensure()
    assert layout.job_status.is_dir() and layout.results_zips.is_dir()


def test_collect_env_has_key_fields(tmp_path):
    report = env.collect_env(disk_path=tmp_path)
    for key in ("python", "platform", "cpu_count", "disk_free_gb", "gpu", "packages", "git_sha"):
        assert key in report
    assert "numpy" in report["packages"]
    json.dumps(report)  # must be serialisable
