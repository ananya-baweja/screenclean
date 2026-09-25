"""Shared fixtures: tiny fake UHDM-style datasets built on the fly."""

import io
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def make_image(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    """Smooth random colours plus stripes, so JPEG crops aren't trivial."""
    base = rng.integers(0, 256, size=(h // 8 + 1, w // 8 + 1, 3), dtype=np.uint8)
    img = np.kron(base, np.ones((8, 8, 1), dtype=np.uint8))[:h, :w]
    img[:, ::7] = 255 - img[:, ::7]
    return img


def jpeg_bytes(img: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def make_uhdm_tree(root: Path, folders: dict[str, int], size=(160, 240), seed=0) -> list[str]:
    """Create ``<root>/<folder>/NNNN_{gt,moire}.jpg`` pairs. Prefixes repeat across folders on purpose."""
    rng = np.random.default_rng(seed)
    keys = []
    for folder, n in folders.items():
        d = root / folder
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            gt = make_image(rng, *size)
            moire = np.clip(gt.astype(int) + rng.integers(-20, 20, size=gt.shape), 0, 255).astype(np.uint8)
            (d / f"{i:04d}_gt.jpg").write_bytes(jpeg_bytes(gt))
            (d / f"{i:04d}_moire.jpg").write_bytes(jpeg_bytes(moire))
            keys.append(f"{folder}/{i:04d}" if folder else f"{i:04d}")
    return sorted(keys)


def make_zip(src_root: Path, out: Path, arc_prefix: dict[str, str]) -> Path:
    """Zip folders of ``src_root`` like the mirror.

    ``{"train": "train/train"}`` stores ``src/train/...`` as ``train/train/...``.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for folder, prefix in arc_prefix.items():
            for f in sorted((src_root / folder).rglob("*")):
                if f.is_file():
                    zf.write(f, f"{prefix}/{f.relative_to(src_root / folder).as_posix()}")
    return out


@pytest.fixture
def fake_uhdm(tmp_path):
    """A fake UHDM zip laid out like the mirror: 12 train pairs (3 folders), 6 test, 2 test_origin."""
    src = tmp_path / "src"
    train_keys = make_uhdm_tree(src / "train", {"pair_00": 4, "pair_01": 4, "pair_02": 4}, seed=1)
    test_keys = make_uhdm_tree(src / "test", {"": 6}, seed=2)
    make_uhdm_tree(src / "test_origin", {"": 2}, seed=3)
    zip_path = make_zip(
        src,
        tmp_path / "archives" / "uhdm.zip",
        {"train": "train/train", "test": "test/test", "test_origin": "test_origin/test_origin"},
    )
    return {"src": src, "zip": zip_path, "train_keys": train_keys, "test_keys": test_keys}


def synthetic_photo(
    page: np.ndarray,
    rng: np.random.Generator,
    photo_size=(2400, 1600),
    blur_max=1.5,
    noise_max=4.0,
    jpeg_quality=60,
):
    """A fake phone photo of ``page``: random perspective, background, blur, noise and JPEG.

    Returns ``(photo uint8 RGB, H_true photo->page)``.
    """
    import cv2

    ph, pw = page.shape[:2]
    W, H = photo_size
    scale = rng.uniform(0.6, 0.85) * W / pw
    cx, cy = W / 2 + rng.uniform(-0.08, 0.08) * W, H / 2 + rng.uniform(-0.08, 0.08) * H
    ang = np.deg2rad(rng.uniform(-8, 8))
    rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    src = np.array([[0, 0], [pw, 0], [pw, ph], [0, ph]], np.float32)
    quad = (src - [pw / 2, ph / 2]) * scale @ rot.T + [cx, cy]
    quad += rng.uniform(-0.05, 0.05, size=(4, 2)) * [pw * scale, ph * scale]  # perspective (tilt)
    Hp = cv2.getPerspectiveTransform(src, quad.astype(np.float32))
    bg = np.full((H, W, 3), rng.integers(20, 200, size=3), np.uint8)
    photo = cv2.warpPerspective(page, Hp, (W, H), dst=bg, borderMode=cv2.BORDER_TRANSPARENT)
    sigma = rng.uniform(0, blur_max)
    if sigma > 0.1:
        photo = cv2.GaussianBlur(photo, (0, 0), sigma)
    photo = np.clip(photo + rng.normal(0, rng.uniform(0, noise_max), photo.shape), 0, 255).astype(np.uint8)
    ok, enc = cv2.imencode(".jpg", photo, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    photo = cv2.imdecode(enc, cv2.IMREAD_UNCHANGED)
    return photo, np.linalg.inv(Hp)
