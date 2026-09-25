"""Image reading and writing.

Internally every image is a float32 RGB array in [0, 1] with shape (H, W, 3).
Phone photos carry an EXIF orientation tag, which is always applied on read.
"""

from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from screenclean.utils.image import bgr_to_rgb, to_float01, to_uint8

JPEG_EXTS = {".jpg", ".jpeg"}


def decode_rgb(data: bytes) -> np.ndarray:
    """Decode encoded image bytes (JPEG/PNG) to uint8 RGB, HWC. EXIF orientation is ignored.

    Used for dataset files, where input and ground truth must stay pixel-aligned exactly as stored.
    """
    flags = cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), flags)
    if bgr is None:
        raise ValueError("could not decode image bytes")
    return bgr_to_rgb(bgr)


def encode_jpeg(img: np.ndarray, quality: int = 95) -> bytes:
    """Encode an RGB image (uint8 or float in [0, 1]) as JPEG bytes with 4:4:4 chroma."""
    buf = io.BytesIO()
    Image.fromarray(to_uint8(img), mode="RGB").save(buf, format="JPEG", quality=quality, subsampling=0)
    return buf.getvalue()


def load_pil(path: str | Path) -> Image.Image:
    """Open an image with PIL, apply its EXIF orientation and convert it to RGB."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        return im.convert("RGB")


def read_image(path: str | Path) -> np.ndarray:
    """Read an image file as float32 RGB in [0, 1], shape (H, W, 3), EXIF-oriented."""
    return to_float01(np.asarray(load_pil(path)))


def write_image(path: str | Path, img: np.ndarray, quality: int = 95) -> Path:
    """Write an RGB image (float in [0, 1] or uint8) to ``path``.

    JPEGs are saved without chroma subsampling (4:4:4) so thin coloured text survives.
    Parent folders are created as needed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pil = Image.fromarray(to_uint8(img), mode="RGB")
    if path.suffix.lower() in JPEG_EXTS:
        pil.save(path, quality=quality, subsampling=0)
    else:
        pil.save(path)
    return path
