"""Image reading and writing.

Internally every image is a float32 RGB array in [0, 1] with shape (H, W, 3).
Phone photos carry an EXIF orientation tag, which is always applied on read.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from screenclean.utils.image import to_float01, to_uint8

JPEG_EXTS = {".jpg", ".jpeg"}


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
