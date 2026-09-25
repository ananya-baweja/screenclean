"""Fonts for rendered pages.

The default set is the fonts that ship with matplotlib (DejaVu and STIX), so pages look
exactly the same on a laptop, in CI and on Colab. Extra system fonts can be added
explicitly, but they are never picked up silently.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from PIL import ImageFont

# (file name, family, style) of matplotlib's bundled fonts that cover all printable ASCII.
BUNDLED = (
    ("DejaVuSans.ttf", "sans", "regular"),
    ("DejaVuSans-Bold.ttf", "sans", "bold"),
    ("DejaVuSans-Oblique.ttf", "sans", "italic"),
    ("DejaVuSerif.ttf", "serif", "regular"),
    ("DejaVuSerif-Bold.ttf", "serif", "bold"),
    ("DejaVuSerif-Italic.ttf", "serif", "italic"),
    ("STIXGeneral.ttf", "serif", "regular"),
    ("STIXGeneralBol.ttf", "serif", "bold"),
    ("DejaVuSansMono.ttf", "mono", "regular"),
    ("DejaVuSansMono-Bold.ttf", "mono", "bold"),
)
FAMILIES = ("sans", "serif", "mono")


@dataclass(frozen=True)
class FontSpec:
    name: str
    path: str
    family: str
    style: str


def matplotlib_font_dir() -> Path:
    import matplotlib

    return Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"


def covers_ascii(path: str | Path) -> bool:
    """True if the font has a glyph for every printable ASCII character."""
    from matplotlib.ft2font import FT2Font

    charmap = FT2Font(str(path)).get_charmap()
    return all(c in charmap for c in range(32, 127))


@lru_cache(maxsize=4)
def available_fonts(extra: tuple[str, ...] = ()) -> tuple[FontSpec, ...]:
    """The bundled fonts, plus ``extra`` font files (family guessed from the name), all ASCII-complete."""
    fonts = []
    base = matplotlib_font_dir()
    for fname, family, style in BUNDLED:
        path = base / fname
        if path.exists():
            fonts.append(FontSpec(Path(fname).stem, str(path), family, style))
    for p in extra:
        name = Path(p).stem
        low = name.lower()
        family = "mono" if "mono" in low or "courier" in low else "serif" if "serif" in low else "sans"
        style = "bold" if "bold" in low else "italic" if ("italic" in low or "oblique" in low) else "regular"
        fonts.append(FontSpec(name, str(p), family, style))
    fonts = [f for f in fonts if covers_ascii(f.path)]
    if not fonts:
        raise RuntimeError("no usable fonts found (is matplotlib installed?)")
    return tuple(fonts)


def fonts_of(family: str, style: str | None = None, extra: tuple[str, ...] = ()) -> list[FontSpec]:
    """Fonts of one family (optionally one style), falling back to the whole family, then to sans."""
    fonts = available_fonts(extra)
    exact = [f for f in fonts if f.family == family and (style is None or f.style == style)]
    return exact or [f for f in fonts if f.family == family] or [f for f in fonts if f.family == "sans"]


@lru_cache(maxsize=512)
def load(path: str, size: int) -> ImageFont.FreeTypeFont:
    """Load a font with Pillow's basic layout engine.

    Pillow uses the "raqm" engine (kerning, complex scripts) when it's available, which it is on
    Linux but often not on Windows. Forcing the basic engine makes pages identical everywhere.
    """
    return ImageFont.truetype(path, size, layout_engine=ImageFont.Layout.BASIC)
