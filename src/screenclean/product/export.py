"""Outputs of a scan: Markdown / plain text, and a searchable PDF.

**How a searchable PDF works.** Each PDF page shows the cleaned page as an image. On top of it,
every OCR line is written as *invisible* text (PDF text render mode 3: neither filled nor
stroked), placed over the line's box and stretched to its width. You see the picture; search,
select and copy-paste use the hidden text. The PDF is made with ``reportlab`` (BSD licence).
The text layer uses DejaVu Sans (bundled with matplotlib), so non-ASCII characters survive.
"""

from __future__ import annotations

import io
from collections.abc import Sequence
from functools import lru_cache

from screenclean.utils.io import encode_jpeg

FONT_NAME = "DejaVuSans"


@lru_cache(maxsize=1)
def _font() -> str:
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    from screenclean.render.fonts import matplotlib_font_dir

    try:
        pdfmetrics.registerFont(TTFont(FONT_NAME, str(matplotlib_font_dir() / "DejaVuSans.ttf")))
        return FONT_NAME
    except Exception:  # font file missing: fall back to a standard PDF font (Latin-1 only)
        return "Helvetica"


def to_markdown(results: Sequence, title: str = "Scanned screens") -> str:
    """One section per photo, with the recognised text."""
    out = [f"# {title}", ""]
    for i, r in enumerate(results, 1):
        out += [f"## {i}. {r.name or f'Photo {i}'}", ""]
        if not r.detected:
            out += ["_Screen not found: the whole photo was used._", ""]
        out += [r.text.strip() or "_No text found._", ""]
    return "\n".join(out)


def to_text(results: Sequence) -> str:
    """Plain text, photos separated by a header line."""
    parts = []
    for i, r in enumerate(results, 1):
        parts.append(f"===== {i}. {r.name or f'Photo {i}'} =====\n{r.text.strip()}\n")
    return "\n".join(parts)


def searchable_pdf(results: Sequence, dpi: float = 150.0, jpeg_quality: int = 85, title: str = "") -> bytes:
    """A PDF with one page per result: the page image plus an invisible, searchable text layer."""
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfgen import canvas

    font = _font()
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pageCompression=1)
    c.setCreator("ScreenClean")
    if title:
        c.setTitle(title)
    k = 72.0 / dpi  # points per page pixel
    for r in results:
        h, w = r.page.shape[:2]
        pw, ph = w * k, h * k
        c.setPageSize((pw, ph))
        c.drawImage(ImageReader(io.BytesIO(encode_jpeg(r.page, jpeg_quality))), 0, 0, width=pw, height=ph)
        for line in r.lines:
            if not line.text.strip():
                continue
            x0, y0, x1, y1 = line.box
            size = max(0.85 * (y1 - y0) * k, 1.0)
            natural = pdfmetrics.stringWidth(line.text, font, size)
            t = c.beginText()
            t.setTextRenderMode(3)  # invisible
            t.setFont(font, size)
            if natural > 0:
                t.setHorizScale(100.0 * (x1 - x0) * k / natural)
            # PDF y grows upwards; put the baseline just above the bottom of the box
            t.setTextOrigin(x0 * k, ph - y1 * k + 0.2 * size)
            t.textLine(line.text)
            c.drawText(t)
        c.showPage()
    c.save()
    return buf.getvalue()
