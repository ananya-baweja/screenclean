"""Render clean 1920x1080 "screen pages" with ground truth for every line of text.

Templates: ``slide``, ``doc``, ``code_dark``, ``code_light``, ``table``, ``chat`` and ``mixed``.
Fonts, sizes and colours are random but reproducible from the seed. Each drawn line is
recorded with its text, pixel box, font and size, so OCR on a photo of the page can be
scored without anyone typing ground truth.

Colours always keep strong contrast between text and background: the benchmark should
measure moiré, not unreadable colour choices.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from screenclean.render import corpus as corpus_mod
from screenclean.render.fonts import FontSpec, fonts_of, load

PAGE_SIZE = (1920, 1080)
TEMPLATES = ("slide", "doc", "code_dark", "code_light", "table", "chat", "mixed")
MARKER_MARGIN = 180  # content stays clear of the corner markers and their white margins (160 px)
PLAIN_MARGIN = 70

LIGHT_THEMES = [
    ((255, 255, 255), (20, 20, 20), (30, 90, 170)),
    ((248, 246, 240), (40, 40, 40), (160, 60, 30)),
    ((235, 241, 250), (15, 30, 60), (0, 110, 120)),
    ((250, 250, 250), (60, 60, 60), (120, 40, 140)),
    ((255, 252, 235), (30, 30, 30), (20, 110, 40)),
]
DARK_THEMES = [
    ((30, 30, 30), (230, 230, 230), (110, 190, 255)),
    ((40, 44, 52), (220, 223, 228), (229, 192, 123)),
    ((13, 17, 23), (201, 209, 217), (126, 231, 135)),
    ((20, 30, 60), (240, 240, 240), (255, 200, 90)),
]
CODE_COLORS_DARK = [(220, 220, 220), (152, 195, 121), (229, 192, 123), (97, 175, 239), (198, 120, 221)]
CODE_COLORS_LIGHT = [(30, 30, 30), (0, 92, 197), (163, 21, 21), (0, 128, 0), (111, 66, 193)]
TABLE_HEADERS = ["Item", "Region", "Owner", "Date", "Status", "Amount", "Qty", "Score", "Notes", "Code"]


@dataclass
class Line:
    """One drawn line of text and where it is on the page."""

    text: str
    box: tuple[int, int, int, int]  # x0, y0, x1, y1 in page pixels
    font: str
    size: int


@dataclass
class Page:
    image: np.ndarray  # uint8 RGB, H x W x 3
    template: str
    seed: int
    lines: list[Line] = field(default_factory=list)
    page_id: int | None = None

    def gt(self) -> dict[str, Any]:
        h, w = self.image.shape[:2]
        return {
            "page_id": self.page_id,
            "template": self.template,
            "seed": self.seed,
            "size": [w, h],
            "lines": [asdict(line) for line in self.lines],
        }

    def text(self) -> str:
        """Ground-truth text in reading order (top to bottom, then left to right within a row)."""
        return "\n".join(line.text for line in reading_order(self.lines))


def reading_order(lines: list[Line]) -> list[Line]:
    """Sort lines into rows by vertical overlap, then left to right."""
    rows: list[list[Line]] = []
    for line in sorted(lines, key=lambda ln: (ln.box[1] + ln.box[3]) / 2):
        cy = (line.box[1] + line.box[3]) / 2
        if rows and rows[-1][0].box[1] <= cy <= rows[-1][0].box[3]:
            rows[-1].append(line)
        else:
            rows.append([line])
    return [ln for row in rows for ln in sorted(row, key=lambda ln: ln.box[0])]


class _Canvas:
    """PIL drawing with ground-truth bookkeeping and reproducible randomness."""

    def __init__(
        self, bg: tuple[int, int, int], rng: np.random.Generator, size=PAGE_SIZE, margin=PLAIN_MARGIN
    ):
        self.img = Image.new("RGB", size, bg)
        self.draw = ImageDraw.Draw(self.img)
        self.rng = rng
        self.w, self.h = size
        self.margin = margin
        self.lines: list[Line] = []

    def text(
        self, xy: tuple[float, float], text: str, spec: FontSpec, size: int, fill
    ) -> tuple[int, int, int, int]:
        """Draw one line at ``xy`` (top-left) and record it; returns its box."""
        font = load(spec.path, size)
        x, y = int(round(xy[0])), int(round(xy[1]))
        self.draw.text((x, y), text, font=font, fill=fill, anchor="la")
        box = tuple(int(v) for v in self.draw.textbbox((x, y), text, font=font, anchor="la"))
        if text.strip():
            self.lines.append(Line(text, box, spec.name, size))
        return box

    def width(self, text: str, spec: FontSpec, size: int) -> float:
        return self.draw.textlength(text, font=load(spec.path, size))

    def wrap(self, text: str, spec: FontSpec, size: int, max_w: float) -> list[str]:
        """Greedy word wrap; a single over-long word is cut with an ellipsis-free hard break."""
        out, cur = [], ""
        for word in text.split():
            cand = f"{cur} {word}".strip()
            if self.width(cand, spec, size) <= max_w:
                cur = cand
                continue
            if cur:
                out.append(cur)
            while self.width(word, spec, size) > max_w and len(word) > 1:
                cut = len(word)
                while cut > 1 and self.width(word[:cut], spec, size) > max_w:
                    cut -= 1
                out.append(word[:cut])
                word = word[cut:]
            cur = word
        if cur:
            out.append(cur)
        return out

    def fit(self, text: str, spec: FontSpec, size: int, max_w: float) -> str:
        """Shorten ``text`` word by word (then character by character) until it fits on one line."""
        words = text.split()
        while len(words) > 1 and self.width(" ".join(words), spec, size) > max_w:
            words.pop()
        s = " ".join(words)
        while len(s) > 1 and self.width(s, spec, size) > max_w:
            s = s[:-1]
        return s.rstrip()

    def line_height(self, size: int, spacing: float = 1.35) -> int:
        return int(round(size * spacing))


class Renderer:
    """Draws pages from a list of text lines (normally one corpus split)."""

    def __init__(self, lines: list[str], size: tuple[int, int] = PAGE_SIZE):
        if not lines:
            raise ValueError("need at least one text line")
        self.lines = list(lines)
        self.code = [ln for ln in self.lines if corpus_mod.is_code(ln)] or self.lines
        self.prose = [ln for ln in self.lines if not corpus_mod.is_code(ln)] or self.lines
        self.size = size

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _pick(rng: np.random.Generator, seq):
        return seq[int(rng.integers(len(seq)))]

    def _take(self, rng: np.random.Generator, pool: list[str], n: int) -> list[str]:
        """``n`` lines from ``pool``, starting at a random point and wrapping around."""
        start = int(rng.integers(len(pool)))
        return [pool[(start + i) % len(pool)] for i in range(n)]

    def _font(self, rng, family: str, style: str | None = None) -> FontSpec:
        return self._pick(rng, fonts_of(family, style))

    # ------------------------------------------------------------------ templates

    def _slide(self, c: _Canvas, rng, base: int, theme) -> None:
        bg, fg, accent = theme
        family = self._pick(rng, ["sans", "sans", "serif"])
        title_size = min(int(base * rng.uniform(1.4, 1.9)), 72)
        tfont = self._font(rng, family, "bold")
        x0, y = c.margin, c.margin
        title = c.fit(self._take(rng, self.prose, 1)[0].rstrip("."), tfont, title_size, c.w - 2 * c.margin)
        box = c.text((x0, y), title, tfont, title_size, accent)
        y = box[3] + int(title_size * 0.5)
        c.draw.rectangle([x0, y, x0 + int(c.w * 0.25), y + max(3, title_size // 12)], fill=accent)
        y += int(base * 1.6)
        bfont = self._font(rng, family, "regular")
        # Bullets until the slide is full (at most 12), so small fonts still give a full page of text.
        for text in self._take(rng, self.prose, 12):
            r = max(3, base // 5)
            for i, part in enumerate(c.wrap(text, bfont, base, c.w - 2 * c.margin - base * 2)):
                if y + c.line_height(base) > c.h - c.margin:
                    return
                if i == 0:
                    cy = y + base * 0.55
                    c.draw.ellipse([x0, cy - r, x0 + 2 * r, cy + r], fill=fg)
                c.text((x0 + base * 1.5, y), part, bfont, base, fg)
                y += c.line_height(base)
            y += int(base * 0.5)

    def _doc(self, c: _Canvas, rng, base: int, theme) -> None:
        bg, fg, accent = theme
        family = self._pick(rng, ["serif", "sans"])
        body = self._font(rng, family, "regular")
        head = self._font(rng, family, "bold")
        cols = 2 if base <= 18 and rng.random() < 0.5 else 1
        gap = 60
        col_w = (c.w - 2 * c.margin - gap * (cols - 1)) / cols
        y0 = c.margin
        hs = min(int(base * 1.6), 60)
        box = c.text(
            (c.margin, y0),
            c.fit(self._take(rng, self.prose, 1)[0], head, hs, c.w - 2 * c.margin),
            head,
            hs,
            fg,
        )
        y0 = box[3] + int(base * 1.2)
        for col in range(cols):
            x, y = c.margin + col * (col_w + gap), y0
            while True:
                para = " ".join(self._take(rng, self.prose, int(rng.integers(2, 5))))
                parts = c.wrap(para, body, base, col_w)
                if y + c.line_height(base) > c.h - c.margin:
                    break
                for part in parts:
                    if y + c.line_height(base) > c.h - c.margin:
                        break
                    c.text((x, y), part, body, base, fg)
                    y += c.line_height(base, 1.45)
                y += int(base * 0.9)

    def _code(self, c: _Canvas, rng, base: int, theme, dark: bool) -> None:
        bg, fg, accent = theme
        mono = self._font(rng, "mono")
        colors = CODE_COLORS_DARK if dark else CODE_COLORS_LIGHT
        numbers = rng.random() < 0.6
        gutter = c.width("000 ", mono, base) if numbers else 0
        x, y = c.margin, c.margin
        start = int(rng.integers(1, 200))
        dim = tuple(int(0.55 * v + 0.45 * b) for v, b in zip(fg, bg, strict=True))
        i = 0
        while y + c.line_height(base, 1.4) <= c.h - c.margin:
            text = self._take(rng, self.code, 1)[0]
            text = c.fit(text, mono, base, c.w - 2 * c.margin - gutter)
            if numbers:
                c.text((x, y), f"{start + i:>3}", mono, base, dim)
            c.text((x + gutter, y), text, mono, base, self._pick(rng, colors))
            y += c.line_height(base, 1.4)
            i += 1

    def _table(self, c: _Canvas, rng, base: int, theme) -> None:
        bg, fg, accent = theme
        family = self._pick(rng, ["sans", "serif", "mono"])
        font = self._font(rng, family, "regular")
        bold = self._font(rng, family, "bold")
        n_cols = int(rng.integers(3, 6))
        headers = list(rng.choice(TABLE_HEADERS, size=n_cols, replace=False))
        col_w = (c.w - 2 * c.margin) / n_cols
        row_h = c.line_height(base, 2.0)
        x0, y = c.margin, c.margin
        title_size = min(int(base * 1.5), 60)
        tfont = self._font(rng, family, "bold")
        box = c.text(
            (x0, y),
            c.fit(self._take(rng, self.prose, 1)[0], tfont, title_size, c.w - 2 * c.margin),
            tfont,
            title_size,
            fg,
        )
        y = box[3] + int(base * 1.2)
        light_band = tuple(int(0.85 * b + 0.15 * a) for b, a in zip(bg, accent, strict=True))
        c.draw.rectangle([x0, y, c.w - c.margin, y + row_h], fill=light_band)
        for j, h in enumerate(headers):
            c.text((x0 + j * col_w + base * 0.5, y + (row_h - base) / 2), h, bold, base, fg)
        y += row_h
        grid = tuple(int(0.7 * b + 0.3 * f) for b, f in zip(bg, fg, strict=True))
        while y + row_h <= c.h - c.margin:
            c.draw.line([x0, y, c.w - c.margin, y], fill=grid, width=1)
            for j in range(n_cols):
                words = self._take(rng, self.lines, 1)[0].split()
                k = int(rng.integers(1, 4))
                start = int(rng.integers(max(1, len(words) - k + 1)))
                cell = c.fit(" ".join(words[start : start + k]), font, base, col_w - base)
                c.text((x0 + j * col_w + base * 0.5, y + (row_h - base) / 2), cell, font, base, fg)
            y += row_h

    def _chat(self, c: _Canvas, rng, base: int, theme) -> None:
        bg, fg, accent = theme
        font = self._font(rng, "sans")
        max_w = c.w * 0.55
        y = c.margin
        pad = int(base * 0.6)
        mine = tuple(int(0.8 * b + 0.2 * a) for b, a in zip(bg, accent, strict=True))
        theirs = tuple(int(0.9 * b + 0.1 * f) for b, f in zip(bg, fg, strict=True))
        while True:
            parts = c.wrap(self._take(rng, self.prose, 1)[0], font, base, max_w - 2 * pad)
            bubble_h = len(parts) * c.line_height(base) + 2 * pad
            if y + bubble_h > c.h - c.margin:
                break
            right = rng.random() < 0.5
            bw = max(c.width(p, font, base) for p in parts) + 2 * pad
            bx = c.w - c.margin - bw if right else c.margin
            c.draw.rounded_rectangle(
                [bx, y, bx + bw, y + bubble_h], radius=pad, fill=mine if right else theirs
            )
            ty = y + pad
            for p in parts:
                c.text((bx + pad, ty), p, font, base, fg)
                ty += c.line_height(base)
            y += bubble_h + int(base * 0.8)

    def _mixed(self, c: _Canvas, rng, base: int, theme) -> None:
        bg, fg, accent = theme
        family = self._pick(rng, ["sans", "serif"])
        body, head = self._font(rng, family, "regular"), self._font(rng, family, "bold")
        hs = min(int(base * 1.7), 64)
        box = c.text(
            (c.margin, c.margin),
            c.fit(self._take(rng, self.prose, 1)[0], head, hs, c.w - 2 * c.margin),
            head,
            hs,
            accent,
        )
        y = box[3] + base
        left_w = (c.w - 2 * c.margin) * 0.48
        while y + c.line_height(base) <= c.h - c.margin:  # left half: paragraphs down to the bottom
            for part in c.wrap(" ".join(self._take(rng, self.prose, 3)), body, base, left_w):
                if y + c.line_height(base) > c.h - c.margin:
                    break
                c.text((c.margin, y), part, body, base, fg)
                y += c.line_height(base, 1.45)
            y += int(base * 0.9)
        # right half: a code box
        mono = self._font(rng, "mono")
        bx0 = c.margin + (c.w - 2 * c.margin) * 0.52
        by = box[3] + base
        box_bg = tuple(int(0.9 * b + 0.1 * f) for b, f in zip(bg, fg, strict=True))
        cs = max(12, int(base * 0.9))
        n = max(1, min(12, int((c.h - c.margin - by - cs) / c.line_height(cs, 1.4))))
        c.draw.rectangle(
            [bx0 - cs * 0.6, by - cs * 0.6, c.w - c.margin, by + n * c.line_height(cs, 1.4)], fill=box_bg
        )
        for text in self._take(rng, self.code, n):
            c.text((bx0, by), c.fit(text, mono, cs, c.w - c.margin - bx0 - cs), mono, cs, fg)
            by += c.line_height(cs, 1.4)

    # ------------------------------------------------------------------ public

    def render(
        self,
        template: str,
        seed: int,
        base_size: int | None = None,
        size_range: tuple[int, int] = (12, 48),
        page_id: int | None = None,
    ) -> Page:
        """Render one page. With ``page_id``, corner markers are added and content keeps clear of them."""
        if template not in TEMPLATES:
            raise ValueError(f"unknown template {template!r}; choose from {TEMPLATES}")
        rng = np.random.default_rng(seed)
        base = int(base_size or rng.integers(size_range[0], size_range[1] + 1))
        dark = template == "code_dark" or (template not in ("code_light",) and rng.random() < 0.3)
        theme = self._pick(rng, DARK_THEMES if dark else LIGHT_THEMES)
        c = _Canvas(theme[0], rng, self.size, MARKER_MARGIN if page_id is not None else PLAIN_MARGIN)
        if template == "slide":
            self._slide(c, rng, base, theme)
        elif template == "doc":
            self._doc(c, rng, base, theme)
        elif template in ("code_dark", "code_light"):
            self._code(c, rng, base, theme, dark=template == "code_dark")
        elif template == "table":
            self._table(c, rng, base, theme)
        elif template == "chat":
            self._chat(c, rng, base, theme)
        else:
            self._mixed(c, rng, base, theme)
        image = np.asarray(c.img, dtype=np.uint8).copy()
        if page_id is not None:
            from screenclean.render.aruco import draw_markers

            image = draw_markers(image, page_id)
        return Page(image=image, template=template, seed=seed, lines=c.lines, page_id=page_id)
