"""OCR text metrics: character / word error rate and word-level F1.

Both texts are normalised first (Unicode NFKC, curly quotes and dashes to ASCII, stand-alone
bullet symbols removed, whitespace collapsed, lower-case by default), so the scores measure
recognition rather than typography.

CER and WER also count reading order, so both sides are put in the same order first
(:func:`ordered_text`: rows by vertical overlap, then left to right). Word F1 ignores order.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence

import jiwer

_TRANSLATE = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'",
    "“": '"', "”": '"', "„": '"', "″": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "−": "-",
    " ": " ", "…": "...",
})  # fmt: skip
_WS = re.compile(r"\s+")
# A bullet drawn as a shape has no ground-truth text, but OCR reads it as "*", "•", "¢", "°" or U+FFFD.
_BULLET = re.compile(r"(?<!\S)[•·◦▪▫■□●○‣⁃∙*¢°�](?!\S)")
Box = tuple[float, float, float, float]  # x0, y0, x1, y1


def normalize(text: str, lowercase: bool = True) -> str:
    text = unicodedata.normalize("NFKC", text).translate(_TRANSLATE)
    text = _WS.sub(" ", _BULLET.sub(" ", text)).strip()
    return text.lower() if lowercase else text


def cer(reference: str, hypothesis: str, lowercase: bool = True) -> float:
    """Character error rate: edits needed / characters in the reference (can exceed 1)."""
    ref, hyp = normalize(reference, lowercase), normalize(hypothesis, lowercase)
    if not ref:
        return 0.0 if not hyp else 1.0
    return float(jiwer.cer(ref, hyp))


def wer(reference: str, hypothesis: str, lowercase: bool = True) -> float:
    ref, hyp = normalize(reference, lowercase), normalize(hypothesis, lowercase)
    if not ref:
        return 0.0 if not hyp else 1.0
    return float(jiwer.wer(ref, hyp))


def word_f1(reference: str, hypothesis: str, lowercase: bool = True) -> float:
    """F1 over word multisets: order-free, so reading-order mistakes don't count."""
    ref = Counter(normalize(reference, lowercase).split())
    hyp = Counter(normalize(hypothesis, lowercase).split())
    if not ref and not hyp:
        return 1.0
    overlap = sum((ref & hyp).values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / sum(hyp.values()), overlap / sum(ref.values())
    return 2 * precision * recall / (precision + recall)


def score(reference: str, hypothesis: str, lowercase: bool = True) -> dict[str, float]:
    """CER, WER and word F1 of one page, plus the reference length in characters."""
    return {
        "cer": cer(reference, hypothesis, lowercase),
        "wer": wer(reference, hypothesis, lowercase),
        "word_f1": word_f1(reference, hypothesis, lowercase),
        "ref_chars": len(normalize(reference, lowercase)),
    }


def quad_box(quad: Sequence[Sequence[float]]) -> Box:
    """Axis-aligned box around a polygon's corners."""
    xs, ys = [float(p[0]) for p in quad], [float(p[1]) for p in quad]
    return min(xs), min(ys), max(xs), max(ys)


def order_lines(items: Iterable[tuple[str, Box]]) -> list[str]:
    """Texts of ``(text, box)`` lines in reading order.

    Lines are taken top to bottom by their centre; a line joins the current row when its
    centre lies inside the row's first line vertically. Each row is read left to right.
    This is the rule that orders the pages' ground truth (``render.pages.reading_order``),
    applied to OCR output too, so neither side gains from a different convention.
    """
    rows: list[list[tuple[str, Box]]] = []
    for text, box in sorted(items, key=lambda it: (it[1][1] + it[1][3]) / 2):
        cy = (box[1] + box[3]) / 2
        if rows and rows[-1][0][1][1] <= cy <= rows[-1][0][1][3]:
            rows[-1].append((text, box))
        else:
            rows.append([(text, box)])
    return [text for row in rows for text, _ in sorted(row, key=lambda it: it[1][0])]


def ordered_text(items: Iterable[tuple[str, Box]]) -> str:
    return "\n".join(order_lines(items))
