"""OCR text metrics: character / word error rate and word-level F1.

Both texts are normalised first (Unicode NFKC, curly quotes and dashes to ASCII, whitespace
collapsed, lower-case by default), so the scores measure recognition rather than typography.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

import jiwer

_TRANSLATE = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'",
    "“": '"', "”": '"', "„": '"', "″": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "−": "-",
    " ": " ", "…": "...",
})  # fmt: skip
_WS = re.compile(r"\s+")


def normalize(text: str, lowercase: bool = True) -> str:
    text = unicodedata.normalize("NFKC", text).translate(_TRANSLATE)
    text = _WS.sub(" ", text).strip()
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
