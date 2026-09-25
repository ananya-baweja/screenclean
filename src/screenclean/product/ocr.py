"""OCR engines behind one small interface: ``engine.read(page) -> list[OcrLine]``.

Tesseract runs locally (it is called directly, see ``eval/tesseract.py``). The web app (P11)
adds an ONNX Runtime engine (RapidOCR / PaddleOCR models) with the same interface.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from screenclean.utils.image import to_float01, to_uint8


@dataclass
class OcrLine:
    text: str
    box: tuple[float, float, float, float]  # x0, y0, x1, y1 in page pixels
    conf: float  # 0..1
    block: int = 0  # paragraph number: a blank line separates paragraphs in the text output

    def as_dict(self) -> dict:
        return asdict(self)


def prepare_for_ocr(page: np.ndarray) -> np.ndarray:
    """Grey uint8 with dark text on a light background (dark-mode pages are inverted)."""
    x = to_float01(page)
    grey = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) if x.ndim == 3 else x
    if float(np.median(grey)) < 0.5:
        grey = 1.0 - grey
    return to_uint8(grey)


def lines_to_text(lines: list[OcrLine]) -> str:
    """Lines in reading order, with a blank line between paragraphs."""
    out: list[str] = []
    for i, line in enumerate(lines):
        if i and line.block != lines[i - 1].block:
            out.append("")
        out.append(line.text)
    return "\n".join(out)


class TesseractOCR:
    name = "tesseract"

    def __init__(self, lang: str = "eng", psm: int = 3):
        self.lang, self.psm = lang, psm

    def read(self, page: np.ndarray) -> list[OcrLine]:
        from screenclean.eval.tesseract import ocr_words

        words = ocr_words(prepare_for_ocr(page), psm=self.psm, lang=self.lang)
        groups: dict[tuple[int, int, int], list[dict]] = {}
        for w in words:
            groups.setdefault(w["line"], []).append(w)
        paragraphs: dict[tuple[int, int], int] = {}
        lines = []
        for key, ws in groups.items():  # dicts keep Tesseract's reading order
            boxes = np.array([w["box"] for w in ws], float)
            box = (boxes[:, 0].min(), boxes[:, 1].min(), boxes[:, 2].max(), boxes[:, 3].max())
            conf = float(np.mean([max(w["conf"], 0.0) for w in ws]) / 100.0)
            block = paragraphs.setdefault(key[:2], len(paragraphs))
            lines.append(OcrLine(" ".join(w["text"] for w in ws), tuple(float(v) for v in box), conf, block))
        return lines


ENGINES = {"tesseract": TesseractOCR}


def get_engine(engine):
    """An engine instance from a name (``"none"`` -> None) or an object with ``read``."""
    if engine is None or engine == "none":
        return None
    if isinstance(engine, str):
        try:
            return ENGINES[engine]()
        except KeyError:
            raise KeyError(
                f"unknown OCR engine {engine!r}; choose from {sorted(ENGINES)} or 'none'"
            ) from None
    return engine
