"""
src/pipeline/ocr.py
--------------------
OCR step: given a PIL image (or a crop thereof) → raw text string.

Backends
--------
- easyocr   : EasyOCR (GPU-friendly, good for handwriting)
- tesseract : pytesseract wrapper (needs system tesseract)
"""
from __future__ import annotations

import logging
from typing import Literal

from PIL import Image

logger = logging.getLogger(__name__)

# Module-level caches so we don't reload models on every call
_easyocr_reader: "easyocr.Reader | None" = None  # type: ignore[name-defined]


def ocr_image(
    pil_img: Image.Image,
    backend: Literal["easyocr", "tesseract"] = "easyocr",
    language: str = "it",
) -> str:
    """
    Run OCR on *pil_img* and return the recognised text.

    Parameters
    ----------
    pil_img  : PIL.Image (RGB)
    backend  : which OCR engine to use
    language : ISO-639-1 language code (e.g. 'it', 'en')
    """
    if backend == "easyocr":
        return _ocr_easyocr(pil_img, language)
    if backend == "tesseract":
        return _ocr_tesseract(pil_img, language)
    raise ValueError(f"Unknown OCR backend: {backend!r}")


def ocr_regions(
    pil_img: Image.Image,
    regions: list,  # list[TextRegion] from layout.py
    backend: Literal["easyocr", "tesseract"] = "easyocr",
    language: str = "it",
) -> str:
    """
    OCR each region crop separately and join with newlines.
    Falls back to full-image OCR if regions is empty.
    """
    if not regions:
        logger.warning("No regions provided; running OCR on full image")
        return ocr_image(pil_img, backend, language)

    lines: list[str] = []
    for region in regions:
        crop = region.crop(pil_img)
        text = ocr_image(crop, backend, language).strip()
        if text:
            lines.append(text)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# EasyOCR backend
# ---------------------------------------------------------------------------

def _ocr_easyocr(pil_img: Image.Image, language: str) -> str:
    global _easyocr_reader

    try:
        import easyocr
        import numpy as np
    except ImportError as e:
        raise ImportError("Run: pip install easyocr") from e

    # Map common ISO codes to EasyOCR language codes
    _lang_map = {"it": "it", "en": "en", "fr": "fr", "de": "de", "es": "es", "la": "la"}
    lang_code = _lang_map.get(language, "en")

    if _easyocr_reader is None or _easyocr_reader.lang_list != [lang_code]:
        logger.info("Loading EasyOCR reader for language '%s'", lang_code)
        _easyocr_reader = easyocr.Reader([lang_code], gpu=_has_gpu())

    arr = np.array(pil_img)
    results = _easyocr_reader.readtext(arr, detail=0, paragraph=False)
    return " ".join(results)


# ---------------------------------------------------------------------------
# Tesseract backend
# ---------------------------------------------------------------------------

def _ocr_tesseract(pil_img: Image.Image, language: str) -> str:
    try:
        import pytesseract
    except ImportError as e:
        raise ImportError("Run: pip install pytesseract  (and install tesseract-ocr system package)") from e

    # Tesseract uses 3-letter codes; map common ones
    _lang_map = {"it": "ita", "en": "eng", "fr": "fra", "de": "deu", "es": "spa", "la": "lat"}
    tess_lang = _lang_map.get(language, "eng")

    config = "--psm 6"  # Assume a single uniform block of text
    return pytesseract.image_to_string(pil_img, lang=tess_lang, config=config).strip()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_gpu() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
