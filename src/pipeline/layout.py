"""
src/pipeline/layout.py
-----------------------
Layout detection: returns ordered list of (x1, y1, x2, y2) bounding boxes
representing text lines / words, sorted top-to-bottom, left-to-right.

Backends
--------
- doctr   : uses python-doctr to detect text blocks
- none    : treats the whole image as one region (fallback)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class TextRegion:
    """A detected text region with its bounding box (pixel coords)."""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float = 1.0

    def crop(self, image: Image.Image) -> Image.Image:
        return image.crop((self.x1, self.y1, self.x2, self.y2))

    def to_dict(self) -> dict:
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2,
                "confidence": self.confidence}


def detect_layout(
    image_path: str | Path,
    backend: Literal["doctr", "none"] = "doctr",
    save_debug: bool = False,
) -> tuple[Image.Image, list[TextRegion]]:
    """
    Detect text regions in *image_path*.

    Returns
    -------
    (pil_image, regions)
        pil_image : original image as PIL.Image
        regions   : list of TextRegion sorted top → bottom, left → right
    """
    image_path = Path(image_path)
    pil_img = Image.open(image_path).convert("RGB")

    if backend == "none":
        w, h = pil_img.size
        regions = [TextRegion(0, 0, w, h)]
        return pil_img, regions

    if backend == "doctr":
        return _detect_doctr(pil_img, image_path, save_debug)

    raise ValueError(f"Unknown layout backend: {backend!r}")


# ---------------------------------------------------------------------------
# DocTR backend
# ---------------------------------------------------------------------------

def _detect_doctr(
    pil_img: Image.Image,
    image_path: Path,
    save_debug: bool,
) -> tuple[Image.Image, list[TextRegion]]:
    try:
        from doctr.io import DocumentFile
        from doctr.models import detection_predictor
    except ImportError as e:
        raise ImportError(
            "python-doctr is required for layout detection. "
            "Install with: pip install 'python-doctr[torch]'"
        ) from e

    logger.info("Running DocTR layout detection on %s", image_path)

    doc = DocumentFile.from_images([str(image_path)])

    # Lightweight detection-only model (fast, CPU-friendly)
    predictor = detection_predictor(arch="db_resnet50", pretrained=True)
    result = predictor(doc)

    w, h = pil_img.size
    regions: list[TextRegion] = []

    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                # DocTR returns relative coords [0,1]
                (x1r, y1r), (x2r, y2r) = line.geometry
                regions.append(TextRegion(
                    x1=int(x1r * w),
                    y1=int(y1r * h),
                    x2=int(x2r * w),
                    y2=int(y2r * h),
                    confidence=line.words[0].confidence if line.words else 1.0,
                ))

    # Sort: top → bottom (primary), left → right (secondary)
    regions.sort(key=lambda r: (r.y1, r.x1))

    logger.info("DocTR found %d regions", len(regions))

    if save_debug:
        _save_debug_image(pil_img, regions, image_path)

    return pil_img, regions


def _save_debug_image(
    img: Image.Image,
    regions: list[TextRegion],
    source_path: Path,
) -> None:
    """Draw bounding boxes and save alongside source image."""
    import cv2

    arr = np.array(img)
    arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    for i, r in enumerate(regions):
        cv2.rectangle(arr_bgr, (r.x1, r.y1), (r.x2, r.y2), (0, 200, 50), 2)
        cv2.putText(arr_bgr, str(i), (r.x1, max(r.y1 - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 50), 1)

    debug_path = source_path.parent / f"{source_path.stem}_layout_debug.jpg"
    cv2.imwrite(str(debug_path), arr_bgr)
    logger.info("Layout debug image saved to %s", debug_path)
