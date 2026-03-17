"""
src/pipeline/preprocess.py
---------------------------
Image preprocessing helpers to improve OCR quality on handwritten scans.

Steps (all optional, configurable):
  1. Deskew       — correct rotation using Hough transform
  2. Denoise      — bilateral filter (preserves edges)
  3. Binarize     — Sauvola adaptive thresholding
  4. Upscale      — 2× bicubic if image is small
  5. Contrast     — CLAHE on luminance channel

Call `preprocess(image_path, **options)` → returns a PIL.Image ready for OCR.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def preprocess(
    image: Image.Image | str | Path,
    *,
    deskew: bool = True,
    denoise: bool = True,
    binarize: bool = False,       # off by default — hurts VLM colour input
    upscale_min_dim: int = 1200,  # upscale if shortest dim < this
    enhance_contrast: bool = True,
) -> Image.Image:
    """
    Apply a configurable preprocessing pipeline to the input image.

    Parameters
    ----------
    image            : PIL.Image or path to image file
    deskew           : correct page rotation
    denoise          : apply bilateral noise filter
    binarize         : adaptive binarization (good for classic OCR)
    upscale_min_dim  : minimum shortest side in pixels; 0 = skip
    enhance_contrast : CLAHE contrast enhancement
    """
    if not isinstance(image, Image.Image):
        image = Image.open(image).convert("RGB")

    arr = np.array(image)

    if upscale_min_dim:
        arr = _upscale(arr, upscale_min_dim)

    if enhance_contrast:
        arr = _clahe(arr)

    if denoise:
        arr = _denoise(arr)

    if deskew:
        arr = _deskew(arr)

    if binarize:
        arr = _binarize(arr)

    return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# Individual steps
# ---------------------------------------------------------------------------

def _upscale(arr: np.ndarray, min_dim: int) -> np.ndarray:
    h, w = arr.shape[:2]
    short = min(h, w)
    if short >= min_dim:
        return arr
    scale = min_dim / short
    new_w, new_h = int(w * scale), int(h * scale)
    logger.debug("Upscaling %dx%d → %dx%d", w, h, new_w, new_h)
    return cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


def _clahe(arr: np.ndarray) -> np.ndarray:
    """CLAHE on the L channel of LAB colour space."""
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)


def _denoise(arr: np.ndarray) -> np.ndarray:
    """Bilateral filter — removes noise while keeping ink edges sharp."""
    return cv2.bilateralFilter(arr, d=9, sigmaColor=75, sigmaSpace=75)


def _deskew(arr: np.ndarray) -> np.ndarray:
    """
    Estimate and correct document skew angle using minAreaRect on
    the thresholded image edges.
    Skips correction if the detected angle is <0.5° (effectively straight).
    """
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 10:
        return arr
    angle = cv2.minAreaRect(coords)[-1]
    # minAreaRect returns angles in [-90, 0); map to [-45, 45]
    if angle < -45:
        angle = 90 + angle
    if abs(angle) < 0.5:
        return arr
    logger.debug("Deskewing by %.2f°", angle)
    h, w = arr.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(arr, M, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def _binarize(arr: np.ndarray) -> np.ndarray:
    """
    Sauvola adaptive thresholding → grayscale binary image (3-channel RGB
    so it stays compatible with the rest of the pipeline).
    """
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    # OpenCV does not have Sauvola natively; use niblack + manual threshold
    thresh = cv2.ximgproc.niBlackThreshold(
        gray, maxValue=255, type=cv2.THRESH_BINARY,
        blockSize=25, k=0.2,
        binarizationMethod=cv2.ximgproc.BINARIZATION_SAUVOLA,
    ) if _has_ximgproc() else _otsu_fallback(gray)
    binary_rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
    return binary_rgb


def _otsu_fallback(gray: np.ndarray) -> np.ndarray:
    """Simple Otsu fallback when opencv-contrib is unavailable."""
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def _has_ximgproc() -> bool:
    try:
        _ = cv2.ximgproc
        return True
    except AttributeError:
        return False
