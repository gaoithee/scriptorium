"""
tests/test_preprocess.py
-------------------------
Unit tests for the preprocessing pipeline.
No GPU / Ollama required.
"""
import numpy as np
import pytest
from PIL import Image

from src.pipeline.preprocess import (
    preprocess,
    _upscale,
    _clahe,
    _denoise,
    _deskew,
)


def _make_image(w: int = 400, h: int = 300) -> Image.Image:
    arr = np.random.randint(100, 200, (h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr)


class TestUpscale:
    def test_small_image_is_upscaled(self):
        arr = np.zeros((100, 80, 3), dtype=np.uint8)
        out = _upscale(arr, min_dim=400)
        h, w = out.shape[:2]
        assert min(h, w) >= 400

    def test_large_image_unchanged(self):
        arr = np.zeros((1200, 1600, 3), dtype=np.uint8)
        out = _upscale(arr, min_dim=400)
        assert out.shape == arr.shape


class TestCLAHE:
    def test_output_shape_preserved(self):
        img = _make_image()
        arr = np.array(img)
        out = _clahe(arr)
        assert out.shape == arr.shape
        assert out.dtype == np.uint8


class TestDenoise:
    def test_output_shape_preserved(self):
        arr = np.array(_make_image())
        out = _denoise(arr)
        assert out.shape == arr.shape


class TestDeskew:
    def test_straight_image_unchanged_shape(self):
        arr = np.ones((300, 400, 3), dtype=np.uint8) * 255
        out = _deskew(arr)
        assert out.shape == arr.shape


class TestPreprocess:
    def test_returns_pil_image(self):
        img = _make_image()
        result = preprocess(img)
        assert isinstance(result, Image.Image)

    def test_binarize_false_default(self):
        img = _make_image()
        # Should not raise even with all options on
        result = preprocess(img, deskew=True, denoise=True, binarize=False)
        assert isinstance(result, Image.Image)

    def test_accepts_path(self, tmp_path):
        img = _make_image()
        p = tmp_path / "test.jpg"
        img.save(p)
        result = preprocess(p)
        assert isinstance(result, Image.Image)
