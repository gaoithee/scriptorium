"""
tests/test_metrics.py
----------------------
Unit tests for the evaluation metrics module.
Run with: pytest tests/
"""
import pytest
from src.eval.metrics import evaluate, _compute_cer, _compute_wer, _compute_bleu, _char_diff


# ---------------------------------------------------------------------------
# CER
# ---------------------------------------------------------------------------

class TestCER:
    def test_perfect(self):
        assert _compute_cer("hello", "hello") == pytest.approx(0.0, abs=1e-4)

    def test_one_char_wrong(self):
        # "hEllo" vs "hello" → 1 substitution / 5 chars = 0.2
        cer = _compute_cer("hEllo", "hello")
        assert cer == pytest.approx(0.2, abs=0.01)

    def test_empty_hypothesis(self):
        cer = _compute_cer("", "hello")
        assert cer >= 1.0


# ---------------------------------------------------------------------------
# WER
# ---------------------------------------------------------------------------

class TestWER:
    def test_perfect(self):
        assert _compute_wer("the cat sat", "the cat sat") == pytest.approx(0.0, abs=1e-4)

    def test_one_word_wrong(self):
        wer = _compute_wer("the dog sat", "the cat sat")
        assert wer == pytest.approx(1 / 3, abs=0.02)

    def test_empty_hypothesis(self):
        wer = _compute_wer("", "the cat sat")
        assert wer >= 1.0


# ---------------------------------------------------------------------------
# BLEU
# ---------------------------------------------------------------------------

class TestBLEU:
    def test_perfect(self):
        bleu = _compute_bleu("the quick brown fox", "the quick brown fox")
        assert bleu == pytest.approx(100.0, abs=0.1)

    def test_worse_than_perfect(self):
        bleu = _compute_bleu("the slow brown fox", "the quick brown fox")
        assert bleu < 100.0

    def test_empty_hypothesis(self):
        bleu = _compute_bleu("", "the quick brown fox")
        assert bleu == pytest.approx(0.0, abs=1.0)


# ---------------------------------------------------------------------------
# Char diff
# ---------------------------------------------------------------------------

class TestCharDiff:
    def test_identical(self):
        diff = _char_diff("hello", "hello")
        assert "[" not in diff          # no insertions / deletions

    def test_insertion(self):
        diff = _char_diff("helllo", "hello")
        assert "[+" in diff or "[-" in diff

    def test_deletion(self):
        diff = _char_diff("helo", "hello")
        assert "[-" in diff


# ---------------------------------------------------------------------------
# EvalResult integration
# ---------------------------------------------------------------------------

class TestEvaluate:
    def test_returns_all_fields(self):
        result = evaluate("ciao mondo", "ciao mondo")
        assert result.cer  == pytest.approx(0.0, abs=1e-4)
        assert result.wer  == pytest.approx(0.0, abs=1e-4)
        assert result.bleu == pytest.approx(100.0, abs=0.1)
        assert isinstance(result.char_diff, str)

    def test_to_dict_has_metrics(self):
        d = evaluate("foo bar", "foo baz").to_dict()
        for key in ("cer", "wer", "bleu", "hypothesis", "reference"):
            assert key in d

    def test_worse_transcription(self):
        r = evaluate("ciaoo mondoo", "ciao mondo")
        assert r.cer > 0
        assert r.wer > 0
