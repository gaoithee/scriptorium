"""
src/eval/metrics.py
--------------------
Evaluation utilities: CER, WER, BLEU and character-level diff.
"""
from __future__ import annotations

import difflib
import logging
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    hypothesis: str
    reference: str
    cer: float
    wer: float
    bleu: float
    char_diff: str = field(repr=False)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("char_diff")          # keep JSON clean; store separately
        return d

    def summary_line(self, label: str = "") -> str:
        tag = f"[{label}] " if label else ""
        return (
            f"{tag}CER={self.cer:.3f}  WER={self.wer:.3f}  BLEU={self.bleu:.2f}"
        )


def evaluate(hypothesis: str, reference: str) -> EvalResult:
    """
    Compute CER, WER, BLEU and a character-level diff.

    Parameters
    ----------
    hypothesis : model-produced transcription
    reference  : gold / ground-truth string
    """
    cer  = _compute_cer(hypothesis, reference)
    wer  = _compute_wer(hypothesis, reference)
    bleu = _compute_bleu(hypothesis, reference)
    diff = _char_diff(hypothesis, reference)

    return EvalResult(
        hypothesis=hypothesis,
        reference=reference,
        cer=cer,
        wer=wer,
        bleu=bleu,
        char_diff=diff,
    )


# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------

def _compute_cer(hyp: str, ref: str) -> float:
    """Character Error Rate via jiwer (treats each char as a 'word')."""
    try:
        from jiwer import cer
        return float(cer(ref, hyp))
    except ImportError:
        logger.warning("jiwer not installed; falling back to edit-distance CER")
        return _edit_distance_ratio(hyp, ref, unit="char")


def _compute_wer(hyp: str, ref: str) -> float:
    """Word Error Rate via jiwer."""
    try:
        from jiwer import wer
        return float(wer(ref, hyp))
    except ImportError:
        logger.warning("jiwer not installed; falling back to edit-distance WER")
        return _edit_distance_ratio(hyp, ref, unit="word")


def _compute_bleu(hyp: str, ref: str) -> float:
    """Sentence BLEU (1–4 gram) via sacrebleu."""
    try:
        from sacrebleu.metrics import BLEU
        bleu = BLEU(effective_order=True)
        score = bleu.sentence_score(hyp, [ref])
        return float(score.score)          # 0–100
    except ImportError:
        logger.warning("sacrebleu not installed; BLEU set to -1")
        return -1.0


def _char_diff(hyp: str, ref: str) -> str:
    """
    Generate a coloured character-level diff string (ANSI or plain).
    '+' = insertion, '-' = deletion, ' ' = match.
    """
    matcher = difflib.SequenceMatcher(None, ref, hyp)
    diff_parts: list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            diff_parts.append(ref[i1:i2])
        elif tag == "replace":
            diff_parts.append(f"[-{ref[i1:i2]}+{hyp[j1:j2]}]")
        elif tag == "delete":
            diff_parts.append(f"[-{ref[i1:i2]}]")
        elif tag == "insert":
            diff_parts.append(f"[+{hyp[j1:j2]}]")
    return "".join(diff_parts)


# ---------------------------------------------------------------------------
# Fallback edit-distance helpers (no deps)
# ---------------------------------------------------------------------------

def _edit_distance_ratio(a: str, b: str, unit: str) -> float:
    tokens_a = list(a) if unit == "char" else a.split()
    tokens_b = list(b) if unit == "char" else b.split()
    dist = _levenshtein(tokens_a, tokens_b)
    return dist / max(len(tokens_b), 1)


def _levenshtein(a: list, b: list) -> int:
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[j] = prev[j - 1]
            else:
                dp[j] = 1 + min(prev[j], dp[j - 1], prev[j - 1])
    return dp[m]
