"""
src/pipeline/correction.py
---------------------------
Small-LM post-correction step.

Uses a local Ollama model (default: qwen2.5:3b) to clean up the noisy
OCR output before metric evaluation.

The prompt is deliberately minimal: we tell the model to fix OCR errors
without paraphrasing or adding content.
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an expert proofreader specialised in correcting OCR errors in \
handwritten text transcriptions.

Rules:
- Fix obvious character recognition errors (e.g. "rn" → "m", "1" → "l", "0" → "o").
- Restore missing spaces and punctuation only when clearly needed.
- Do NOT paraphrase, summarise, or add any content.
- Return ONLY the corrected text, nothing else — no explanations, no labels.
"""


def correct_with_llm(
    raw_text: str,
    model: str = "qwen2.5:3b",
    ollama_base_url: str = "http://localhost:11434",
    language: str = "it",
) -> str:
    """
    Send *raw_text* (noisy OCR output) to a local Ollama model and return
    the corrected transcription string.

    Parameters
    ----------
    raw_text        : the OCR-produced string to clean up
    model           : Ollama model tag (e.g. 'qwen2.5:3b', 'qwen2.5:14b')
    ollama_base_url : base URL of the running Ollama instance
    language        : hint passed in the user message
    """
    try:
        import ollama
    except ImportError as e:
        raise ImportError("Run: pip install ollama") from e

    if not raw_text.strip():
        logger.warning("correction.py received empty OCR text — skipping LM call")
        return raw_text

    user_message = (
        f"The following text was produced by an OCR system reading a handwritten "
        f"document in {language}. Please correct any OCR errors and return ONLY the "
        f"corrected text.\n\n---\n{raw_text}\n---"
    )

    logger.info("Calling Ollama model '%s' for post-correction", model)

    client = ollama.Client(host=ollama_base_url)
    response = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        options={"temperature": 0.0},  # deterministic
    )

    corrected = response["message"]["content"]
    corrected = _strip_markdown_fences(corrected).strip()
    logger.info("Post-correction done (%d → %d chars)", len(raw_text), len(corrected))
    return corrected


def _strip_markdown_fences(text: str) -> str:
    """Remove ```...``` wrappers that some models add despite instructions."""
    return re.sub(r"^```[^\n]*\n?", "", re.sub(r"\n?```$", "", text.strip()))
