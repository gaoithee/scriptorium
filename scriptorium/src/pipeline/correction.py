"""
src/pipeline/correction.py
---------------------------
Small-LM post-correction step via any OpenAI-compatible endpoint.

Uses Qwen3.5-9B (or any other model) to clean up noisy OCR output before
metric evaluation. Defaults to a lightweight local model so it does not
compete for GPU memory with the VLM.

Compatible servers: vLLM, SGLang, Ollama (/v1 endpoint), LM Studio, etc.
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
    api_base: str = "http://localhost:11434/v1",
    api_key: str = "EMPTY",
    language: str = "it",
) -> str:
    """
    Send *raw_text* (noisy OCR output) to an OpenAI-compatible LM endpoint
    and return the corrected transcription string.

    Parameters
    ----------
    raw_text : the OCR-produced string to clean up
    model    : model name as registered in the serving engine
    api_base : base URL of an OpenAI-compatible server
               - Ollama:  http://localhost:11434/v1
               - vLLM:   http://localhost:8000/v1
               - SGLang: http://localhost:8000/v1
    api_key  : API key (use "EMPTY" for local servers)
    language : hint passed in the user message
    """
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError("Run: pip install openai") from e

    if not raw_text.strip():
        logger.warning("correction.py received empty OCR text — skipping LM call")
        return raw_text

    user_message = (
        f"The following text was produced by an OCR system reading a handwritten "
        f"document in {language}. Please correct any OCR errors and return ONLY the "
        f"corrected text.\n\n---\n{raw_text}\n---"
    )

    logger.info("Calling '%s' @ %s for post-correction", model, api_base)

    client = OpenAI(base_url=api_base, api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        temperature=0.0,
        max_tokens=2048,
    )

    corrected = response.choices[0].message.content or raw_text
    corrected = _strip_markdown_fences(corrected).strip()
    logger.info("Post-correction: %d → %d chars", len(raw_text), len(corrected))
    return corrected


def _strip_markdown_fences(text: str) -> str:
    """Remove ```...``` wrappers that some models add despite instructions."""
    return re.sub(r"^```[^\n]*\n?", "", re.sub(r"\n?```$", "", text.strip()))
