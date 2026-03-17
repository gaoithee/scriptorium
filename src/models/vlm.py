"""
src/models/vlm.py
------------------
VLM end-to-end transcription using Qwen2.5-VL via Ollama.

The model receives the raw image and returns a plain-text transcription
with no intermediate OCR steps.

Qwen2.5-VL supports native vision input; we pass the image as a base64
blob through the Ollama multimodal API.
"""
from __future__ import annotations

import base64
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an expert in reading and transcribing handwritten documents.
Your task is to produce a verbatim transcription of the handwritten text
visible in the image.

Rules:
- Transcribe EXACTLY what is written — do not correct spelling or grammar.
- Preserve line breaks where you can identify them.
- If a word is illegible, write [illegible] in its place.
- Return ONLY the transcription, with no explanations, no labels, no markdown.
"""


def transcribe_with_vlm(
    image_path: str | Path,
    model: str = "qwen2.5vl:7b",
    ollama_base_url: str = "http://localhost:11434",
    language: str = "it",
) -> str:
    """
    Send *image_path* to a Qwen2.5-VL model running in Ollama and return
    the transcribed text.

    Parameters
    ----------
    image_path      : path to the handwriting image
    model           : Ollama model tag, e.g. 'qwen2.5vl:7b' or 'qwen2.5vl:72b'
    ollama_base_url : base URL of the running Ollama instance
    language        : language hint included in the user prompt
    """
    try:
        import ollama
    except ImportError as e:
        raise ImportError("Run: pip install ollama") from e

    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image_b64 = _encode_image(image_path)

    user_message = (
        f"Please transcribe all handwritten text in this image. "
        f"The text is written in {language}. "
        f"Return ONLY the transcription."
    )

    logger.info("Calling Ollama VLM '%s' on image '%s'", model, image_path.name)

    client = ollama.Client(host=ollama_base_url)
    response = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": user_message,
                "images": [image_b64],   # Ollama multimodal API
            },
        ],
        options={"temperature": 0.0},
    )

    transcription = response["message"]["content"]
    transcription = _strip_markdown_fences(transcription).strip()
    logger.info("VLM transcription: %d chars", len(transcription))
    return transcription


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _encode_image(path: Path) -> str:
    """Return the image as a base64-encoded string."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _strip_markdown_fences(text: str) -> str:
    return re.sub(r"^```[^\n]*\n?", "", re.sub(r"\n?```$", "", text.strip()))
