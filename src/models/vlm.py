"""
src/models/vlm.py
------------------
VLM end-to-end transcription using Qwen3.5-9B.

Qwen3.5 is a unified vision-language model (early fusion, native multimodal).
It exposes an OpenAI-compatible API and thinks by default; we disable thinking
for OCR tasks (instruct/non-thinking mode) to get clean plain-text output.

Inference backends supported:
  - vLLM  (recommended):  vllm serve Qwen/Qwen3.5-9B --reasoning-parser qwen3
  - SGLang:               python -m sglang.launch_server ... --reasoning-parser qwen3
  - HuggingFace Transformers:  transformers serve Qwen/Qwen3.5-9B

All expose an OpenAI-compatible endpoint at http://localhost:8000/v1 by default.
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

# Recommended sampling parameters for instruct (non-thinking) mode — Qwen3.5 docs
_GENERATION_DEFAULTS = dict(
    temperature=0.7,
    top_p=0.8,
    max_tokens=4096,
)


def transcribe_with_vlm(
    image_path: str | Path,
    model: str = "Qwen/Qwen3.5-9B",
    api_base: str = "http://localhost:8000/v1",
    api_key: str = "EMPTY",
    language: str = "it",
    thinking: bool = False,
) -> str:
    """
    Send *image_path* to Qwen3.5-9B via its OpenAI-compatible endpoint and
    return the transcribed text.

    Parameters
    ----------
    image_path : path to the handwriting image
    model      : model name as registered in the serving engine
    api_base   : base URL of the OpenAI-compatible server (vLLM / SGLang / HF)
    api_key    : API key (use "EMPTY" for local servers)
    language   : language hint included in the user prompt
    thinking   : if True, keep the <think>…</think> reasoning block in the
                 output (useful for debugging); default False for clean output
    """
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError("Run: pip install openai") from e

    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image_b64 = _encode_image(image_path)
    image_url = f"data:image/jpeg;base64,{image_b64}"

    user_content = [
        {"type": "image_url", "image_url": {"url": image_url}},
        {
            "type": "text",
            "text": (
                f"Transcribe all handwritten text in this image. "
                f"The text is in {language}. Return ONLY the transcription."
            ),
        },
    ]

    # Disable thinking mode for OCR — faster and cleaner output
    extra_body: dict = {}
    if not thinking:
        extra_body["chat_template_kwargs"] = {"enable_thinking": False}

    logger.info("Calling Qwen3.5 VLM '%s' on '%s' (thinking=%s)", model, image_path.name, thinking)

    client = OpenAI(base_url=api_base, api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ],
        extra_body=extra_body or None,
        **_GENERATION_DEFAULTS,
    )

    transcription = response.choices[0].message.content or ""

    # Strip any residual <think>…</think> block if thinking was left on
    transcription = re.sub(r"<think>.*?</think>", "", transcription, flags=re.DOTALL)
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
