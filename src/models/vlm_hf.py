"""
src/models/vlm_hf.py
---------------------
HuggingFace Transformers fallback for Qwen2.5-VL when Ollama is not available.

Supports:
  - Qwen/Qwen2.5-VL-7B-Instruct   (default)
  - Qwen/Qwen2.5-VL-72B-Instruct  (requires ~160 GB VRAM, use quantized)

Usage
-----
from src.models.vlm_hf import transcribe_with_vlm_hf
text = transcribe_with_vlm_hf("page.jpg", model_id="Qwen/Qwen2.5-VL-7B-Instruct")

Notes
-----
- Requires: pip install transformers accelerate qwen-vl-utils
- On CPU-only machines this will be very slow; use Ollama backend instead.
- 4-bit quantization is enabled automatically when bitsandbytes is installed.
"""
from __future__ import annotations

import base64
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an expert in reading and transcribing handwritten documents.
Produce a verbatim transcription of the handwritten text in the image.
Rules:
- Transcribe EXACTLY what is written. Do not correct spelling or grammar.
- Preserve line breaks where identifiable.
- Mark illegible words as [illegible].
- Return ONLY the transcription, no explanations, no markdown.
"""


def transcribe_with_vlm_hf(
    image_path: str | Path,
    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    language: str = "it",
    max_new_tokens: int = 1024,
    quantize_4bit: bool = True,
) -> str:
    """
    Transcribe handwritten text using a locally loaded Qwen2.5-VL model
    via HuggingFace Transformers.

    Parameters
    ----------
    image_path      : path to the handwriting image
    model_id        : HuggingFace model ID
    language        : language hint for the prompt
    max_new_tokens  : maximum tokens to generate
    quantize_4bit   : load in 4-bit (requires bitsandbytes)
    """
    try:
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        from qwen_vl_utils import process_vision_info
    except ImportError as e:
        raise ImportError(
            "Install HuggingFace dependencies: "
            "pip install transformers accelerate qwen-vl-utils"
        ) from e

    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    logger.info("Loading HuggingFace model: %s", model_id)

    load_kwargs: dict = {"device_map": "auto", "torch_dtype": "auto"}
    if quantize_4bit:
        try:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            logger.info("Using 4-bit quantization")
        except ImportError:
            logger.warning("bitsandbytes not found; loading in full precision")

    model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **load_kwargs)
    processor = AutoProcessor.from_pretrained(model_id)

    user_content = [
        {"type": "image", "image": str(image_path)},
        {
            "type": "text",
            "text": (
                f"Please transcribe all handwritten text in this image. "
                f"The text is in {language}. Return ONLY the transcription."
            ),
        },
    ]

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]

    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text_input],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    logger.info("Generating transcription …")
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    # Strip prompt tokens from output
    trimmed = [
        out[len(inp):]
        for inp, out in zip(inputs.input_ids, output_ids)
    ]
    transcription = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    transcription = _strip_markdown_fences(transcription).strip()
    logger.info("HF VLM transcription: %d chars", len(transcription))
    return transcription


def _strip_markdown_fences(text: str) -> str:
    return re.sub(r"^```[^\n]*\n?", "", re.sub(r"\n?```$", "", text.strip()))
