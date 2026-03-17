"""
src/models/vlm_hf.py
---------------------
HuggingFace Transformers backend for Qwen3.5-9B.

Use this when you want to load the model weights directly (no inference server).
For production use, prefer the OpenAI-compatible API via vLLM or SGLang (vlm.py).

Model: Qwen/Qwen3.5-9B  (unified vision-language, native multimodal)

Usage
-----
from src.models.vlm_hf import transcribe_with_vlm_hf
text = transcribe_with_vlm_hf("page.jpg")

Notes
-----
- Requires: pip install "transformers[serving] @ git+https://github.com/huggingface/transformers.git@main"
            pip install accelerate qwen-vl-utils pillow torchvision
- 4-bit quantization is enabled automatically when bitsandbytes is installed.
- Thinking mode is disabled by default (instruct mode) for clean OCR output.
"""
from __future__ import annotations

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
    model_id: str = "Qwen/Qwen3.5-9B",
    language: str = "it",
    max_new_tokens: int = 4096,
    quantize_4bit: bool = True,
    thinking: bool = False,
) -> str:
    """
    Transcribe handwritten text using Qwen3.5-9B loaded directly via
    HuggingFace Transformers.

    Parameters
    ----------
    image_path      : path to the handwriting image
    model_id        : HuggingFace model ID (default: Qwen/Qwen3.5-9B)
    language        : language hint for the prompt
    max_new_tokens  : maximum tokens to generate
    quantize_4bit   : load in 4-bit (requires bitsandbytes)
    thinking        : enable chain-of-thought reasoning (slower, not needed for OCR)
    """
    try:
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText
        from qwen_vl_utils import process_vision_info
    except ImportError as e:
        raise ImportError(
            "Install: pip install 'transformers[serving] @ git+https://github.com/"
            "huggingface/transformers.git@main' accelerate qwen-vl-utils torchvision"
        ) from e

    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    logger.info("Loading %s from HuggingFace", model_id)

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

    model = AutoModelForImageTextToText.from_pretrained(model_id, **load_kwargs)
    processor = AutoProcessor.from_pretrained(model_id)

    user_content = [
        {"type": "image", "image": str(image_path)},
        {
            "type": "text",
            "text": (
                f"Transcribe all handwritten text in this image. "
                f"The text is in {language}. Return ONLY the transcription."
            ),
        },
    ]

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]

    # Disable thinking for instruct/OCR mode — faster, cleaner output
    chat_template_kwargs = {"enable_thinking": thinking}

    text_input = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        chat_template_kwargs=chat_template_kwargs,
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text_input],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    # Instruct mode sampling params (from Qwen3.5 docs)
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        do_sample=True,
    )

    logger.info("Generating transcription …")
    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)

    # Strip prompt tokens from output
    trimmed = [
        out[len(inp):]
        for inp, out in zip(inputs.input_ids, output_ids)
    ]
    transcription = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    # Remove any residual <think> block
    transcription = re.sub(r"<think>.*?</think>", "", transcription, flags=re.DOTALL)
    transcription = _strip_markdown_fences(transcription).strip()
    logger.info("HF VLM transcription: %d chars", len(transcription))
    return transcription


def _strip_markdown_fences(text: str) -> str:
    return re.sub(r"^```[^\n]*\n?", "", re.sub(r"\n?```$", "", text.strip()))
