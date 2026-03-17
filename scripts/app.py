"""
scripts/app.py
---------------
Gradio web interface for the handwriting OCR benchmark.

Launch with:
    python scripts/app.py
    python scripts/app.py --port 7861 --share
"""
from __future__ import annotations

import json
import logging
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
import yaml

try:
    import gradio as gr
except ImportError as e:
    raise SystemExit("Install gradio: pip install gradio") from e

from src.eval.metrics import evaluate
from src.models.vlm import transcribe_with_vlm
from src.pipeline.correction import correct_with_llm
from src.pipeline.layout import detect_layout
from src.pipeline.ocr import ocr_regions
from src.pipeline.preprocess import preprocess

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("app")

DEFAULT_CONFIG = {
    "vlm_api_base":  "http://localhost:8000/v1",
    "vlm_api_key":   "EMPTY",
    "vlm_model":     "Qwen/Qwen3.5-9B",
    "vlm_thinking":  False,
    "corrector_api_base": "http://localhost:11434/v1",
    "corrector_api_key":  "EMPTY",
    "corrector_model":    "qwen2.5:3b",
    "ocr_backend":    "easyocr",
    "layout_backend": "doctr",
    "language":       "it",
}


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------

def run_inference(
    image_pil,
    gold_string: str,
    run_pipeline: bool,
    run_vlm: bool,
    language: str,
    ocr_backend: str,
    vlm_model: str,
    vlm_api_base: str,
    corrector_model: str,
    corrector_api_base: str,
    do_preprocess: bool,
    vlm_thinking: bool,
) -> tuple[str, str, str, str, str, str]:
    if image_pil is None:
        return "", "", "", "⚠️ Upload an image first.", "", ""

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        image_pil.save(tmp_path, format="JPEG")

    results: dict = {"gold": gold_string}
    raw_ocr = pipeline_text = vlm_text = ""
    pipeline_metrics_str = vlm_metrics_str = ""

    try:
        working_image = preprocess(image_pil) if do_preprocess else image_pil

        if run_pipeline:
            pil_img, regions = detect_layout(tmp_path, backend="doctr")
            raw_ocr = ocr_regions(
                working_image, regions,
                backend=ocr_backend,
                language=language,
            )
            pipeline_text = correct_with_llm(
                raw_ocr,
                model=corrector_model,
                api_base=corrector_api_base,
                api_key="EMPTY",
                language=language,
            )
            if gold_string.strip():
                r = evaluate(pipeline_text, gold_string)
                pipeline_metrics_str = (
                    f"CER: {r.cer:.3f}  |  WER: {r.wer:.3f}  |  BLEU: {r.bleu:.2f}\n\n"
                    f"Char diff:\n{r.char_diff}"
                )
                results["pipeline"] = {
                    "raw_ocr": raw_ocr,
                    "corrected": pipeline_text,
                    "metrics": r.to_dict(),
                }

        if run_vlm:
            vlm_text = transcribe_with_vlm(
                tmp_path,
                model=vlm_model,
                api_base=vlm_api_base,
                api_key="EMPTY",
                language=language,
                thinking=vlm_thinking,
            )
            if gold_string.strip():
                r = evaluate(vlm_text, gold_string)
                vlm_metrics_str = (
                    f"CER: {r.cer:.3f}  |  WER: {r.wer:.3f}  |  BLEU: {r.bleu:.2f}\n\n"
                    f"Char diff:\n{r.char_diff}"
                )
                results["vlm"] = {
                    "transcription": vlm_text,
                    "metrics": r.to_dict(),
                }

    except Exception as exc:
        logger.exception("Inference error")
        return str(exc), "", "", "Error", "Error", ""
    finally:
        tmp_path.unlink(missing_ok=True)

    json_tmp = tempfile.NamedTemporaryFile(
        suffix=".json", delete=False, mode="w", encoding="utf-8"
    )
    json.dump(results, json_tmp, ensure_ascii=False, indent=2)
    json_tmp.close()

    return raw_ocr, pipeline_text, vlm_text, pipeline_metrics_str, vlm_metrics_str, json_tmp.name


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="scriptorium", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# scriptorium\n"
            "Classic pipeline (DocTR → EasyOCR → LM correction) "
            "vs **Qwen3.5-9B** end-to-end."
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="Handwriting image")
                gold_input  = gr.Textbox(
                    label="Gold transcription (optional — needed for metrics)",
                    placeholder="Paste the ground-truth text here …",
                    lines=4,
                )
                with gr.Accordion("⚙️ Options", open=False):
                    language        = gr.Dropdown(
                        ["it", "en", "fr", "de", "es", "la"],
                        value="it", label="Language"
                    )
                    ocr_backend     = gr.Dropdown(
                        ["easyocr", "tesseract"], value="easyocr", label="OCR backend"
                    )
                    vlm_model       = gr.Textbox(
                        value="Qwen/Qwen3.5-9B", label="VLM model"
                    )
                    vlm_api_base    = gr.Textbox(
                        value="http://localhost:8000/v1", label="VLM API base (vLLM / SGLang)"
                    )
                    corrector_model = gr.Textbox(
                        value="qwen2.5:3b", label="Corrector model"
                    )
                    corrector_api   = gr.Textbox(
                        value="http://localhost:11434/v1", label="Corrector API base"
                    )
                    vlm_thinking    = gr.Checkbox(
                        value=False, label="Enable VLM thinking mode (slower)"
                    )
                    do_preproc      = gr.Checkbox(
                        value=True, label="Preprocess image (deskew, denoise, CLAHE)"
                    )

                with gr.Row():
                    run_pipe = gr.Checkbox(value=True,  label="Run Pipeline")
                    run_vlm  = gr.Checkbox(value=True,  label="Run VLM")

                submit_btn = gr.Button("▶ Run", variant="primary")

            with gr.Column(scale=2):
                with gr.Tab("Pipeline"):
                    raw_ocr_out   = gr.Textbox(label="Raw OCR",             lines=6, interactive=False)
                    pipe_text_out = gr.Textbox(label="After LM correction",  lines=6, interactive=False)
                    pipe_metrics  = gr.Textbox(label="Metrics vs gold",      lines=5, interactive=False)

                with gr.Tab("VLM — Qwen3.5-9B"):
                    vlm_text_out  = gr.Textbox(label="VLM transcription",   lines=8, interactive=False)
                    vlm_metrics   = gr.Textbox(label="Metrics vs gold",      lines=5, interactive=False)

                json_download = gr.File(label="Download JSON result")

        submit_btn.click(
            fn=run_inference,
            inputs=[
                image_input, gold_input,
                run_pipe, run_vlm,
                language, ocr_backend,
                vlm_model, vlm_api_base,
                corrector_model, corrector_api,
                do_preproc, vlm_thinking,
            ],
            outputs=[
                raw_ocr_out, pipe_text_out, vlm_text_out,
                pipe_metrics, vlm_metrics,
                json_download,
            ],
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

cli = typer.Typer()

@cli.command()
def main(
    port:  int  = typer.Option(7860, "--port", "-p"),
    share: bool = typer.Option(False, "--share"),
):
    """Launch the Gradio web UI."""
    build_ui().launch(server_port=port, share=share)


if __name__ == "__main__":
    cli()
