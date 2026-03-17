"""
scripts/app.py
---------------
Gradio web interface for the handwriting OCR benchmark.

Launch with:
    python scripts/app.py
    # or with custom port:
    python scripts/app.py --port 7861

Features
--------
- Upload a handwriting image
- Paste your gold string
- Choose which approaches to run (pipeline, VLM, or both)
- See side-by-side transcriptions and metrics
- Download the JSON result
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
    "ollama_base_url": "http://localhost:11434",
    "vlm_model": "qwen2.5vl:7b",
    "corrector_model": "qwen2.5:3b",
    "ocr_backend": "easyocr",
    "layout_backend": "doctr",
    "language": "it",
    "save_layout_debug": False,
}


def load_cfg(config_path: str | None = None) -> dict:
    cfg = DEFAULT_CONFIG.copy()
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            cfg.update(yaml.safe_load(f) or {})
    return cfg


# ---------------------------------------------------------------------------
# Core inference function called by Gradio
# ---------------------------------------------------------------------------

def run_inference(
    image_pil,
    gold_string: str,
    run_pipeline: bool,
    run_vlm: bool,
    language: str,
    ocr_backend: str,
    vlm_model: str,
    corrector_model: str,
    ollama_url: str,
    do_preprocess: bool,
) -> tuple[str, str, str, str, str, str]:
    """
    Returns (raw_ocr, pipeline_text, vlm_text, pipeline_metrics, vlm_metrics, json_download_path)
    """
    if image_pil is None:
        return "", "", "", "⚠️ Upload an image first.", "", ""

    cfg = {
        "ollama_base_url": ollama_url,
        "vlm_model": vlm_model,
        "corrector_model": corrector_model,
        "ocr_backend": ocr_backend,
        "layout_backend": "doctr",
        "language": language,
        "save_layout_debug": False,
    }

    # Save uploaded PIL image to a temp file (needed by DocTR / VLM)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        image_pil.save(tmp_path, format="JPEG")

    results: dict = {"gold": gold_string, "config": cfg}
    raw_ocr = pipeline_text = vlm_text = ""
    pipeline_metrics_str = vlm_metrics_str = ""

    try:
        working_image = preprocess(image_pil) if do_preprocess else image_pil

        # ── Pipeline ──────────────────────────────────────────────────────
        if run_pipeline:
            pil_img, regions = detect_layout(tmp_path, backend=cfg["layout_backend"])
            raw_ocr = ocr_regions(
                working_image, regions,
                backend=cfg["ocr_backend"],
                language=cfg["language"],
            )
            pipeline_text = correct_with_llm(
                raw_ocr,
                model=cfg["corrector_model"],
                ollama_base_url=cfg["ollama_base_url"],
                language=cfg["language"],
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

        # ── VLM ───────────────────────────────────────────────────────────
        if run_vlm:
            vlm_text = transcribe_with_vlm(
                tmp_path,
                model=cfg["vlm_model"],
                ollama_base_url=cfg["ollama_base_url"],
                language=cfg["language"],
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

    # Write JSON to a temp file so Gradio can serve it as a download
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
    with gr.Blocks(title="Handwriting OCR Benchmark", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# ✍️ Handwriting OCR Benchmark\n"
            "Compare **Classic Pipeline** (DocTR → EasyOCR → Qwen2.5:3b) "
            "vs **Qwen2.5-VL** end-to-end on your handwriting scans."
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
                    language    = gr.Dropdown(
                        ["it", "en", "fr", "de", "es", "la"],
                        value="it", label="Language"
                    )
                    ocr_backend = gr.Dropdown(
                        ["easyocr", "tesseract"], value="easyocr", label="OCR backend"
                    )
                    vlm_model   = gr.Textbox(value="qwen2.5vl:7b",  label="VLM model (Ollama tag)")
                    corrector   = gr.Textbox(value="qwen2.5:3b",    label="Corrector model (Ollama tag)")
                    ollama_url  = gr.Textbox(value="http://localhost:11434", label="Ollama URL")
                    do_preproc  = gr.Checkbox(value=True, label="Apply image preprocessing (deskew, denoise, CLAHE)")

                with gr.Row():
                    run_pipe = gr.Checkbox(value=True,  label="Run Pipeline")
                    run_vlm  = gr.Checkbox(value=True,  label="Run VLM")

                submit_btn = gr.Button("▶ Run", variant="primary")

            with gr.Column(scale=2):
                with gr.Tab("Pipeline"):
                    raw_ocr_out  = gr.Textbox(label="Raw OCR output",             lines=6, interactive=False)
                    pipe_text_out = gr.Textbox(label="After LM post-correction",   lines=6, interactive=False)
                    pipe_metrics  = gr.Textbox(label="Metrics vs gold",            lines=5, interactive=False)

                with gr.Tab("VLM (Qwen2.5-VL)"):
                    vlm_text_out  = gr.Textbox(label="VLM transcription",          lines=8, interactive=False)
                    vlm_metrics   = gr.Textbox(label="Metrics vs gold",            lines=5, interactive=False)

                json_download = gr.File(label="Download JSON result")

        submit_btn.click(
            fn=run_inference,
            inputs=[
                image_input, gold_input,
                run_pipe, run_vlm,
                language, ocr_backend, vlm_model, corrector, ollama_url,
                do_preproc,
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
    port: int = typer.Option(7860, "--port", "-p"),
    share: bool = typer.Option(False, "--share", help="Create a public Gradio link"),
    config: str = typer.Option(None, "--config", "-c"),
):
    """Launch the Gradio web UI."""
    app = build_ui()
    app.launch(server_port=port, share=share)


if __name__ == "__main__":
    cli()
