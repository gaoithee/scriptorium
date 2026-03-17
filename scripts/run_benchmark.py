#!/usr/bin/env python
"""
scripts/run_benchmark.py
-------------------------
Main entry point for the handwriting OCR benchmark.

Usage examples
--------------
# With gold file:
python scripts/run_benchmark.py \
    --image data/samples/page.jpg \
    --gold  data/gold/page.txt

# With inline gold string:
python scripts/run_benchmark.py \
    --image data/samples/page.jpg \
    --gold-string "Il cielo è azzurro sopra il monte"

# Run all samples in data/samples/:
python scripts/run_benchmark.py --all

# Custom config:
python scripts/run_benchmark.py --image img.jpg --gold g.txt --config myconfig.yaml
"""
from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import typer
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Ensure project root is on path when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.eval.metrics import evaluate, EvalResult
from src.models.vlm import transcribe_with_vlm
from src.pipeline.correction import correct_with_llm
from src.pipeline.layout import detect_layout
from src.pipeline.ocr import ocr_regions

app = typer.Typer(pretty_exceptions_enable=False)
console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("benchmark")


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "ollama_base_url": "http://localhost:11434",
    "vlm_model": "qwen2.5vl:7b",
    "corrector_model": "qwen2.5:3b",
    "ocr_backend": "easyocr",
    "layout_backend": "doctr",
    "language": "it",
    "save_layout_debug": False,
}


def load_config(config_path: Path | None) -> dict:
    cfg = DEFAULT_CONFIG.copy()
    if config_path and config_path.exists():
        with open(config_path) as f:
            cfg.update(yaml.safe_load(f) or {})
    return cfg


# ---------------------------------------------------------------------------
# Core benchmark logic
# ---------------------------------------------------------------------------

def run_single(
    image_path: Path,
    gold: str,
    cfg: dict,
    output_dir: Path,
) -> dict:
    """
    Run both approaches on *image_path*, compare to *gold*, return results dict.
    """
    results: dict = {
        "image": str(image_path),
        "gold": gold,
        "timestamp": datetime.utcnow().isoformat(),
        "config": cfg,
    }

    # ── 1. Classic pipeline ────────────────────────────────────────────────
    console.rule("[bold blue]Approach 1 — Classic Pipeline (Layout → OCR → LM)")

    t0 = time.perf_counter()

    console.print(f"  [cyan]Layout detection[/cyan] ({cfg['layout_backend']}) …")
    pil_img, regions = detect_layout(
        image_path,
        backend=cfg["layout_backend"],
        save_debug=cfg.get("save_layout_debug", False),
    )
    console.print(f"  → {len(regions)} region(s) detected")

    console.print(f"  [cyan]OCR[/cyan] ({cfg['ocr_backend']}) …")
    raw_ocr = ocr_regions(pil_img, regions, backend=cfg["ocr_backend"], language=cfg["language"])
    console.print(f"  → Raw OCR output:\n{_preview(raw_ocr)}")

    console.print(f"  [cyan]LM post-correction[/cyan] ({cfg['corrector_model']}) …")
    pipeline_text = correct_with_llm(
        raw_ocr,
        model=cfg["corrector_model"],
        ollama_base_url=cfg["ollama_base_url"],
        language=cfg["language"],
    )
    console.print(f"  → Corrected:\n{_preview(pipeline_text)}")

    pipeline_time = time.perf_counter() - t0

    pipeline_eval = evaluate(pipeline_text, gold)
    results["pipeline"] = {
        "raw_ocr": raw_ocr,
        "corrected": pipeline_text,
        "time_s": round(pipeline_time, 2),
        "metrics": pipeline_eval.to_dict(),
        "char_diff": pipeline_eval.char_diff,
    }

    # ── 2. VLM end-to-end ─────────────────────────────────────────────────
    console.rule(f"[bold green]Approach 2 — VLM end-to-end ({cfg['vlm_model']})")

    t0 = time.perf_counter()
    vlm_text = transcribe_with_vlm(
        image_path,
        model=cfg["vlm_model"],
        ollama_base_url=cfg["ollama_base_url"],
        language=cfg["language"],
    )
    vlm_time = time.perf_counter() - t0

    console.print(f"  → VLM transcription:\n{_preview(vlm_text)}")

    vlm_eval = evaluate(vlm_text, gold)
    results["vlm"] = {
        "transcription": vlm_text,
        "time_s": round(vlm_time, 2),
        "metrics": vlm_eval.to_dict(),
        "char_diff": vlm_eval.char_diff,
    }

    # ── 3. Summary table ──────────────────────────────────────────────────
    _print_summary(pipeline_eval, vlm_eval, pipeline_time, vlm_time)

    # ── 4. Save artefacts ─────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    ts   = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    json_path = output_dir / f"{stem}_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    md_path = output_dir / f"{stem}_{ts}.md"
    _write_markdown_report(results, pipeline_eval, vlm_eval, md_path)

    console.print(f"\n[bold]Results saved to:[/bold] {json_path}")
    console.print(f"[bold]Markdown report:[/bold]  {md_path}")

    return results


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

@app.command()
def main(
    image: Path = typer.Option(..., "--image", "-i", help="Path to the handwriting image"),
    gold: Path  = typer.Option(None,  "--gold",  "-g", help="Path to the gold .txt file"),
    gold_string: str = typer.Option(None, "--gold-string", help="Gold string passed inline"),
    output: Path = typer.Option(Path("results"), "--output", "-o", help="Output directory"),
    config: Path = typer.Option(None, "--config", "-c", help="YAML config file"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Run the OCR benchmark on a single image."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    cfg = load_config(config)

    # Resolve gold string
    if gold_string:
        gold_text = gold_string
    elif gold:
        gold_text = gold.read_text(encoding="utf-8").strip()
    else:
        typer.echo("ERROR: provide either --gold <file> or --gold-string <text>", err=True)
        raise typer.Exit(1)

    if not image.exists():
        typer.echo(f"ERROR: image not found: {image}", err=True)
        raise typer.Exit(1)

    run_single(image, gold_text, cfg, output)


@app.command(name="all")
def run_all(
    samples_dir: Path = typer.Option(Path("data/samples"), "--samples"),
    gold_dir: Path    = typer.Option(Path("data/gold"),    "--gold-dir"),
    output: Path      = typer.Option(Path("results"),      "--output", "-o"),
    config: Path      = typer.Option(None,                 "--config", "-c"),
):
    """Run the benchmark on all images in *samples_dir*."""
    cfg = load_config(config)

    images = sorted(samples_dir.glob("*.[jp][pn][g]"))
    if not images:
        typer.echo(f"No images found in {samples_dir}", err=True)
        raise typer.Exit(1)

    all_results = []
    for img in images:
        gold_file = gold_dir / f"{img.stem}.txt"
        if not gold_file.exists():
            console.print(f"[yellow]Skipping {img.name} — no gold file found[/yellow]")
            continue
        gold_text = gold_file.read_text(encoding="utf-8").strip()
        r = run_single(img, gold_text, cfg, output)
        all_results.append(r)

    _print_aggregate_summary(all_results)


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def _preview(text: str, max_chars: int = 200) -> str:
    text = text.replace("\n", "↵ ")
    if len(text) > max_chars:
        return text[:max_chars] + " …"
    return text


def _print_summary(
    pe: EvalResult, ve: EvalResult,
    p_time: float, v_time: float,
) -> None:
    table = Table(title="Benchmark Summary", show_lines=True)
    table.add_column("Metric",   style="bold")
    table.add_column("Pipeline", justify="right")
    table.add_column("VLM",      justify="right")

    def _fmt(val: float) -> str:
        return f"{val:.3f}" if val >= 0 else "N/A"

    table.add_row("CER  ↓",  _fmt(pe.cer),  _fmt(ve.cer))
    table.add_row("WER  ↓",  _fmt(pe.wer),  _fmt(ve.wer))
    table.add_row("BLEU ↑",  _fmt(pe.bleu), _fmt(ve.bleu))
    table.add_row("Time (s)", f"{p_time:.1f}", f"{v_time:.1f}")

    console.print()
    console.print(table)

    # Character diff panels
    console.print(Panel(pe.char_diff, title="Pipeline char-diff  ([-del][+ins])", border_style="blue"))
    console.print(Panel(ve.char_diff, title="VLM char-diff",                      border_style="green"))


def _write_markdown_report(
    results: dict,
    pe: EvalResult,
    ve: EvalResult,
    path: Path,
) -> None:
    lines = [
        f"# OCR Benchmark — {results['image']}\n",
        f"**Timestamp:** {results['timestamp']}  \n",
        f"**Gold:** {results['gold']}\n",
        "",
        "## Metrics\n",
        "| Metric | Pipeline | VLM |",
        "|--------|----------|-----|",
        f"| CER ↓  | {pe.cer:.3f} | {ve.cer:.3f} |",
        f"| WER ↓  | {pe.wer:.3f} | {ve.wer:.3f} |",
        f"| BLEU ↑ | {pe.bleu:.2f} | {ve.bleu:.2f} |",
        f"| Time s | {results['pipeline']['time_s']} | {results['vlm']['time_s']} |",
        "",
        "## Transcriptions\n",
        "### Raw OCR",
        f"```\n{results['pipeline']['raw_ocr']}\n```",
        "### Pipeline (after LM correction)",
        f"```\n{results['pipeline']['corrected']}\n```",
        "### VLM end-to-end",
        f"```\n{results['vlm']['transcription']}\n```",
        "",
        "## Character diffs\n",
        "### Pipeline",
        f"```\n{pe.char_diff}\n```",
        "### VLM",
        f"```\n{ve.char_diff}\n```",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _print_aggregate_summary(all_results: list[dict]) -> None:
    if not all_results:
        return
    console.rule("[bold]Aggregate summary")
    table = Table(show_lines=True)
    table.add_column("Image")
    table.add_column("Pipeline CER", justify="right")
    table.add_column("VLM CER",      justify="right")
    table.add_column("Pipeline WER", justify="right")
    table.add_column("VLM WER",      justify="right")

    for r in all_results:
        table.add_row(
            Path(r["image"]).name,
            f"{r['pipeline']['metrics']['cer']:.3f}",
            f"{r['vlm']['metrics']['cer']:.3f}",
            f"{r['pipeline']['metrics']['wer']:.3f}",
            f"{r['vlm']['metrics']['wer']:.3f}",
        )
    console.print(table)


if __name__ == "__main__":
    app()
