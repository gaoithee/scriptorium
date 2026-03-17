"""
scripts/eval_dataset.py
------------------------
Evaluate both approaches on an entire dataset and produce an aggregate
Markdown + CSV report.

Expected layout
---------------
data/
  samples/  ← *.jpg / *.png
  gold/     ← <same_stem>.txt

Usage
-----
python scripts/eval_dataset.py
python scripts/eval_dataset.py --samples data/samples --gold data/gold --output results/
"""
from __future__ import annotations

import csv
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import typer
import yaml
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.eval.metrics import evaluate
from src.models.vlm import transcribe_with_vlm
from src.pipeline.correction import correct_with_llm
from src.pipeline.layout import detect_layout
from src.pipeline.ocr import ocr_regions
from src.pipeline.preprocess import preprocess

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("eval_dataset")
console = Console()

DEFAULT_CONFIG = {
    "ollama_base_url": "http://localhost:11434",
    "vlm_model": "qwen2.5vl:7b",
    "corrector_model": "qwen2.5:3b",
    "ocr_backend": "easyocr",
    "layout_backend": "doctr",
    "language": "it",
}

app = typer.Typer()


@app.command()
def main(
    samples: Path = typer.Option(Path("data/samples"), "--samples"),
    gold_dir: Path = typer.Option(Path("data/gold"),   "--gold"),
    output:   Path = typer.Option(Path("results"),     "--output", "-o"),
    config:   Path = typer.Option(None,                "--config", "-c"),
    preprocess_images: bool = typer.Option(True, "--preprocess/--no-preprocess"),
):
    cfg = DEFAULT_CONFIG.copy()
    if config and config.exists():
        with open(config) as f:
            cfg.update(yaml.safe_load(f) or {})

    images = sorted(samples.glob("*.[jp][pn][g]"))
    if not images:
        console.print(f"[red]No images found in {samples}[/red]")
        raise typer.Exit(1)

    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for img_path in images:
        gold_file = gold_dir / f"{img_path.stem}.txt"
        if not gold_file.exists():
            console.print(f"[yellow]Skip {img_path.name} — no gold file[/yellow]")
            continue

        gold = gold_file.read_text(encoding="utf-8").strip()
        console.rule(f"[bold]{img_path.name}")

        row = {"image": img_path.name, "gold": gold}

        try:
            pil_img, regions = detect_layout(img_path, backend=cfg["layout_backend"])
            if preprocess_images:
                pil_img = preprocess(pil_img)

            # Pipeline
            t0 = time.perf_counter()
            raw_ocr = ocr_regions(pil_img, regions, backend=cfg["ocr_backend"], language=cfg["language"])
            corrected = correct_with_llm(raw_ocr, model=cfg["corrector_model"],
                                         ollama_base_url=cfg["ollama_base_url"],
                                         language=cfg["language"])
            pipe_t = round(time.perf_counter() - t0, 2)
            pipe_eval = evaluate(corrected, gold)

            row.update({
                "pipeline_raw":       raw_ocr,
                "pipeline_corrected": corrected,
                "pipeline_cer":       round(pipe_eval.cer,  4),
                "pipeline_wer":       round(pipe_eval.wer,  4),
                "pipeline_bleu":      round(pipe_eval.bleu, 2),
                "pipeline_time_s":    pipe_t,
            })

            # VLM
            t0 = time.perf_counter()
            vlm_text = transcribe_with_vlm(img_path, model=cfg["vlm_model"],
                                           ollama_base_url=cfg["ollama_base_url"],
                                           language=cfg["language"])
            vlm_t = round(time.perf_counter() - t0, 2)
            vlm_eval = evaluate(vlm_text, gold)

            row.update({
                "vlm_transcription": vlm_text,
                "vlm_cer":           round(vlm_eval.cer,  4),
                "vlm_wer":           round(vlm_eval.wer,  4),
                "vlm_bleu":          round(vlm_eval.bleu, 2),
                "vlm_time_s":        vlm_t,
            })

        except Exception as exc:
            logger.exception("Error processing %s", img_path.name)
            row["error"] = str(exc)

        rows.append(row)
        _print_row_summary(row)

    # ── Aggregate ──────────────────────────────────────────────────────────
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    # CSV
    csv_path = output / f"benchmark_{ts}.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        console.print(f"\n[green]CSV saved:[/green] {csv_path}")

    # JSON
    json_path = output / f"benchmark_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"config": cfg, "results": rows}, f, ensure_ascii=False, indent=2)
    console.print(f"[green]JSON saved:[/green] {json_path}")

    # Markdown aggregate report
    md_path = output / f"benchmark_{ts}.md"
    _write_aggregate_md(rows, cfg, md_path)
    console.print(f"[green]Report saved:[/green] {md_path}")

    _print_aggregate_table(rows)


def _print_row_summary(row: dict) -> None:
    if "error" in row:
        console.print(f"  [red]ERROR: {row['error']}[/red]")
        return
    console.print(
        f"  Pipeline — CER={row.get('pipeline_cer','?'):.3f}  "
        f"WER={row.get('pipeline_wer','?'):.3f}  "
        f"BLEU={row.get('pipeline_bleu','?'):.2f}"
    )
    console.print(
        f"  VLM      — CER={row.get('vlm_cer','?'):.3f}  "
        f"WER={row.get('vlm_wer','?'):.3f}  "
        f"BLEU={row.get('vlm_bleu','?'):.2f}"
    )


def _print_aggregate_table(rows: list[dict]) -> None:
    valid = [r for r in rows if "error" not in r]
    if not valid:
        return

    def mean(key: str) -> float:
        vals = [r[key] for r in valid if key in r]
        return sum(vals) / len(vals) if vals else 0.0

    table = Table(title=f"Aggregate ({len(valid)} samples)", show_lines=True)
    table.add_column("Metric")
    table.add_column("Pipeline", justify="right")
    table.add_column("VLM", justify="right")

    table.add_row("CER ↓",   f"{mean('pipeline_cer'):.3f}",  f"{mean('vlm_cer'):.3f}")
    table.add_row("WER ↓",   f"{mean('pipeline_wer'):.3f}",  f"{mean('vlm_wer'):.3f}")
    table.add_row("BLEU ↑",  f"{mean('pipeline_bleu'):.2f}", f"{mean('vlm_bleu'):.2f}")
    table.add_row("Time s",  f"{mean('pipeline_time_s'):.1f}", f"{mean('vlm_time_s'):.1f}")

    console.print()
    console.print(table)


def _write_aggregate_md(rows: list[dict], cfg: dict, path: Path) -> None:
    valid = [r for r in rows if "error" not in r]

    def mean(key: str) -> float:
        vals = [r[key] for r in valid if key in r]
        return sum(vals) / len(vals) if vals else 0.0

    lines = [
        "# Benchmark Aggregate Report\n",
        f"**Date:** {datetime.utcnow().isoformat()}  ",
        f"**Samples:** {len(valid)} / {len(rows)}  ",
        f"**VLM model:** `{cfg.get('vlm_model')}`  ",
        f"**Corrector:** `{cfg.get('corrector_model')}`  ",
        f"**OCR backend:** `{cfg.get('ocr_backend')}`  \n",
        "## Aggregate Metrics\n",
        "| Metric | Pipeline | VLM |",
        "|--------|----------|-----|",
        f"| CER ↓  | {mean('pipeline_cer'):.3f} | {mean('vlm_cer'):.3f} |",
        f"| WER ↓  | {mean('pipeline_wer'):.3f} | {mean('vlm_wer'):.3f} |",
        f"| BLEU ↑ | {mean('pipeline_bleu'):.2f} | {mean('vlm_bleu'):.2f} |",
        f"| Time s | {mean('pipeline_time_s'):.1f} | {mean('vlm_time_s'):.1f} |\n",
        "## Per-sample Results\n",
        "| Image | Pipe CER | VLM CER | Pipe WER | VLM WER | Pipe BLEU | VLM BLEU |",
        "|-------|----------|---------|----------|---------|-----------|----------|",
    ]
    for r in rows:
        if "error" in r:
            lines.append(f"| {r['image']} | ERROR | — | — | — | — | — |")
        else:
            lines.append(
                f"| {r['image']} "
                f"| {r.get('pipeline_cer', '?'):.3f} "
                f"| {r.get('vlm_cer', '?'):.3f} "
                f"| {r.get('pipeline_wer', '?'):.3f} "
                f"| {r.get('vlm_wer', '?'):.3f} "
                f"| {r.get('pipeline_bleu', '?'):.2f} "
                f"| {r.get('vlm_bleu', '?'):.2f} |"
            )

    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    app()
