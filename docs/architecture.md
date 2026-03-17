# Architecture

## Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    run_benchmark.py (CLI)                        │
│                                                                  │
│   --image  path/to/scan.jpg                                      │
│   --gold   path/to/gold.txt  (or --gold-string "…")             │
└────────────────────┬────────────────────────────────────────────┘
                     │
          ┌──────────┴──────────┐
          │                     │
          ▼                     ▼
 ┌─────────────────┐   ┌──────────────────────┐
 │ PIPELINE        │   │ VLM END-TO-END        │
 │                 │   │                       │
 │ layout.py       │   │ vlm.py                │
 │  └─ DocTR       │   │  └─ Qwen2.5-VL (7B)  │
 │     bboxes      │   │     via Ollama        │
 │                 │   │                       │
 │ ocr.py          │   │  image → base64       │
 │  └─ EasyOCR /   │   │  → Ollama chat API    │
 │     Tesseract   │   │  → plain text         │
 │     per region  │   └──────────┬────────────┘
 │                 │              │
 │ correction.py   │              │
 │  └─ Qwen2.5:3b  │              │
 │     (Ollama)    │              │
 └────────┬────────┘              │
          │                       │
          └──────────┬────────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │   eval/metrics.py   │
          │                     │
          │  CER  (jiwer)       │
          │  WER  (jiwer)       │
          │  BLEU (sacrebleu)   │
          │  char diff          │
          └─────────────────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │  results/           │
          │   *.json            │
          │   *.md              │
          └─────────────────────┘
```

## Component responsibilities

### `src/pipeline/layout.py`
- Wraps python-doctr's detection predictor
- Returns `TextRegion` objects (pixel bounding boxes, sorted top→bottom)
- Optionally saves a debug image with drawn boxes

### `src/pipeline/ocr.py`
- Crops each `TextRegion` and runs OCR
- Backends: EasyOCR (default, GPU-friendly) or pytesseract
- Falls back to full-image OCR if no regions are found

### `src/pipeline/correction.py`
- Sends the raw OCR string to `qwen2.5:3b` via the Ollama Python client
- System prompt instructs the model to fix OCR errors without paraphrasing
- Temperature 0 for determinism

### `src/models/vlm.py`
- Encodes the image as base64 and sends it to `qwen2.5vl:7b` (or `:72b`)
- Uses the Ollama multimodal chat API
- Returns a clean plain-text transcription

### `src/eval/metrics.py`
- `evaluate(hypothesis, reference)` → `EvalResult`
- Pure Python fallbacks (Levenshtein) if jiwer/sacrebleu are unavailable

## Data flow example

```
scan.jpg  ──▶  DocTR  ──▶  5 line regions
                              │
                              ▼
                        EasyOCR (per region)
                              │
                          raw OCR string
                              │
                        Qwen2.5:3b (correction)
                              │
                         pipeline_text ──┐
                                         ├──▶  evaluate(hyp, gold)
scan.jpg  ──▶  Qwen2.5-VL:7b            │         │
                    │                   │       CER / WER / BLEU
               vlm_text ────────────────┘
```

## Adding a new OCR backend

1. Implement a function `_ocr_mybackend(pil_img, language) -> str` in `src/pipeline/ocr.py`
2. Add it to the `if/elif` chain in `ocr_image()`
3. Update the `ocr_backend` option in `config.example.yaml`

## Adding a new VLM

1. Add a new file `src/models/my_vlm.py` mirroring `vlm.py`
2. In `run_benchmark.py`, import and call it alongside the existing VLM approach
3. Add its results to the metrics table
