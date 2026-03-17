<p align="center">
  <img src="assets/mascot.png" width="600" alt="Scriptorium Mascot">
</p>

# scriptorium

> **Note on the name:** In medieval times, the *scriptorium* was the dedicated room in a monastery where monks (amanuenses) painstakingly transcribed and preserved manuscripts. This benchmark honors that tradition by testing how modern "digital scribes" handle the complexities of human handwriting.

**scriptorium** is a benchmark pipeline for handwritten text transcription. It compares a classic modular OCR pipeline against a **VLM end-to-end** approach using **Qwen3.5-9B**, evaluated against a gold standard string.

| Approach | Description |
|---|---|
| **Classic pipeline** | Layout detection (DocTR) → OCR (EasyOCR / Tesseract) → LM post-correction |
| **VLM end-to-end** | Qwen3.5-9B reads the image directly via OpenAI-compatible API |

**Metrics:** CER, WER, and BLEU calculated against your gold transcription.

---

## Structure

```

scriptorium/
├── assets/          \# Project images and mascots
├── src/
│   ├── pipeline/
│   │   ├── preprocess.py    \# deskew, denoise, CLAHE, upscale
│   │   ├── layout.py        \# DocTR bounding-box detection
│   │   ├── ocr.py           \# EasyOCR / Tesseract, per-region loop
│   │   └── correction.py    \# LM post-correction (OpenAI-compat API)
│   ├── models/
│   │   ├── vlm.py           \# Qwen3.5-9B via OpenAI-compat API (vLLM / SGLang)
│   │   └── vlm\_hf.py        \# Qwen3.5-9B via HuggingFace Transformers (direct load)
│   └── eval/
│       └── metrics.py       \# CER, WER, BLEU, char diff
├── scripts/
│   ├── run\_benchmark.py     \# single-image CLI
│   ├── eval\_dataset.py      \# batch evaluation → CSV + Markdown
│   └── app.py               \# Gradio web UI
├── data/
│   ├── samples/             \# input images (.jpg / .png)
│   └── gold/                \# matching \<stem\>.txt ground-truth files
├── results/                 \# auto-generated reports (git-ignored)
├── tests/
├── config.example.yaml
└── pyproject.toml

````

---

## Quickstart

### 1. Install

```bash
git clone [https://github.com/gaoithee/scriptorium](https://github.com/gaoithee/scriptorium)
cd scriptorium
pip install -e ".[dev]"
````

### 2\. Serve Qwen3.5-9B

```bash
# vLLM (recommended)
pip install vllm --torch-backend=auto --extra-index-url [https://wheels.vllm.ai/nightly](https://wheels.vllm.ai/nightly)
vllm serve Qwen/Qwen3.5-9B --reasoning-parser qwen3 --port 8000

# SGLang
pip install 'git+[https://github.com/sgl-project/sglang.git#subdirectory=python&egg=sglang](https://github.com/sgl-project/sglang.git#subdirectory=python&egg=sglang)[all]'
python -m sglang.launch_server --model-path Qwen/Qwen3.5-9B --reasoning-parser qwen3 --port 8000
```

Both expose an OpenAI-compatible endpoint at `http://localhost:8000/v1`.

For the **classic pipeline corrector**, any small model via Ollama works:

```bash
ollama pull qwen2.5:3b   # serves at http://localhost:11434/v1
```

### 3\. Configure

```bash
cp config.example.yaml config.yaml
# edit if your ports / models differ
```

### 4\. Run

```bash
# single image
python scripts/run_benchmark.py \
  --image data/samples/my_scan.jpg \
  --gold  data/gold/my_scan.txt

# or inline gold
python scripts/run_benchmark.py \
  --image data/samples/my_scan.jpg \
  --gold-string "testo originale"

# full dataset
python scripts/eval_dataset.py

# web UI
python scripts/app.py   →  http://localhost:7860
```

-----

## Pipelines

**Classic**

```
image → preprocess (deskew/denoise/CLAHE)
      → DocTR (line bboxes)
      → EasyOCR per region
      → LM post-correction (any OpenAI-compat model)
      → string
```

**VLM**

```
image (base64) → Qwen3.5-9B (thinking disabled) → string
```

Qwen3.5-9B is a **native multimodal model** (early fusion vision encoder), not a separate vision adapter. Thinking mode is disabled by default for OCR — faster and cleaner output.

-----

## Metrics

| Metric | Library |
|---|---|
| CER | `jiwer` |
| WER | `jiwer` |
| BLEU | `sacrebleu` |
| char diff | `difflib` |

-----

## Configuration

```yaml
# config.example.yaml
vlm_api_base:  "http://localhost:8000/v1"
vlm_api_key:   "EMPTY"
vlm_model:     "Qwen/Qwen3.5-9B"
vlm_thinking:  false               # keep false for OCR

corrector_api_base: "http://localhost:11434/v1"
corrector_api_key:  "EMPTY"
corrector_model:    "qwen2.5:3b"

ocr_backend:    "easyocr"          # easyocr | tesseract
layout_backend: "doctr"            # doctr | none
language:       "it"
preprocess:     true
```

-----

## Adding samples

```bash
cp my_scan.jpg data/samples/
echo "testo gold" > data/gold/my_scan.txt
python scripts/eval_dataset.py
```

-----

## License

MIT
