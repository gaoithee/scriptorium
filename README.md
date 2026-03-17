# handwriting-ocr-benchmark

A benchmark pipeline that compares **two approaches** to transcribing handwritten text from images:

| Approach | Description |
|---|---|
| **Pipeline** | Layout detection (DocTR / YOLO) → OCR (Tesseract / EasyOCR) → small LM post-correction |
| **VLM end-to-end** | Qwen2.5-VL (7B/72B) reads the image directly and returns the transcription |

Results are compared against a user-supplied **gold string** using CER, WER, and BLEU.

---

## Repo structure

```
handwriting-ocr-benchmark/
├── src/
│   ├── pipeline/
│   │   ├── layout.py        # layout detection (DocTR bounding boxes)
│   │   ├── ocr.py           # OCR backends (Tesseract, EasyOCR)
│   │   └── correction.py    # small-LM post-correction via Ollama
│   ├── models/
│   │   └── vlm.py           # Qwen2.5-VL inference (local via Ollama or HF)
│   └── eval/
│       └── metrics.py       # CER, WER, BLEU, pretty diff
├── data/
│   ├── samples/             # input images (put your .jpg/.png here)
│   └── gold/                # gold .txt files (same stem as the image)
├── results/                 # JSON + Markdown reports (auto-generated)
├── scripts/
│   └── run_benchmark.py     # main CLI entry point
├── tests/
│   └── test_metrics.py
├── docs/
│   └── architecture.md
├── requirements.txt
├── pyproject.toml
└── .github/
    └── workflows/
        └── ci.yml
```

---

## Quick start

```bash
# 1. clone
git clone https://github.com/your-org/handwriting-ocr-benchmark
cd handwriting-ocr-benchmark

# 2. install (Python ≥ 3.10)
pip install -e ".[dev]"

# 3. install Ollama + models (for local inference)
ollama pull qwen2.5vl:7b       # VLM end-to-end
ollama pull qwen2.5:3b         # small-LM post-corrector

# 4. place your image and gold string
cp my_scan.jpg data/samples/
echo "my gold text here" > data/gold/my_scan.txt

# 5. run
python scripts/run_benchmark.py \
    --image data/samples/my_scan.jpg \
    --gold  data/gold/my_scan.txt \
    --output results/
```

Or pass gold inline:

```bash
python scripts/run_benchmark.py \
    --image  data/samples/my_scan.jpg \
    --gold-string "Il cielo è azzurro" \
    --output results/
```

---

## Approaches in detail

### 1. Classic pipeline

```
Image → DocTR layout (line bounding boxes)
      → EasyOCR (per line)
      → concatenate raw OCR string
      → Qwen2.5:3b (post-correction prompt)
      → final string
```

### 2. VLM end-to-end (Qwen2.5-VL)

The image is sent directly to `qwen2.5vl:7b` (or `:72b`) with a structured prompt asking for a verbatim transcription. No preprocessing required.

---

## Metrics

| Metric | Library | Notes |
|---|---|---|
| **CER** | `jiwer` | Character Error Rate |
| **WER** | `jiwer` | Word Error Rate |
| **BLEU** | `sacrebleu` | unigram-4gram |
| **Diff** | `difflib` | coloured character diff |

---

## Configuration

Copy and edit `config.example.yaml`:

```yaml
ollama_base_url: "http://localhost:11434"
vlm_model: "qwen2.5vl:7b"          # or qwen2.5vl:72b
corrector_model: "qwen2.5:3b"       # post-correction LM
ocr_backend: "easyocr"              # easyocr | tesseract
layout_backend: "doctr"             # doctr | none
language: "it"                      # passed to OCR
```

---

## Adding new samples

1. Drop the image in `data/samples/`
2. Create a matching `data/gold/<stem>.txt` with the gold transcription
3. Run `python scripts/run_benchmark.py --all`

---

## License

MIT
