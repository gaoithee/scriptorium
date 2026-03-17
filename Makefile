.PHONY: install dev test lint format serve-vlm serve-corrector run app eval-all

## Install package in editable mode
install:
	pip install -e .

## Install with dev + runtime extras
dev:
	pip install -e ".[dev]" jiwer sacrebleu gradio openai

## Run unit tests
test:
	pytest tests/ -v --tb=short

## Lint
lint:
	ruff check src/ scripts/ tests/

## Format
format:
	black src/ scripts/ tests/

## Serve Qwen3.5-9B with vLLM (requires a GPU)
serve-vlm:
	vllm serve Qwen/Qwen3.5-9B \
		--reasoning-parser qwen3 \
		--port 8000 \
		--max-model-len 32768

## Serve corrector via Ollama
serve-corrector:
	ollama pull qwen2.5:3b

## Run benchmark on a single image (set IMAGE and GOLD env vars)
run:
	python scripts/run_benchmark.py \
		--image  "$(IMAGE)" \
		--gold   "$(GOLD)" \
		--output results/

## Launch Gradio web UI
app:
	python scripts/app.py

## Run full dataset evaluation
eval-all:
	python scripts/eval_dataset.py \
		--samples data/samples \
		--gold    data/gold \
		--output  results/
