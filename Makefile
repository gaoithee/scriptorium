.PHONY: install dev test lint format run app pull-models

## Install package in editable mode
install:
	pip install -e .

## Install with dev extras
dev:
	pip install -e ".[dev]" jiwer sacrebleu gradio

## Run unit tests
test:
	pytest tests/ -v --tb=short

## Lint with ruff
lint:
	ruff check src/ scripts/ tests/

## Auto-format with black
format:
	black src/ scripts/ tests/

## Pull required Ollama models
pull-models:
	ollama pull qwen2.5vl:7b
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
