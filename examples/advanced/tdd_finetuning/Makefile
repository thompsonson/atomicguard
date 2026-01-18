.PHONY: help install baseline extract finetune evaluate clean

help:
	@echo "TDD Fine-Tuning Experiment - Makefile"
	@echo ""
	@echo "Usage:"
	@echo "  make install          Install dependencies"
	@echo "  make baseline         Run baseline benchmark (Phase 1)"
	@echo "  make extract          Extract training data (Phase 2, step 1)"
	@echo "  make finetune         Fine-tune model (Phase 2, step 2)"
	@echo "  make finetuned        Run fine-tuned benchmark (Phase 3)"
	@echo "  make evaluate         Compare results (Phase 3)"
	@echo "  make all              Run complete pipeline"
	@echo "  make clean            Remove results and data"
	@echo ""

install:
	pip install -r requirements.txt

baseline:
	python run_benchmark.py --model baseline --trials 50

extract:
	python extract_training_data.py --prepare-dataset

finetune:
	@echo "Fine-tuning requires manual conversion to Ollama format."
	@echo "See README.md for detailed instructions."
	python -m src.training.finetune \
		--train-file data/prepared/train.jsonl \
		--val-file data/prepared/val.jsonl \
		--output-dir models/finetuned

finetuned:
	python run_benchmark.py --model finetuned --trials 50

evaluate:
	python compare_results.py --output results/analysis/comparison_report.json

all: baseline extract finetuned evaluate

clean:
	rm -rf results/ data/ models/finetuned/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
