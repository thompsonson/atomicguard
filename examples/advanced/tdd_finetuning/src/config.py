"""Configuration for TDD Fine-tuning Experiment."""

from pathlib import Path
from typing import TypedDict


class ModelConfig(TypedDict):
    """Model configuration."""
    name: str
    base_url: str
    api_key: str
    temperature: float
    max_tokens: int


class ExperimentConfig(TypedDict):
    """Experiment configuration."""
    num_trials: int
    retry_budget: int
    random_seed: int
    output_dir: Path


# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
TASKS_DIR = PROJECT_ROOT / "tasks"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

# Default model configurations
BASELINE_MODEL: ModelConfig = {
    "name": "qwen2.5-coder:7b",
    "base_url": "http://localhost:11434/v1",  # Ollama endpoint
    "api_key": "ollama",  # Ollama doesn't require real API key
    "temperature": 0.2,
    "max_tokens": 2048,
}

FINETUNED_MODEL: ModelConfig = {
    "name": "qwen2.5-coder:7b-tdd-finetuned",
    "base_url": "http://localhost:11434/v1",
    "api_key": "ollama",
    "temperature": 0.2,
    "max_tokens": 2048,
}

# Experiment settings
EXPERIMENT: ExperimentConfig = {
    "num_trials": 50,
    "retry_budget": 3,
    "random_seed": 42,
    "output_dir": RESULTS_DIR,
}

# Training configuration
TRAINING_CONFIG = {
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "learning_rate": 2e-4,
    "num_epochs": 3,
    "batch_size": 4,
    "warmup_steps": 100,
}

# Quality standards for training data extraction
QUALITY_STANDARDS = {
    "first_attempt_success": True,  # Must pass on first attempt
    "validated_success": True,       # Subsequent g_impl must succeed
}
