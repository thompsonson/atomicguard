"""Configuration loading utilities for AtomicGuard examples."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from atomicguard import PromptTemplate

from .exceptions import ConfigurationError


def _require_field(data: dict[str, Any], field: str, step_id: str) -> str:
    """Extract a required field from prompt data.

    Args:
        data: Prompt data dictionary
        field: Field name to extract
        step_id: Step ID for error messages

    Returns:
        The field value

    Raises:
        ConfigurationError: If field is missing or empty
    """
    value = data.get(field)
    if not value:
        raise ConfigurationError(
            f"prompts.json: '{step_id}' missing required field '{field}'"
        )
    return value


def load_prompts(path: Path) -> dict[str, PromptTemplate]:
    """
    Load prompt templates from JSON file.

    Args:
        path: Path to prompts.json

    Returns:
        Dict mapping step ID to PromptTemplate

    Raises:
        ConfigurationError: If file is missing or invalid
    """
    if not path.exists():
        raise ConfigurationError(f"Prompts file not found: {path}")

    try:
        with open(path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ConfigurationError(f"Invalid JSON in {path}: {e}") from e

    if not isinstance(data, dict):
        raise ConfigurationError(f"Expected dict in {path}, got {type(data).__name__}")

    prompts = {}
    for step_id, prompt_data in data.items():
        if not isinstance(prompt_data, dict):
            raise ConfigurationError(
                f"Invalid prompt config for '{step_id}': expected dict"
            )
        prompts[step_id] = PromptTemplate(
            role=_require_field(prompt_data, "role", step_id),
            constraints=prompt_data.get("constraints", ""),  # Optional
            task=_require_field(prompt_data, "task", step_id),
            feedback_wrapper=prompt_data.get("feedback_wrapper", "{feedback}"),
        )
    return prompts


def load_workflow_config(
    path: Path,
    required_fields: tuple[str, ...] = ("name",),
) -> dict[str, Any]:
    """
    Load workflow configuration from JSON file.

    Args:
        path: Path to workflow.json
        required_fields: Fields that must be present (varies by workflow type)

    Returns:
        Workflow configuration dict

    Raises:
        ConfigurationError: If file is missing or invalid
    """
    if not path.exists():
        raise ConfigurationError(f"Workflow file not found: {path}")

    try:
        with open(path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ConfigurationError(f"Invalid JSON in {path}: {e}") from e

    # Validate required fields
    missing = [f for f in required_fields if f not in data]
    if missing:
        raise ConfigurationError(
            f"Missing required fields in {path}: {', '.join(missing)}"
        )

    # Validate action_pairs if present
    if "action_pairs" in data:
        if not isinstance(data["action_pairs"], dict):
            raise ConfigurationError("'action_pairs' must be a dict")
        if not data["action_pairs"]:
            raise ConfigurationError("'action_pairs' cannot be empty")

    result: dict[str, Any] = data
    return result


def normalize_base_url(url: str) -> str:
    """
    Normalize base URL for OpenAI-compatible API.

    Ensures /v1 suffix for Ollama/OpenAI compatible endpoints.

    Args:
        url: Raw URL from CLI or config

    Returns:
        URL with /v1 suffix
    """
    base_url = url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    return base_url


def normalize_model_name(model: str) -> str:
    """
    Normalize model string for PydanticAI.

    Ensures proper provider prefix (ollama:, openai:, anthropic:).

    Args:
        model: Raw model name

    Returns:
        Model name with provider prefix
    """
    if not model.startswith(("ollama:", "openai:", "anthropic:")):
        model = f"ollama:{model}"
    return model
