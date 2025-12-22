"""AtomicGuard JSON Schema definitions and validation utilities.

This module provides JSON Schema definitions for AtomicGuard configuration files,
aligned with the formal framework defined in the paper.

Schemas:
    - workflow.schema.json: Workflow definition (action pairs, guards, preconditions)
    - prompts.schema.json: Prompt templates for generators
    - artifact.schema.json: Artifact storage format in DAG

Usage:
    from atomicguard.schemas import validate_workflow, validate_prompts

    # Validate a workflow configuration
    with open("workflow.json") as f:
        data = json.load(f)
    validate_workflow(data)  # Raises jsonschema.ValidationError if invalid
"""

from __future__ import annotations

import json
from importlib.resources import files
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import types

try:
    import jsonschema as _jsonschema

    jsonschema: types.ModuleType | None = _jsonschema
except ImportError:
    jsonschema = None


def _load_schema(name: str) -> dict[str, Any]:
    """Load a JSON schema from the schemas package.

    Args:
        name: Schema filename (e.g., 'workflow.schema.json')

    Returns:
        Parsed JSON schema as a dictionary
    """
    schema_text = files("atomicguard.schemas").joinpath(name).read_text()
    result: dict[str, Any] = json.loads(schema_text)
    return result


def get_workflow_schema() -> dict[str, Any]:
    """Get the workflow.json schema.

    Returns:
        JSON Schema for workflow configuration
    """
    return _load_schema("workflow.schema.json")


def get_prompts_schema() -> dict[str, Any]:
    """Get the prompts.json schema.

    Returns:
        JSON Schema for prompt templates
    """
    return _load_schema("prompts.schema.json")


def get_artifact_schema() -> dict[str, Any]:
    """Get the artifact.json schema.

    Returns:
        JSON Schema for artifact storage
    """
    return _load_schema("artifact.schema.json")


def validate_workflow(data: dict[str, Any]) -> None:
    """Validate a workflow configuration against the schema.

    Args:
        data: Workflow configuration dictionary

    Raises:
        jsonschema.ValidationError: If validation fails
        ImportError: If jsonschema is not installed
    """
    if jsonschema is None:
        raise ImportError(
            "jsonschema is required for validation. "
            "Install it with: pip install jsonschema"
        )
    schema = get_workflow_schema()
    jsonschema.validate(data, schema)


def validate_prompts(data: dict[str, Any]) -> None:
    """Validate prompt templates against the schema.

    Args:
        data: Prompts configuration dictionary

    Raises:
        jsonschema.ValidationError: If validation fails
        ImportError: If jsonschema is not installed
    """
    if jsonschema is None:
        raise ImportError(
            "jsonschema is required for validation. "
            "Install it with: pip install jsonschema"
        )
    schema = get_prompts_schema()
    jsonschema.validate(data, schema)


def validate_artifact(data: dict[str, Any]) -> None:
    """Validate an artifact against the schema.

    Args:
        data: Artifact dictionary

    Raises:
        jsonschema.ValidationError: If validation fails
        ImportError: If jsonschema is not installed
    """
    if jsonschema is None:
        raise ImportError(
            "jsonschema is required for validation. "
            "Install it with: pip install jsonschema"
        )
    schema = get_artifact_schema()
    jsonschema.validate(data, schema)


__all__ = [
    "get_workflow_schema",
    "get_prompts_schema",
    "get_artifact_schema",
    "validate_workflow",
    "validate_prompts",
    "validate_artifact",
]
