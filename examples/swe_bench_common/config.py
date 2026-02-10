"""Shared configuration utilities for SWE-bench experiment runners."""

import json
from pathlib import Path
from typing import Any

from atomicguard.domain.prompts import PromptTemplate


def topological_sort(action_pairs: dict[str, Any]) -> list[str]:
    """Sort action pairs by their ``requires`` dependencies.

    Args:
        action_pairs: Dictionary mapping action pair IDs to their configs.
            Each config may have a ``requires`` list of dependency IDs.

    Returns:
        List of action pair IDs in topological order (dependencies first).
    """
    result: list[str] = []
    visited: set[str] = set()

    def visit(ap_id: str) -> None:
        if ap_id in visited:
            return
        visited.add(ap_id)
        for dep in action_pairs.get(ap_id, {}).get("requires", []):
            visit(dep)
        result.append(ap_id)

    for ap_id in action_pairs:
        visit(ap_id)
    return result


def load_prompts(prompts_file: Path) -> dict[str, PromptTemplate]:
    """Load prompt templates from a prompts.json file.

    Args:
        prompts_file: Path to the prompts.json file.

    Returns:
        Dictionary mapping prompt IDs to PromptTemplate instances.
        Returns empty dict if file doesn't exist.
    """
    if not prompts_file.exists():
        return {}

    data = json.loads(prompts_file.read_text())
    templates = {}
    for key, value in data.items():
        templates[key] = PromptTemplate(
            role=value.get("role", ""),
            constraints=value.get("constraints", ""),
            task=value.get("task", ""),
            feedback_wrapper=value.get("feedback_wrapper", "Feedback: {feedback}"),
        )
    return templates


def load_workflow_config(workflow_dir: Path, variant: str) -> dict[str, Any]:
    """Load workflow configuration from JSON file.

    Args:
        workflow_dir: Directory containing workflow JSON files.
        variant: Name of the workflow variant (without .json extension).

    Returns:
        Parsed workflow configuration dictionary.

    Raises:
        FileNotFoundError: If the workflow file doesn't exist.
    """
    workflow_file = workflow_dir / f"{variant}.json"
    if not workflow_file.exists():
        raise FileNotFoundError(f"Workflow not found: {workflow_file}")
    return json.loads(workflow_file.read_text())
