"""
ReconGuard: Validates codebase reconnaissance output from g_recon.

Predicates validated:
    G_recon(r) = parseable_json(r) ^ all_fields_are_lists(r)
                  ^ at_least_one_nonempty(r)

Complexity: O(1) — fixed number of field checks.
"""

from __future__ import annotations

import json
from typing import Any

from atomicguard.domain.interfaces import GuardInterface
from atomicguard.domain.models import Artifact, GuardResult

REQUIRED_LIST_FIELDS = [
    "mentioned_files",
    "stack_traces",
    "apis_involved",
    "test_references",
    "reproduction_steps",
    "constraints_mentioned",
]


class ReconGuard(GuardInterface):
    """
    Validates codebase reconnaissance JSON produced by LLMJsonGenerator.

    Expected schema:
    {
        "mentioned_files": ["file paths or module names"],
        "stack_traces": ["summarized stack traces"],
        "apis_involved": ["function/class/API names"],
        "test_references": ["test files or patterns"],
        "reproduction_steps": ["steps to reproduce"],
        "constraints_mentioned": ["performance, compatibility constraints"]
    }

    All fields must be arrays. At least one must be non-empty (the recon
    extracted something). O(1) checks.
    """

    def __init__(self, **_kwargs: Any):
        pass

    def validate(self, artifact: Artifact, **_deps: Artifact) -> GuardResult:
        """Validate reconnaissance artifact."""
        errors: list[str] = []

        # 1. Parseable JSON
        try:
            data = json.loads(artifact.content)
        except (json.JSONDecodeError, TypeError) as e:
            return GuardResult(
                passed=False,
                feedback=f"Recon not parseable as JSON: {e}",
                guard_name="ReconGuard",
            )

        if not isinstance(data, dict):
            return GuardResult(
                passed=False,
                feedback=f"Recon must be a JSON object, got {type(data).__name__}",
                guard_name="ReconGuard",
            )

        # 2. All required fields present and are lists
        for field in REQUIRED_LIST_FIELDS:
            value = data.get(field)
            if value is None:
                errors.append(f"Missing required field: {field}")
            elif not isinstance(value, list):
                errors.append(
                    f"{field} must be a list, got {type(value).__name__}"
                )

        if errors:
            return GuardResult(
                passed=False,
                feedback="Recon validation failed:\n- " + "\n- ".join(errors),
                guard_name="ReconGuard",
            )

        # 3. At least one field is non-empty (recon found something)
        total_items = sum(len(data[f]) for f in REQUIRED_LIST_FIELDS)
        if total_items == 0:
            return GuardResult(
                passed=False,
                feedback=(
                    "Recon validation failed:\n- All fields are empty. "
                    "The reconnaissance must extract at least one signal."
                ),
                guard_name="ReconGuard",
            )

        # Build summary
        nonempty = [f for f in REQUIRED_LIST_FIELDS if len(data[f]) > 0]
        return GuardResult(
            passed=True,
            feedback=(
                f"Recon valid: {total_items} items across "
                f"{len(nonempty)} fields ({', '.join(nonempty)})"
            ),
            guard_name="ReconGuard",
        )
