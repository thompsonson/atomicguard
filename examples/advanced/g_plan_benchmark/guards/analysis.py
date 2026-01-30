"""
AnalysisGuard: Validates problem classification output from g_analysis.

Predicates validated:
    G_analysis(a) = parseable_json(a) ^ valid_problem_type(a)
                     ^ language_present(a) ^ key_signals_nonempty(a)
                     ^ severity_valid(a) ^ affected_area_present(a)
                     ^ rationale_present(a)

Complexity: O(1) — fixed number of field checks.
"""

from __future__ import annotations

import json
from typing import Any

from atomicguard.domain.interfaces import GuardInterface
from atomicguard.domain.models import Artifact, GuardResult

VALID_PROBLEM_TYPES = {"bug_fix", "feature", "refactoring", "performance"}
VALID_SEVERITIES = {"low", "medium", "high"}
VALID_LANGUAGES = {"python", "java", "javascript", "typescript", "unknown"}


class AnalysisGuard(GuardInterface):
    """
    Validates problem analysis JSON produced by LLMJsonGenerator.

    Expected schema:
    {
        "problem_type": "bug_fix" | "feature" | "refactoring" | "performance",
        "language": "python" | "java" | "javascript" | "typescript" | "unknown",
        "severity": "low" | "medium" | "high",
        "key_signals": ["list of signals"],
        "affected_area": "description",
        "rationale": "one sentence"
    }

    All checks are O(1) — lightweight structural validation.
    """

    def __init__(self, **_kwargs: Any):
        pass

    def validate(self, artifact: Artifact, **_deps: Artifact) -> GuardResult:
        """Validate analysis artifact."""
        errors: list[str] = []

        # 1. Parseable JSON
        try:
            data = json.loads(artifact.content)
        except (json.JSONDecodeError, TypeError) as e:
            return GuardResult(
                passed=False,
                feedback=f"Analysis not parseable as JSON: {e}",
                guard_name="AnalysisGuard",
            )

        if not isinstance(data, dict):
            return GuardResult(
                passed=False,
                feedback=f"Analysis must be a JSON object, got {type(data).__name__}",
                guard_name="AnalysisGuard",
            )

        # 2. problem_type is valid enum
        problem_type = data.get("problem_type")
        if problem_type is None:
            errors.append("Missing required field: problem_type")
        elif problem_type not in VALID_PROBLEM_TYPES:
            errors.append(
                f"Invalid problem_type '{problem_type}'. "
                f"Must be one of: {', '.join(sorted(VALID_PROBLEM_TYPES))}"
            )

        # 3. language is present
        language = data.get("language")
        if language is None:
            errors.append("Missing required field: language")
        elif not isinstance(language, str) or not language.strip():
            errors.append("language must be a non-empty string")
        elif language not in VALID_LANGUAGES:
            errors.append(
                f"Invalid language '{language}'. "
                f"Must be one of: {', '.join(sorted(VALID_LANGUAGES))}"
            )

        # 4. key_signals is a non-empty list
        key_signals = data.get("key_signals")
        if key_signals is None:
            errors.append("Missing required field: key_signals")
        elif not isinstance(key_signals, list):
            errors.append(
                f"key_signals must be a list, got {type(key_signals).__name__}"
            )
        elif len(key_signals) == 0:
            errors.append("key_signals must be non-empty")

        # 5. severity is valid enum
        severity = data.get("severity")
        if severity is None:
            errors.append("Missing required field: severity")
        elif severity not in VALID_SEVERITIES:
            errors.append(
                f"Invalid severity '{severity}'. "
                f"Must be one of: {', '.join(sorted(VALID_SEVERITIES))}"
            )

        # 6. affected_area is present and non-empty
        affected_area = data.get("affected_area")
        if affected_area is None:
            errors.append("Missing required field: affected_area")
        elif not isinstance(affected_area, str) or not affected_area.strip():
            errors.append("affected_area must be a non-empty string")

        # 7. rationale is present and non-empty
        rationale = data.get("rationale")
        if rationale is None:
            errors.append("Missing required field: rationale")
        elif not isinstance(rationale, str) or not rationale.strip():
            errors.append("rationale must be a non-empty string")

        if errors:
            return GuardResult(
                passed=False,
                feedback="Analysis validation failed:\n- " + "\n- ".join(errors),
                guard_name="AnalysisGuard",
            )

        return GuardResult(
            passed=True,
            feedback=(
                f"Analysis valid: {problem_type}/{language}, "
                f"severity={severity}, {len(key_signals)} signals"
            ),
            guard_name="AnalysisGuard",
        )
