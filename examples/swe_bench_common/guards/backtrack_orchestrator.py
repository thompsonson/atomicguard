"""BacktrackOrchestrator: Cross-pair feedback loop with selective backtracking.

Implements the search strategy from plan_search_feedback_loop.md. Wraps
a standard Workflow execution with backtracking logic that can re-execute
earlier steps when guard feedback indicates the root cause lies upstream.

Two heuristic modes:
- LLM-based (Arm 18): Uses ap_diff_review's backtrack_target field
- Rule-based (Arm 19): Pattern-matches guard feedback strings

Used by: Arms 18, 19, 21
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from atomicguard.domain.models import GuardResult

logger = logging.getLogger("swe_bench_ablation.backtrack")


@dataclass
class BacktrackState:
    """Tracks backtracking state for a single workflow execution."""

    backtrack_counts: dict[str, int] = field(default_factory=dict)
    backtrack_budget: dict[str, int] = field(default_factory=dict)
    failure_summaries: list[dict[str, Any]] = field(default_factory=list)

    def can_backtrack(self, target: str) -> bool:
        """Check if backtracking to target is within budget."""
        budget = self.backtrack_budget.get(target, 0)
        used = self.backtrack_counts.get(target, 0)
        return used < budget

    def record_backtrack(self, target: str, reason: str) -> None:
        """Record a backtrack event."""
        self.backtrack_counts[target] = self.backtrack_counts.get(target, 0) + 1
        self.failure_summaries.append({"target": target, "reason": reason})

    def get_amended_context(self, target: str) -> str:
        """Build amended context for a backtracked step."""
        relevant = [s for s in self.failure_summaries if s["target"] == target]
        if not relevant:
            return ""
        parts = ["Previous attempts at this step failed downstream:"]
        for i, summary in enumerate(relevant, 1):
            parts.append(f"  Attempt {i}: {summary['reason']}")
        return "\n".join(parts)


def rule_based_heuristic(
    step_id: str,  # noqa: ARG001
    guard_result: GuardResult,
    retry_count: int,
    history: list[GuardResult],
) -> int:
    """Rule-based backtrack heuristic from plan_search_feedback_loop.md Section 4.1.

    Returns the backtrack depth:
    - 0: retry same step
    - 1: backtrack one step (e.g., from patch to test)
    - 2: backtrack two steps (e.g., from patch to analysis)

    Args:
        step_id: Current step identifier
        guard_result: The guard result that triggered backtracking
        retry_count: Number of retries already attempted at this step
        history: Previous guard results for this step
    """
    feedback = guard_result.feedback.lower()

    # Structural errors -> retry same step
    if "syntax error" in feedback or "not parseable" in feedback:
        return 0

    # Repeated identical failures -> escalate
    if _same_feedback_repeated(history, guard_result, threshold=2):
        return min(retry_count, 2)

    # Import/module errors after multiple retries -> analysis wrong
    if retry_count >= 2 and (
        "importerror" in feedback or "modulenotfounderror" in feedback
    ):
        return 2  # backtrack to analysis

    # Test discrimination failure -> retry test (already at test step)
    if "passed on buggy code" in feedback:
        return 0

    # Patch doesn't fix the bug -> maybe test is wrong
    if "still fails after patch" in feedback and retry_count >= 3:
        return 1  # backtrack from patch to test

    # Default: retry at same level
    return 0


def llm_review_heuristic(
    review_artifact_content: str,
    step_order: list[str],
    current_step: str,
) -> tuple[str | None, str]:
    """LLM-based backtrack heuristic using ap_diff_review output.

    Parses the review artifact's backtrack_target field to determine
    where to backtrack. Returns the target step ID and the reason.

    Args:
        review_artifact_content: JSON content of the diff review artifact
        step_order: Ordered list of step IDs in the workflow
        current_step: The current step that produced the review

    Returns:
        Tuple of (backtrack_target or None, reason string)
    """
    try:
        data = json.loads(review_artifact_content)
    except (json.JSONDecodeError, TypeError):
        return None, "Could not parse review output"

    verdict = data.get("verdict", "")
    backtrack_target = data.get("backtrack_target")
    reasoning = data.get("reasoning", "No reasoning provided")

    if verdict == "approve":
        return None, "Review approved the patch"

    if verdict == "revise":
        # Revise means retry the current patch generation
        return current_step, f"Review requested revision: {reasoning}"

    if verdict == "backtrack" and backtrack_target:
        if backtrack_target in step_order:
            return backtrack_target, f"Review recommended backtracking: {reasoning}"
        # Target not in workflow — fall back to retry
        return current_step, (
            f"Review suggested backtracking to '{backtrack_target}' "
            f"but it's not in the workflow. Retrying current step. "
            f"Reason: {reasoning}"
        )

    return None, f"Unrecognised review verdict: {verdict}"


def resolve_backtrack_target(
    depth: int,
    step_order: list[str],
    current_step_index: int,
) -> str:
    """Convert a backtrack depth (0, 1, 2) into a step ID.

    Args:
        depth: How many steps to backtrack (0 = current, 1 = previous, etc.)
        step_order: Ordered list of step IDs
        current_step_index: Index of the current step in step_order

    Returns:
        The step ID to backtrack to
    """
    target_index = max(0, current_step_index - depth)
    return step_order[target_index]


def _same_feedback_repeated(
    history: list[GuardResult],
    current: GuardResult,
    threshold: int = 2,
) -> bool:
    """Check if the same feedback has been repeated threshold times."""
    if not history:
        return False
    current_feedback = current.feedback.strip().lower()
    count = sum(1 for h in history if h.feedback.strip().lower() == current_feedback)
    return count >= threshold
