"""DiffReviewGuard: Validates diff review output schema.

G_val guard for ap_diff_review. Checks that the review output is valid JSON
matching the DiffReview schema with a valid verdict and, when the verdict
is 'backtrack', a valid backtrack_target.

Used by: Arms 17, 18
"""

import json
import logging
from typing import Any

from examples.swe_bench_common.models import DiffReview, DiffReviewVerdict

from atomicguard.domain.interfaces import GuardInterface
from atomicguard.domain.models import Artifact, GuardResult

logger = logging.getLogger("swe_bench_ablation.guards")

VALID_BACKTRACK_TARGETS = {"ap_gen_patch", "ap_gen_test", "ap_analysis"}


class DiffReviewGuard(GuardInterface):
    """Validates diff review output.

    Checks:
    - Valid JSON matching DiffReview schema
    - Non-empty reasoning
    - If verdict is 'backtrack', backtrack_target must be a valid action pair ID
    - If verdict is 'approve', no critical issues should be present
    """

    def __init__(self, **kwargs: Any):  # noqa: ARG002
        pass

    def validate(
        self,
        artifact: Artifact,
        **deps: Artifact,  # noqa: ARG002
    ) -> GuardResult:
        """Validate the diff review artifact."""
        logger.info(
            "[DiffReviewGuard] Validating artifact %s...", artifact.artifact_id[:8]
        )

        try:
            data = json.loads(artifact.content)
        except json.JSONDecodeError as e:
            return GuardResult(
                passed=False,
                feedback=f"Invalid JSON: {e}",
                guard_name="DiffReviewGuard",
            )

        try:
            review = DiffReview.model_validate(data)
        except Exception as e:
            return GuardResult(
                passed=False,
                feedback=f"Schema validation failed: {e}",
                guard_name="DiffReviewGuard",
            )

        errors: list[str] = []

        if not review.reasoning.strip():
            errors.append("reasoning is empty")

        if review.verdict == DiffReviewVerdict.BACKTRACK:
            if not review.backtrack_target:
                errors.append(
                    "verdict is 'backtrack' but backtrack_target is null. "
                    "Must specify one of: " + ", ".join(sorted(VALID_BACKTRACK_TARGETS))
                )
            elif review.backtrack_target not in VALID_BACKTRACK_TARGETS:
                errors.append(
                    f"backtrack_target '{review.backtrack_target}' is not valid. "
                    "Must be one of: " + ", ".join(sorted(VALID_BACKTRACK_TARGETS))
                )

        if review.verdict == DiffReviewVerdict.APPROVE:
            critical_issues = [i for i in review.issues if i.severity == "critical"]
            if critical_issues:
                errors.append(
                    f"verdict is 'approve' but {len(critical_issues)} critical "
                    "issues found — use 'revise' or 'backtrack' instead"
                )

        if errors:
            feedback = "Review validation failed:\n- " + "\n- ".join(errors)
            logger.info("[DiffReviewGuard] REJECTED: %s", feedback)
            return GuardResult(
                passed=False,
                feedback=feedback,
                guard_name="DiffReviewGuard",
            )

        feedback = (
            f"Review valid: verdict={review.verdict.value}, {len(review.issues)} issues"
        )
        if review.backtrack_target:
            feedback += f", backtrack_target={review.backtrack_target}"
        logger.info("[DiffReviewGuard] PASSED: %s", feedback)

        return GuardResult(
            passed=True,
            feedback=feedback,
            guard_name="DiffReviewGuard",
        )
