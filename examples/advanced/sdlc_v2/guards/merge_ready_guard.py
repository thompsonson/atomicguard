"""
MergeReadyGuard: Composite check that all prior gates passed.

Final validation step that verifies all gates passed and the
implementation is ready to be merged/deployed.
"""

import json
import logging
from typing import Any

from atomicguard.domain.interfaces import GuardInterface
from atomicguard.domain.models import Artifact, ArtifactStatus, GuardResult

logger = logging.getLogger("sdlc_checkpoint")


class MergeReadyGuard(GuardInterface):
    """
    Composite check verifying all prior gates passed.

    This guard is sensing-only: it only reads artifact status,
    does not perform any file I/O.

    Checks:
    - All required dependency artifacts are ACCEPTED
    - The merge ready report is valid JSON
    """

    def __init__(
        self,
        required_gates: list[str] | None = None,
    ):
        """
        Initialize the merge ready guard.

        Args:
            required_gates: List of gate IDs that must be ACCEPTED.
                           Defaults to standard SDLC gates.
        """
        self._required_gates = required_gates or [
            "g_config",
            "g_add",
            "g_bdd",
            "g_rules",
            "g_coder",
            "g_quality",
            "g_arch_validate",
        ]

    def validate(self, artifact: Artifact, **deps: Any) -> GuardResult:
        """
        Validate that all prior gates passed.

        Args:
            artifact: The merge ready report artifact
            **deps: All dependency artifacts

        Returns:
            GuardResult indicating pass/fail with detailed feedback
        """
        logger.debug("[MergeReadyGuard] Checking all gates...")

        # Parse the merge ready report
        try:
            json.loads(artifact.content)  # Validate JSON structure
        except json.JSONDecodeError as e:
            return GuardResult(
                passed=False,
                feedback=f"Invalid merge ready report JSON: {e}",
            )

        # Check each required gate
        gate_status: dict[str, bool] = {}
        missing_gates: list[str] = []
        failed_gates: list[str] = []

        for gate_id in self._required_gates:
            dep_artifact = deps.get(gate_id)

            if dep_artifact is None:
                missing_gates.append(gate_id)
                gate_status[gate_id] = False
            elif dep_artifact.status != ArtifactStatus.ACCEPTED:
                failed_gates.append(gate_id)
                gate_status[gate_id] = False
            else:
                gate_status[gate_id] = True

        # Build feedback
        all_passed = not missing_gates and not failed_gates
        feedback_parts = ["## Gate Status\n"]

        for gate_id in self._required_gates:
            status = "✓" if gate_status.get(gate_id, False) else "✗"
            feedback_parts.append(f"- {status} {gate_id}")

        feedback_parts.append("")

        if missing_gates:
            feedback_parts.append(f"### Missing Gates: {', '.join(missing_gates)}")
        if failed_gates:
            feedback_parts.append(f"### Failed Gates: {', '.join(failed_gates)}")

        feedback = "\n".join(feedback_parts)

        if all_passed:
            logger.debug("[MergeReadyGuard] ✓ All gates passed - ready to merge")
            return GuardResult(
                passed=True,
                feedback=f"All {len(self._required_gates)} gates passed. Ready to merge!\n\n{feedback}",
            )
        else:
            logger.debug(
                f"[MergeReadyGuard] ✗ Not ready: missing={missing_gates}, failed={failed_gates}"
            )
            return GuardResult(
                passed=False,
                feedback=f"Not ready to merge.\n\n{feedback}",
            )
