"""
FeedbackSummarizer: Stagnation detection and failure summarization.

Implements Extension 09 Definitions:
- Definition 44: Stagnation Detection
- Definition 48: Context Injection (failure summary generation)
"""

from dataclasses import dataclass
from difflib import SequenceMatcher

from atomicguard.domain.models import Artifact


@dataclass(frozen=True)
class StagnationInfo:
    """Result of stagnation detection (Definition 44).

    Captures whether consecutive failures are similar enough to trigger
    escalation, along with summary information for context injection.
    """

    detected: bool  # True if r_patience consecutive similar failures
    similar_count: int  # Number of consecutive similar failures
    error_signature: str  # Deduplicated error type/pattern
    approaches_tried: list[str]  # Summary of what was attempted
    failure_summary: str  # Full summary for context injection (Definition 48)


class FeedbackSummarizer:
    """Implements Definition 44: Stagnation Detection.

    Detects when the retry loop is producing similar failures repeatedly,
    indicating that local retries are insufficient and escalation to
    upstream action pairs is needed.
    """

    SIMILARITY_THRESHOLD = 0.7  # Feedback similarity ratio to consider "same"

    def detect_stagnation(
        self,
        feedback_history: list[tuple[Artifact, str]],
        r_patience: int,
    ) -> StagnationInfo:
        """Check if last r_patience feedbacks are similar.

        Definition 44: Stagnation detected when r_patience consecutive
        feedbacks have similarity > SIMILARITY_THRESHOLD.

        Args:
            feedback_history: List of (artifact, feedback) tuples from failed attempts
            r_patience: Number of consecutive similar failures to trigger escalation

        Returns:
            StagnationInfo with detection result and summary
        """
        if len(feedback_history) < r_patience:
            return StagnationInfo(
                detected=False,
                similar_count=len(feedback_history),
                error_signature="",
                approaches_tried=[],
                failure_summary="",
            )

        # Get the last r_patience feedbacks
        recent = feedback_history[-r_patience:]
        recent_feedback = [fb for _, fb in recent]

        # Check pairwise similarity
        all_similar = True
        for i in range(len(recent_feedback) - 1):
            ratio = SequenceMatcher(
                None, recent_feedback[i], recent_feedback[i + 1]
            ).ratio()
            if ratio < self.SIMILARITY_THRESHOLD:
                all_similar = False
                break

        if not all_similar:
            return StagnationInfo(
                detected=False,
                similar_count=0,
                error_signature="",
                approaches_tried=[],
                failure_summary="",
            )

        # Stagnation detected - generate summary
        error_signature = self._extract_error_signature(recent_feedback[-1])
        approaches_tried = self._extract_approaches(recent)
        failure_summary = self.generate_failure_summary(feedback_history)

        return StagnationInfo(
            detected=True,
            similar_count=r_patience,
            error_signature=error_signature,
            approaches_tried=approaches_tried,
            failure_summary=failure_summary,
        )

    def generate_failure_summary(
        self,
        feedback_history: list[tuple[Artifact, str]],
    ) -> str:
        """Generate summary for context injection (Definition 48).

        Creates a concise summary of all failure attempts that can be
        injected into upstream generator constraints to inform the
        re-generation with knowledge of what didn't work.

        Args:
            feedback_history: Full history of (artifact, feedback) tuples

        Returns:
            Formatted failure summary string for constraint injection
        """
        if not feedback_history:
            return ""

        lines = ["## Previous Downstream Failures", ""]
        lines.append(
            f"The following {len(feedback_history)} attempts failed with similar patterns:"
        )
        lines.append("")

        # Extract unique error signatures
        signatures = set()
        for _, feedback in feedback_history:
            sig = self._extract_error_signature(feedback)
            if sig:
                signatures.add(sig)

        if signatures:
            lines.append("### Error Patterns")
            for sig in sorted(signatures):
                lines.append(f"- {sig}")
            lines.append("")

        # Extract approaches tried (from artifact content patterns)
        approaches = self._extract_approaches(feedback_history)
        if approaches:
            lines.append("### Approaches Already Tried")
            for approach in approaches[:5]:  # Limit to 5 approaches
                lines.append(f"- {approach}")
            lines.append("")

        # Latest feedback detail
        _, latest_feedback = feedback_history[-1]
        lines.append("### Latest Feedback")
        # Truncate if too long
        if len(latest_feedback) > 500:
            lines.append(latest_feedback[:500] + "...")
        else:
            lines.append(latest_feedback)
        lines.append("")

        lines.append(
            "**Constraint**: Your regenerated output must enable a different "
            "approach that avoids these failure patterns."
        )

        return "\n".join(lines)

    def _extract_error_signature(self, feedback: str) -> str:
        """Extract a short error signature from feedback.

        Attempts to identify the core error type from feedback text.
        """
        # Look for common error patterns
        patterns = [
            "TypeError:",
            "AttributeError:",
            "ImportError:",
            "SyntaxError:",
            "NameError:",
            "ValueError:",
            "KeyError:",
            "IndexError:",
            "AssertionError:",
            "FileNotFoundError:",
            "ModuleNotFoundError:",
            "test failed",
            "tests failed",
            "FAILED",
            "ERROR",
            "patch does not apply",
            "git apply failed",
        ]

        feedback_lower = feedback.lower()
        for pattern in patterns:
            if pattern.lower() in feedback_lower:
                # Extract the line containing the pattern
                for line in feedback.split("\n"):
                    if pattern.lower() in line.lower():
                        # Clean up and truncate
                        clean = line.strip()[:100]
                        return clean
                return pattern

        # Fallback: first non-empty line
        for line in feedback.split("\n"):
            stripped = line.strip()
            if stripped:
                return stripped[:80]

        return "Unknown error"

    def _extract_approaches(
        self, feedback_history: list[tuple[Artifact, str]]
    ) -> list[str]:
        """Extract summary of approaches tried from artifacts.

        Looks at artifact content to identify what strategies were attempted.
        """
        approaches = []
        seen = set()

        for artifact, _ in feedback_history:
            content = artifact.content if artifact else ""
            approach = self._describe_approach(content)
            if approach and approach not in seen:
                seen.add(approach)
                approaches.append(approach)

        return approaches

    def _describe_approach(self, content: str) -> str:
        """Generate a short description of the approach taken in content."""
        if not content:
            return ""

        # Look for distinctive patterns
        content_lower = content.lower()

        if "def test_" in content_lower:
            # Count test methods
            test_count = content_lower.count("def test_")
            return f"Test with {test_count} test method(s)"

        if "--- a/" in content and "+++ b/" in content:
            # It's a diff/patch
            file_changes = content.count("--- a/")
            return f"Patch modifying {file_changes} file(s)"

        if "import " in content_lower:
            # Code with imports
            imports = [
                line.strip()
                for line in content.split("\n")
                if line.strip().startswith("import ")
                or line.strip().startswith("from ")
            ]
            if imports:
                return f"Code using {imports[0][:40]}"

        # Fallback: first significant line
        for line in content.split("\n"):
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and len(stripped) > 10:
                return f"Approach: {stripped[:50]}..."

        return ""
