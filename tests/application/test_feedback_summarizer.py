"""Tests for FeedbackSummarizer - stagnation detection and failure summarization.

Extension 09: Escalation via Informed Backtracking (Definitions 44, 48).
"""

import pytest

from atomicguard.application.feedback_summarizer import FeedbackSummarizer, StagnationInfo
from atomicguard.domain.models import Artifact, ArtifactStatus, ContextSnapshot, GuardResult


def _make_artifact(content: str = "test", artifact_id: str = "a-001") -> Artifact:
    """Create a minimal artifact for testing."""
    return Artifact(
        artifact_id=artifact_id,
        workflow_id="w-001",
        content=content,
        previous_attempt_id=None,
        parent_action_pair_id=None,
        action_pair_id="ap-001",
        created_at="2025-01-01T00:00:00Z",
        attempt_number=1,
        status=ArtifactStatus.REJECTED,
        guard_result=None,
        context=ContextSnapshot(
            workflow_id="w-001",
            specification="test spec",
            constraints="",
            feedback_history=(),
            dependency_artifacts=(),
        ),
    )


class TestStagnationDetection:
    """Tests for Definition 44: Stagnation Detection."""

    def test_no_stagnation_below_r_patience(self) -> None:
        """No stagnation if feedback count < r_patience."""
        summarizer = FeedbackSummarizer()
        history = [
            (_make_artifact("code1", "a-001"), "Error: test failed"),
        ]

        result = summarizer.detect_stagnation(history, r_patience=2)

        assert result.detected is False

    def test_no_stagnation_if_feedback_differs(self) -> None:
        """No stagnation if feedbacks are different."""
        summarizer = FeedbackSummarizer()
        history = [
            (_make_artifact("code1", "a-001"), "TypeError: expected int"),
            (_make_artifact("code2", "a-002"), "ImportError: no module named foo"),
        ]

        result = summarizer.detect_stagnation(history, r_patience=2)

        assert result.detected is False

    def test_stagnation_detected_when_similar(self) -> None:
        """Stagnation detected when r_patience consecutive feedbacks are similar."""
        summarizer = FeedbackSummarizer()
        history = [
            (_make_artifact("code1", "a-001"), "Test failed: expected 5, got 4"),
            (_make_artifact("code2", "a-002"), "Test failed: expected 5, got 3"),
            (_make_artifact("code3", "a-003"), "Test failed: expected 5, got 2"),
        ]

        result = summarizer.detect_stagnation(history, r_patience=3)

        assert result.detected is True
        assert result.similar_count == 3

    def test_stagnation_uses_last_r_patience_only(self) -> None:
        """Only last r_patience feedbacks are checked."""
        summarizer = FeedbackSummarizer()
        history = [
            (_make_artifact("code1", "a-001"), "Different error at start"),
            (_make_artifact("code2", "a-002"), "Test failed: expected 5, got 4"),
            (_make_artifact("code3", "a-003"), "Test failed: expected 5, got 3"),
        ]

        # r_patience=2 checks only last 2, which are similar
        result = summarizer.detect_stagnation(history, r_patience=2)

        assert result.detected is True
        assert result.similar_count == 2

    def test_stagnation_returns_error_signature(self) -> None:
        """Stagnation result includes extracted error signature."""
        summarizer = FeedbackSummarizer()
        history = [
            (_make_artifact("code1", "a-001"), "TypeError: cannot add str to int"),
            (_make_artifact("code2", "a-002"), "TypeError: cannot add str to float"),
        ]

        result = summarizer.detect_stagnation(history, r_patience=2)

        assert result.detected is True
        assert "TypeError" in result.error_signature

    def test_stagnation_returns_failure_summary(self) -> None:
        """Stagnation result includes failure summary for context injection."""
        summarizer = FeedbackSummarizer()
        history = [
            (_make_artifact("def test_foo():", "a-001"), "Test failed"),
            (_make_artifact("def test_bar():", "a-002"), "Test failed"),
        ]

        result = summarizer.detect_stagnation(history, r_patience=2)

        assert result.detected is True
        assert "Previous Downstream Failures" in result.failure_summary
        assert "Test failed" in result.failure_summary


class TestFailureSummaryGeneration:
    """Tests for Definition 48: Context Injection."""

    def test_empty_history_returns_empty_summary(self) -> None:
        """Empty history produces empty summary."""
        summarizer = FeedbackSummarizer()

        result = summarizer.generate_failure_summary([])

        assert result == ""

    def test_summary_includes_error_patterns(self) -> None:
        """Summary includes deduplicated error patterns."""
        summarizer = FeedbackSummarizer()
        history = [
            (_make_artifact(), "TypeError: cannot add str to int"),
            (_make_artifact(), "TypeError: cannot add str to float"),
        ]

        result = summarizer.generate_failure_summary(history)

        assert "Error Patterns" in result
        assert "TypeError" in result

    def test_summary_includes_approaches_tried(self) -> None:
        """Summary includes approaches extracted from artifacts."""
        summarizer = FeedbackSummarizer()
        history = [
            (_make_artifact("def test_case_a(): pass"), "Failed"),
            (_make_artifact("def test_case_b(): pass"), "Failed"),
        ]

        result = summarizer.generate_failure_summary(history)

        assert "Approaches Already Tried" in result

    def test_summary_includes_latest_feedback(self) -> None:
        """Summary includes truncated latest feedback."""
        summarizer = FeedbackSummarizer()
        feedback = "Very long feedback " * 50  # >500 chars
        history = [
            (_make_artifact(), feedback),
        ]

        result = summarizer.generate_failure_summary(history)

        assert "Latest Feedback" in result
        assert "..." in result  # Truncation indicator

    def test_summary_includes_constraint_instruction(self) -> None:
        """Summary includes constraint for re-generation."""
        summarizer = FeedbackSummarizer()
        history = [
            (_make_artifact(), "Test failed"),
        ]

        result = summarizer.generate_failure_summary(history)

        assert "Constraint" in result
        assert "avoids these failure patterns" in result


class TestErrorSignatureExtraction:
    """Tests for error signature extraction from feedback."""

    def test_extracts_python_exception_types(self) -> None:
        """Recognizes common Python exception types."""
        summarizer = FeedbackSummarizer()
        test_cases = [
            ("TypeError: 'str' object is not callable", "TypeError"),
            ("AttributeError: 'NoneType' has no attribute 'x'", "AttributeError"),
            ("ImportError: No module named 'foo'", "ImportError"),
            ("SyntaxError: invalid syntax", "SyntaxError"),
            ("NameError: name 'x' is not defined", "NameError"),
        ]

        for feedback, expected in test_cases:
            result = summarizer._extract_error_signature(feedback)
            assert expected in result, f"Failed for {feedback}"

    def test_extracts_test_failure_patterns(self) -> None:
        """Recognizes test failure patterns."""
        summarizer = FeedbackSummarizer()
        test_cases = [
            "3 tests failed",
            "test_foo FAILED",
            "ERROR: test collection failed",
        ]

        for feedback in test_cases:
            result = summarizer._extract_error_signature(feedback)
            assert result != "Unknown error"

    def test_extracts_git_apply_failures(self) -> None:
        """Recognizes git apply failures."""
        summarizer = FeedbackSummarizer()
        feedback = "error: patch does not apply"

        result = summarizer._extract_error_signature(feedback)

        assert "patch does not apply" in result.lower()

    def test_fallback_to_first_line(self) -> None:
        """Falls back to first non-empty line for unknown errors."""
        summarizer = FeedbackSummarizer()
        feedback = "Some completely unexpected error format\nMore details"

        result = summarizer._extract_error_signature(feedback)

        assert "unexpected error" in result


class TestApproachExtraction:
    """Tests for approach extraction from artifacts."""

    def test_extracts_test_method_count(self) -> None:
        """Extracts test method count from test code."""
        summarizer = FeedbackSummarizer()
        content = """
def test_foo():
    pass

def test_bar():
    pass
"""
        history = [(_make_artifact(content), "Failed")]

        result = summarizer._extract_approaches(history)

        assert any("test method" in a.lower() for a in result)

    def test_extracts_patch_file_count(self) -> None:
        """Extracts file count from patch content."""
        summarizer = FeedbackSummarizer()
        content = """
--- a/foo.py
+++ b/foo.py
@@ -1,3 +1,4 @@
+# fixed
"""
        history = [(_make_artifact(content), "Failed")]

        result = summarizer._extract_approaches(history)

        assert any("patch" in a.lower() for a in result)

    def test_deduplicates_approaches(self) -> None:
        """Duplicate approaches are deduplicated."""
        summarizer = FeedbackSummarizer()
        content = "def test_foo(): pass"
        history = [
            (_make_artifact(content, "a-001"), "Failed"),
            (_make_artifact(content, "a-002"), "Failed"),
        ]

        result = summarizer._extract_approaches(history)

        # Same content should produce same approach description
        # so it should be deduplicated
        assert len(result) == 1


class TestStagnationEdgeCases:
    """Edge case tests for stagnation detection (Definition 44)."""

    def test_r_patience_of_2_minimum(self) -> None:
        """r_patience must be at least 2 (need 2 to detect pattern)."""
        summarizer = FeedbackSummarizer()
        history = [
            (_make_artifact("code1", "a-001"), "Test failed: expected 5"),
            (_make_artifact("code2", "a-002"), "Test failed: expected 5"),
        ]

        result = summarizer.detect_stagnation(history, r_patience=2)

        assert result.detected is True

    def test_dissimilar_feedback_resets_count(self) -> None:
        """Dissimilar feedback breaks the stagnation pattern."""
        summarizer = FeedbackSummarizer()
        history = [
            (_make_artifact("code1", "a-001"), "Test failed: expected 5"),
            (_make_artifact("code2", "a-002"), "Test failed: expected 5"),
            (_make_artifact("code3", "a-003"), "Completely different TypeError"),
            (_make_artifact("code4", "a-004"), "Test failed: expected 5"),
        ]

        # Last 3 are not all similar (different error in middle)
        result = summarizer.detect_stagnation(history, r_patience=3)

        assert result.detected is False

    def test_summary_deduplicates_identical_errors(self) -> None:
        """Failure summary deduplicates identical error messages."""
        summarizer = FeedbackSummarizer()
        history = [
            (_make_artifact("c1", "a-001"), "TypeError: expected int"),
            (_make_artifact("c2", "a-002"), "TypeError: expected int"),
            (_make_artifact("c3", "a-003"), "TypeError: expected int"),
            (_make_artifact("c4", "a-004"), "TypeError: expected int"),
            (_make_artifact("c5", "a-005"), "TypeError: expected int"),
        ]

        result = summarizer.generate_failure_summary(history)

        # Should mention TypeError only once in error patterns, not 5 times
        error_section = result.split("### Error Patterns")[1].split("###")[0]
        assert error_section.count("TypeError") == 1

    def test_stagnation_triggers_exactly_at_r_patience(self) -> None:
        """Stagnation triggers exactly when r_patience consecutive similar failures."""
        summarizer = FeedbackSummarizer()

        # At r_patience=3, exactly 3 similar should trigger
        history = [
            (_make_artifact("c1", "a-001"), "Test failed: expected 5, got 1"),
            (_make_artifact("c2", "a-002"), "Test failed: expected 5, got 2"),
            (_make_artifact("c3", "a-003"), "Test failed: expected 5, got 3"),
        ]

        result = summarizer.detect_stagnation(history, r_patience=3)
        assert result.detected is True

        # At r_patience=4, only 3 similar should not trigger
        result = summarizer.detect_stagnation(history, r_patience=4)
        assert result.detected is False


# =============================================================================
# Per-Guard Stagnation Detection (Definition 44, escalation_by_guard)
# =============================================================================


def _make_artifact_with_guard(
    content: str = "test",
    artifact_id: str = "a-001",
    guard_name: str | None = None,
    passed: bool = False,
) -> Artifact:
    """Create a minimal artifact with guard result for per-guard testing."""
    return Artifact(
        artifact_id=artifact_id,
        workflow_id="w-001",
        content=content,
        previous_attempt_id=None,
        parent_action_pair_id=None,
        action_pair_id="ap-001",
        created_at="2025-01-01T00:00:00Z",
        attempt_number=1,
        status=ArtifactStatus.REJECTED if not passed else ArtifactStatus.ACCEPTED,
        guard_result=GuardResult(passed=passed, feedback="test", guard_name=guard_name),
        context=ContextSnapshot(
            workflow_id="w-001",
            specification="test spec",
            constraints="",
            feedback_history=(),
            dependency_artifacts=(),
        ),
    )


class TestDetectStagnationByGuard:
    """Tests for per-guard stagnation detection (Definition 44)."""

    def test_groups_feedback_by_guard_name(self) -> None:
        """Per-guard detection groups feedback by guard_name."""
        summarizer = FeedbackSummarizer()
        history = [
            (_make_artifact_with_guard(guard_name="SyntaxGuard"), "Syntax error"),
            (_make_artifact_with_guard(guard_name="TypeGuard"), "Type error"),
            (_make_artifact_with_guard(guard_name="SyntaxGuard"), "Syntax error"),
            (_make_artifact_with_guard(guard_name="TypeGuard"), "Type error"),
            (_make_artifact_with_guard(guard_name="SyntaxGuard"), "Syntax error similar"),
        ]

        # With r_patience=3, only SyntaxGuard has 3 entries
        result = summarizer.detect_stagnation_by_guard(history, r_patience=3)

        assert result.detected
        assert result.stagnant_guard == "SyntaxGuard"

    def test_catches_oscillation_pattern(self) -> None:
        """Detects stagnation hidden by alternating guard failures."""
        summarizer = FeedbackSummarizer()
        # Alternating failures that hide stagnation in global view
        history = [
            (_make_artifact_with_guard(guard_name="Guard1"), "Same error A"),
            (_make_artifact_with_guard(guard_name="Guard2"), "Different error B"),
            (_make_artifact_with_guard(guard_name="Guard1"), "Same error A"),
            (_make_artifact_with_guard(guard_name="Guard2"), "Another error C"),
            (_make_artifact_with_guard(guard_name="Guard1"), "Same error A similar"),
        ]

        # Global detection would not find stagnation
        global_result = summarizer.detect_stagnation(history, r_patience=3)
        assert not global_result.detected

        # Per-guard detection finds Guard1 stagnation
        per_guard_result = summarizer.detect_stagnation_by_guard(history, r_patience=3)
        assert per_guard_result.detected
        assert per_guard_result.stagnant_guard == "Guard1"

    def test_no_stagnation_when_guard_streams_short(self) -> None:
        """No stagnation when each guard's stream is below r_patience."""
        summarizer = FeedbackSummarizer()
        history = [
            (_make_artifact_with_guard(guard_name="Guard1"), "Error 1"),
            (_make_artifact_with_guard(guard_name="Guard2"), "Error 2"),
            (_make_artifact_with_guard(guard_name="Guard3"), "Error 3"),
        ]

        result = summarizer.detect_stagnation_by_guard(history, r_patience=2)

        assert not result.detected

    def test_stagnation_info_includes_stagnant_guard(self) -> None:
        """StagnationInfo carries which guard stagnated."""
        summarizer = FeedbackSummarizer()
        history = [
            (_make_artifact_with_guard(guard_name="TestGuard"), "Same error"),
            (_make_artifact_with_guard(guard_name="TestGuard"), "Same error"),
        ]

        result = summarizer.detect_stagnation_by_guard(history, r_patience=2)

        assert result.detected
        assert result.stagnant_guard == "TestGuard"


class TestStagnationInfoImmutability:
    """Tests for StagnationInfo dataclass properties."""

    def test_stagnation_info_is_frozen(self) -> None:
        """StagnationInfo is immutable."""
        info = StagnationInfo(
            detected=True,
            similar_count=3,
            error_signature="test",
            approaches_tried=("a", "b"),
            failure_summary="summary",
            stagnant_guard="Guard1",
        )

        with pytest.raises(AttributeError):
            info.detected = False  # type: ignore

    def test_stagnation_info_approaches_is_tuple(self) -> None:
        """StagnationInfo.approaches_tried is a tuple (immutable)."""
        info = StagnationInfo(
            detected=False,
            similar_count=0,
            error_signature="",
            approaches_tried=("a", "b", "c"),
            failure_summary="",
        )

        assert isinstance(info.approaches_tried, tuple)
