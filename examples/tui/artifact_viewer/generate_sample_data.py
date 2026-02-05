"""Generate sample artifact data for testing the TUI."""

from __future__ import annotations

import sys
from pathlib import Path


def generate_sample_artifacts(output_dir: str) -> None:
    """Generate sample artifacts for testing the TUI."""
    # Import atomicguard models
    from atomicguard.domain.models import (
        Artifact,
        ArtifactStatus,
        ContextSnapshot,
        FeedbackEntry,
        GuardResult,
        SubGuardOutcome,
    )
    from atomicguard.infrastructure.persistence.filesystem import FilesystemArtifactDAG

    # Create DAG
    dag = FilesystemArtifactDAG(output_dir)

    # === Workflow 1: Simple function generation ===
    workflow_id_1 = "wf-simple-func-001"
    context_1 = ContextSnapshot(
        workflow_id=workflow_id_1,
        specification="Write a function that calculates the factorial of a number",
        constraints="Use recursion. Handle edge cases. Include type hints.",
        feedback_history=(),
        dependency_artifacts=(),
    )

    # Attempt 1 - Rejected (no type hints)
    artifact_1_1 = Artifact(
        artifact_id="art-001-attempt-1",
        workflow_id=workflow_id_1,
        content='''def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
''',
        previous_attempt_id=None,
        parent_action_pair_id=None,
        action_pair_id="g_factorial",
        created_at="2025-02-05T10:00:00Z",
        attempt_number=1,
        status=ArtifactStatus.REJECTED,
        guard_result=GuardResult(
            passed=False,
            feedback="Missing type hints. Function should use type annotations.",
            fatal=False,
            guard_name="TypeHintGuard",
            sub_results=(),
        ),
        context=context_1,
    )

    # Attempt 2 - Rejected (no edge case handling)
    context_1_2 = ContextSnapshot(
        workflow_id=workflow_id_1,
        specification="Write a function that calculates the factorial of a number",
        constraints="Use recursion. Handle edge cases. Include type hints.",
        feedback_history=(
            FeedbackEntry(
                artifact_id="art-001-attempt-1",
                feedback="Missing type hints. Function should use type annotations.",
            ),
        ),
        dependency_artifacts=(),
    )

    artifact_1_2 = Artifact(
        artifact_id="art-001-attempt-2",
        workflow_id=workflow_id_1,
        content='''def factorial(n: int) -> int:
    if n <= 1:
        return 1
    return n * factorial(n - 1)
''',
        previous_attempt_id="art-001-attempt-1",
        parent_action_pair_id=None,
        action_pair_id="g_factorial",
        created_at="2025-02-05T10:01:00Z",
        attempt_number=2,
        status=ArtifactStatus.REJECTED,
        guard_result=GuardResult(
            passed=False,
            feedback="Does not handle negative numbers. Add validation for n < 0.",
            fatal=False,
            guard_name="EdgeCaseGuard",
            sub_results=(),
        ),
        context=context_1_2,
    )

    # Attempt 3 - Accepted
    context_1_3 = ContextSnapshot(
        workflow_id=workflow_id_1,
        specification="Write a function that calculates the factorial of a number",
        constraints="Use recursion. Handle edge cases. Include type hints.",
        feedback_history=(
            FeedbackEntry(
                artifact_id="art-001-attempt-1",
                feedback="Missing type hints. Function should use type annotations.",
            ),
            FeedbackEntry(
                artifact_id="art-001-attempt-2",
                feedback="Does not handle negative numbers. Add validation for n < 0.",
            ),
        ),
        dependency_artifacts=(),
    )

    artifact_1_3 = Artifact(
        artifact_id="art-001-attempt-3",
        workflow_id=workflow_id_1,
        content='''def factorial(n: int) -> int:
    """Calculate the factorial of a non-negative integer.

    Args:
        n: A non-negative integer.

    Returns:
        The factorial of n.

    Raises:
        ValueError: If n is negative.
    """
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    if n <= 1:
        return 1
    return n * factorial(n - 1)
''',
        previous_attempt_id="art-001-attempt-2",
        parent_action_pair_id=None,
        action_pair_id="g_factorial",
        created_at="2025-02-05T10:02:00Z",
        attempt_number=3,
        status=ArtifactStatus.ACCEPTED,
        guard_result=GuardResult(
            passed=True,
            feedback="All checks passed.",
            fatal=False,
            guard_name="CompositeGuard",
            sub_results=(
                SubGuardOutcome(
                    guard_name="SyntaxGuard",
                    passed=True,
                    feedback="",
                    execution_time_ms=5.2,
                ),
                SubGuardOutcome(
                    guard_name="TypeHintGuard",
                    passed=True,
                    feedback="",
                    execution_time_ms=12.3,
                ),
                SubGuardOutcome(
                    guard_name="EdgeCaseGuard",
                    passed=True,
                    feedback="",
                    execution_time_ms=8.7,
                ),
            ),
        ),
        context=context_1_3,
    )

    # Store workflow 1 artifacts
    dag.store(artifact_1_1)
    dag.store(artifact_1_2)
    dag.store(artifact_1_3)

    # === Workflow 2: Multi-step data processing ===
    workflow_id_2 = "wf-data-pipeline-002"

    # Step 1: Data loader
    context_2_1 = ContextSnapshot(
        workflow_id=workflow_id_2,
        specification="Create a data loader that reads CSV files",
        constraints="Use pathlib. Return list of dicts.",
        feedback_history=(),
        dependency_artifacts=(),
    )

    artifact_2_1 = Artifact(
        artifact_id="art-002-loader-1",
        workflow_id=workflow_id_2,
        content='''import csv
from pathlib import Path


def load_csv(filepath: Path) -> list[dict[str, str]]:
    """Load a CSV file and return list of row dictionaries."""
    with filepath.open() as f:
        reader = csv.DictReader(f)
        return list(reader)
''',
        previous_attempt_id=None,
        parent_action_pair_id=None,
        action_pair_id="g_data_loader",
        created_at="2025-02-05T11:00:00Z",
        attempt_number=1,
        status=ArtifactStatus.ACCEPTED,
        guard_result=GuardResult(
            passed=True,
            feedback="Clean implementation.",
            fatal=False,
            guard_name="SyntaxGuard",
            sub_results=(),
        ),
        context=context_2_1,
    )

    # Step 2: Data transformer (depends on loader)
    context_2_2 = ContextSnapshot(
        workflow_id=workflow_id_2,
        specification="Create a transformer that filters and maps data",
        constraints="Use functional style. Support custom predicates.",
        feedback_history=(),
        dependency_artifacts=(("g_data_loader", "art-002-loader-1"),),
    )

    artifact_2_2_1 = Artifact(
        artifact_id="art-002-transform-1",
        workflow_id=workflow_id_2,
        content='''from typing import Callable, TypeVar

T = TypeVar("T")


def transform_data(
    data: list[dict],
    predicate: Callable[[dict], bool],
    mapper: Callable[[dict], T],
) -> list[T]:
    """Filter and transform data using predicate and mapper."""
    return [mapper(item) for item in data if predicate(item)]
''',
        previous_attempt_id=None,
        parent_action_pair_id="g_data_loader",
        action_pair_id="g_transformer",
        created_at="2025-02-05T11:01:00Z",
        attempt_number=1,
        status=ArtifactStatus.ACCEPTED,
        guard_result=GuardResult(
            passed=True,
            feedback="Good use of generics.",
            fatal=False,
            guard_name="SyntaxGuard",
            sub_results=(),
        ),
        context=context_2_2,
    )

    # Step 3: Data exporter (depends on transformer)
    context_2_3 = ContextSnapshot(
        workflow_id=workflow_id_2,
        specification="Create an exporter that writes to JSON",
        constraints="Support pretty printing option.",
        feedback_history=(),
        dependency_artifacts=(
            ("g_data_loader", "art-002-loader-1"),
            ("g_transformer", "art-002-transform-1"),
        ),
    )

    artifact_2_3_1 = Artifact(
        artifact_id="art-002-export-1",
        workflow_id=workflow_id_2,
        content='''import json
from pathlib import Path
from typing import Any


def export_json(
    data: list[Any],
    filepath: Path,
    pretty: bool = True,
) -> None:
    """Export data to JSON file."""
    with filepath.open("w") as f:
        if pretty:
            json.dump(data, f, indent=2)
        else:
            json.dump(data, f)
''',
        previous_attempt_id=None,
        parent_action_pair_id="g_transformer",
        action_pair_id="g_exporter",
        created_at="2025-02-05T11:02:00Z",
        attempt_number=1,
        status=ArtifactStatus.ACCEPTED,
        guard_result=GuardResult(
            passed=True,
            feedback="Simple and effective.",
            fatal=False,
            guard_name="SyntaxGuard",
            sub_results=(),
        ),
        context=context_2_3,
    )

    # Store workflow 2 artifacts
    dag.store(artifact_2_1)
    dag.store(artifact_2_2_1)
    dag.store(artifact_2_3_1)

    # === Workflow 3: Failed generation (fatal error) ===
    workflow_id_3 = "wf-failed-003"

    context_3 = ContextSnapshot(
        workflow_id=workflow_id_3,
        specification="Write a function that divides two numbers",
        constraints="Handle all edge cases.",
        feedback_history=(),
        dependency_artifacts=(),
    )

    artifact_3_1 = Artifact(
        artifact_id="art-003-div-1",
        workflow_id=workflow_id_3,
        content='''def divide(a, b):
    return a / b  # This will fail for b=0
''',
        previous_attempt_id=None,
        parent_action_pair_id=None,
        action_pair_id="g_divide",
        created_at="2025-02-05T12:00:00Z",
        attempt_number=1,
        status=ArtifactStatus.REJECTED,
        guard_result=GuardResult(
            passed=False,
            feedback="FATAL: Division by zero not handled. This is a critical security issue.",
            fatal=True,
            guard_name="SecurityGuard",
            sub_results=(),
        ),
        context=context_3,
    )

    dag.store(artifact_3_1)

    print(f"Generated {len(dag.get_all())} sample artifacts in {output_dir}")
    print("\nWorkflows:")
    print(f"  - {workflow_id_1}: Factorial function (3 attempts)")
    print(f"  - {workflow_id_2}: Data pipeline (3 steps)")
    print(f"  - {workflow_id_3}: Failed division (fatal error)")


def main() -> None:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate sample artifact data")
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="./sample_artifact_dag",
        help="Output directory for artifacts (default: ./sample_artifact_dag)",
    )
    args = parser.parse_args()

    output_path = Path(args.output_dir).resolve()
    generate_sample_artifacts(str(output_path))
    print(f"\nTo view in TUI, run:")
    print(f"  python -m examples.tui.artifact_viewer.app {output_path}")


if __name__ == "__main__":
    main()
