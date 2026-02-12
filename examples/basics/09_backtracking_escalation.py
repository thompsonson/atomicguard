#!/usr/bin/env python3
"""
Extension 09: Backtracking and Escalation (Definitions 44-48).

Demonstrates informed backtracking when stagnation is detected:
- Context injection from feedback - How feedback flows back during retries
- DAG storage - How artifacts store feedback_history and escalation_feedback
- Escalation feedback routing - How failure summaries are passed to upstream steps

Scenario:
  Two-step workflow: spec (generate specification) -> impl (implement code)
  - spec generates an incomplete specification (missing edge case handling)
  - impl fails repeatedly with similar TypeErrors (stagnation detected after r_patience=2)
  - Workflow escalates: failure summary injected into spec step
  - spec re-runs with escalation context, generates improved specification
  - impl succeeds with the improved input

Run: python -m examples.basics.09_backtracking_escalation
"""

import logging

from atomicguard import (
    ActionPair,
    Artifact,
    GuardInterface,
    GuardResult,
    InMemoryArtifactDAG,
    MockGenerator,
    PromptTemplate,
    Workflow,
    WorkflowStatus,
    export_workflow_html,
)

# Configure logging to see escalation events
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# CUSTOM GUARDS
# =============================================================================


class AlwaysPassGuard(GuardInterface):
    """Guard that always passes - used for the spec step."""

    def validate(self, artifact: Artifact, **dependencies: Artifact) -> GuardResult:
        """Always pass validation."""
        return GuardResult(passed=True, feedback="", guard_name="AlwaysPassGuard")


class RequiresEdgeCaseGuard(GuardInterface):
    """
    Guard that simulates downstream failures.

    Fails first 2 calls with similar feedback (triggers stagnation).
    Passes on 3rd call if content mentions edge cases.
    """

    def __init__(self) -> None:
        self._call_count = 0

    def validate(self, artifact: Artifact, **dependencies: Artifact) -> GuardResult:
        """Validate that implementation handles edge cases."""
        self._call_count += 1

        # Check if spec dependency mentions edge cases
        spec_artifact = dependencies.get("spec")
        spec_has_edge_cases = spec_artifact and "edge case" in spec_artifact.content.lower()

        # Check if impl mentions edge case handling
        impl_has_edge_handling = "edge case" in artifact.content.lower() or "none" in artifact.content.lower()

        logger.info(
            "[RequiresEdgeCaseGuard] Call %d: spec_has_edge_cases=%s, impl_has_edge_handling=%s",
            self._call_count,
            spec_has_edge_cases,
            impl_has_edge_handling,
        )

        # If spec has edge cases AND impl handles them, pass
        if spec_has_edge_cases and impl_has_edge_handling:
            return GuardResult(
                passed=True,
                feedback="Implementation correctly handles edge cases.",
                guard_name="RequiresEdgeCaseGuard",
            )

        # Otherwise fail with similar TypeError feedback (to trigger stagnation)
        return GuardResult(
            passed=False,
            feedback="TypeError: cannot handle None input - missing null check in implementation",
            guard_name="RequiresEdgeCaseGuard",
        )


# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

SPEC_TEMPLATE = PromptTemplate(
    role="Specification Writer",
    constraints="Write clear, complete specifications.",
    task="Generate a specification for a function that processes user input.",
    feedback_wrapper="Previous attempt failed:\n{feedback}\n\nFix the issues and try again.",
    escalation_feedback_wrapper=(
        "CRITICAL: Downstream implementation step failed repeatedly.\n"
        "Failure details:\n{feedback}\n\n"
        "You MUST update the specification to address this failure."
    ),
)

IMPL_TEMPLATE = PromptTemplate(
    role="Python Developer",
    constraints="Write clean, robust Python code.",
    task="Implement the function according to the specification.",
    feedback_wrapper="Previous attempt failed:\n{feedback}\n\nFix the issues and try again.",
)


# =============================================================================
# MOCK RESPONSES
# =============================================================================

# Spec responses:
# 1st: Incomplete spec (no edge cases mentioned)
# 2nd: After escalation - complete spec with edge cases
SPEC_RESPONSES = [
    """## Specification: process_input(value)

The function should:
1. Accept a string value as input
2. Convert the value to uppercase
3. Return the processed string

Example:
  process_input("hello") -> "HELLO"
""",
    """## Specification: process_input(value)

The function should:
1. Accept a string value as input (may be None)
2. Handle EDGE CASE: If value is None, return empty string
3. Handle EDGE CASE: If value is not a string, convert to string first
4. Convert the value to uppercase
5. Return the processed string

Examples:
  process_input("hello") -> "HELLO"
  process_input(None) -> ""
  process_input(123) -> "123"
""",
]

# Impl responses:
# 1st & 2nd: Naive implementation (fails guard - no null handling)
# 3rd: After escalation - implementation with edge case handling
IMPL_RESPONSES = [
    """def process_input(value):
    return value.upper()
""",
    """def process_input(value):
    result = value.upper()
    return result
""",
    """def process_input(value):
    # Handle edge case: None input
    if value is None:
        return ""
    # Handle edge case: non-string input
    if not isinstance(value, str):
        value = str(value)
    return value.upper()
""",
]


# =============================================================================
# MAIN DEMO
# =============================================================================


def main() -> None:
    """Demonstrate backtracking and escalation workflow."""
    print("\n" + "=" * 70)
    print("EXTENSION 09: BACKTRACKING AND ESCALATION")
    print("Definitions 44-48 from the formal specification")
    print("=" * 70)

    # Create shared artifact DAG
    dag = InMemoryArtifactDAG()

    # Create generators with predefined responses
    spec_generator = MockGenerator(responses=SPEC_RESPONSES)
    impl_generator = MockGenerator(responses=IMPL_RESPONSES)

    # Create guards
    spec_guard = AlwaysPassGuard()
    impl_guard = RequiresEdgeCaseGuard()

    # Create action pairs
    spec_action_pair = ActionPair(
        generator=spec_generator,
        guard=spec_guard,
        prompt_template=SPEC_TEMPLATE,
    )
    impl_action_pair = ActionPair(
        generator=impl_generator,
        guard=impl_guard,
        prompt_template=IMPL_TEMPLATE,
    )

    # Create workflow with escalation configuration
    # rmax=5 allows enough retries for the demo
    # r_patience=2 means stagnation detected after 2 similar failures
    workflow = Workflow(artifact_dag=dag, rmax=5)

    workflow.add_step(
        guard_id="spec",
        action_pair=spec_action_pair,
    )

    workflow.add_step(
        guard_id="impl",
        action_pair=impl_action_pair,
        requires=("spec",),
        r_patience=2,  # Stagnation detected after 2 similar failures
        e_max=1,  # Allow 1 escalation attempt
        escalate_feedback_to=("spec",),  # Send failure summary to spec step
    )

    print("\nWorkflow Configuration:")
    print("  Steps: spec -> impl")
    print("  rmax=5, r_patience=2 for impl, e_max=1")
    print("  Escalation target: impl escalates to spec")
    print()
    print("-" * 70)
    print("EXECUTION TIMELINE")
    print("-" * 70)

    # Execute the workflow
    result = workflow.execute("Create a function to process user input")

    print("-" * 70)
    print("WORKFLOW RESULT")
    print("-" * 70)
    print(f"\nStatus: {result.status.value}")

    if result.status == WorkflowStatus.SUCCESS:
        print("\n=== SUCCESS ===")
        print("\nFinal Artifacts:")
        for guard_id, artifact in result.artifacts.items():
            print(f"\n--- {guard_id} ---")
            print(f"Artifact ID: {artifact.artifact_id}")
            print(f"Attempt: {artifact.attempt_number}")
            print(f"Content:\n{artifact.content}")

        # Show provenance in DAG
        print("\n" + "-" * 70)
        print("DAG PROVENANCE (Artifact History)")
        print("-" * 70)
        all_artifacts = dag.get_all()
        print(f"\nTotal artifacts in DAG: {len(all_artifacts)}")
        print("\nExecution trace:")
        for i, art in enumerate(all_artifacts, 1):
            escalation_info = ""
            if art.context.escalation_feedback:
                escalation_info = " [has escalation_feedback]"
            feedback_info = ""
            if art.context.feedback_history:
                feedback_info = f" [feedback_history: {len(art.context.feedback_history)}]"
            status_symbol = "PASS" if art.status.value == "accepted" else "FAIL"
            print(
                f"  {i}. [{art.action_pair_id:4}] attempt={art.attempt_number} "
                f"-> {status_symbol:4}{escalation_info}{feedback_info}"
            )

        # Explain what happened
        print("\n" + "-" * 70)
        print("WHAT HAPPENED")
        print("-" * 70)
        print("""
1. spec generated incomplete specification (no edge cases) -> PASSED
2. impl generated naive implementation -> FAILED (TypeError)
3. impl retry with feedback -> FAILED (same TypeError - stagnation!)
4. ESCALATION: Stagnation detected after 2 similar failures
   - Workflow invalidates 'spec' step
   - Failure summary injected into spec's escalation_feedback
5. spec re-runs with escalation context -> generates improved spec with edge cases
6. impl runs against new spec -> PASSED (now handles edge cases)
""")

    elif result.status == WorkflowStatus.ESCALATION:
        print(f"\n=== ESCALATION (Human intervention needed) ===")
        print(f"Failed step: {result.failed_step}")
        print(f"Feedback: {result.escalation_feedback}")

    else:
        print(f"\n=== FAILED ===")
        print(f"Failed step: {result.failed_step}")

    # Export HTML visualization
    print("\n" + "-" * 70)
    print("HTML VISUALIZATION")
    print("-" * 70)

    # Get workflow_id from the first artifact
    all_artifacts = dag.get_all()
    if all_artifacts:
        workflow_id = all_artifacts[0].workflow_id
        output_path = export_workflow_html(dag, workflow_id, "workflow_visualization.html")
        print(f"\nVisualization exported to: {output_path}")
        print("Open this file in a browser to explore the workflow DAG interactively.")

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
