"""Integration tests for the Classify-then-Plan pipeline (Option A).

Tests the two-step pipeline: g_analysis → g_plan_conditioned,
verifying that analysis output enriches the plan generation context.
"""

import json

import pytest

from atomicguard.domain.models import (
    AmbientEnvironment,
    Artifact,
    ArtifactStatus,
    Context,
    ContextSnapshot,
)
from atomicguard.domain.prompts import PromptTemplate
from atomicguard.infrastructure.persistence.memory import InMemoryArtifactDAG

from examples.advanced.g_plan_benchmark.guards.analysis import AnalysisGuard
from examples.advanced.g_plan_benchmark.guards.minimal import MinimalPlanGuard
from examples.advanced.g_plan_benchmark.guards.medium import MediumPlanGuard


VALID_ANALYSIS = {
    "problem_type": "bug_fix",
    "language": "python",
    "severity": "high",
    "key_signals": ["TypeError in line 42"],
    "affected_area": "authentication middleware",
    "rationale": "Stack trace indicates type mismatch in auth module.",
}

VALID_PLAN = {
    "plan_id": "bug-fix-auth",
    "initial_state": ["intent_received"],
    "goal_state": ["g_merge_ready"],
    "total_retry_budget": 18,
    "steps": [
        {
            "step_id": "g_config",
            "name": "Extract project configuration",
            "generator": "ConfigExtractorGenerator",
            "guard": "config_extracted",
            "retry_budget": 3,
            "preconditions": ["intent_received"],
            "effects": ["config_ready"],
            "dependencies": [],
        },
        {
            "step_id": "g_search",
            "name": "Locate the bug in auth module",
            "generator": "CoderGenerator",
            "guard": "syntax",
            "retry_budget": 3,
            "preconditions": ["config_ready"],
            "effects": ["bug_located"],
            "dependencies": ["g_config"],
        },
        {
            "step_id": "g_chartest",
            "name": "Write characterization test for TypeError",
            "generator": "CoderGenerator",
            "guard": "dynamic_test",
            "retry_budget": 3,
            "preconditions": ["bug_located"],
            "effects": ["char_test_ready"],
            "dependencies": ["g_search"],
        },
        {
            "step_id": "g_fix",
            "name": "Fix the TypeError in auth middleware",
            "generator": "CoderGenerator",
            "guard": "composite_validation",
            "retry_budget": 6,
            "preconditions": ["char_test_ready"],
            "effects": ["fix_applied"],
            "dependencies": ["g_chartest"],
        },
        {
            "step_id": "g_merge_ready",
            "name": "Final validation",
            "generator": "IdentityGenerator",
            "guard": "merge_ready",
            "retry_budget": 3,
            "preconditions": ["fix_applied"],
            "effects": ["g_merge_ready"],
            "dependencies": ["g_fix"],
        },
    ],
}


@pytest.fixture
def analysis_guard() -> AnalysisGuard:
    return AnalysisGuard()


@pytest.fixture
def minimal_guard() -> MinimalPlanGuard:
    return MinimalPlanGuard()


@pytest.fixture
def medium_guard() -> MediumPlanGuard:
    return MediumPlanGuard()


@pytest.fixture
def snapshot() -> ContextSnapshot:
    return ContextSnapshot(
        workflow_id="test",
        specification="test",
        constraints="",
        feedback_history=(),
    )


def _make_artifact(
    content: str, action_pair_id: str, snapshot: ContextSnapshot
) -> Artifact:
    return Artifact(
        artifact_id="test-001",
        workflow_id="test",
        content=content,
        previous_attempt_id=None,
        parent_action_pair_id=None,
        action_pair_id=action_pair_id,
        created_at="2026-01-30T00:00:00",
        attempt_number=1,
        status=ArtifactStatus.PENDING,
        guard_result=None,
        context=snapshot,
    )


class TestClassifyThenPlanPipeline:
    """Integration tests for the two-step pipeline."""

    def test_analysis_passes_guard(self, analysis_guard, snapshot):
        """Valid analysis artifact passes the analysis_valid guard."""
        artifact = _make_artifact(
            json.dumps(VALID_ANALYSIS), "g_analysis", snapshot
        )
        result = analysis_guard.validate(artifact)
        assert result.passed is True

    def test_conditioned_plan_passes_minimal(self, minimal_guard, snapshot):
        """A plan conditioned on analysis passes minimal validation."""
        artifact = _make_artifact(
            json.dumps(VALID_PLAN), "g_plan_conditioned", snapshot
        )
        result = minimal_guard.validate(artifact)
        assert result.passed is True

    def test_conditioned_plan_passes_medium(self, medium_guard, snapshot):
        """A plan conditioned on analysis passes medium validation."""
        artifact = _make_artifact(
            json.dumps(VALID_PLAN), "g_plan_conditioned", snapshot
        )
        result = medium_guard.validate(artifact)
        assert result.passed is True

    def test_context_amend_includes_analysis(self):
        """Context.amend() correctly injects analysis into constraints."""
        base_context = Context(
            ambient=AmbientEnvironment(
                repository=InMemoryArtifactDAG(), constraints=""
            ),
            specification="Fix the TypeError in login handler.",
            current_artifact=None,
            feedback_history=(),
            dependency_artifacts=(),
        )

        analysis_json = json.dumps(VALID_ANALYSIS)
        enriched = base_context.amend(
            delta_constraints=(
                f"## Problem Analysis (from g_analysis)\n{analysis_json}"
            )
        )

        # Analysis content should appear in ambient constraints
        assert "Problem Analysis" in enriched.ambient.constraints
        assert "bug_fix" in enriched.ambient.constraints
        assert "authentication middleware" in enriched.ambient.constraints

    def test_enriched_context_renders_in_prompt(self):
        """Analysis content appears in the rendered prompt via CONTEXT section."""
        template = PromptTemplate(
            role="Workflow planner",
            constraints="Design a DAG plan.",
            task="Generate a plan.",
        )

        base_context = Context(
            ambient=AmbientEnvironment(
                repository=InMemoryArtifactDAG(), constraints=""
            ),
            specification="Fix login bug.",
            current_artifact=None,
            feedback_history=(),
            dependency_artifacts=(),
        )

        analysis_json = json.dumps(VALID_ANALYSIS, indent=2)
        enriched = base_context.amend(
            delta_constraints=(
                f"## Problem Analysis (from g_analysis)\n{analysis_json}"
            )
        )

        rendered = template.render(enriched)

        # The rendered prompt should contain the analysis in CONTEXT section
        assert "Problem Analysis" in rendered
        assert "bug_fix" in rendered
        assert "authentication middleware" in rendered

    def test_analysis_failure_blocks_plan(self, analysis_guard, snapshot):
        """Invalid analysis should fail, preventing plan generation."""
        bad_analysis = {"problem_type": "invalid_type"}
        artifact = _make_artifact(
            json.dumps(bad_analysis), "g_analysis", snapshot
        )
        result = analysis_guard.validate(artifact)
        assert result.passed is False
        # In the pipeline, this would prevent g_plan_conditioned from running

    def test_pipeline_with_all_problem_types(
        self, analysis_guard, minimal_guard, snapshot
    ):
        """Each problem type produces a valid analysis."""
        for pt in ["bug_fix", "feature", "refactoring", "performance"]:
            data = {**VALID_ANALYSIS, "problem_type": pt}
            artifact = _make_artifact(
                json.dumps(data), "g_analysis", snapshot
            )
            result = analysis_guard.validate(artifact)
            assert result.passed is True, f"problem_type={pt} should pass"


class TestWorkflowConfig:
    """Tests for workflow.json configuration."""

    def test_workflow_json_has_analysis_pair(self):
        """workflow.json should define g_analysis action pair."""
        import pathlib

        wf_path = (
            pathlib.Path(__file__).parent.parent.parent
            / "examples"
            / "advanced"
            / "g_plan_benchmark"
            / "workflow.json"
        )
        with open(wf_path) as f:
            wf = json.load(f)

        assert "g_analysis" in wf["action_pairs"]
        assert wf["action_pairs"]["g_analysis"]["guard"] == "analysis_valid"
        assert (
            wf["action_pairs"]["g_analysis"]["generator"] == "LLMJsonGenerator"
        )

    def test_workflow_json_has_conditioned_pair(self):
        """workflow.json should define g_plan_conditioned with requires."""
        import pathlib

        wf_path = (
            pathlib.Path(__file__).parent.parent.parent
            / "examples"
            / "advanced"
            / "g_plan_benchmark"
            / "workflow.json"
        )
        with open(wf_path) as f:
            wf = json.load(f)

        pair = wf["action_pairs"]["g_plan_conditioned"]
        assert pair["generator"] == "LLMPlanGenerator"
        assert pair["guard"] == "plan_medium"
        assert "g_analysis" in pair["requires"]

    def test_prompts_json_has_analysis_template(self):
        """prompts.json should define g_analysis template."""
        import pathlib

        prompts_path = (
            pathlib.Path(__file__).parent.parent.parent
            / "examples"
            / "advanced"
            / "g_plan_benchmark"
            / "prompts.json"
        )
        with open(prompts_path) as f:
            prompts = json.load(f)

        assert "g_analysis" in prompts
        entry = prompts["g_analysis"]
        assert "role" in entry
        assert "constraints" in entry
        assert "task" in entry
        assert "feedback_wrapper" in entry
        assert "problem_type" in entry["constraints"]
