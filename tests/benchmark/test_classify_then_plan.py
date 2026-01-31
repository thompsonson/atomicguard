"""Integration tests for the decomposed planning pipelines.

Tests:
- Two-step pipeline: g_analysis → g_plan_conditioned (classify-then-plan)
- Full pipeline: g_analysis → g_recon → g_strategy → g_plan_full

Verifies that pre-step outputs enrich the plan generation context.
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
from examples.advanced.g_plan_benchmark.guards.recon import ReconGuard
from examples.advanced.g_plan_benchmark.guards.strategy import StrategyGuard


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

    def test_workflow_json_has_full_pipeline(self):
        """workflow.json should define the full 4-step pipeline."""
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

        # Full pipeline action pairs
        assert "g_recon" in wf["action_pairs"]
        assert wf["action_pairs"]["g_recon"]["guard"] == "recon_valid"
        assert "g_analysis" in wf["action_pairs"]["g_recon"]["requires"]

        assert "g_strategy" in wf["action_pairs"]
        assert wf["action_pairs"]["g_strategy"]["guard"] == "strategy_valid"
        assert "g_analysis" in wf["action_pairs"]["g_strategy"]["requires"]
        assert "g_recon" in wf["action_pairs"]["g_strategy"]["requires"]

        assert "g_plan_full" in wf["action_pairs"]
        assert wf["action_pairs"]["g_plan_full"]["guard"] == "plan_medium"
        requires = wf["action_pairs"]["g_plan_full"]["requires"]
        assert "g_analysis" in requires
        assert "g_recon" in requires
        assert "g_strategy" in requires

    def test_prompts_json_has_recon_template(self):
        """prompts.json should define g_recon template."""
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

        assert "g_recon" in prompts
        entry = prompts["g_recon"]
        assert "mentioned_files" in entry["constraints"]
        assert "stack_traces" in entry["constraints"]

    def test_prompts_json_has_strategy_template(self):
        """prompts.json should define g_strategy template."""
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

        assert "g_strategy" in prompts
        entry = prompts["g_strategy"]
        assert "S1_locate_and_fix" in entry["constraints"]
        assert "S2_tdd_feature" in entry["constraints"]


VALID_RECON = {
    "mentioned_files": ["src/auth/handler.py"],
    "stack_traces": ["TypeError: NoneType has no attribute 'login'"],
    "apis_involved": ["AuthHandler.login"],
    "test_references": ["test_login_success"],
    "reproduction_steps": ["Call login with None"],
    "constraints_mentioned": [],
}

VALID_STRATEGY = {
    "strategy_id": "S1_locate_and_fix",
    "strategy_name": "Locate and Fix Bug",
    "rationale": "Problem is a TypeError in auth module.",
    "key_steps": ["Locate defect", "Write characterization test", "Fix", "Verify"],
    "expected_guards": ["syntax", "dynamic_test"],
    "risk_factors": [],
}


class TestFullPipeline:
    """Integration tests for the full 4-step pipeline."""

    def test_full_context_chain(self):
        """Context.amend() chains correctly across all 4 steps."""
        base = Context(
            ambient=AmbientEnvironment(
                repository=InMemoryArtifactDAG(), constraints=""
            ),
            specification="Fix TypeError in auth handler.",
            current_artifact=None,
            feedback_history=(),
            dependency_artifacts=(),
        )

        # Step 1: analysis
        ctx1 = base.amend(
            delta_constraints=(
                f"## Problem Analysis (from g_analysis)\n"
                f"{json.dumps(VALID_ANALYSIS)}"
            )
        )
        assert "bug_fix" in ctx1.ambient.constraints

        # Step 2: recon
        ctx2 = ctx1.amend(
            delta_constraints=(
                f"## Codebase Reconnaissance (from g_recon)\n"
                f"{json.dumps(VALID_RECON)}"
            )
        )
        assert "bug_fix" in ctx2.ambient.constraints
        assert "handler.py" in ctx2.ambient.constraints

        # Step 3: strategy
        ctx3 = ctx2.amend(
            delta_constraints=(
                f"## Selected Strategy (from g_strategy)\n"
                f"{json.dumps(VALID_STRATEGY)}"
            )
        )
        assert "bug_fix" in ctx3.ambient.constraints
        assert "handler.py" in ctx3.ambient.constraints
        assert "S1_locate_and_fix" in ctx3.ambient.constraints

    def test_full_context_renders_in_prompt(self):
        """All 3 pre-step outputs appear in the rendered plan prompt."""
        template = PromptTemplate(
            role="Workflow planner",
            constraints="Design a DAG plan.",
            task="Generate a plan.",
        )

        base = Context(
            ambient=AmbientEnvironment(
                repository=InMemoryArtifactDAG(), constraints=""
            ),
            specification="Fix TypeError in auth handler.",
            current_artifact=None,
            feedback_history=(),
            dependency_artifacts=(),
        )

        ctx = base.amend(
            delta_constraints=(
                f"## Problem Analysis\n{json.dumps(VALID_ANALYSIS)}"
            )
        ).amend(
            delta_constraints=(
                f"## Codebase Reconnaissance\n{json.dumps(VALID_RECON)}"
            )
        ).amend(
            delta_constraints=(
                f"## Selected Strategy\n{json.dumps(VALID_STRATEGY)}"
            )
        )

        rendered = template.render(ctx)
        assert "Problem Analysis" in rendered
        assert "Codebase Reconnaissance" in rendered
        assert "Selected Strategy" in rendered
        assert "S1_locate_and_fix" in rendered

    def test_recon_guard_validates(self):
        """ReconGuard accepts valid recon JSON."""
        guard = ReconGuard()
        artifact = _make_artifact(
            json.dumps(VALID_RECON), "g_recon", ContextSnapshot(
                workflow_id="test", specification="test",
                constraints="", feedback_history=(),
            )
        )
        result = guard.validate(artifact)
        assert result.passed is True

    def test_strategy_guard_validates(self):
        """StrategyGuard accepts valid strategy JSON."""
        guard = StrategyGuard()
        artifact = _make_artifact(
            json.dumps(VALID_STRATEGY), "g_strategy", ContextSnapshot(
                workflow_id="test", specification="test",
                constraints="", feedback_history=(),
            )
        )
        result = guard.validate(artifact)
        assert result.passed is True

    def test_strategy_failure_blocks_plan(self):
        """Invalid strategy fails guard, blocking plan generation."""
        guard = StrategyGuard()
        bad = {"strategy_id": "S99_invalid"}
        artifact = _make_artifact(
            json.dumps(bad), "g_strategy", ContextSnapshot(
                workflow_id="test", specification="test",
                constraints="", feedback_history=(),
            )
        )
        result = guard.validate(artifact)
        assert result.passed is False
