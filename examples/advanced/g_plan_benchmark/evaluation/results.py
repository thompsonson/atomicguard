"""Structured result types for the evaluation harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExperimentConfig:
    """Configuration for an evaluation experiment.

    Attributes:
        pipelines: Pipeline modes to evaluate (subset of single, classify-then-plan, full).
        trials_per_problem: Number of trials per problem per pipeline.
        model: LLM model identifier.
        backend: LLM backend (ollama or huggingface).
        host: Ollama host URL (ollama backend only).
    """

    pipelines: list[str] = field(default_factory=lambda: ["single", "full"])
    trials_per_problem: int = 1
    model: str = "qwen2.5-coder:14b"
    backend: str = "ollama"
    host: str = "http://localhost:11434"

    def to_dict(self) -> dict[str, Any]:
        return {
            "pipelines": self.pipelines,
            "trials_per_problem": self.trials_per_problem,
            "model": self.model,
            "backend": self.backend,
            "host": self.host,
        }


@dataclass
class StepResult:
    """Result of a single pipeline step (analysis, recon, strategy, or plan)."""

    passed: bool
    content: str = ""
    feedback: str = ""
    time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "content": self.content,
            "feedback": self.feedback,
            "time_ms": self.time_ms,
        }


@dataclass
class PlanValidation:
    """G_plan guard validation results for a generated plan."""

    minimal_passed: bool = False
    medium_passed: bool = False
    expansive_passed: bool = False
    minimal_feedback: str = ""
    medium_feedback: str = ""
    expansive_feedback: str = ""
    plan_steps: int = 0
    plan_content: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "minimal_passed": self.minimal_passed,
            "medium_passed": self.medium_passed,
            "expansive_passed": self.expansive_passed,
            "minimal_feedback": self.minimal_feedback,
            "medium_feedback": self.medium_feedback,
            "expansive_feedback": self.expansive_feedback,
            "plan_steps": self.plan_steps,
            "plan_content": self.plan_content,
        }


@dataclass
class ProblemTrialResult:
    """Result of running one trial of one problem through one pipeline.

    A trial includes all pipeline steps plus plan validation.
    """

    problem_id: str
    pipeline: str
    trial_num: int
    total_time_ms: float = 0.0

    # Pre-step results (None = step not part of this pipeline)
    analysis: StepResult | None = None
    recon: StepResult | None = None
    strategy: StepResult | None = None

    # Plan generation and validation
    plan: PlanValidation = field(default_factory=PlanValidation)

    # Pipeline-level outcome
    pipeline_succeeded: bool = False  # all pre-steps passed AND plan passed Medium
    errors: list[str] = field(default_factory=list)

    # Strategy alignment (extracted from analysis/strategy outputs)
    classified_type: str = ""  # problem_type from g_analysis
    selected_strategy: str = ""  # strategy_id from g_strategy

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "problem_id": self.problem_id,
            "pipeline": self.pipeline,
            "trial_num": self.trial_num,
            "total_time_ms": self.total_time_ms,
            "pipeline_succeeded": self.pipeline_succeeded,
            "errors": self.errors,
            "plan": self.plan.to_dict(),
        }
        if self.analysis is not None:
            d["analysis"] = self.analysis.to_dict()
        if self.recon is not None:
            d["recon"] = self.recon.to_dict()
        if self.strategy is not None:
            d["strategy"] = self.strategy.to_dict()
        if self.classified_type:
            d["classified_type"] = self.classified_type
        if self.selected_strategy:
            d["selected_strategy"] = self.selected_strategy
        return d


@dataclass
class PipelineResult:
    """Aggregated results for one pipeline across all problems."""

    pipeline: str
    trials: list[ProblemTrialResult] = field(default_factory=list)

    @property
    def total_trials(self) -> int:
        return len(self.trials)

    @property
    def succeeded_trials(self) -> int:
        return sum(1 for t in self.trials if t.pipeline_succeeded)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pipeline": self.pipeline,
            "total_trials": self.total_trials,
            "succeeded_trials": self.succeeded_trials,
            "trials": [t.to_dict() for t in self.trials],
        }


@dataclass
class ExperimentResult:
    """Complete result of an evaluation experiment."""

    config: ExperimentConfig
    pipeline_results: dict[str, PipelineResult] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "pipelines": {
                name: pr.to_dict() for name, pr in self.pipeline_results.items()
            },
        }
