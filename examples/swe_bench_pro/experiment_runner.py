"""Experiment runner for SWE-Bench Pro.

Orchestrates running AtomicGuard workflow arms across SWE-Bench Pro
instances with language-aware generator and guard selection.
"""

import json
import logging
import os
import subprocess
import tempfile
import threading
import time
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from examples.swe_bench_common import (
    ArmResult,
    load_existing_results,
    topological_sort,
)
from examples.swe_bench_common import load_prompts as _load_prompts
from examples.swe_bench_common import load_workflow_config as _load_workflow_config

from atomicguard import ActionPair, CompositeGuard, Workflow
from atomicguard.domain.prompts import PromptTemplate
from atomicguard.infrastructure.persistence.filesystem import FilesystemArtifactDAG

from .dataset import SWEBenchProInstance, load_swe_bench_pro
from .generators import (
    AnalysisGenerator,
    ClassificationGenerator,
    ContextReadGenerator,
    DiffReviewGenerator,
    FixApproachGenerator,
    ImpactAnalysisGenerator,
    LocalizationGenerator,
    MultiLangPatchGenerator,
    MultiLangTestGenerator,
    PatchGenerator,
    RootCauseGenerator,
    StructureGenerator,
    TestGenerator,
    TestLocalizationGenerator,
    WorkflowGenerator,
)
from .guards import (
    AnalysisGuard,
    ClassificationGuard,
    ContextGuard,
    DiffReviewGuard,
    FixApproachGuard,
    FullEvalGuard,
    ImpactGuard,
    LintGuard,
    LocalizationGuard,
    MultiLangTestSyntaxGuard,
    PatchGuard,
    RootCauseGuard,
    StructureGuard,
    TestGreenGuard,
    TestLocalizationGuard,
    TestRedGuard,
    TestSetupVerificationGuard,
    TestSyntaxGuard,
    WorkflowGuard,
)
from .language import LanguageConfig, get_language_config

logger = logging.getLogger("swe_bench_pro.experiment")

# Workflow configs live in the common directory.
_COMMON_DIR = Path(__file__).parent.parent / "swe_bench_common"
_WORKFLOW_DIR = _COMMON_DIR / "workflows"

_SKIP_DIRS = {"__pycache__", ".git", "node_modules", "vendor", "venv", ".venv", ".tox"}


def _list_repo_files(
    repo_root: str,
    extensions: tuple[str, ...] = (".py",),
) -> list[str]:
    """Return source file paths relative to *repo_root*.

    Walks the repository tree, filtering by *extensions* and skipping
    common non-source directories. Returns all matching files sorted
    alphabetically.
    """
    root = Path(repo_root)
    found: list[str] = []
    for dirpath, dirnames, filenames in root.walk():
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for fname in sorted(filenames):
            if any(fname.endswith(ext) for ext in extensions):
                rel = (dirpath / fname).relative_to(root)
                found.append(str(rel))
    return found


# =========================================================================
# Workflow construction helpers
# =========================================================================


def load_workflow_config(variant: str) -> dict[str, Any]:
    """Load a workflow configuration JSON from the ablation workflows dir."""
    return _load_workflow_config(_WORKFLOW_DIR, variant)


def load_prompts() -> dict[str, PromptTemplate]:
    """Load prompt templates from this example's ``prompts.json``.

    Raises:
        FileNotFoundError: If ``prompts.json`` is missing.
    """
    prompts_file = Path(__file__).parent / "prompts.json"
    if not prompts_file.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_file}")
    return _load_prompts(prompts_file)


def _get_generator_registry(
    lang_config: LanguageConfig,
) -> dict[str, type]:
    """Build a generator class registry for a given language."""
    is_python = lang_config.name == "python"
    return {
        "AnalysisGenerator": AnalysisGenerator,
        "ClassificationGenerator": ClassificationGenerator,
        "ContextReadGenerator": ContextReadGenerator,
        "DiffReviewGenerator": DiffReviewGenerator,
        "FixApproachGenerator": FixApproachGenerator,
        "ImpactAnalysisGenerator": ImpactAnalysisGenerator,
        "LocalizationGenerator": LocalizationGenerator,
        "PatchGenerator": PatchGenerator if is_python else MultiLangPatchGenerator,
        "RootCauseGenerator": RootCauseGenerator,
        "StructureGenerator": StructureGenerator,
        "TestGenerator": TestGenerator if is_python else MultiLangTestGenerator,
        "TestLocalizationGenerator": TestLocalizationGenerator,
        "WorkflowGenerator": WorkflowGenerator,
    }


def _get_guard_registry(
    lang_config: LanguageConfig,
) -> dict[str, type]:
    """Build a guard class registry for a given language."""
    is_python = lang_config.name == "python"
    return {
        "analysis": AnalysisGuard,
        "classification_schema": ClassificationGuard,
        "context": ContextGuard,
        "fix_approach": FixApproachGuard,
        "impact": ImpactGuard,
        "lint": LintGuard,
        "localization": LocalizationGuard,
        "patch": PatchGuard,
        "review_schema": DiffReviewGuard,
        "root_cause": RootCauseGuard,
        "structure": StructureGuard,
        "test_localization": TestLocalizationGuard,
        "test_setup_verification": TestSetupVerificationGuard,
        "workflow_schema": WorkflowGuard,
        "test_syntax": TestSyntaxGuard if is_python else MultiLangTestSyntaxGuard,
        # TDD verification guards (Docker-based)
        "test_red": TestRedGuard,
        "test_green": TestGreenGuard,
        "full_eval": FullEvalGuard,
    }


# Guards that require instance context for Docker execution
_DOCKER_GUARDS = {TestRedGuard, TestGreenGuard, FullEvalGuard}


def build_workflow(
    config: dict[str, Any],
    prompts: dict[str, PromptTemplate],
    lang_config: LanguageConfig,
    model: str,
    base_url: str,
    artifact_dag: FilesystemArtifactDAG,
    repo_root: str | None = None,
    api_key: str = "ollama",
    provider: str = "ollama",
    instance: SWEBenchProInstance | None = None,
) -> Workflow:
    """Build a :class:`Workflow` with language-aware registries.

    Mirrors the ablation ``build_workflow`` but selects multi-language
    generator / guard subclasses when ``lang_config`` is not Python.

    Args:
        config: Workflow configuration dict from JSON.
        prompts: Prompt templates keyed by action pair ID.
        lang_config: Language configuration for the instance.
        model: LLM model identifier.
        base_url: API base URL.
        artifact_dag: DAG for artifact storage.
        repo_root: Path to the cloned repository.
        api_key: API key for LLM provider.
        provider: LLM provider name.
        instance: SWE-Bench Pro instance (required for Docker-based guards).

    Returns:
        Configured Workflow ready for execution.
    """
    generator_registry = _get_generator_registry(lang_config)
    guard_registry = _get_guard_registry(lang_config)

    rmax = config.get("rmax", 3)
    workflow = Workflow(artifact_dag=artifact_dag, rmax=rmax)

    action_pairs = config.get("action_pairs")
    if not action_pairs:
        raise ValueError("Workflow config has no 'action_pairs' section")
    sorted_pairs = topological_sort(action_pairs)

    def _build_guard(guard_name: str, guard_config: dict[str, Any]) -> Any:
        """Build a single guard instance with proper configuration."""
        if guard_name not in guard_registry:
            raise ValueError(f"Unknown guard: {guard_name}")
        guard_cls = guard_registry[guard_name]

        config = dict(guard_config)
        if repo_root:
            config["repo_root"] = repo_root
        if guard_cls is MultiLangTestSyntaxGuard:
            config["language_config"] = lang_config
        # Docker-based guards need the instance for image selection
        if guard_cls in _DOCKER_GUARDS:
            if instance is None:
                raise ValueError(
                    f"Guard '{guard_name}' requires instance context for Docker "
                    "execution, but no instance was provided."
                )
            config["instance"] = instance

        return guard_cls(**config)

    for ap_id in sorted_pairs:
        ap_config = action_pairs[ap_id]

        # ---- Generator ----
        gen_name = ap_config["generator"]

        if gen_name in generator_registry:
            gen_cls = generator_registry[gen_name]

            gen_kwargs: dict[str, Any] = {
                "model": model,
                "base_url": base_url,
                "api_key": api_key,
                "provider": provider,
            }
            # AnalysisGenerator needs repo_root for code-aware analysis.
            if repo_root and issubclass(gen_cls, AnalysisGenerator):
                gen_kwargs["repo_root"] = repo_root
            # Patch generators need repo_root to produce unified diffs.
            if repo_root and issubclass(gen_cls, PatchGenerator):
                gen_kwargs["repo_root"] = repo_root
                gen_kwargs["code_block_tag"] = lang_config.code_block_tag
            # Multi-language subclasses need the language config.
            if gen_cls in (MultiLangPatchGenerator, MultiLangTestGenerator):
                gen_kwargs["language_config"] = lang_config

            generator = gen_cls(**gen_kwargs)
        else:
            raise ValueError(f"Unknown generator: {gen_name}")

        # ---- Guard ----
        guard_name = ap_config["guard"]
        guard_config = dict(ap_config.get("guard_config", {}))

        # Check if this is a composite guard (explicit configuration)
        if guard_name == "composite":
            guard_names = ap_config.get("guards", [])
            if not guard_names:
                raise ValueError(f"Composite guard {ap_id} requires 'guards' array")

            sub_guards = [_build_guard(name, guard_config) for name in guard_names]
            guard = CompositeGuard(*sub_guards)
        else:
            guard = _build_guard(guard_name, guard_config)

        # ---- Prompt template ----
        template = prompts.get(ap_id)

        # ---- Assemble action pair ----
        action_pair = ActionPair(
            generator=generator,
            guard=guard,
            prompt_template=template,
        )
        requires = tuple(ap_config.get("requires", []))

        # Extension 09: Backtracking parameters
        r_patience = ap_config.get("r_patience")
        e_max = ap_config.get("e_max", 1)
        escalate_feedback_to = tuple(ap_config.get("escalate_feedback_to", []))

        # Extension 09: Guard-specific feedback routing
        raw_ebg = ap_config.get("escalate_feedback_by_guard")
        escalate_feedback_by_guard = (
            {k: tuple(v) for k, v in raw_ebg.items()} if raw_ebg else None
        )

        workflow.add_step(
            ap_id,
            action_pair,
            requires=requires,
            r_patience=r_patience,
            e_max=e_max,
            escalate_feedback_to=escalate_feedback_to,
            escalate_feedback_by_guard=escalate_feedback_by_guard,
        )

    return workflow


# =========================================================================
# Progress Tracking
# =========================================================================


def _format_duration(seconds: float) -> str:
    """Format duration in human-readable form."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.0f}m"
    else:
        hours = seconds / 3600
        remaining_minutes = (seconds % 3600) / 60
        if remaining_minutes > 0:
            return f"{hours:.0f}h {remaining_minutes:.0f}m"
        return f"{hours:.0f}h"


@dataclass
class ArmStats:
    """Per-arm statistics for progress tracking.

    IMPORTANT: The PRIMARY metric is `eval_resolved` (patches that pass
    SWE-Bench evaluation). Guard failures are tracked separately to show
    which guards are blocking progress.
    """

    eval_resolved: int = 0  # PRIMARY: Patches that pass evaluation
    eval_failed: int = 0  # Patches generated but failed evaluation (resolved=False)
    eval_pending: int = 0  # Awaiting evaluation (resolved is None)
    errors: int = 0  # Exceptions during execution
    # Guard failure tracking:
    failed_by_guard: dict[str, int] = field(default_factory=dict)  # {guard_name: count}
    total_retries: int = 0  # Total retry attempts across all runs
    total_tokens: int = 0
    total_wall_time: float = 0.0

    @property
    def total(self) -> int:
        """Total runs = resolved + failed + pending + errors + guard failures."""
        return (
            self.eval_resolved
            + self.eval_failed
            + self.eval_pending
            + self.errors
            + sum(self.failed_by_guard.values())
        )


@dataclass
class GuardFailureStats:
    """Guard failure tracking for warning detection."""

    total: int = 0
    failures: int = 0
    failure_reasons: dict[str, int] = field(default_factory=dict)


class ProgressTracker:
    """Thread-safe progress tracker for experiment runs."""

    def __init__(self, total_runs: int, log_interval: int = 10):
        """Initialize progress tracker.

        Args:
            total_runs: Total number of (instance, arm) pairs to run.
            log_interval: Log progress every N completed runs.
        """
        self._total_runs = total_runs
        self._log_interval = log_interval
        self._start_time = time.time()
        self._completed = 0
        self._lock = threading.Lock()

        # Per-arm statistics
        self._arm_stats: dict[str, ArmStats] = defaultdict(ArmStats)

        # Guard failure tracking (guard_name -> stats)
        self._guard_failures: dict[str, GuardFailureStats] = defaultdict(
            GuardFailureStats
        )

        # Last log time for time-based logging
        self._last_log_time = self._start_time
        self._time_log_interval = 300  # Log every 5 minutes regardless

    def record_result(self, arm: str, result: "ArmResult") -> int:
        """Record a completed run result.

        Args:
            arm: The workflow arm name.
            result: The ArmResult from the run.

        Returns:
            The current completed count (for logging).
        """
        with self._lock:
            self._completed += 1
            finished_count = self._completed
            stats = self._arm_stats[arm]
            stats.total_tokens += result.total_tokens
            stats.total_wall_time += result.wall_time_seconds
            stats.total_retries += result.retry_count

            # Track errors first
            if result.error:
                stats.errors += 1
            # Track guard failures (prefer failed_guard for specific guard name)
            elif result.failed_step:
                # Use guard name if available (e.g., "TestGreenGuard"), else step ID
                failure_key = result.failed_guard or result.failed_step
                stats.failed_by_guard[failure_key] = (
                    stats.failed_by_guard.get(failure_key, 0) + 1
                )
            # Track evaluation result (only if workflow succeeded)
            elif result.resolved is True:
                stats.eval_resolved += 1
            elif result.resolved is False:
                # Patch generated but failed evaluation
                stats.eval_failed += 1
            elif result.resolved is None:
                stats.eval_pending += 1

            # Check if we should log progress
            should_log = (
                self._completed % self._log_interval == 0
                or self._completed == self._total_runs
                or (time.time() - self._last_log_time) > self._time_log_interval
            )

            if should_log:
                self._log_progress()
                self._last_log_time = time.time()

        return finished_count

    def record_guard_failure(
        self, guard_name: str, success: bool, failure_reason: str = ""
    ) -> None:
        """Record a guard check result for failure tracking.

        Args:
            guard_name: Name of the guard.
            success: Whether the guard passed.
            failure_reason: Reason for failure if applicable.
        """
        with self._lock:
            stats = self._guard_failures[guard_name]
            stats.total += 1
            if not success:
                stats.failures += 1
                if failure_reason:
                    # Normalize and truncate the reason
                    reason = failure_reason[:100].strip()
                    stats.failure_reasons[reason] = (
                        stats.failure_reasons.get(reason, 0) + 1
                    )

    def _log_progress(self) -> None:
        """Log current progress (must hold lock)."""
        elapsed = time.time() - self._start_time
        remaining = self._total_runs - self._completed

        # Calculate rate and ETA
        if self._completed > 0:
            rate = self._completed / elapsed * 60  # runs per minute
            eta_seconds = remaining / (self._completed / elapsed)
            eta_str = _format_duration(eta_seconds)
        else:
            rate = 0.0
            eta_str = "calculating..."

        pct = self._completed / self._total_runs * 100

        # Main progress line
        logger.info(
            "Progress: %d/%d runs completed (%.1f%%)",
            self._completed,
            self._total_runs,
            pct,
        )
        logger.info(
            "Elapsed: %s | ETA: %s | Rate: %.1f runs/min",
            _format_duration(elapsed),
            eta_str,
            rate,
        )

        # Per-arm stats with guard failure breakdown
        if self._arm_stats:
            logger.info("Results by arm:")
            for arm in sorted(self._arm_stats.keys()):
                stats = self._arm_stats[arm]
                if stats.total > 0:
                    # Guard failure summary
                    guard_failures = ", ".join(
                        f"{g}:{c}" for g, c in sorted(stats.failed_by_guard.items())
                    )
                    logger.info(
                        "  %s: %d/%d RESOLVED | %d retries | failures: %s",
                        arm,
                        stats.eval_resolved,
                        stats.total,
                        stats.total_retries,
                        guard_failures or "none",
                    )

        # Guard failure warnings (100% fail rate = potential issue)
        high_fail_guards = []
        for guard_name, gstats in self._guard_failures.items():
            if gstats.total >= 10 and gstats.failures / gstats.total > 0.95:
                high_fail_guards.append((guard_name, gstats))

        if high_fail_guards:
            logger.warning("Guards with high failure rate (potential issues):")
            for guard_name, gstats in high_fail_guards:
                fail_pct = gstats.failures / gstats.total * 100
                logger.warning(
                    "  %s: %d/%d failures (%.0f%% fail rate)",
                    guard_name,
                    gstats.failures,
                    gstats.total,
                    fail_pct,
                )
                # Show top failure reasons
                if gstats.failure_reasons:
                    top_reasons = sorted(
                        gstats.failure_reasons.items(), key=lambda x: -x[1]
                    )[:3]
                    for reason, count in top_reasons:
                        logger.warning("    - %s (%d occurrences)", reason, count)

    def get_summary(self) -> dict[str, Any]:
        """Get final summary statistics.

        Returns:
            Dictionary with summary stats for each arm. The PRIMARY metric
            is `eval_resolved` (patches passing SWE-Bench evaluation).
        """
        with self._lock:
            elapsed = time.time() - self._start_time
            summary = {
                "total_runs": self._total_runs,
                "completed_runs": self._completed,
                "elapsed_seconds": round(elapsed, 2),
                "arms": {},
            }
            for arm, stats in self._arm_stats.items():
                summary["arms"][arm] = {
                    "total": stats.total,
                    # PRIMARY METRIC: patches that pass SWE-Bench evaluation
                    "eval_resolved": stats.eval_resolved,
                    "resolve_rate": (
                        round(stats.eval_resolved / stats.total * 100, 1)
                        if stats.total > 0
                        else 0.0
                    ),
                    # Evaluation failures (patch generated but failed eval)
                    "eval_failed": stats.eval_failed,
                    # Guard failures
                    "failed_by_guard": dict(stats.failed_by_guard),
                    "total_retries": stats.total_retries,
                    "errors": stats.errors,
                    "eval_pending": stats.eval_pending,
                    "total_tokens": stats.total_tokens,
                    "avg_wall_time": (
                        round(stats.total_wall_time / stats.total, 1)
                        if stats.total > 0
                        else 0.0
                    ),
                }
            return summary


# =========================================================================
# Runner
# =========================================================================


class SWEBenchProRunner:
    """Runs workflow arms across SWE-Bench Pro instances."""

    def __init__(
        self,
        model: str = "moonshotai/kimi-k2-0905",
        provider: str = "ollama",
        base_url: str = "",
        api_key: str | None = None,
        output_dir: str = "output/swe_bench_pro",
        clone_dir: str | None = None,
    ):
        self._model = model
        self._provider = provider
        self._base_url = base_url
        self._api_key = api_key or os.environ.get("LLM_API_KEY", "")
        self._output_dir = Path(output_dir)
        self._clone_dir = Path(clone_dir) if clone_dir else None

        if not self._api_key:
            logger.warning("LLM_API_KEY not set. API calls will fail.")

    # -----------------------------------------------------------------
    # Single instance
    # -----------------------------------------------------------------

    def run_instance(
        self,
        instance: SWEBenchProInstance,
        arm: str,
    ) -> ArmResult:
        """Run one workflow arm on one instance."""
        logger.info("Running arm=%s on instance=%s", arm, instance.instance_id)
        start_time = time.time()

        try:
            # Track init phase (repo clone + checkout + workflow setup)
            init_start = time.time()

            repo_root = self._prepare_repo(instance)
            lang_config = get_language_config(instance.repo_language)

            config = load_workflow_config(arm)
            prompts = load_prompts()

            dag_dir = self._output_dir / "artifact_dags" / instance.instance_id / arm
            dag_dir.mkdir(parents=True, exist_ok=True)
            artifact_dag = FilesystemArtifactDAG(str(dag_dir))

            workflow = build_workflow(
                config=config,
                prompts=prompts,
                lang_config=lang_config,
                model=self._model,
                base_url=self._base_url,
                artifact_dag=artifact_dag,
                repo_root=repo_root,
                api_key=self._api_key,
                provider=self._provider,
                instance=instance,
            )

            repo_files = _list_repo_files(
                repo_root, extensions=lang_config.file_extensions
            )
            specification = instance.problem_statement

            # Add requirements if present (detailed specs for the solution)
            if instance.requirements:
                specification += f"\n\n## Requirements\n{instance.requirements}"

            # Add interface if present (new interfaces to introduce)
            if instance.interface:
                specification += f"\n\n## Interface\n{instance.interface}"

            if repo_files:
                listing = "\n".join(repo_files)
                specification += (
                    f"\n\n## Repository Structure\n"
                    f"Source files in the repository (use these exact paths):\n"
                    f"```\n{listing}\n```"
                )

            # Add test infrastructure context for TDD workflows
            # This helps the LLM generate tests that follow project patterns
            test_infra = self._get_test_infrastructure(repo_root, lang_config)
            if test_infra:
                specification += f"\n\n## Test Infrastructure\n{test_infra}"

            init_time = time.time() - init_start

            # Track workflow phase (action pair execution)
            workflow_start = time.time()

            result = workflow.execute(specification)

            workflow_time = time.time() - workflow_start
            wall_time = time.time() - start_time

            patch_content = ""
            total_tokens = 0
            per_step_tokens: dict[str, int] = {}

            for step_id, artifact in result.artifacts.items():
                step_tokens = artifact.metadata.get("total_tokens", 0)
                if step_tokens:
                    per_step_tokens[step_id] = step_tokens
                    total_tokens += step_tokens

                if any(
                    x in step_id
                    for x in ("patch", "fix", "singleshot", "eval", "verify_green")
                ):
                    try:
                        data = json.loads(artifact.content)
                        patch_content = data.get("patch", "")
                        if "patch" not in data:
                            logger.warning(
                                "Artifact %s for %s has no 'patch' key (keys: %s)",
                                step_id,
                                instance.instance_id,
                                ", ".join(data.keys()),
                            )
                        elif not patch_content:
                            logger.warning(
                                "Artifact %s for %s has empty 'patch' "
                                "(files not found in repo — LLM likely used "
                                "wrong file paths in edits)",
                                step_id,
                                instance.instance_id,
                            )
                    except (json.JSONDecodeError, TypeError) as e:
                        logger.warning(
                            "Artifact %s for %s is not valid JSON (%s), "
                            "using raw content as patch",
                            step_id,
                            instance.instance_id,
                            e,
                        )
                        patch_content = artifact.content

            # Count tokens from failed attempts (provenance) — these
            # are not in result.artifacts because the step didn't pass.
            # Also extract the guard name from the last failed attempt.
            failed_guard = None
            if result.provenance:
                failed_step = result.failed_step or "unknown"
                for artifact, _feedback in result.provenance:
                    prov_tokens = artifact.metadata.get("total_tokens", 0)
                    if prov_tokens:
                        per_step_tokens[failed_step] = (
                            per_step_tokens.get(failed_step, 0) + prov_tokens
                        )
                        total_tokens += prov_tokens
                # Extract guard name from last failed attempt
                last_artifact, _ = result.provenance[-1]
                if last_artifact.guard_result and last_artifact.guard_result.guard_name:
                    failed_guard = last_artifact.guard_result.guard_name

            return ArmResult(
                instance_id=instance.instance_id,
                arm=arm,
                patch_content=patch_content,
                total_tokens=total_tokens,
                per_step_tokens=per_step_tokens,
                wall_time_seconds=round(wall_time, 2),
                init_time_seconds=round(init_time, 2),
                workflow_time_seconds=round(workflow_time, 2),
                failed_step=result.failed_step,
                failed_guard=failed_guard,
                retry_count=len(result.provenance),
            )

        except Exception as e:
            wall_time = time.time() - start_time
            tb = traceback.format_exc()
            logger.error(
                "Error running arm=%s on instance=%s: %s\n%s",
                arm,
                instance.instance_id,
                e,
                tb,
            )
            return ArmResult(
                instance_id=instance.instance_id,
                arm=arm,
                wall_time_seconds=round(wall_time, 2),
                error=f"{e}\n{tb}",
            )

    # -----------------------------------------------------------------
    # Full experiment
    # -----------------------------------------------------------------

    def run_all(
        self,
        arms: list[str],
        split: str = "test",
        language: str | None = None,
        max_instances: int | None = None,
        resume_from: str | None = None,
        max_workers: int = 1,
        instance_filter: list[str] | None = None,
    ) -> list[ArmResult]:
        """Run all arms across all matching instances.

        Args:
            arms: Workflow variant names.
            split: Dataset split.
            language: Optional language filter (``None`` = all).
            max_instances: Cap on instances.
            resume_from: Path to existing results dir for resume.
            max_workers: Number of parallel workers.  ``1`` (default)
                runs sequentially.  Values > 1 use a thread pool.
            instance_filter: List of instance ID substrings to include.
                An instance is included if any filter matches. ``None`` = all.

        Returns:
            List of :class:`ArmResult` objects.
        """
        if max_workers < 1:
            raise ValueError(f"max_workers must be >= 1, got {max_workers}")

        instances = load_swe_bench_pro(
            split=split,
            language=language,
            max_instances=max_instances,
            instance_filter=instance_filter,
        )

        # Build the work items list, filtering out already-completed runs.
        completed: set[tuple[str, str]] = set()
        results: list[ArmResult] = []

        if resume_from:
            results, completed = load_existing_results(resume_from)
            logger.info("Resuming: %d runs already completed", len(completed))

        work_items: list[tuple[int, SWEBenchProInstance, str]] = []
        for i, instance in enumerate(instances):
            for arm in arms:
                key = (instance.instance_id, arm)
                if key in completed:
                    logger.info(
                        "Skipping %s / %s (already done)",
                        instance.instance_id,
                        arm,
                    )
                    continue
                work_items.append((i, instance, arm))

        total_runs = len(work_items)
        logger.info(
            "Running %d work items (%d arms x %d instances, %d already done) "
            "with max_workers=%d",
            total_runs,
            len(arms),
            len(instances),
            len(completed),
            max_workers,
        )

        if total_runs == 0:
            logger.info("Nothing to run — all items already completed")
            return results

        self._output_dir.mkdir(parents=True, exist_ok=True)
        results_path = self._output_dir / "results.jsonl"
        write_lock = threading.Lock()

        # Initialize progress tracker
        progress = ProgressTracker(total_runs=total_runs, log_interval=10)

        def _execute_and_record(
            idx: int,
            instance: SWEBenchProInstance,
            arm: str,
        ) -> ArmResult:
            logger.info(
                "Starting: instance=%s, arm=%s, lang=%s (%d/%d instances)",
                instance.instance_id,
                arm,
                instance.repo_language,
                idx + 1,
                len(instances),
            )
            arm_result = self.run_instance(instance, arm)

            with write_lock, open(results_path, "a") as f:
                f.write(json.dumps(asdict(arm_result)) + "\n")

            # Record result for progress tracking and get finished count
            finished_count = progress.record_result(arm, arm_result)

            # Determine status string for logging (include guard name if available)
            if arm_result.error:
                status = "error"
            elif arm_result.failed_step:
                guard_info = arm_result.failed_guard or arm_result.failed_step
                status = f"failed:{guard_info}"
            else:
                status = "success"

            logger.info(
                "Finished %d/%d: instance=%s, arm=%s, status=%s (%.1fs)",
                finished_count,
                total_runs,
                instance.instance_id,
                arm,
                status,
                arm_result.wall_time_seconds,
            )

            return arm_result

        if max_workers == 1:
            # Sequential execution — no thread overhead.
            for idx, instance, arm in work_items:
                results.append(_execute_and_record(idx, instance, arm))
        else:
            # Parallel execution with thread pool.
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {
                    pool.submit(_execute_and_record, idx, instance, arm): (
                        instance.instance_id,
                        arm,
                    )
                    for idx, instance, arm in work_items
                }
                for future in as_completed(futures):
                    iid, arm = futures[future]
                    try:
                        arm_result = future.result()
                        results.append(arm_result)
                    except Exception:
                        # run_instance already catches exceptions and returns
                        # an error ArmResult, so this should not happen.
                        # Log it defensively.
                        tb = traceback.format_exc()
                        logger.error(
                            "Unexpected error in worker for %s / %s:\n%s",
                            iid,
                            arm,
                            tb,
                        )

        # Log final summary
        summary = progress.get_summary()
        logger.info(
            "Experiment complete. %d results written to %s",
            len(results),
            results_path,
        )
        logger.info(
            "Total elapsed: %s",
            _format_duration(summary["elapsed_seconds"]),
        )

        # Log per-arm summary
        if summary["arms"]:
            logger.info("Final results by arm:")
            for arm_name in sorted(summary["arms"].keys()):
                arm_data = summary["arms"][arm_name]
                guard_failures = ", ".join(
                    f"{g}:{c}"
                    for g, c in sorted(arm_data.get("failed_by_guard", {}).items())
                )
                logger.info(
                    "  %s: %d/%d RESOLVED | %d retries | failures: %s",
                    arm_name,
                    arm_data["eval_resolved"],
                    arm_data["total"],
                    arm_data["total_retries"],
                    guard_failures or "none",
                )

        # Save summary to file
        summary_path = self._output_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Summary written to %s", summary_path)

        return results

    # -----------------------------------------------------------------
    # Repo preparation
    # -----------------------------------------------------------------

    def _prepare_repo(self, instance: SWEBenchProInstance) -> str:
        """Clone the repository and check out ``base_commit``.

        Raises:
            RuntimeError: If any git operation fails, with details about
                the repo, commit, and stderr from the failed command.
        """
        if self._clone_dir:
            base_dir = self._clone_dir
            base_dir.mkdir(parents=True, exist_ok=True)
        else:
            base_dir = Path(tempfile.mkdtemp(prefix="swe_pro_"))

        repo_dir = base_dir / instance.instance_id.replace("/", "__")
        repo_url = f"https://github.com/{instance.repo}.git"

        def _run_git(
            args: list[str], *, timeout: int = 60, capture: bool = False
        ) -> str:
            """Run a git command, raising *RuntimeError* on failure.

            Args:
                args: Git command and arguments.
                timeout: Maximum seconds to wait.
                capture: If True, return stdout instead of None.

            Returns:
                stdout if capture=True, else empty string.
            """
            try:
                result = subprocess.run(
                    args,
                    cwd=str(repo_dir) if repo_dir.exists() else None,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    check=True,
                )
                return result.stdout.strip() if capture else ""
            except subprocess.CalledProcessError as e:
                raise RuntimeError(
                    f"Git command failed for {instance.instance_id} "
                    f"(repo={instance.repo}, commit={instance.base_commit}): "
                    f"{' '.join(args)}\nstderr: {e.stderr}"
                ) from e
            except subprocess.TimeoutExpired as e:
                raise RuntimeError(
                    f"Git command timed out after {timeout}s for "
                    f"{instance.instance_id}: {' '.join(args)}"
                ) from e

        if repo_dir.exists():
            logger.info("Resetting existing repo: %s", repo_dir)
            # Fetch the target commit in case it's missing from shallow history
            _run_git(
                ["git", "fetch", "--depth=1", "origin", instance.base_commit],
                timeout=120,
            )
            _run_git(["git", "checkout", "-f", instance.base_commit])
            _run_git(["git", "clean", "-fdx"])
        else:
            logger.info("Cloning %s to %s", repo_url, repo_dir)
            _run_git(
                ["git", "clone", "--depth=1", repo_url, str(repo_dir)],
                timeout=300,
            )
            _run_git(
                ["git", "fetch", "--depth=1", "origin", instance.base_commit],
                timeout=120,
            )
            _run_git(["git", "checkout", instance.base_commit])

        # Verify we're at the expected commit
        actual_commit = _run_git(["git", "rev-parse", "HEAD"], capture=True)
        expected_short = instance.base_commit[:12]
        actual_short = actual_commit[:12]

        if not actual_commit.startswith(instance.base_commit[:7]):
            raise RuntimeError(
                f"Commit mismatch for {instance.instance_id}: "
                f"expected {expected_short}, got {actual_short}"
            )

        logger.info(
            "Repo ready at commit %s for %s",
            actual_short,
            instance.instance_id,
        )
        return str(repo_dir)

    # -----------------------------------------------------------------
    # Test infrastructure context
    # -----------------------------------------------------------------

    def _get_test_infrastructure(
        self,
        repo_root: str,
        lang_config: LanguageConfig,
        analysis_files: list[str] | None = None,
    ) -> str:
        """Extract test infrastructure context from the project.

        Reads conftest.py and sample test files to help the LLM understand
        the project's test patterns (fixtures, imports, initialization).

        Args:
            repo_root: Path to the cloned repository.
            lang_config: Language configuration for file extensions.
            analysis_files: Files identified in analysis (to find relevant tests).

        Returns:
            Formatted string with test infrastructure context, or empty string.
        """
        root = Path(repo_root)
        parts: list[str] = []
        max_content_size = 3000  # Limit per file to avoid huge prompts

        # Common conftest.py locations
        conftest_paths = [
            "tests/conftest.py",
            "test/conftest.py",
            "conftest.py",
            "tests/unit/conftest.py",
        ]

        # Find and read conftest.py
        for conftest_path in conftest_paths:
            full_path = root / conftest_path
            if full_path.exists():
                try:
                    content = full_path.read_text(errors="replace")
                    if len(content) > max_content_size:
                        content = content[:max_content_size] + "\n# ... (truncated)"
                    parts.append(
                        f"### Test Configuration ({conftest_path})\n"
                        f"This file defines pytest fixtures and test setup.\n"
                        f"```python\n{content}\n```"
                    )
                    break  # Use first found
                except Exception:
                    pass

        # Find a sample test file to show patterns
        # Priority: tests related to analysis files, then any test file
        sample_test = self._find_sample_test(root, lang_config, analysis_files)
        if sample_test:
            rel_path = sample_test.relative_to(root)
            try:
                content = sample_test.read_text(errors="replace")
                if len(content) > max_content_size:
                    content = content[:max_content_size] + "\n# ... (truncated)"
                parts.append(
                    f"### Sample Test Pattern ({rel_path})\n"
                    f"Follow this file's import and fixture patterns.\n"
                    f"```python\n{content}\n```"
                )
            except Exception:
                pass

        if not parts:
            return ""

        header = (
            "The project has existing test infrastructure. "
            "Your generated test MUST follow these patterns to run correctly."
        )
        return header + "\n\n" + "\n\n".join(parts)

    def _find_sample_test(
        self,
        repo_root: Path,
        lang_config: LanguageConfig,
        analysis_files: list[str] | None = None,
    ) -> Path | None:
        """Find a representative test file from the project.

        Args:
            repo_root: Path to the repository root.
            lang_config: Language configuration.
            analysis_files: Files from analysis to find related tests.

        Returns:
            Path to a sample test file, or None if not found.
        """
        test_dirs = ["tests", "test", "tests/unit", "test/unit"]

        # If we have analysis files, try to find tests for those modules
        if analysis_files:
            for src_file in analysis_files:
                # Convert source file to potential test file
                # e.g., "qutebrowser/utils/qtlog.py" -> "tests/unit/utils/test_qtlog.py"
                src_path = Path(src_file)
                base_name = src_path.stem

                for test_dir in test_dirs:
                    # Try various test file patterns
                    patterns = [
                        f"{test_dir}/**/test_{base_name}.py",
                        f"{test_dir}/**/{base_name}_test.py",
                        f"{test_dir}/**/test_{base_name}*.py",
                    ]
                    for pattern in patterns:
                        matches = list(repo_root.glob(pattern))
                        if matches:
                            return matches[0]

        # Fallback: find any test file
        for test_dir in test_dirs:
            test_path = repo_root / test_dir
            if test_path.exists():
                for ext in lang_config.file_extensions:
                    # Look for test_*.py or *_test.py patterns
                    patterns = [f"**/test_*{ext}", f"**/*_test{ext}"]
                    for pattern in patterns:
                        matches = list(test_path.glob(pattern))
                        if matches:
                            # Return first non-trivial test file
                            for match in matches[:5]:
                                try:
                                    if match.stat().st_size > 200:
                                        return match
                                except Exception:
                                    pass
        return None
