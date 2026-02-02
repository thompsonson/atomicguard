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
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

from atomicguard import ActionPair, Workflow
from atomicguard.domain.prompts import PromptTemplate
from atomicguard.infrastructure.persistence.filesystem import FilesystemArtifactDAG
from examples.swe_bench_ablation.experiment_runner import ArmResult

from .dataset import SWEBenchProInstance, load_swe_bench_pro
from .generators import (
    AnalysisGenerator,
    LocalizationGenerator,
    MultiLangPatchGenerator,
    MultiLangTestGenerator,
    PatchGenerator,
    TestGenerator,
)
from .guards import (
    AnalysisGuard,
    LocalizationGuard,
    MultiLangTestSyntaxGuard,
    PatchGuard,
    TestSyntaxGuard,
)
from .language import LanguageConfig, get_language_config

logger = logging.getLogger("swe_bench_pro.experiment")

# Workflow configs live in the ablation example directory.
_ABLATION_DIR = Path(__file__).parent.parent / "swe_bench_ablation"
_WORKFLOW_DIR = _ABLATION_DIR / "workflows"


# =========================================================================
# Workflow construction helpers
# =========================================================================


def load_workflow_config(variant: str) -> dict[str, Any]:
    """Load a workflow configuration JSON from the ablation workflows dir."""
    workflow_file = _WORKFLOW_DIR / f"{variant}.json"
    if not workflow_file.exists():
        raise FileNotFoundError(f"Workflow not found: {workflow_file}")
    return json.loads(workflow_file.read_text())


def load_prompts() -> dict[str, PromptTemplate]:
    """Load prompt templates from this example's ``prompts.json``.

    Raises:
        FileNotFoundError: If ``prompts.json`` is missing.
    """
    prompts_file = Path(__file__).parent / "prompts.json"
    if not prompts_file.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_file}")
    data = json.loads(prompts_file.read_text())
    return {
        key: PromptTemplate(
            role=val.get("role", ""),
            constraints=val.get("constraints", ""),
            task=val.get("task", ""),
            feedback_wrapper=val.get("feedback_wrapper", "Feedback: {feedback}"),
        )
        for key, val in data.items()
    }


def _topological_sort(action_pairs: dict[str, Any]) -> list[str]:
    """Sort action pairs by their ``requires`` dependencies."""
    result: list[str] = []
    visited: set[str] = set()

    def visit(ap_id: str) -> None:
        if ap_id in visited:
            return
        visited.add(ap_id)
        for dep in action_pairs.get(ap_id, {}).get("requires", []):
            visit(dep)
        result.append(ap_id)

    for ap_id in action_pairs:
        visit(ap_id)
    return result


def _get_generator_registry(
    lang_config: LanguageConfig,
) -> dict[str, type]:
    """Build a generator class registry for a given language."""
    is_python = lang_config.name == "python"
    return {
        "AnalysisGenerator": AnalysisGenerator,
        "LocalizationGenerator": LocalizationGenerator,
        "PatchGenerator": PatchGenerator if is_python else MultiLangPatchGenerator,
        "TestGenerator": TestGenerator if is_python else MultiLangTestGenerator,
    }


def _get_guard_registry(
    lang_config: LanguageConfig,
) -> dict[str, type]:
    """Build a guard class registry for a given language."""
    is_python = lang_config.name == "python"
    return {
        "analysis": AnalysisGuard,
        "localization": LocalizationGuard,
        "patch": PatchGuard,
        "test_syntax": TestSyntaxGuard if is_python else MultiLangTestSyntaxGuard,
    }


def build_workflow(
    config: dict[str, Any],
    prompts: dict[str, PromptTemplate],
    lang_config: LanguageConfig,
    model: str,
    base_url: str,
    artifact_dag: FilesystemArtifactDAG,
    repo_root: str | None = None,
    api_key: str = "ollama",
) -> Workflow:
    """Build a :class:`Workflow` with language-aware registries.

    Mirrors the ablation ``build_workflow`` but selects multi-language
    generator / guard subclasses when ``lang_config`` is not Python.
    """
    generator_registry = _get_generator_registry(lang_config)
    guard_registry = _get_guard_registry(lang_config)

    rmax = config.get("rmax", 3)
    workflow = Workflow(artifact_dag=artifact_dag, rmax=rmax)

    action_pairs = config.get("action_pairs")
    if not action_pairs:
        raise ValueError("Workflow config has no 'action_pairs' section")
    sorted_pairs = _topological_sort(action_pairs)

    for ap_id in sorted_pairs:
        ap_config = action_pairs[ap_id]

        # ---- Generator ----
        gen_name = ap_config["generator"]
        if gen_name not in generator_registry:
            raise ValueError(f"Unknown generator: {gen_name}")
        gen_cls = generator_registry[gen_name]

        gen_kwargs: dict[str, Any] = {
            "model": model,
            "base_url": base_url,
            "api_key": api_key,
        }
        # Multi-language subclasses need the language config.
        if gen_cls in (MultiLangPatchGenerator, MultiLangTestGenerator):
            gen_kwargs["language_config"] = lang_config

        generator = gen_cls(**gen_kwargs)

        # ---- Guard ----
        guard_name = ap_config["guard"]
        if guard_name not in guard_registry:
            raise ValueError(f"Unknown guard: {guard_name}")
        guard_cls = guard_registry[guard_name]

        guard_config = dict(ap_config.get("guard_config", {}))
        if repo_root:
            guard_config["repo_root"] = repo_root
        if guard_cls is MultiLangTestSyntaxGuard:
            guard_config["language_config"] = lang_config

        guard = guard_cls(**guard_config)

        # ---- Prompt template ----
        template = prompts.get(ap_id)

        # ---- Assemble action pair ----
        action_pair = ActionPair(
            generator=generator,
            guard=guard,
            prompt_template=template,
        )
        requires = tuple(ap_config.get("requires", []))
        workflow.add_step(ap_id, action_pair, requires=requires)

    return workflow


# =========================================================================
# Runner
# =========================================================================


class SWEBenchProRunner:
    """Runs workflow arms across SWE-Bench Pro instances."""

    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-Coder-32B-Instruct",
        base_url: str = "https://router.huggingface.co/v1",
        api_key: str | None = None,
        output_dir: str = "output/swe_bench_pro",
        clone_dir: str | None = None,
    ):
        self._model = model
        self._base_url = base_url
        self._api_key = api_key or os.environ.get("HF_TOKEN", "")
        self._output_dir = Path(output_dir)
        self._clone_dir = Path(clone_dir) if clone_dir else None

        if not self._api_key:
            logger.warning("HF_TOKEN not set. API calls will fail.")

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
            repo_root = self._prepare_repo(instance)
            lang_config = get_language_config(instance.repo_language)

            config = load_workflow_config(arm)
            prompts = load_prompts()

            dag_dir = (
                self._output_dir / "artifact_dags" / instance.instance_id / arm
            )
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
            )

            result = workflow.execute(instance.problem_statement)
            wall_time = time.time() - start_time

            patch_content = ""
            total_tokens = 0
            per_step_tokens: dict[str, int] = {}

            for step_id, artifact in result.artifacts.items():
                step_tokens = artifact.metadata.get("total_tokens", 0)
                if step_tokens:
                    per_step_tokens[step_id] = step_tokens
                    total_tokens += step_tokens

                if "patch" in step_id or "fix" in step_id or "singleshot" in step_id:
                    try:
                        data = json.loads(artifact.content)
                        patch_content = data.get("patch", "")
                        if not patch_content:
                            logger.warning(
                                "Artifact %s for %s parsed as JSON but has no 'patch' key",
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

            return ArmResult(
                instance_id=instance.instance_id,
                arm=arm,
                workflow_status=result.status.value,
                patch_content=patch_content,
                total_tokens=total_tokens,
                per_step_tokens=per_step_tokens,
                wall_time_seconds=round(wall_time, 2),
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
                workflow_status="error",
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

        Returns:
            List of :class:`ArmResult` objects.
        """
        if max_workers < 1:
            raise ValueError(f"max_workers must be >= 1, got {max_workers}")

        instances = load_swe_bench_pro(
            split=split,
            language=language,
            max_instances=max_instances,
        )

        # Build the work items list, filtering out already-completed runs.
        completed: set[tuple[str, str]] = set()
        results: list[ArmResult] = []

        if resume_from:
            results, completed = self._load_existing_results(resume_from)
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
        finished_count = 0
        finished_lock = threading.Lock()

        def _execute_and_record(
            idx: int,
            instance: SWEBenchProInstance,
            arm: str,
        ) -> ArmResult:
            nonlocal finished_count
            logger.info(
                "Starting: instance=%s, arm=%s, lang=%s (%d/%d instances)",
                instance.instance_id,
                arm,
                instance.repo_language,
                idx + 1,
                len(instances),
            )
            arm_result = self.run_instance(instance, arm)

            with write_lock:
                with open(results_path, "a") as f:
                    f.write(json.dumps(asdict(arm_result)) + "\n")

            with finished_lock:
                finished_count += 1
                logger.info(
                    "Finished %d/%d: instance=%s, arm=%s, status=%s (%.1fs)",
                    finished_count,
                    total_runs,
                    instance.instance_id,
                    arm,
                    arm_result.workflow_status,
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

        logger.info(
            "Experiment complete. %d results written to %s",
            len(results),
            results_path,
        )
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

        def _run_git(args: list[str], *, timeout: int = 60) -> None:
            """Run a git command, raising *RuntimeError* on failure."""
            try:
                subprocess.run(
                    args,
                    cwd=str(repo_dir) if repo_dir.exists() else None,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    check=True,
                )
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

        return str(repo_dir)

    # -----------------------------------------------------------------
    # Resume helpers
    # -----------------------------------------------------------------

    def _load_existing_results(
        self,
        results_dir: str,
    ) -> tuple[list[ArmResult], set[tuple[str, str]]]:
        results_path = Path(results_dir) / "results.jsonl"
        results: list[ArmResult] = []
        completed: set[tuple[str, str]] = set()

        if not results_path.exists():
            return results, completed

        with open(results_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    arm_result = ArmResult(**data)
                    results.append(arm_result)
                    completed.add((arm_result.instance_id, arm_result.arm))
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning("Skipping malformed result line: %s", e)

        return results, completed
