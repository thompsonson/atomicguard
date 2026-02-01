"""Experiment runner for ISMIS 2026 Experiment 7.2.

Orchestrates running all three arms (single-shot, S1-direct, S1-TDD)
across SWE-PolyBench instances with incremental result persistence.
"""

import json
import logging
import os
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from atomicguard.infrastructure.persistence.filesystem import FilesystemArtifactDAG

from .dataset import SWEInstance, load_swe_polybench
from .demo import build_workflow, load_prompts, load_workflow_config

logger = logging.getLogger("swe_bench_ablation.experiment")


@dataclass
class ArmResult:
    """Result of running one arm on one instance."""

    instance_id: str
    arm: str
    workflow_status: str
    patch_content: str = ""
    total_tokens: int = 0
    per_step_tokens: dict[str, int] = field(default_factory=dict)
    wall_time_seconds: float = 0.0
    error: str | None = None


class ExperimentRunner:
    """Runs Experiment 7.2 across instances and arms."""

    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-Coder-32B-Instruct",
        base_url: str = "https://router.huggingface.co/v1",
        api_key: str | None = None,
        output_dir: str = "output/experiment_7_2",
        clone_dir: str | None = None,
    ):
        """Initialize the experiment runner.

        Args:
            model: HuggingFace model ID
            base_url: HuggingFace Inference API URL
            api_key: HuggingFace API token (defaults to HF_TOKEN env var)
            output_dir: Directory for results output
            clone_dir: Directory for cloning repos (defaults to temp dir)
        """
        self._model = model
        self._base_url = base_url
        self._api_key = api_key or os.environ.get("HF_TOKEN", "")
        self._output_dir = Path(output_dir)
        self._clone_dir = Path(clone_dir) if clone_dir else None

        if not self._api_key:
            logger.warning("HF_TOKEN not set. API calls will fail.")

    def run_instance(
        self,
        instance: SWEInstance,
        arm: str,
    ) -> ArmResult:
        """Run one arm on one instance.

        Args:
            instance: SWE-PolyBench instance
            arm: Workflow variant name (e.g., "02_singleshot")

        Returns:
            ArmResult with status, patch, tokens, and timing
        """
        logger.info("Running arm=%s on instance=%s", arm, instance.instance_id)
        start_time = time.time()

        try:
            # Clone and checkout repo
            repo_root = self._prepare_repo(instance)

            # Load workflow config
            config = load_workflow_config(arm)
            prompts = load_prompts()

            # Initialize artifact DAG for this run
            dag_dir = self._output_dir / "artifact_dags" / instance.instance_id / arm
            dag_dir.mkdir(parents=True, exist_ok=True)
            artifact_dag = FilesystemArtifactDAG(str(dag_dir))

            # Build workflow
            workflow = build_workflow(
                config=config,
                prompts=prompts,
                model=self._model,
                base_url=self._base_url,
                artifact_dag=artifact_dag,
                repo_root=repo_root,
                api_key=self._api_key,
            )

            # Execute workflow with problem statement
            result = workflow.execute(instance.problem_statement)

            wall_time = time.time() - start_time

            # Extract patch from final artifact
            patch_content = ""
            total_tokens = 0
            per_step_tokens: dict[str, int] = {}

            for step_id, artifact in result.artifacts.items():
                # Collect token usage from metadata
                step_tokens = artifact.metadata.get("total_tokens", 0)
                if step_tokens:
                    per_step_tokens[step_id] = step_tokens
                    total_tokens += step_tokens

                # Extract patch content from the last patch step
                if "patch" in step_id or "fix" in step_id or "singleshot" in step_id:
                    try:
                        data = json.loads(artifact.content)
                        patch_content = data.get("patch", artifact.content)
                    except (json.JSONDecodeError, TypeError):
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
            logger.error(
                "Error running arm=%s on instance=%s: %s",
                arm,
                instance.instance_id,
                e,
            )
            return ArmResult(
                instance_id=instance.instance_id,
                arm=arm,
                workflow_status="error",
                wall_time_seconds=round(wall_time, 2),
                error=str(e),
            )

    def run_all(
        self,
        arms: list[str],
        split: str = "test",
        max_instances: int | None = None,
        resume_from: str | None = None,
    ) -> list[ArmResult]:
        """Run all arms across all instances.

        Args:
            arms: List of workflow variant names to run
            split: Dataset split
            max_instances: Maximum number of instances (None for all)
            resume_from: Path to existing results JSONL for resume

        Returns:
            List of all ArmResults
        """
        # Load dataset
        instances = load_swe_polybench(split=split)
        if max_instances:
            instances = instances[:max_instances]

        logger.info(
            "Running %d arms x %d instances = %d total runs",
            len(arms),
            len(instances),
            len(arms) * len(instances),
        )

        # Load existing results for resume
        completed: set[tuple[str, str]] = set()
        results: list[ArmResult] = []

        if resume_from:
            results, completed = self._load_existing_results(resume_from)
            logger.info("Resuming: %d runs already completed", len(completed))

        # Ensure output directory exists
        self._output_dir.mkdir(parents=True, exist_ok=True)
        results_path = self._output_dir / "results.jsonl"

        # Run all combinations
        for i, instance in enumerate(instances):
            for arm in arms:
                key = (instance.instance_id, arm)
                if key in completed:
                    logger.info(
                        "Skipping %s / %s (already done)", instance.instance_id, arm
                    )
                    continue

                logger.info(
                    "Progress: %d/%d instances, arm=%s, instance=%s",
                    i + 1,
                    len(instances),
                    arm,
                    instance.instance_id,
                )

                arm_result = self.run_instance(instance, arm)
                results.append(arm_result)

                # Write incrementally
                with open(results_path, "a") as f:
                    f.write(json.dumps(asdict(arm_result)) + "\n")

        logger.info(
            "Experiment complete. %d results written to %s", len(results), results_path
        )
        return results

    def _prepare_repo(self, instance: SWEInstance) -> str:
        """Clone repo and checkout base commit.

        Args:
            instance: SWE-PolyBench instance with repo and base_commit

        Returns:
            Path to the checked-out repository
        """
        if self._clone_dir:
            base_dir = self._clone_dir
            base_dir.mkdir(parents=True, exist_ok=True)
        else:
            base_dir = Path(tempfile.mkdtemp(prefix="swe_exp_"))

        # Use instance_id as directory name
        repo_dir = base_dir / instance.instance_id.replace("/", "__")

        if repo_dir.exists():
            # Reset to base commit if already cloned
            logger.info("Resetting existing repo: %s", repo_dir)
            subprocess.run(
                ["git", "checkout", "-f", instance.base_commit],
                cwd=str(repo_dir),
                capture_output=True,
                timeout=60,
                check=True,
            )
            subprocess.run(
                ["git", "clean", "-fdx"],
                cwd=str(repo_dir),
                capture_output=True,
                timeout=60,
                check=True,
            )
        else:
            # Clone the repo
            repo_url = f"https://github.com/{instance.repo}.git"
            logger.info("Cloning %s to %s", repo_url, repo_dir)
            subprocess.run(
                ["git", "clone", "--depth=1", repo_url, str(repo_dir)],
                capture_output=True,
                timeout=300,
                check=True,
            )
            # Fetch the specific commit
            subprocess.run(
                ["git", "fetch", "--depth=1", "origin", instance.base_commit],
                cwd=str(repo_dir),
                capture_output=True,
                timeout=120,
                check=True,
            )
            subprocess.run(
                ["git", "checkout", instance.base_commit],
                cwd=str(repo_dir),
                capture_output=True,
                timeout=60,
                check=True,
            )

        return str(repo_dir)

    def _load_existing_results(
        self,
        results_dir: str,
    ) -> tuple[list[ArmResult], set[tuple[str, str]]]:
        """Load existing results from JSONL for resume support."""
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
