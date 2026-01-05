"""
Multi-step Workflow Benchmark using the Dual-State Framework.
Paper: Managing the Stochastic (Thompson, 2025)

This benchmark tests TDD workflows: generate tests → generate implementation.
Uses the Workflow class from core.py for proper dependency management.
"""

import csv
import logging
import multiprocessing
import os
import re
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table

# Application layer
from atomicguard.application import ActionPair, Workflow

# Domain models and interfaces
from atomicguard.domain import (
    Artifact,
    ArtifactStatus,
    Context,
    ContextSnapshot,
    FeedbackEntry,
    GeneratorInterface,
    GuardInterface,
    PromptTemplate,
    RmaxExhausted,
)

# Guards
from atomicguard.guards import (
    CompositeGuard,
    DynamicTestGuard,
    HumanReviewGuard,
    SyntaxGuard,
)

# Infrastructure
from atomicguard.infrastructure.persistence import FilesystemArtifactDAG

# =============================================================================
# CONFIGURATION
# =============================================================================

NUM_TRIALS = 10
DEFAULT_OLLAMA_URL = "http://100.69.76.46:11434/v1"

console = Console()
logger = logging.getLogger("workflow_benchmark")


# =============================================================================
# WORKFLOW LOADING AND VALIDATION
# =============================================================================

# Schema for action_pairs-based workflow JSON
TASK_SCHEMA = {
    "required_task_fields": ["name", "specification", "action_pairs"],
    "required_action_pair_fields": ["prompt", "guard"],
    "valid_guards": ["syntax", "dynamic_test", "human"],
}


class WorkflowValidationError(Exception):
    """Raised when workflow JSON fails schema validation."""

    pass


def validate_workflow_schema(tasks: dict, source: str = "workflow") -> None:
    """
    Validate that the workflow JSON conforms to the action_pairs-based schema.

    Expected structure:
    {
        "task_id": {
            "name": "Task Name",
            "specification": "Task description",
            "action_pairs": {
                "g_test": {"prompt": "...", "guard": "syntax"},
                "g_impl": {"prompt": "...", "guard": "dynamic_test", "requires": ["g_test"]}
            }
        }
    }

    Args:
        tasks: Dict of task_id -> task definition
        source: Description of the source file for error messages

    Raises:
        WorkflowValidationError: If validation fails
    """
    if not isinstance(tasks, dict):
        raise WorkflowValidationError(
            f"{source}: Expected dict, got {type(tasks).__name__}"
        )

    if not tasks:
        raise WorkflowValidationError(f"{source}: No tasks defined")

    errors = []
    for task_id, task_def in tasks.items():
        if not isinstance(task_def, dict):
            errors.append(
                f"Task '{task_id}': Expected dict, got {type(task_def).__name__}"
            )
            continue

        # Check required task fields
        for field in TASK_SCHEMA["required_task_fields"]:
            if field not in task_def:
                errors.append(f"Task '{task_id}': Missing required field '{field}'")

        # Validate name
        if "name" in task_def:
            if not isinstance(task_def["name"], str):
                errors.append(f"Task '{task_id}.name': Expected str")
            elif not task_def["name"].strip():
                errors.append(f"Task '{task_id}.name': Cannot be empty")

        # Validate specification
        if "specification" in task_def and not isinstance(
            task_def["specification"], str
        ):
            errors.append(f"Task '{task_id}.specification': Expected str")

        # Validate action_pairs
        if "action_pairs" not in task_def:
            continue

        action_pairs = task_def["action_pairs"]
        if not isinstance(action_pairs, dict):
            errors.append(f"Task '{task_id}.action_pairs': Expected dict")
            continue

        if not action_pairs:
            errors.append(f"Task '{task_id}.action_pairs': No action_pairs defined")
            continue

        # Validate each action_pair
        action_pair_ids = set(action_pairs.keys())
        for ap_id, ap_def in action_pairs.items():
            if not isinstance(ap_def, dict):
                errors.append(f"ActionPair '{task_id}.{ap_id}': Expected dict")
                continue

            # Check required action_pair fields
            for field in TASK_SCHEMA["required_action_pair_fields"]:
                if field not in ap_def:
                    errors.append(
                        f"ActionPair '{task_id}.{ap_id}': Missing required field '{field}'"
                    )

            # Validate prompt
            if "prompt" in ap_def:
                if not isinstance(ap_def["prompt"], str):
                    errors.append(
                        f"ActionPair '{task_id}.{ap_id}.prompt': Expected str"
                    )
                elif not ap_def["prompt"].strip():
                    errors.append(
                        f"ActionPair '{task_id}.{ap_id}.prompt': Cannot be empty"
                    )

            # Validate guard type
            if "guard" in ap_def and ap_def["guard"] not in TASK_SCHEMA["valid_guards"]:
                errors.append(
                    f"ActionPair '{task_id}.{ap_id}.guard': Invalid guard type "
                    f"'{ap_def['guard']}'. Valid: {TASK_SCHEMA['valid_guards']}"
                )

            # Validate requires (dependencies)
            if "requires" in ap_def:
                requires = ap_def["requires"]
                if not isinstance(requires, list):
                    errors.append(
                        f"ActionPair '{task_id}.{ap_id}.requires': Expected list"
                    )
                else:
                    for dep in requires:
                        if dep not in action_pair_ids:
                            errors.append(
                                f"ActionPair '{task_id}.{ap_id}.requires': "
                                f"Unknown action_pair '{dep}'"
                            )

    if errors:
        raise WorkflowValidationError(
            f"{source} validation failed:\n  - " + "\n  - ".join(errors)
        )


def load_workflow(path: str = None) -> dict:
    """
    Load and validate task definitions from a workflow JSON file.

    Args:
        path: Path to workflow JSON. Defaults to benchmarks/workflows.json

    Returns:
        Dict of task_id -> task definition

    Raises:
        WorkflowValidationError: If schema validation fails
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file isn't valid JSON
    """
    import json
    from pathlib import Path as PathLib

    if path is None:
        workflow_path = PathLib(__file__).parent / "workflows.json"
    else:
        workflow_path = PathLib(path)

    if not workflow_path.exists():
        raise FileNotFoundError(f"Workflow file not found: {workflow_path}")

    with open(workflow_path) as f:
        tasks = json.load(f)

    validate_workflow_schema(tasks, source=str(workflow_path))
    return tasks


# Default workflow path
DEFAULT_WORKFLOW_PATH = None  # Will use benchmarks/workflows.json


# =============================================================================
# GENERATOR: OLLAMA (Workflow-aware)
# =============================================================================


class OllamaGenerator(GeneratorInterface):
    """
    Generator that connects to Ollama and includes dependencies in prompt.

    When dependencies are present in the Context (e.g., test code from prior step),
    they are included in the prompt so the generator can see them.
    """

    def __init__(
        self, model: str, base_url: str, action_pair_id: str, timeout: float = 120.0
    ):
        from openai import OpenAI

        self.model = model
        self.action_pair_id = action_pair_id
        self.client = OpenAI(base_url=base_url, api_key="ollama", timeout=timeout)
        self._attempt_counter = 0

    def generate(
        self, context: Context, _template: PromptTemplate | None = None
    ) -> Artifact:
        # Build prompt with dependencies
        prompt = context.specification

        # Include dependencies (e.g., test code for impl generator)
        if context.dependency_artifacts:
            prompt += "\n\n=== DEPENDENCIES ===\n"
            for key, artifact_id in context.dependency_artifacts:
                # Retrieve full artifact from ℛ
                artifact = context.ambient.repository.get_artifact(artifact_id)
                prompt += f"\n{key.upper()} CODE:\n```python\n{artifact.content}\n```\n"

        # Include feedback history
        if context.feedback_history:
            prompt += "\n\n=== PREVIOUS ATTEMPTS ===\n"
            for i, (_, fb) in enumerate(context.feedback_history, 1):
                prompt += f"\nAttempt {i} failed:\n{fb}\n"
            prompt += "\n=== END FEEDBACK ===\n\nPlease address the issues above and provide corrected code."

        logger.debug(
            f"[{self.model}] Generating for {self.action_pair_id} (prompt: {len(prompt)} chars)"
        )
        start_time = time.time()

        messages = [
            {
                "role": "system",
                "content": "You are a Python programming assistant. Provide complete, runnable code in a markdown block:\n```python\n# code\n```",
            },
            {"role": "user", "content": prompt},
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=0.7
            )
            content = response.choices[0].message.content or ""
            duration = time.time() - start_time
            logger.debug(f"[{self.model}] Generation completed in {duration:.2f}s")
        except Exception as e:
            logger.error(f"Ollama Error: {e}")
            content = ""

        code = self._extract_code(content)
        self._attempt_counter += 1

        # Build context snapshot for the artifact
        context_snapshot = ContextSnapshot(
            specification=context.specification,
            constraints=context.ambient.constraints,
            feedback_history=tuple(
                FeedbackEntry(artifact_id="", feedback=fb)
                for _, fb in context.feedback_history
            ),
            dependency_artifacts=context.dependency_artifacts,
        )

        # Determine previous attempt ID
        previous_attempt_id = None
        if context.current_artifact and self._attempt_counter > 1:
            # We'd need to track this properly; for now, leave as None
            pass

        return Artifact(
            artifact_id=str(uuid.uuid4()),
            workflow_id="benchmark",
            content=code,
            previous_attempt_id=previous_attempt_id,
            parent_action_pair_id=None,
            action_pair_id=self.action_pair_id,
            created_at=datetime.now(UTC).isoformat(),
            attempt_number=self._attempt_counter,
            status=ArtifactStatus.PENDING,
            guard_result=None,
            feedback="",
            context=context_snapshot,
        )

    def _extract_code(self, content: str) -> str:
        """Extract Python code from response."""
        if not content or content.isspace():
            return ""

        # Try python block
        match = re.search(r"```python\n(.*?)\n```", content, re.DOTALL)
        if match:
            return match.group(1)

        # Try generic block
        match = re.search(r"```\n(.*?)\n```", content, re.DOTALL)
        if match:
            return match.group(1)

        # Try first def/import/class
        match = re.search(r"^(def |import |class )", content, re.MULTILINE)
        if match:
            return content[match.start() :]

        return content


# =============================================================================
# WORKFLOW RUNNER
# =============================================================================


@dataclass
class TDDWorkflowResult:
    """Result of a TDD workflow execution."""

    success: bool
    test_artifact: Artifact | None
    impl_artifact: Artifact | None
    total_attempts: int
    duration_seconds: float
    failed_step: str | None = None
    error_message: str = ""


def run_tdd_workflow(
    model: str,
    base_url: str,
    tdd_task: dict,
    r_max: int,
    artifact_dir: str,
    trial_id: int = 0,
    test_guard: GuardInterface | None = None,
    impl_guard: GuardInterface | None = None,
) -> TDDWorkflowResult:
    """
    Execute a TDD workflow: generate tests → generate implementation.

    Args:
        model: Ollama model name
        base_url: Ollama API URL
        tdd_task: Task definition from TDD_TASKS
        r_max: Maximum retries per step
        artifact_dir: Directory for artifact storage
        trial_id: Trial number (for unique artifact paths)
        test_guard: Guard for test generation step (default: SyntaxGuard)
        impl_guard: Guard for implementation step (default: DynamicTestGuard)

    Returns:
        TDDWorkflowResult with success/failure info
    """
    start_time = time.time()

    # Use default guards if not provided
    if test_guard is None:
        test_guard = SyntaxGuard()
    if impl_guard is None:
        impl_guard = DynamicTestGuard()

    # Create artifact DAG for this workflow run
    trial_artifact_dir = os.path.join(artifact_dir, f"trial_{trial_id}")
    dag = FilesystemArtifactDAG(trial_artifact_dir)

    # Create generators for each step
    test_generator = OllamaGenerator(
        model=model, base_url=base_url, action_pair_id="g_test"
    )
    impl_generator = OllamaGenerator(
        model=model, base_url=base_url, action_pair_id="g_impl"
    )

    # Create action pairs
    test_action = ActionPair(generator=test_generator, guard=test_guard)
    impl_action = ActionPair(generator=impl_generator, guard=impl_guard)

    # Build workflow
    workflow = Workflow(artifact_dag=dag, rmax=r_max)
    workflow.add_step("g_test", action_pair=test_action)
    workflow.add_step("g_impl", action_pair=impl_action, requires=("g_test",))

    # Build specification with impl_prompt that will get test code appended
    # The test step uses test_prompt, impl step uses impl_prompt
    # We need to handle this by using different specs per step
    # For now, use the task spec for both and let the generator handle it

    # Extract prompts from step-based structure
    test_prompt = tdd_task["action_pairs"]["g_test"]["prompt"]
    impl_prompt = tdd_task["action_pairs"]["g_impl"]["prompt"]

    # Execute test generation first
    try:
        # Step 1: Generate tests
        test_result = _execute_single_step(
            dag=dag,
            action_pair=test_action,
            specification=test_prompt,
            rmax=r_max,
            dependencies={},
        )

        if not test_result["success"]:
            duration = time.time() - start_time
            return TDDWorkflowResult(
                success=False,
                test_artifact=test_result.get("artifact"),
                impl_artifact=None,
                total_attempts=test_result["attempts"],
                duration_seconds=duration,
                failed_step="g_test",
                error_message=test_result.get("error", "Test generation failed"),
            )

        test_artifact = test_result["artifact"]

        # Mark test artifact as accepted
        dag.update_status(test_artifact.artifact_id, ArtifactStatus.ACCEPTED)

        # Step 2: Generate implementation with test as dependency
        impl_spec = impl_prompt + f"\n```python\n{test_artifact.content}\n```"

        impl_result = _execute_single_step(
            dag=dag,
            action_pair=impl_action,
            specification=impl_spec,
            rmax=r_max,
            dependencies={"test": test_artifact},
        )

        if not impl_result["success"]:
            duration = time.time() - start_time
            return TDDWorkflowResult(
                success=False,
                test_artifact=test_artifact,
                impl_artifact=impl_result.get("artifact"),
                total_attempts=test_result["attempts"] + impl_result["attempts"],
                duration_seconds=duration,
                failed_step="g_impl",
                error_message=impl_result.get(
                    "error", "Implementation generation failed"
                ),
            )

        impl_artifact = impl_result["artifact"]
        dag.update_status(impl_artifact.artifact_id, ArtifactStatus.ACCEPTED)

        duration = time.time() - start_time
        return TDDWorkflowResult(
            success=True,
            test_artifact=test_artifact,
            impl_artifact=impl_artifact,
            total_attempts=test_result["attempts"] + impl_result["attempts"],
            duration_seconds=duration,
        )

    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Workflow error: {e}")
        return TDDWorkflowResult(
            success=False,
            test_artifact=None,
            impl_artifact=None,
            total_attempts=0,
            duration_seconds=duration,
            error_message=str(e),
        )


def _execute_single_step(
    dag: FilesystemArtifactDAG,
    action_pair: ActionPair,
    specification: str,
    rmax: int,
    dependencies: dict[str, Artifact],
) -> dict:
    """Execute a single workflow step with retries."""
    from atomicguard.application import DualStateAgent

    agent = DualStateAgent(action_pair=action_pair, artifact_dag=dag, rmax=rmax)

    try:
        artifact = agent.execute(specification, dependencies)
        return {
            "success": True,
            "artifact": artifact,
            "attempts": artifact.attempt_number,
        }
    except RmaxExhausted as e:
        last_artifact = e.provenance[-1][0] if e.provenance else None
        return {
            "success": False,
            "artifact": last_artifact,
            "attempts": len(e.provenance),
            "error": str(e),
        }


# =============================================================================
# RESULT WRITER
# =============================================================================


class ResultWriter:
    """Writes benchmark results to CSV."""

    def __init__(self, filename: str, resume: bool = False):
        self.filename = filename
        mode = "a" if resume and os.path.exists(filename) else "w"
        self.file = open(filename, mode, newline="")  # noqa: SIM115
        self.writer = csv.DictWriter(
            self.file,
            fieldnames=[
                "model_name",
                "task",
                "trial_num",
                "success",
                "total_attempts",
                "duration_seconds",
                "failed_step",
                "error_message",
                "timestamp",
            ],
        )
        if mode == "w":
            self.writer.writeheader()

    def write_result(self, data: dict) -> None:
        self.writer.writerow(data)
        self.file.flush()

    def close(self) -> None:
        self.file.close()


# =============================================================================
# CLI
# =============================================================================


@click.command()
@click.option("--trials", default=NUM_TRIALS, help="Number of trials per task")
@click.option(
    "--task",
    default=None,
    help="Task ID to run (or 'all' for all tasks in workflow)",
)
@click.option(
    "--workflow",
    default=None,
    type=click.Path(exists=True),
    help="Path to workflow JSON file (default: benchmarks/workflows.json)",
)
@click.option(
    "--model", default=None, help="Single Ollama model (deprecated, use --models)"
)
@click.option(
    "--models",
    default="qwen2.5-coder:7b",
    help="Comma-separated list of Ollama models to benchmark",
)
@click.option("--host", default=DEFAULT_OLLAMA_URL, help="Ollama API URL")
@click.option("--rmax", default=3, help="Maximum retries per step")
@click.option("--output", default="workflow_results.csv", help="Output CSV file")
@click.option(
    "--artifact-dir", default="./artifacts", help="Directory for artifact storage"
)
@click.option(
    "--human-review", is_flag=True, help="Enable human-in-the-loop review (G_21)"
)
@click.option("--verbose", is_flag=True, help="Enable debug logging")
@click.option(
    "--log-file", default=None, help="Log file path (default: derived from --output)"
)
def main(
    trials,
    task,
    workflow,
    model,
    models,
    host,
    rmax,
    output,
    artifact_dir,
    human_review,
    verbose,
    log_file,
):
    """Run TDD workflow benchmark."""

    # Load workflow (validates schema)
    try:
        tasks_dict = load_workflow(workflow)
        workflow_source = workflow if workflow else "benchmarks/workflows.json"
        console.print(f"[dim]Loaded workflow: {workflow_source}[/dim]")
    except WorkflowValidationError as e:
        console.print(f"[bold red]Workflow validation error:[/bold red]\n{e}")
        raise SystemExit(1) from e
    except FileNotFoundError as e:
        console.print(f"[bold red]Workflow file not found:[/bold red] {e}")
        raise SystemExit(1) from e

    available_tasks = list(tasks_dict.keys())

    # Parse models list (--model takes precedence for backwards compatibility)
    if model:
        models_to_run = [model]
    else:
        models_to_run = [m.strip() for m in models.split(",") if m.strip()]

    if not models_to_run:
        console.print("[bold red]No models specified.[/bold red]")
        raise SystemExit(1)

    # Determine tasks to run
    if task is None:
        # Default to first task if not specified
        tasks_to_run = [available_tasks[0]]
        console.print(
            f"[dim]No --task specified, using first task: {tasks_to_run[0]}[/dim]"
        )
    elif task == "all":
        tasks_to_run = available_tasks
    elif task in available_tasks:
        tasks_to_run = [task]
    else:
        console.print(
            f"[bold red]Unknown task '{task}'.[/bold red] "
            f"Available tasks: {', '.join(available_tasks)}, all"
        )
        raise SystemExit(1)

    # Setup logging
    # Console logging (respects --verbose)
    console_level = logging.DEBUG if verbose else logging.INFO
    console_handler = RichHandler(console=console, rich_tracebacks=True)
    console_handler.setLevel(console_level)

    # File logging (always DEBUG for our code)
    log_path = log_file if log_file else os.path.splitext(output)[0] + ".log"
    file_handler = logging.FileHandler(log_path, mode="a")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    # Configure our logger with both handlers
    logger.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # Suppress noisy 3rd party loggers
    for noisy in ["httpx", "openai", "httpcore", "urllib3"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    console.print(f"[dim]Logging to: {log_path}[/dim]")

    # Configure guards based on --human-review flag
    # G_22: ALL_STREAMS_VALIDATED → PRODUCTION_READY = G_automated ∧ G_human
    if human_review:
        test_guard = CompositeGuard(
            SyntaxGuard(), HumanReviewGuard("Review Generated Tests")
        )
        impl_guard = CompositeGuard(
            DynamicTestGuard(), HumanReviewGuard("Review Implementation")
        )
        console.print("\n[bold yellow]Human-in-the-loop mode enabled[/bold yellow]")
        console.print("[dim]You will be prompted to approve each artifact.[/dim]")
    else:
        test_guard = SyntaxGuard()
        impl_guard = DynamicTestGuard()

    # Print summary
    console.print("\n[bold]═══ TDD WORKFLOW BENCHMARK ═══[/bold]")
    console.print(f"\n[bold]Tasks:[/bold] {', '.join(tasks_to_run)}")
    console.print(f"[bold]Models:[/bold] {', '.join(models_to_run)}")
    console.print(f"[bold]Trials:[/bold] {trials}")
    console.print(f"[bold]R_max:[/bold] {rmax}")
    console.print(f"[bold]Artifacts:[/bold] {artifact_dir}")
    console.print(f"[bold]Output:[/bold] {output}")
    console.print(
        f"[bold]Human Review:[/bold] {'Enabled' if human_review else 'Disabled'}"
    )
    total_runs = len(models_to_run) * len(tasks_to_run) * trials
    console.print(f"[bold]Total runs:[/bold] {total_runs}")

    console.print("\n[dim]Starting in 3 seconds... (Ctrl+C to cancel)[/dim]")
    time.sleep(3)

    writer = ResultWriter(output)
    results = []

    for current_model in models_to_run:
        console.print(f"\n[bold magenta]═══ Model: {current_model} ═══[/bold magenta]")

        for current_task in tasks_to_run:
            tdd_task = tasks_dict[current_task]
            task_artifact_dir = os.path.join(artifact_dir, current_model, current_task)

            console.print(f"\n[bold cyan]═══ Task: {tdd_task['name']} ═══[/bold cyan]")

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn(),
                console=console,
            ) as progress:
                task_id = progress.add_task(
                    f"Running {tdd_task['name']}...", total=trials
                )

                successes = 0
                total_attempts = 0
                total_duration = 0.0

                for trial in range(trials):
                    result = run_tdd_workflow(
                        model=current_model,
                        base_url=host,
                        tdd_task=tdd_task,
                        r_max=rmax,
                        artifact_dir=task_artifact_dir,
                        trial_id=trial,
                        test_guard=test_guard,
                        impl_guard=impl_guard,
                    )

                    if result.success:
                        successes += 1

                    total_attempts += result.total_attempts
                    total_duration += result.duration_seconds

                    writer.write_result(
                        {
                            "model_name": current_model,
                            "task": current_task,
                            "trial_num": trial + 1,
                            "success": result.success,
                            "total_attempts": result.total_attempts,
                            "duration_seconds": round(result.duration_seconds, 2),
                            "failed_step": result.failed_step or "",
                            "error_message": result.error_message,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        }
                    )

                    progress.advance(task_id)
                    progress.update(
                        task_id,
                        description=f"Running {tdd_task['name']}... (Success: {successes}/{trial + 1})",
                    )

            # Summary for this task
            success_rate = (successes / trials) * 100 if trials > 0 else 0
            avg_attempts = total_attempts / trials if trials > 0 else 0
            avg_duration = total_duration / trials if trials > 0 else 0

            results.append(
                {
                    "model": current_model,
                    "task": current_task,
                    "success_rate": success_rate,
                    "avg_attempts": avg_attempts,
                    "avg_duration": avg_duration,
                }
            )

            console.print(
                f"  [dim]Success: {success_rate:.0f}%, Avg Attempts: {avg_attempts:.1f}, Avg Duration: {avg_duration:.1f}s[/dim]"
            )

    writer.close()

    # Print results table
    console.print("\n[bold]Results Summary[/bold]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Model")
    table.add_column("Task")
    table.add_column("Success Rate", justify="right")
    table.add_column("Avg Attempts", justify="right")
    table.add_column("Avg Duration", justify="right")

    for row in results:
        success_val = row["success_rate"]
        if success_val >= 90:
            success_str = f"[green]{success_val:.0f}%[/green]"
        elif success_val >= 70:
            success_str = f"[yellow]{success_val:.0f}%[/yellow]"
        else:
            success_str = f"[red]{success_val:.0f}%[/red]"

        table.add_row(
            row["model"],
            row["task"],
            success_str,
            f"{row['avg_attempts']:.1f}",
            f"{row['avg_duration']:.1f}s",
        )

    console.print(table)
    console.print(f"\n[green]✓ Results saved to {output}[/green]")
    console.print(f"[green]✓ Artifacts saved to {artifact_dir}/[/green]")


if __name__ == "__main__":
    multiprocessing.set_start_method("fork", force=True)
    main()
