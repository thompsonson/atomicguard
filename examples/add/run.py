#!/usr/bin/env python3
"""
ADD Agent Example Runner.

Demonstrates using ADDGenerator to extract architecture gates from
documentation and generate pytest-arch tests.

Usage:
    python -m examples.add.run
    python -m examples.add.run --docs examples/add/ml_agents/specification.md
    python -m examples.add.run --docs examples/add/atomicguard/specification.md
    python -m examples.add.run --host http://gpu:11434
    python -m examples.add.run --model qwen2.5-coder:14b
    python -m examples.add.run --output results/add_experiment.json

Configuration:
    - prompts.json: Externalized prompt templates for each action pair
    - workflow.json: Workflow configuration (model, rmax, min_gates, etc.)

Requirements:
    - Ollama running locally with qwen2.5-coder:14b model
    - pydantic-ai installed
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click
from examples.base import (
    AgentRunner,
    ConfigurationError,
    add_options,
    common_options,
    console,
    display_workflow_result,
    load_prompts,
    load_workflow_config,
    normalize_base_url,
    normalize_model_name,
    print_error,
    print_failure,
    print_header,
    print_provenance,
    print_workflow_info,
    save_workflow_results,
    setup_logging,
)

from atomicguard import Artifact, FilesystemArtifactDAG
from atomicguard.application.action_pair import ActionPair
from atomicguard.domain.exceptions import EscalationRequired, RmaxExhausted

from .add_generator import ADDGenerator
from .guards import ArtifactStructureGuard


def load_docs(docs_path: Path | None) -> str:
    """Load architecture documentation.

    Args:
        docs_path: Path to documentation file (use --docs CLI arg)

    Returns:
        Documentation content as string
    """
    if docs_path and docs_path.exists():
        return docs_path.read_text()

    # Default to sample_docs/architecture.md
    script_dir = Path(__file__).parent
    default_path = script_dir / "sample_docs" / "architecture.md"
    if default_path.exists():
        return default_path.read_text()

    # Inline fallback
    return """
# Architecture Documentation

## Layer Structure

This system follows a clean architecture with three layers:
- **Domain**: Core business logic and entities
- **Application**: Use cases and orchestration
- **Infrastructure**: External adapters (database, API, filesystem)

## Architecture Gates

### Gate 1: Domain Independence
The domain layer MUST NOT import from infrastructure.
Domain entities should be pure Python with no external dependencies.

### Gate 2: Application Layer Boundaries
The application layer can import from domain but NOT from infrastructure directly.
Use cases should depend on abstractions (ports), not implementations.

### Gate 3: Dependency Direction
Dependencies flow inward: Infrastructure → Application → Domain
Never the reverse.
"""


class ADDAgentRunner(AgentRunner):
    """ADD-specific runner with documentation loading and custom generator."""

    def __init__(
        self,
        workflow_config: dict,
        prompts: dict,
        artifact_dag: FilesystemArtifactDAG,
        action_pair: ActionPair,
        docs: str,
        host: str | None = None,
        model_override: str | None = None,
        logger: logging.Logger | None = None,
    ):
        super().__init__(
            workflow_config=workflow_config,
            prompts=prompts,
            artifact_dag=artifact_dag,
            action_pair=action_pair,
            host=host,
            model_override=model_override,
            logger=logger,
        )
        self._docs = docs

    def get_specification(self) -> str:
        """Return loaded documentation as specification."""
        return self._docs


@click.command()
@common_options
@add_options
def main(
    host: str,
    model: str | None,
    prompts: str | None,
    workflow: str | None,
    output: str | None,
    artifact_dir: str | None,
    log_file: str | None,
    verbose: bool,
    docs: str | None,
    workdir: str | None,
    rmax: int,
    min_gates: int,
    min_tests: int,
) -> None:
    """ADD Agent - Architecture-Driven Development."""
    script_dir = Path(__file__).parent

    # Resolve paths
    prompts_path = Path(prompts) if prompts else script_dir / "prompts.json"
    workflow_path = Path(workflow) if workflow else script_dir / "workflow.json"
    output_dir = Path(workdir) if workdir else script_dir / "output"
    output_dir.mkdir(exist_ok=True, parents=True)

    output_path = Path(output) if output else None
    artifact_path = Path(artifact_dir) if artifact_dir else output_dir / "artifacts"
    artifact_path.mkdir(exist_ok=True, parents=True)

    log_path = log_file
    if not log_path and output_path:
        log_path = str(output_path.with_suffix(".log"))
    elif not log_path:
        log_path = str(output_dir / "run.log")

    # Setup logging
    logger = setup_logging("add_workflow", log_path, verbose)
    logger.info("Starting ADD Agent Example")

    # Load configuration
    try:
        logger.debug(f"Loading prompts from: {prompts_path}")
        prompts_data = load_prompts(prompts_path)
        logger.debug(f"Loading workflow from: {workflow_path}")
        workflow_config = load_workflow_config(workflow_path, required_fields=("name",))
        logger.info("Configuration loaded successfully")
    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        print_error(str(e), "Check that your JSON files exist and are valid.")
        sys.exit(1)

    # Load documentation
    docs_path = Path(docs) if docs else None
    docs_content = load_docs(docs_path)
    spec_source = docs if docs else "sample_docs/architecture.md"
    logger.info(
        f"Loaded documentation from '{spec_source}' ({len(docs_content)} chars)"
    )

    # Get config values (CLI args override workflow.json)
    effective_model = model or workflow_config.get("model", "qwen2.5-coder:14b")
    effective_rmax = rmax if rmax != 3 else workflow_config.get("rmax", 3)
    effective_min_gates = (
        min_gates if min_gates != 3 else workflow_config.get("min_gates", 3)
    )
    effective_min_tests = (
        min_tests if min_tests != 3 else workflow_config.get("min_tests", 3)
    )
    effective_host = normalize_base_url(host)
    normalized_model = normalize_model_name(effective_model)

    # Display workflow info
    print_header(f"Workflow: {workflow_config.get('name', 'ADD Agent')}")
    print_workflow_info(
        workflow_name=workflow_config.get("name", "ADD Agent"),
        model=normalized_model,
        host=effective_host,
        rmax=effective_rmax,
        output_path=str(output_path) if output_path else "N/A",
        artifact_dir=str(artifact_path),
        log_file=log_path,
        extra_info={
            "Min gates": effective_min_gates,
            "Min tests": effective_min_tests,
            "Spec": spec_source,
        },
    )

    # Create artifact storage
    artifact_dag = FilesystemArtifactDAG(str(artifact_path))
    logger.debug(f"Artifact storage initialized: {artifact_path}")

    # Update workflow config with effective values
    workflow_config["rmax"] = 2  # Outer retry budget
    workflow_config["constraints"] = ""  # Ω starts empty - AP0 populates it

    # Create ADDGenerator with artifact_dag for internal persistence
    add_generator = ADDGenerator(
        model=normalized_model,
        base_url=effective_host,
        rmax=effective_rmax,
        workdir=output_dir,
        min_gates=effective_min_gates,
        min_tests=effective_min_tests,
        artifact_dag=artifact_dag,
        prompts=prompts_data,
    )

    # Create action pair with ADD generator
    action_pair = ActionPair(
        generator=add_generator,
        guard=ArtifactStructureGuard(min_tests=effective_min_tests),
    )

    # Create runner
    runner = ADDAgentRunner(
        workflow_config=workflow_config,
        prompts=prompts_data,
        artifact_dag=artifact_dag,
        action_pair=action_pair,
        docs=docs_content,
        host=host,
        model_override=model,
        logger=logger,
    )

    console.print("\nExecuting ADD workflow...\n")

    try:
        result, duration = runner.execute()

        # Save results
        if output_path:
            save_workflow_results(
                str(output_path), workflow_config, result, normalized_model, duration
            )
            logger.info(f"Results saved to: {output_path}")

        # Display result
        exit_code = display_workflow_result(
            result,
            str(output_path) if output_path else "N/A",
            str(artifact_path),
            log_path,
        )

        # Show generated files (result is always Artifact for AgentRunner)
        if isinstance(result, Artifact):
            try:
                manifest = json.loads(result.content)
                console.print(
                    f"\n[bold]Test count:[/bold] {manifest.get('test_count', 0)}"
                )
                console.print(
                    f"[bold]Gates covered:[/bold] {manifest.get('gates_covered', [])}"
                )
                console.print(
                    f"\nCheck [cyan]{output_dir}[/cyan] to see the generated files"
                )
            except json.JSONDecodeError:
                pass

        sys.exit(exit_code)

    except EscalationRequired as e:
        logger.error(f"Escalation required: {e.feedback}")
        print_failure(
            "Escalation Required",
            details="Fatal error requiring human intervention",
        )
        console.print(f"\n[bold]Feedback:[/bold] {e.feedback}")
        if output_path:
            _save_error_result(str(output_path), workflow_config, e.feedback)
        sys.exit(1)

    except RmaxExhausted as e:
        logger.error(f"Max retries exhausted after {len(e.provenance)} attempts")
        print_failure(
            f"Failed after {len(e.provenance)} attempts",
            details="Could not generate valid output",
        )
        print_provenance(e.provenance)
        if output_path:
            _save_error_result(str(output_path), workflow_config, str(e))
        sys.exit(1)

    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        click.echo("\n\nInterrupted by user.")
        sys.exit(130)

    except Exception as e:
        logger.error(f"Workflow error: {type(e).__name__}: {e}")
        _handle_error(e, effective_host, normalized_model)
        if output_path:
            _save_error_result(
                str(output_path), workflow_config, f"{type(e).__name__}: {e}"
            )
        sys.exit(1)


def _save_error_result(output_path: str, workflow_config: dict, error: str) -> None:
    """Save error result to JSON file."""
    import os
    from datetime import datetime

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    data = {
        "workflow_name": workflow_config.get("name", "Unknown"),
        "timestamp": datetime.now().isoformat(),
        "success": False,
        "error": error,
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def _handle_error(e: Exception, host: str, model: str) -> None:
    """Handle and display execution errors."""
    error_str = str(e).lower()
    if "connection" in error_str or "refused" in error_str:
        print_error(
            f"Cannot connect to Ollama at {host}",
            hint=(
                "Make sure Ollama is running:\n"
                "  1. Start Ollama: ollama serve\n"
                "  2. Or specify a different host: --host http://your-server:11434\n"
                "  3. Verify the model is available: ollama list"
            ),
        )
    elif "model" in error_str and "not found" in error_str:
        print_error(
            f"Model '{model}' not found",
            hint=f"Pull the model first: ollama pull {model}",
        )
    elif "timeout" in error_str:
        print_error(
            "Request timed out",
            hint="The model may be loading or the server is slow. Try again.",
        )
    else:
        print_error(f"Unexpected error: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
