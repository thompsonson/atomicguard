#!/usr/bin/env python3
"""
TDD Workflow with Import Guard - Automated Test Validation.

This example demonstrates automated import validation before human review:
1. Step 1: Generate tests with SyntaxGuard + ImportGuard + HumanReviewGuard
2. Step 2: Generate implementation validated by DynamicTestGuard

The ImportGuard catches missing imports (like `import pytest`) before
the human review step, ensuring the human only reviews valid test code.

Unlike TestCollectionGuard (which runs pytest --collect-only in a subprocess),
ImportGuard uses pure AST analysis to validate that all used names are
either imported, defined, or builtin. This is faster and has a single
responsibility (import validation only).

Usage:
    python -m examples.tdd_import_guard.run
    python -m examples.tdd_import_guard.run --host http://gpu:11434
    python -m examples.tdd_import_guard.run --model llama3.2:3b
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import click
from examples.base import (
    ConfigurationError,
    WorkflowRunner,
    common_options,
    display_workflow_result,
    load_prompts,
    load_workflow_config,
    normalize_base_url,
    print_error,
    print_header,
    print_steps,
    print_workflow_info,
    save_workflow_results,
    setup_logging,
)

from atomicguard import FilesystemArtifactDAG


@click.command()
@common_options
def main(
    host: str,
    model: str | None,
    prompts: str | None,
    workflow: str | None,
    output: str | None,
    artifact_dir: str | None,
    log_file: str | None,
    verbose: bool,
) -> None:
    """TDD Workflow with Import Guard."""
    script_dir = Path(__file__).parent

    # Resolve paths
    prompts_path = Path(prompts) if prompts else script_dir / "prompts.json"
    workflow_path = Path(workflow) if workflow else script_dir / "workflow.json"
    output_dir = script_dir / "output"
    output_path = Path(output) if output else output_dir / "results.json"
    artifact_path = Path(artifact_dir) if artifact_dir else output_dir / "artifacts"
    log_path = log_file or str(output_path.with_suffix(".log"))

    # Setup logging
    logger = setup_logging("tdd_import_guard", log_path, verbose)
    logger.info("Starting TDD workflow with import guard")

    # Load configuration
    try:
        logger.debug(f"Loading prompts from: {prompts_path}")
        prompts_data = load_prompts(prompts_path)
        logger.debug(f"Loading workflow from: {workflow_path}")
        workflow_config = load_workflow_config(
            workflow_path,
            required_fields=("name", "specification", "action_pairs"),
        )
        logger.info("Configuration loaded successfully")
    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        print_error(str(e), "Check that your JSON files exist and are valid.")
        sys.exit(1)

    # Display workflow info
    effective_model = model or workflow_config.get("model", "qwen2.5-coder:14b")
    effective_host = normalize_base_url(host)

    print_header(f"Workflow: {workflow_config['name']}")
    print_steps(workflow_config["action_pairs"])
    print_workflow_info(
        workflow_name=workflow_config["name"],
        model=effective_model,
        host=effective_host,
        rmax=workflow_config.get("rmax", 3),
        output_path=str(output_path),
        artifact_dir=str(artifact_path),
        log_file=log_path,
    )

    # Create artifact storage
    os.makedirs(artifact_path, exist_ok=True)
    artifact_dag = FilesystemArtifactDAG(str(artifact_path))
    logger.debug(f"Artifact storage initialized: {artifact_path}")

    # Create and execute workflow runner
    runner = WorkflowRunner(
        workflow_config=workflow_config,
        prompts=prompts_data,
        artifact_dag=artifact_dag,
        host=host,
        model_override=model,
        logger=logger,
    )

    try:
        result, duration = runner.execute()

        # Save results
        save_workflow_results(
            str(output_path), workflow_config, result, effective_model, duration
        )
        logger.info(f"Results saved to: {output_path}")

        # Display and exit
        exit_code = display_workflow_result(
            result, str(output_path), str(artifact_path), log_path
        )
        sys.exit(exit_code)

    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        click.echo("\n\nInterrupted by user.")
        sys.exit(130)

    except Exception as e:
        logger.error(f"Workflow error: {type(e).__name__}: {e}")
        _handle_error(e, effective_host, effective_model)
        sys.exit(1)


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
