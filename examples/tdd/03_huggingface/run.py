#!/usr/bin/env python3
"""
Fully Automated TDD Workflow with HuggingFace.

This example demonstrates a configuration-driven TDD workflow where:
1. Step 1 (g_test): Generate tests validated by SyntaxGuard + ImportGuard
2. Step 2 (g_impl): Generate implementation validated by DynamicTestGuard

Unlike 01_human_review and 02_import_guard, this workflow is fully automated —
no human review step is needed. The guard chain catches syntax errors and
undefined imports automatically, and the DynamicTestGuard runs the generated
tests against the implementation.

Usage:
    uv run python -m examples.tdd.03_huggingface.run
    uv run python -m examples.tdd.03_huggingface.run --model Qwen/Qwen2.5-Coder-32B-Instruct
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
    host: str,  # noqa: ARG001 — provided by common_options, unused for HuggingFace
    model: str | None,
    prompts: str | None,
    workflow: str | None,
    output: str | None,
    artifact_dir: str | None,
    log_file: str | None,
    verbose: bool,
) -> None:
    """Fully Automated TDD Workflow with HuggingFace."""
    # Validate HF_TOKEN early
    if not os.environ.get("HF_TOKEN"):
        print_error(
            "HF_TOKEN environment variable not set",
            hint=(
                "Set your HuggingFace API token:\n"
                '  export HF_TOKEN="hf_your_token_here"\n\n'
                "Get a token at: https://huggingface.co/settings/tokens"
            ),
        )
        sys.exit(1)

    script_dir = Path(__file__).parent

    # Resolve paths
    prompts_path = Path(prompts) if prompts else script_dir / "prompts.json"
    workflow_path = Path(workflow) if workflow else script_dir / "workflow.json"
    output_dir = script_dir / "output"
    output_path = Path(output) if output else output_dir / "results.json"
    artifact_path = Path(artifact_dir) if artifact_dir else output_dir / "artifacts"
    log_path = log_file or str(output_path.with_suffix(".log"))

    # Setup logging
    logger = setup_logging("tdd_huggingface", log_path, verbose)
    logger.info("Starting fully automated TDD workflow with HuggingFace")

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
    effective_model = model or workflow_config.get(
        "model", "Qwen/Qwen2.5-Coder-32B-Instruct"
    )

    print_header(f"Workflow: {workflow_config['name']}")
    print_steps(workflow_config["action_pairs"])
    print_workflow_info(
        workflow_name=workflow_config["name"],
        model=effective_model,
        host="HuggingFace Inference API",
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
    # host=None prevents base_url injection — HuggingFace uses HF_TOKEN, not a host URL
    runner = WorkflowRunner(
        workflow_config=workflow_config,
        prompts=prompts_data,
        artifact_dag=artifact_dag,
        host=None,
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
        _handle_error(e)
        sys.exit(1)


def _handle_error(e: Exception) -> None:
    """Handle and display execution errors."""
    error_str = str(e).lower()
    if "hf_token" in error_str or "unauthorized" in error_str or "401" in error_str:
        print_error(
            "HuggingFace authentication failed",
            hint=(
                "Check your HF_TOKEN is valid:\n"
                "  1. Verify at: https://huggingface.co/settings/tokens\n"
                "  2. Ensure the token has Inference API access"
            ),
        )
    elif "rate" in error_str and "limit" in error_str:
        print_error(
            "HuggingFace rate limit exceeded",
            hint="Wait a moment and try again, or use a different model/provider.",
        )
    elif "model" in error_str and "not found" in error_str:
        print_error(
            "Model not found on HuggingFace",
            hint="Check the model name in workflow.json is correct and accessible.",
        )
    elif "huggingface_hub" in error_str:
        print_error(
            "huggingface_hub library not installed",
            hint="Install it with: uv pip install huggingface_hub",
        )
    else:
        print_error(f"Unexpected error: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
