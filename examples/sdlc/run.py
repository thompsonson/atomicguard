#!/usr/bin/env python3
"""
Multi-Agent SDLC Workflow Runner.

Demonstrates hierarchical composition (Remark 5) by orchestrating multiple
semantic agents in a pipeline:

    g_config → g_add ──┐
                       ├→ g_coder
    g_config → g_bdd ──┘

Each generator (ADDGenerator, BDDGenerator, CoderGenerator) may be an
autonomous Semantic Agent with its own internal retry loops and context.

Usage:
    python -m examples.sdlc.run
    python -m examples.sdlc.run --docs examples/sdlc/sample_input/requirements.md
    python -m examples.sdlc.run --host http://gpu:11434
    python -m examples.sdlc.run --model qwen2.5-coder:14b

Configuration:
    - workflow.json: Multi-agent SDLC workflow configuration
    - prompts.json: Prompt templates for each generator

Requirements:
    - Ollama running with qwen2.5-coder model
    - pydantic-ai installed
"""

from __future__ import annotations

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

# Register SDLC-specific guards
from .guards import register_sdlc_guards


def load_docs(docs_path: Path | None, script_dir: Path) -> str:
    """Load requirements/architecture documentation.

    Args:
        docs_path: Path to documentation file (use --docs CLI arg)
        script_dir: Script directory for default path resolution

    Returns:
        Documentation content as string
    """
    if docs_path and docs_path.exists():
        return docs_path.read_text()

    # Default to sample_input/requirements.md
    default_path = script_dir / "sample_input" / "requirements.md"
    if default_path.exists():
        return default_path.read_text()

    # Inline fallback
    return """
# Feature Requirements

## Overview

Implement a user authentication system with the following capabilities:

1. User registration with email validation
2. Login with JWT token generation
3. Password reset functionality

## Architecture

### Layer Structure
- **Domain**: User entity, AuthToken value object
- **Application**: RegisterUser, LoginUser, ResetPassword use cases
- **Infrastructure**: PostgreSQL repository, SMTP email adapter

### Architecture Gates
- Domain MUST NOT import from infrastructure
- Application depends on domain abstractions only
- All external I/O through infrastructure adapters

## Acceptance Criteria

### User Registration
- Given a valid email and password
- When registering a new user
- Then the user is created and confirmation email is sent

### User Login
- Given valid credentials
- When logging in
- Then a JWT token is returned with 24h expiry

### Password Reset
- Given a registered email
- When requesting password reset
- Then a reset link is sent to the email
"""


@click.command()
@common_options
@click.option(
    "--docs",
    default=None,
    type=click.Path(exists=True),
    help="Path to requirements/architecture documentation",
)
@click.option(
    "--workdir",
    default=None,
    type=click.Path(),
    help="Output directory for generated files",
)
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
) -> None:
    """Multi-Agent SDLC Workflow - End-to-end development pipeline."""
    script_dir = Path(__file__).parent

    # Register SDLC guards
    register_sdlc_guards()

    # Resolve paths
    prompts_path = Path(prompts) if prompts else script_dir / "prompts.json"
    workflow_path = Path(workflow) if workflow else script_dir / "workflow.json"
    output_dir = Path(workdir) if workdir else script_dir / "output"
    output_dir.mkdir(exist_ok=True, parents=True)

    output_path = Path(output) if output else output_dir / "results.json"
    artifact_path = Path(artifact_dir) if artifact_dir else output_dir / "artifacts"
    artifact_path.mkdir(exist_ok=True, parents=True)

    log_path = log_file
    if not log_path and output_path:
        log_path = str(output_path.with_suffix(".log"))
    elif not log_path:
        log_path = str(output_dir / "run.log")

    # Setup logging - include child loggers for ADDGenerator and guards
    logger = setup_logging(
        "sdlc_workflow",
        log_path,
        verbose,
        child_loggers=["add_workflow", "examples.sdlc.guards.sdlc_guards"],
    )
    logger.info("Starting Multi-Agent SDLC Workflow")

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

    # Load documentation
    docs_path = Path(docs) if docs else None
    docs_content = load_docs(docs_path, script_dir)
    spec_source = docs if docs else "sample_input/requirements.md"
    logger.info(
        f"Loaded documentation from '{spec_source}' ({len(docs_content)} chars)"
    )

    # Override specification with loaded docs
    workflow_config["specification"] = docs_content

    # Get config values (CLI args override workflow.json)
    effective_model = model or workflow_config.get("model", "qwen2.5-coder:14b")
    effective_host = normalize_base_url(host)

    # Display workflow info
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
        extra_info={
            "Spec": spec_source,
            "Steps": ", ".join(workflow_config["action_pairs"].keys()),
        },
    )

    # Create artifact storage
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
            result,
            str(output_path),
            str(artifact_path),
            log_path,
            artifact_keys=("g_config", "g_add", "g_bdd", "g_coder"),
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
