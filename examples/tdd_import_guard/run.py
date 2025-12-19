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

Setup:
    1. Install Ollama: https://ollama.ai/
    2. Pull a model: ollama pull qwen2.5-coder:14b
    3. Run this script: python examples/tdd_import_guard/run.py

Options:
    --host URL    Ollama API URL (default: http://localhost:11434)
    --model NAME  Override model from workflow.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from atomicguard import (
    ActionPair,
    CompositeGuard,
    DynamicTestGuard,
    FilesystemArtifactDAG,
    GuardInterface,
    HumanReviewGuard,
    ImportGuard,
    OllamaGenerator,
    PromptTemplate,
    SyntaxGuard,
    Workflow,
    WorkflowStatus,
)

# =============================================================================
# Exceptions
# =============================================================================


class ConfigurationError(Exception):
    """Raised when configuration files are invalid or missing."""

    pass


# =============================================================================
# Logging Setup
# =============================================================================


def setup_logging(
    log_file: str | None = None,
    verbose: bool = False,
) -> logging.Logger:
    """
    Configure dual-handler logging (console + file).

    Args:
        log_file: Path to log file (None for no file logging)
        verbose: Enable DEBUG level on console (default INFO)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("tdd_import_guard")
    logger.setLevel(logging.DEBUG)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(levelname)-8s | %(message)s"))
    logger.addHandler(console_handler)

    # File handler (if path provided)
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(file_handler)

    # Suppress noisy 3rd party loggers
    for noisy in ["httpx", "openai", "httpcore", "urllib3"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return logger


# =============================================================================
# JSON Loaders
# =============================================================================


def load_prompts(path: Path) -> dict[str, PromptTemplate]:
    """
    Load prompt templates from JSON file.

    Args:
        path: Path to prompts.json

    Returns:
        Dict mapping step ID to PromptTemplate

    Raises:
        ConfigurationError: If file is missing or invalid
    """
    if not path.exists():
        raise ConfigurationError(f"Prompts file not found: {path}")

    try:
        with open(path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ConfigurationError(f"Invalid JSON in {path}: {e}") from e

    if not isinstance(data, dict):
        raise ConfigurationError(f"Expected dict in {path}, got {type(data).__name__}")

    prompts = {}
    for step_id, prompt_data in data.items():
        if not isinstance(prompt_data, dict):
            raise ConfigurationError(
                f"Invalid prompt config for '{step_id}': expected dict"
            )
        prompts[step_id] = PromptTemplate(
            role=prompt_data.get("role", ""),
            constraints=prompt_data.get("constraints", ""),
            task=prompt_data.get("task", ""),
            feedback_wrapper=prompt_data.get("feedback_wrapper", "{feedback}"),
        )
    return prompts


def load_workflow_config(path: Path) -> dict:
    """
    Load workflow configuration from JSON file.

    Args:
        path: Path to workflow.json

    Returns:
        Workflow configuration dict

    Raises:
        ConfigurationError: If file is missing or invalid
    """
    if not path.exists():
        raise ConfigurationError(f"Workflow file not found: {path}")

    try:
        with open(path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ConfigurationError(f"Invalid JSON in {path}: {e}") from e

    # Validate required fields
    required_fields = ["name", "specification", "action_pairs"]
    missing = [f for f in required_fields if f not in data]
    if missing:
        raise ConfigurationError(
            f"Missing required fields in {path}: {', '.join(missing)}"
        )

    if not isinstance(data["action_pairs"], dict):
        raise ConfigurationError("'action_pairs' must be a dict")

    if not data["action_pairs"]:
        raise ConfigurationError("'action_pairs' cannot be empty")

    return data


# =============================================================================
# Guard Builder
# =============================================================================

VALID_GUARD_TYPES = ["syntax", "import", "human", "dynamic_test", "composite"]


def build_guard(config: dict) -> GuardInterface:
    """
    Build a guard from configuration.

    Supported guard types:
    - "syntax": SyntaxGuard for Python AST validation
    - "import": ImportGuard for undefined name detection (pure AST-based)
    - "human": HumanReviewGuard for human approval
    - "dynamic_test": DynamicTestGuard for test execution
    - "composite": CompositeGuard combining multiple guards

    Args:
        config: Guard configuration dict with "guard" key

    Returns:
        Configured GuardInterface instance

    Raises:
        ConfigurationError: If guard type is unknown or config is invalid
    """
    if "guard" not in config:
        raise ConfigurationError("Guard config missing 'guard' key")

    guard_type = config["guard"]

    if guard_type == "syntax":
        return SyntaxGuard()

    elif guard_type == "import":
        return ImportGuard()

    elif guard_type == "human":
        return HumanReviewGuard(
            prompt_title=config.get("human_prompt_title", "HUMAN REVIEW REQUIRED")
        )

    elif guard_type == "dynamic_test":
        return DynamicTestGuard()

    elif guard_type == "composite":
        if "guards" not in config:
            raise ConfigurationError("Composite guard missing 'guards' list")
        if not isinstance(config["guards"], list):
            raise ConfigurationError("Composite 'guards' must be a list")
        if not config["guards"]:
            raise ConfigurationError("Composite 'guards' cannot be empty")

        # Build each sub-guard, passing parent config for shared settings
        guards = []
        for g in config["guards"]:
            sub_config = {"guard": g}
            # Pass through human_prompt_title if present
            if g == "human" and "human_prompt_title" in config:
                sub_config["human_prompt_title"] = config["human_prompt_title"]
            guards.append(build_guard(sub_config))
        return CompositeGuard(*guards)

    else:
        raise ConfigurationError(
            f"Unknown guard type: '{guard_type}'. "
            f"Valid types: {', '.join(VALID_GUARD_TYPES)}"
        )


# =============================================================================
# Workflow Builder
# =============================================================================


def create_workflow(
    workflow_config: dict,
    prompts: dict[str, PromptTemplate],
    artifact_dag: FilesystemArtifactDAG | None = None,
    host: str | None = None,
    model_override: str | None = None,
    logger: logging.Logger | None = None,  # noqa: ARG001
) -> Workflow:
    """
    Build a Workflow from configuration and prompts.

    Args:
        workflow_config: Loaded workflow.json data
        prompts: Loaded prompts keyed by step ID
        artifact_dag: Optional artifact storage (FilesystemArtifactDAG)
        host: Optional Ollama host URL override
        model_override: Optional model name override
        logger: Optional logger for debug output

    Returns:
        Configured Workflow instance
    """
    if artifact_dag is None:
        raise ConfigurationError("artifact_dag is required")

    model = model_override or workflow_config.get("model", "qwen2.5-coder:14b")
    rmax = workflow_config.get("rmax", 3)

    # Create generator with optional host override
    generator_kwargs = {"model": model}
    if host:
        # Ensure /v1 suffix for OpenAI-compatible API
        base_url = host.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"
        generator_kwargs["base_url"] = base_url
    generator = OllamaGenerator(**generator_kwargs)

    workflow = Workflow(artifact_dag=artifact_dag, rmax=rmax)

    # Build action pairs and add steps
    action_pairs_config = workflow_config["action_pairs"]

    for step_id, ap_config in action_pairs_config.items():
        guard = build_guard(ap_config)
        prompt_template = prompts.get(step_id)

        action_pair = ActionPair(
            generator=generator,
            guard=guard,
            prompt_template=prompt_template,
        )

        requires = tuple(ap_config.get("requires", []))

        workflow.add_step(
            guard_id=step_id,
            action_pair=action_pair,
            requires=requires,
            deps=requires,
        )

    return workflow


# =============================================================================
# Results Saving
# =============================================================================


def save_results(
    output_path: str,
    workflow_config: dict,
    result: Any,
    model: str,
    duration: float,
) -> None:
    """
    Save workflow results to JSON file.

    Args:
        output_path: Path to save results
        workflow_config: Original workflow configuration
        result: WorkflowResult from execution
        model: Model name used
        duration: Execution duration in seconds
    """
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Calculate total attempts: successful artifacts + failed attempts in provenance
    total_attempts = len(result.artifacts) + len(result.provenance)

    data: dict[str, Any] = {
        "workflow_name": workflow_config["name"],
        "model": model,
        "timestamp": datetime.now().isoformat(),
        "duration_seconds": round(duration, 2),
        "success": result.status == WorkflowStatus.SUCCESS,
        "status": result.status.value,
        "failed_step": result.failed_step,
        "total_attempts": total_attempts,
        "artifacts": {},
    }

    # Include artifact content and metadata
    for step_id, artifact in result.artifacts.items():
        data["artifacts"][step_id] = {
            "artifact_id": artifact.artifact_id,
            "content": artifact.content,
            "attempt_number": artifact.attempt_number,
            "status": artifact.status.value if artifact.status else None,
        }

    # Include provenance (attempt history) if any failed attempts
    if result.provenance:
        data["provenance"] = [
            {"attempt": i + 1, "content": artifact.content, "feedback": feedback}
            for i, (artifact, feedback) in enumerate(result.provenance)
        ]

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="TDD Workflow with Import Guard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                          # Use defaults from workflow.json
  python run.py --host http://gpu:11434  # Use remote Ollama server
  python run.py --model llama3.2:3b      # Override model
  python run.py -v                       # Verbose logging
  python run.py --output results.json    # Custom output path
        """,
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Ollama API URL (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override model specified in workflow.json",
    )
    parser.add_argument(
        "--prompts",
        default=None,
        help="Path to prompts.json (default: ./prompts.json)",
    )
    parser.add_argument(
        "--workflow",
        default=None,
        help="Path to workflow.json (default: ./workflow.json)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save results JSON (default: ./output/results.json)",
    )
    parser.add_argument(
        "--artifact-dir",
        default=None,
        help="Directory for artifact storage (default: ./output/artifacts)",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Path to log file (default: derived from --output)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging to console",
    )
    return parser.parse_args()


def print_error(message: str, hint: str | None = None) -> None:
    """Print formatted error message to stderr."""
    print(f"\n{'=' * 60}", file=sys.stderr)
    print(f"ERROR: {message}", file=sys.stderr)
    if hint:
        print(f"\nHint: {hint}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)


# =============================================================================
# Main
# =============================================================================


def main() -> int:
    """
    Execute TDD workflow with import guard from JSON configuration.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    args = parse_args()

    # Resolve paths relative to this script
    script_dir = Path(__file__).parent
    prompts_path = Path(args.prompts) if args.prompts else script_dir / "prompts.json"
    workflow_path = (
        Path(args.workflow) if args.workflow else script_dir / "workflow.json"
    )

    # Resolve output paths
    output_dir = script_dir / "output"
    output_path = Path(args.output) if args.output else output_dir / "results.json"
    artifact_dir = (
        Path(args.artifact_dir) if args.artifact_dir else output_dir / "artifacts"
    )
    log_file = args.log_file or str(output_path.with_suffix(".log"))

    # Setup logging
    logger = setup_logging(log_file=log_file, verbose=args.verbose)
    logger.info("Starting TDD workflow with import guard")

    # Load configuration
    try:
        logger.debug(f"Loading prompts from: {prompts_path}")
        prompts = load_prompts(prompts_path)
        logger.debug(f"Loading workflow from: {workflow_path}")
        workflow_config = load_workflow_config(workflow_path)
        logger.info("Configuration loaded successfully")
    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        print_error(str(e), "Check that your JSON files exist and are valid.")
        return 1

    # Display workflow info
    model = args.model or workflow_config.get("model", "qwen2.5-coder:14b")
    host_input = args.host or "http://localhost:11434"
    # Normalize host for display (matches what OllamaGenerator will use)
    host = host_input.rstrip("/")
    if not host.endswith("/v1"):
        host = f"{host}/v1"

    print("=" * 60)
    print(f"Workflow: {workflow_config['name']}")
    print("=" * 60)
    print("\nSteps:")
    for step_id, ap_config in workflow_config["action_pairs"].items():
        guard_desc = ap_config["guard"]
        if guard_desc == "composite":
            guard_desc = f"composite({', '.join(ap_config['guards'])})"
        requires = ap_config.get("requires", [])
        req_str = f" (requires: {requires})" if requires else ""
        print(f"  {step_id}: {guard_desc}{req_str}")

    print(f"\nModel: {model}")
    print(f"Host: {host}")
    print(f"Max attempts per step: {workflow_config.get('rmax', 3)}")
    print(f"Output: {output_path}")
    print(f"Artifacts: {artifact_dir}")
    print(f"Log file: {log_file}")
    print("-" * 60)

    # Create artifact storage directory
    os.makedirs(artifact_dir, exist_ok=True)
    artifact_dag = FilesystemArtifactDAG(str(artifact_dir))
    logger.debug(f"Artifact storage initialized: {artifact_dir}")

    # Build workflow
    try:
        workflow = create_workflow(
            workflow_config,
            prompts,
            artifact_dag=artifact_dag,
            host=args.host,
            model_override=args.model,
            logger=logger,
        )
        logger.debug(
            f"Workflow built with {len(workflow_config['action_pairs'])} steps"
        )
    except ConfigurationError as e:
        logger.error(f"Failed to build workflow: {e}")
        print_error(f"Failed to build workflow: {e}")
        return 1

    # Execute workflow
    specification = workflow_config["specification"]
    start_time = datetime.now()
    logger.info(f"Executing workflow: {workflow_config['name']}")

    try:
        result = workflow.execute(specification)
        duration = (datetime.now() - start_time).total_seconds()

        # Log result
        if result.status == WorkflowStatus.SUCCESS:
            logger.info(f"Workflow completed successfully in {duration:.2f}s")
        else:
            logger.warning(
                f"Workflow {result.status.value} at step '{result.failed_step}' after {duration:.2f}s"
            )

        # Save results
        save_results(str(output_path), workflow_config, result, model, duration)
        logger.info(f"Results saved to: {output_path}")

    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        print("\n\nInterrupted by user.")
        return 130  # Standard exit code for SIGINT

    except Exception as e:
        logger.error(f"Workflow error: {type(e).__name__}: {e}")
        # Handle connection errors specifically
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
        return 1

    # Display results
    if result.status == WorkflowStatus.SUCCESS:
        print("\n" + "=" * 60)
        print("SUCCESS: TDD Workflow Complete")
        print("=" * 60)

        print("\n--- Generated Tests (g_test) ---")
        print(result.artifacts["g_test"].content)

        print("\n--- Generated Implementation (g_impl) ---")
        print(result.artifacts["g_impl"].content)

        print("\n" + "-" * 60)
        print(f"Results saved to: {output_path}")
        print(f"Artifacts saved to: {artifact_dir}")
        print(f"Log file: {log_file}")
        return 0

    else:
        print("\n" + "=" * 60)
        print(f"FAILED at step: {result.failed_step}")
        print("=" * 60)

        if result.provenance:
            print("\nAttempt history:")
            for i, (_content, feedback) in enumerate(result.provenance, 1):
                print(f"\n--- Attempt {i} ---")
                print(f"Feedback: {feedback}")

        print("\n" + "-" * 60)
        print(f"Results saved to: {output_path}")
        print(f"Log file: {log_file}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
