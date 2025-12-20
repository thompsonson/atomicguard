#!/usr/bin/env python3
"""
ADD Agent Example Runner.

Demonstrates using ADDGenerator to extract architecture gates from
documentation and generate pytest-arch tests.

Usage:
    python -m examples.add.run
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

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from atomicguard.application.action_pair import ActionPair
from atomicguard.application.agent import DualStateAgent
from atomicguard.domain.exceptions import EscalationRequired, RmaxExhausted
from atomicguard.domain.prompts import PromptTemplate
from atomicguard.infrastructure.persistence import FilesystemArtifactDAG

from .add_generator import ADDGenerator
from .guards import ArtifactStructureGuard

# =============================================================================
# Exceptions
# =============================================================================


class ConfigurationError(Exception):
    """Raised when configuration files are invalid or missing."""

    pass


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
    required_fields = ["name"]
    missing = [f for f in required_fields if f not in data]
    if missing:
        raise ConfigurationError(
            f"Missing required fields in {path}: {', '.join(missing)}"
        )

    return data


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
    logger = logging.getLogger("add_workflow")
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


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="ADD Agent - Architecture-Driven Development",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m examples.add.run
  python -m examples.add.run --host http://gpu:11434
  python -m examples.add.run --model openai/qwen2.5-coder:14b
  python -m examples.add.run --output results/add_experiment.json
  python -m examples.add.run --docs path/to/architecture.md
        """,
    )
    parser.add_argument(
        "--host",
        default="http://localhost:11434",
        help="Ollama API URL (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--model",
        default="ollama:qwen2.5-coder:14b",
        help="Model to use (default: ollama:qwen2.5-coder:14b)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save results JSON (optional)",
    )
    parser.add_argument(
        "--docs",
        default=None,
        help="Path to architecture documentation (default: sample_docs/architecture.md)",
    )
    parser.add_argument(
        "--workdir",
        default=None,
        help="Output directory for generated tests (default: examples/add/output)",
    )
    parser.add_argument(
        "--rmax",
        type=int,
        default=3,
        help="Maximum retry attempts per action pair (default: 3)",
    )
    parser.add_argument(
        "--min-gates",
        type=int,
        default=3,
        help="Minimum number of gates required (default: 3)",
    )
    parser.add_argument(
        "--min-tests",
        type=int,
        default=3,
        help="Minimum number of tests required (default: 3)",
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
    return parser.parse_args()


def load_docs(docs_path: Path | None) -> str:
    """Load architecture documentation."""
    if docs_path and docs_path.exists():
        return docs_path.read_text()

    # Default to sample docs
    default_path = Path(__file__).parent / "sample_docs" / "architecture.md"
    if default_path.exists():
        return default_path.read_text()

    # Fallback to inline sample
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


def save_results(
    output_path: str,
    artifact: object,
    duration: float,
    success: bool,
    error: str | None = None,
) -> None:
    """Save results to JSON file."""
    import os

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    data = {
        "timestamp": datetime.now().isoformat(),
        "duration_seconds": round(duration, 2),
        "success": success,
    }

    if success and artifact:
        data["artifact_id"] = artifact.artifact_id  # type: ignore[attr-defined]
        data["attempt_number"] = artifact.attempt_number  # type: ignore[attr-defined]
        data["manifest"] = json.loads(artifact.content)  # type: ignore[attr-defined]
    elif error:
        data["error"] = error

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def main() -> int:
    """Run the ADD example."""
    args = parse_args()

    # Resolve paths relative to this script
    script_dir = Path(__file__).parent
    prompts_path = Path(args.prompts) if args.prompts else script_dir / "prompts.json"
    workflow_path = (
        Path(args.workflow) if args.workflow else script_dir / "workflow.json"
    )

    # Resolve output paths
    output_dir = Path(args.workdir) if args.workdir else script_dir / "output"
    output_dir.mkdir(exist_ok=True, parents=True)

    # Resolve log file path
    log_file = args.log_file
    if not log_file and args.output:
        log_file = str(Path(args.output).with_suffix(".log"))
    elif not log_file:
        log_file = str(output_dir / "run.log")

    # Setup logging
    logger = setup_logging(log_file=log_file, verbose=args.verbose)
    logger.info("Starting ADD Agent Example")

    # Load configuration files
    try:
        logger.debug(f"Loading prompts from: {prompts_path}")
        prompts = load_prompts(prompts_path)
        logger.debug(f"Loading workflow from: {workflow_path}")
        workflow_config = load_workflow_config(workflow_path)
        logger.info("Configuration loaded successfully")
    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        print(f"ERROR: {e}")
        return 1

    # Resolve artifact directory
    artifact_dir = (
        Path(args.artifact_dir) if args.artifact_dir else output_dir / "artifacts"
    )
    artifact_dir.mkdir(exist_ok=True, parents=True)
    logger.debug(f"Artifact storage: {artifact_dir}")

    # Load documentation
    docs_path = Path(args.docs) if args.docs else None
    docs = load_docs(docs_path)
    logger.info(f"Loaded documentation ({len(docs)} chars)")

    # Get config values (CLI args override workflow.json)
    model = args.model or workflow_config.get("model", "qwen2.5-coder:14b")
    rmax = args.rmax if args.rmax != 3 else workflow_config.get("rmax", 3)
    min_gates = (
        args.min_gates if args.min_gates != 3 else workflow_config.get("min_gates", 3)
    )
    min_tests = (
        args.min_tests if args.min_tests != 3 else workflow_config.get("min_tests", 3)
    )

    # Normalize model string for PydanticAI
    # PydanticAI expects "ollama:model" format
    if not model.startswith(("ollama:", "openai:", "anthropic:")):
        model = f"ollama:{model}"

    # Normalize base_url (ensure /v1 suffix for OpenAI-compatible API)
    base_url = args.host.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"

    # Create agent with filesystem artifact storage
    dag = FilesystemArtifactDAG(str(artifact_dir))

    # Create ADDGenerator with artifact_dag for internal persistence
    add_generator = ADDGenerator(
        model=model,
        base_url=base_url,
        rmax=rmax,
        workdir=output_dir,
        min_gates=min_gates,
        min_tests=min_tests,
        artifact_dag=dag,
        prompts=prompts,
    )

    # Create action pair with ADD generator
    # The outer guard validates the final manifest
    action_pair = ActionPair(
        generator=add_generator,
        guard=ArtifactStructureGuard(min_tests=args.min_tests),
    )
    agent = DualStateAgent(
        action_pair=action_pair,
        artifact_dag=dag,
        rmax=2,  # Outer retry budget (in addition to internal retries)
        constraints="Generate pytest-arch compatible tests",
    )

    print("=" * 60)
    print(f"Workflow: {workflow_config.get('name', 'ADD Agent')}")
    print("=" * 60)
    print(f"Model: {model}")
    print(f"Base URL: {base_url}")
    print(f"Output dir: {output_dir}")
    print(f"Prompts: {prompts_path}")
    print(f"Min gates: {min_gates}")
    print(f"Min tests: {min_tests}")
    print(f"Max retries: {rmax}")
    if args.output:
        print(f"Results: {args.output}")
    print("=" * 60)
    print("\nExecuting ADD workflow...\n")

    start_time = datetime.now()
    artifact = None

    try:
        artifact = agent.execute(docs)
        duration = (datetime.now() - start_time).total_seconds()

        print("=== SUCCESS ===\n")
        print(f"Artifact ID: {artifact.artifact_id}")
        print(f"Attempt: {artifact.attempt_number}")
        print(f"Duration: {duration:.2f}s")

        # Print manifest
        manifest = json.loads(artifact.content)
        print(f"\nTest count: {manifest.get('test_count', 0)}")
        print(f"Gates covered: {manifest.get('gates_covered', [])}")

        print("\n--- Generated Files ---")
        for f in manifest.get("files", []):
            print(f"  - {f['path']}")

        print(f"\nCheck {output_dir} to see the generated files")

        if args.output:
            save_results(args.output, artifact, duration, success=True)
            print(f"\nResults saved to: {args.output}")

        return 0

    except EscalationRequired as e:
        duration = (datetime.now() - start_time).total_seconds()
        print("=== ESCALATION ===")
        print("Fatal error requiring human intervention")
        print(f"Duration: {duration:.2f}s")
        print(f"Feedback: {e.feedback}")

        if args.output:
            save_results(args.output, None, duration, success=False, error=e.feedback)

        return 1

    except RmaxExhausted as e:
        duration = (datetime.now() - start_time).total_seconds()
        print("=== FAILED ===")
        print(f"Could not generate valid output after {len(e.provenance)} attempts")
        print(f"Duration: {duration:.2f}s")
        print("\nAttempt history:")
        for i, (_artifact, feedback) in enumerate(e.provenance, 1):
            print(f"\n--- Attempt {i} ---")
            print(f"Feedback: {feedback}")

        if args.output:
            save_results(args.output, None, duration, success=False, error=str(e))

        return 1

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        return 130

    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        print(f"\nError: {type(e).__name__}: {e}")

        if args.output:
            save_results(
                args.output,
                None,
                duration,
                success=False,
                error=f"{type(e).__name__}: {e}",
            )

        return 1


if __name__ == "__main__":
    sys.exit(main())
