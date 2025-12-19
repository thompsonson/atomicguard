"""
OpenHands semantic agent generator implementation.

Connects to LLM instances via the OpenHands SDK for agentic code generation
with filesystem tools.
"""

import asyncio
import json
import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from atomicguard.domain.interfaces import GeneratorInterface
from atomicguard.domain.models import (
    Artifact,
    ArtifactStatus,
    Context,
    ContextSnapshot,
)
from atomicguard.domain.prompts import PromptTemplate

DEFAULT_OPENHANDS_BASE_URL = "http://localhost:11434/v1"


def _run_async(coro: Any) -> Any:
    """Run an async coroutine from sync code."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # We're inside an async context, need to run in new thread
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    else:
        return asyncio.run(coro)


class SemanticAgentGenerator(GeneratorInterface):
    """
    Semantic agent generator using OpenHands SDK.

    This generator operates as an autonomous agent with filesystem tools.
    Unlike OllamaGenerator (single inference), this performs multi-step
    reasoning and tool use within a workspace directory.

    The artifact content is a JSON manifest of filesystem operations performed,
    enabling guards to verify the environmental state rather than raw code.

    Note (Semantic Agency):
        Per GeneratorInterface contract, this agentic process is atomic from
        the workflow's perspective. The generator may perform multiple tool
        invocations internally, but returns a single artifact manifest.

    Note (Idempotency):
        For retry safety, agent operations should be convergent:
        - Directory creation uses mkdir -p semantics
        - File writes are deterministic overwrites
        - Guards verify final state, not intermediate steps
    """

    def __init__(
        self,
        model: str = "openai/qwen3-coder:30b",
        base_url: str = DEFAULT_OPENHANDS_BASE_URL,
        workspace: str | Path | None = None,
        timeout: float = 300.0,
    ):
        """
        Initialize the semantic agent generator.

        Args:
            model: LiteLLM model identifier (e.g., "openai/qwen3-coder:30b")
            base_url: LLM API base URL (Ollama endpoint)
            workspace: Working directory for agent operations.
                       If None, creates a temp directory per invocation.
            timeout: Request timeout in seconds (default 5 minutes for agentic tasks)
        """
        self._model = model
        self._base_url = base_url
        self._workspace = Path(workspace) if workspace else None
        self._timeout = timeout
        self._version_counter = 0

        # Verify SDK is available (defer actual import to generate())
        self._verify_sdk_available()

    def _verify_sdk_available(self) -> None:
        """Verify OpenHands SDK is installed."""
        try:
            from openhands.core.config import LLMConfig  # noqa: F401
            from openhands.core.main import create_runtime  # noqa: F401
        except ImportError as err:
            raise ImportError(
                "openhands-ai required: pip install openhands-ai"
            ) from err

    def generate(
        self, context: Context, template: PromptTemplate | None = None
    ) -> Artifact:
        """
        Generate an artifact via agentic tool use.

        The agent operates in a workspace directory, using terminal and file
        editing tools to create/modify files based on the specification.

        Args:
            context: Generation context with specification and feedback
            template: Optional prompt template for structured generation

        Returns:
            Artifact containing JSON manifest of filesystem operations
        """
        from openhands.core.config import (
            LLMConfig,
            OpenHandsConfig,
            SandboxConfig,
        )
        from openhands.core.main import create_runtime, run_controller
        from openhands.events.action import MessageAction

        # Build the prompt
        if template:
            prompt = template.render(context)
        else:
            prompt = self._build_agentic_prompt(context)

        # Resolve workspace directory
        workspace_dir = self._resolve_workspace()

        # Configure OpenHands
        llm_config = LLMConfig(
            model=self._model,
            api_key="ollama",  # Required but unused for Ollama
            base_url=self._base_url,
            timeout=int(self._timeout),
        )

        sandbox_config = SandboxConfig(
            use_host_network=True,
        )

        config = OpenHandsConfig(
            workspace_base=str(workspace_dir),
            workspace_mount_path=str(workspace_dir),
            llms={"default": llm_config},
            sandbox=sandbox_config,
        )

        # Create runtime and run agent (async API)
        runtime = create_runtime(config, headless_mode=True)
        try:
            initial_action = MessageAction(content=prompt)
            _run_async(
                run_controller(
                    config=config,
                    initial_user_action=initial_action,
                    runtime=runtime,
                    headless_mode=True,
                )
            )
        finally:
            runtime.close()

        # Scan workspace to build manifest
        manifest = self._build_manifest(workspace_dir)

        self._version_counter += 1

        return Artifact(
            artifact_id=str(uuid.uuid4()),
            content=json.dumps(manifest, indent=2),
            previous_attempt_id=None,
            action_pair_id="openhands",
            created_at=datetime.now().isoformat(),
            attempt_number=self._version_counter,
            status=ArtifactStatus.PENDING,
            guard_result=None,
            feedback="",
            context=ContextSnapshot(
                specification=context.specification,
                constraints=context.ambient.constraints,
                feedback_history=tuple(
                    (aid, fb) for aid, fb in context.feedback_history
                ),
                dependency_ids=tuple(key for key, _ in context.dependencies),
            ),
        )

    def _resolve_workspace(self) -> Path:
        """
        Resolve the workspace directory for agent operations.

        Returns:
            Path to workspace directory
        """
        if self._workspace:
            self._workspace.mkdir(parents=True, exist_ok=True)
            return self._workspace

        # Create temp directory that persists beyond invocation
        # (caller is responsible for cleanup if needed)
        temp_dir = Path(tempfile.mkdtemp(prefix="atomicguard_agent_"))
        return temp_dir

    def _build_agentic_prompt(self, context: Context) -> str:
        """
        Build a prompt optimized for agentic code generation.

        This prompt instructs the agent to create files in the workspace
        rather than just outputting code.
        """
        parts = [
            "You are a software engineering agent working in a project workspace.",
            "Your task is to create or modify files to implement the specification.",
            "",
            "## Specification",
            context.specification,
        ]

        if context.ambient.constraints:
            parts.extend(
                [
                    "",
                    "## Constraints",
                    context.ambient.constraints,
                ]
            )

        if context.dependencies:
            parts.extend(
                [
                    "",
                    "## Dependencies (artifacts from prior steps)",
                ]
            )
            for key, artifact in context.dependencies:
                parts.append(f"### {key}")
                parts.append(artifact.content)

        if context.feedback_history:
            parts.extend(
                [
                    "",
                    "## Previous Attempt Feedback (address these issues)",
                ]
            )
            for i, (_, feedback) in enumerate(context.feedback_history, 1):
                parts.append(f"### Attempt {i} rejection:")
                parts.append(feedback)

        parts.extend(
            [
                "",
                "## Instructions",
                "1. Analyze the specification and any feedback from prior attempts",
                "2. Create the necessary files in the current directory",
                "3. Ensure all imports are valid and code is syntactically correct",
                "4. Use descriptive filenames that match the specification",
            ]
        )

        return "\n".join(parts)

    def _build_manifest(self, workspace: Path) -> dict[str, Any]:
        """
        Scan the workspace and build a JSON manifest of operations.

        The manifest captures:
        - Files created/modified with their content
        - Directory structure
        - Metadata about the generation

        Args:
            workspace: Path to the workspace directory

        Returns:
            Manifest dictionary suitable for JSON serialization
        """
        files: dict[str, dict[str, Any]] = {}
        directories: list[str] = []

        for item in workspace.rglob("*"):
            relative_path = str(item.relative_to(workspace))

            # Skip hidden files and __pycache__
            if any(
                part.startswith(".") or part == "__pycache__"
                for part in relative_path.split(os.sep)
            ):
                continue

            if item.is_dir():
                directories.append(relative_path)
            elif item.is_file():
                try:
                    content = item.read_text(encoding="utf-8")
                    files[relative_path] = {
                        "content": content,
                        "size": item.stat().st_size,
                        "extension": item.suffix,
                    }
                except (UnicodeDecodeError, PermissionError):
                    files[relative_path] = {
                        "content": None,
                        "binary": True,
                        "size": item.stat().st_size,
                    }

        return {
            "workspace": str(workspace),
            "files": files,
            "directories": sorted(directories),
            "file_count": len(files),
            "generated_at": datetime.now().isoformat(),
        }
