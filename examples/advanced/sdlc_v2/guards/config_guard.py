"""
ConfigGuard: Validates project configuration extraction.

Validates that the ConfigExtractorGenerator produced valid ProjectConfig.
"""

import json
import logging
from typing import Any

from atomicguard.domain.interfaces import GuardInterface
from atomicguard.domain.models import Artifact, GuardResult

from ..models import ProjectConfig

logger = logging.getLogger("sdlc_checkpoint")


class ConfigGuard(GuardInterface):
    """
    Validates ProjectConfig extraction (Action Pair 0).

    Checks:
    - Valid JSON structure
    - source_root is non-empty
    - No validation errors in content
    """

    def validate(self, artifact: Artifact, **_deps: Any) -> GuardResult:
        """Validate extracted project configuration."""
        logger.debug("[ConfigGuard] Validating project configuration...")

        try:
            data = json.loads(artifact.content)
            logger.debug("[ConfigGuard] Parsed JSON successfully")
        except json.JSONDecodeError as e:
            logger.debug(f"[ConfigGuard] Invalid JSON: {e}")
            return GuardResult(passed=False, feedback=f"Invalid JSON: {e}")

        # Check for validation errors from generator
        if "error" in data:
            logger.debug(f"[ConfigGuard] Generator returned error: {data.get('error')}")
            return GuardResult(
                passed=False,
                feedback=f"Generation error: {data.get('details', data['error'])}",
            )

        # Parse as ProjectConfig
        try:
            config = ProjectConfig.model_validate(data)
            logger.debug(
                f"[ConfigGuard] Schema valid, source_root={config.source_root}"
            )
        except Exception as e:
            logger.debug(f"[ConfigGuard] Schema invalid: {e}")
            return GuardResult(passed=False, feedback=f"Schema validation failed: {e}")

        # Check source_root is non-empty
        if not config.source_root:
            logger.debug("[ConfigGuard] source_root is empty")
            return GuardResult(
                passed=False,
                feedback="source_root not found. Look for 'Source Root' in Package Configuration "
                "or infer from layer paths (e.g., 'src/taskmanager/domain/' → 'src/taskmanager').",
            )

        logger.debug("[ConfigGuard] ✓ All checks passed")
        return GuardResult(
            passed=True,
            feedback=f"Config extracted: source_root={config.source_root}",
        )
