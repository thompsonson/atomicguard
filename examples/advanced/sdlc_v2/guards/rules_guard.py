"""
RulesGuard: Validates extracted architecture rules.

Validates that the RulesExtractorGenerator produced valid ArchitectureRules.
"""

import logging
from typing import Any

from atomicguard.domain.interfaces import GuardInterface
from atomicguard.domain.models import Artifact, GuardResult

from ..models import ArchitectureRules

logger = logging.getLogger("sdlc_checkpoint")


class RulesGuard(GuardInterface):
    """
    Validates ArchitectureRules extraction.

    Checks:
    - Valid JSON matching ArchitectureRules schema
    - At least one import rule extracted
    - At least one folder structure entry
    """

    def __init__(self, min_rules: int = 1, min_folders: int = 1):
        self._min_rules = min_rules
        self._min_folders = min_folders

    def validate(self, artifact: Artifact, **_deps: Any) -> GuardResult:
        """Validate extracted architecture rules."""
        logger.debug("[RulesGuard] Validating architecture rules...")

        try:
            rules = ArchitectureRules.model_validate_json(artifact.content)
            logger.debug("[RulesGuard] Parsed ArchitectureRules successfully")
        except Exception as e:
            logger.debug(f"[RulesGuard] Invalid schema: {e}")
            return GuardResult(
                passed=False,
                feedback=f"Invalid ArchitectureRules schema: {e}",
            )

        # Check minimum import rules
        if len(rules.import_rules) < self._min_rules:
            logger.debug(
                f"[RulesGuard] Not enough import rules: {len(rules.import_rules)} < {self._min_rules}"
            )
            return GuardResult(
                passed=False,
                feedback=f"Expected at least {self._min_rules} import rule(s), "
                f"found {len(rules.import_rules)}. "
                "Check that g_add artifact contains tests with names like "
                "'test_domain_no_infrastructure_imports'.",
            )

        # Check minimum folder structure
        if len(rules.folder_structure) < self._min_folders:
            logger.debug(
                f"[RulesGuard] Not enough folders: {len(rules.folder_structure)} < {self._min_folders}"
            )
            return GuardResult(
                passed=False,
                feedback=f"Expected at least {self._min_folders} folder structure(s), "
                f"found {len(rules.folder_structure)}. "
                "Check that specification contains a Package Structure section.",
            )

        # Validate import rules have forbidden targets
        for rule in rules.import_rules:
            if not rule.forbidden_targets:
                logger.debug(
                    f"[RulesGuard] Rule for {rule.source_layer} has no targets"
                )
                return GuardResult(
                    passed=False,
                    feedback=f"Import rule for '{rule.source_layer}' has no forbidden targets.",
                )

        # Validate folder structure has paths
        for folder in rules.folder_structure:
            if not folder.path:
                logger.debug(f"[RulesGuard] Folder for {folder.layer} has no path")
                return GuardResult(
                    passed=False,
                    feedback=f"Folder structure for '{folder.layer}' has no path.",
                )

        logger.debug("[RulesGuard] ✓ All checks passed")

        # Build summary
        rule_summary = ", ".join(
            f"{r.source_layer}→¬{r.forbidden_targets}" for r in rules.import_rules
        )
        folder_summary = ", ".join(f.layer for f in rules.folder_structure)

        return GuardResult(
            passed=True,
            feedback=f"Valid rules: {len(rules.import_rules)} import rules ({rule_summary}), "
            f"{len(rules.folder_structure)} folders ({folder_summary})",
        )
