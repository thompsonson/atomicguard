"""
RulesExtractorGenerator: Extracts structured architecture rules.

Deterministic extraction from g_add tests and specification.
No LLM needed - parses test names and specification text.
"""

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import uuid4

from atomicguard.domain.interfaces import GeneratorInterface
from atomicguard.domain.models import Artifact, ArtifactStatus, Context, ContextSnapshot
from atomicguard.domain.prompts import PromptTemplate

from ..models import (
    ArchitectureRules,
    FolderStructure,
    ImportRule,
    TestSuite,
)

logger = logging.getLogger("sdlc_checkpoint")


@dataclass
class RulesExtractorConfig:
    """Configuration for RulesExtractorGenerator."""

    # No LLM config needed - this is deterministic


class RulesExtractorGenerator(GeneratorInterface):
    """
    Extracts structured architecture rules from g_add tests and specification.

    This is a DETERMINISTIC generator - no LLM needed. It parses:
    - Test names from g_add → import rules
    - Specification text → folder structure
    - g_config → package name and source root

    The output provides explicit, actionable constraints for the coder.
    """

    config_class = RulesExtractorConfig

    def __init__(
        self,
        config: RulesExtractorConfig | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ):
        self._version_counter = 0

    def generate(
        self,
        context: Context,
        template: PromptTemplate | None = None,  # noqa: ARG002
        action_pair_id: str = "g_rules",
        workflow_id: str = "unknown",
    ) -> Artifact:
        """Extract architecture rules from g_add artifact and specification."""
        logger.debug("[RulesExtractor] Extracting rules...")

        # Get dependencies
        g_add_content = None
        g_config_content = None

        if context.dependency_artifacts:
            for dep_name, dep_id in context.dependency_artifacts:
                try:
                    dep_artifact = context.ambient.repository.get_artifact(dep_id)
                    if dep_name == "g_add":
                        g_add_content = dep_artifact.content
                    elif dep_name == "g_config":
                        g_config_content = dep_artifact.content
                except Exception as e:
                    logger.warning(
                        f"[RulesExtractor] Could not load dep {dep_name}: {e}"
                    )

        # Extract package info from g_config FIRST (needed for folder structure extraction)
        package_name = ""
        source_root = ""
        if g_config_content:
            try:
                config_data = json.loads(g_config_content)
                package_name = config_data.get("package_name", "")
                source_root = config_data.get("source_root", "")
            except json.JSONDecodeError:
                pass

        # Extract import rules from g_add tests
        import_rules = []
        if g_add_content:
            import_rules = self._extract_import_rules(g_add_content)

        # Extract folder structure from specification
        folder_structure = self._extract_folder_structure(
            context.specification, source_root, package_name
        )

        # Build layer descriptions
        layer_descriptions = self._extract_layer_descriptions(context.specification)

        # Build rules artifact
        rules = ArchitectureRules(
            import_rules=import_rules,
            folder_structure=folder_structure,
            dependency_direction="infrastructure → application → domain",
            layer_descriptions=layer_descriptions,
            package_name=package_name,
            source_root=source_root,
        )

        self._version_counter += 1

        return Artifact(
            artifact_id=str(uuid4()),
            workflow_id=workflow_id,
            content=rules.model_dump_json(indent=2),
            previous_attempt_id=None,
            parent_action_pair_id=None,
            action_pair_id=action_pair_id,
            created_at=datetime.now().isoformat(),
            attempt_number=self._version_counter,
            status=ArtifactStatus.PENDING,
            guard_result=None,
            context=ContextSnapshot(
                workflow_id=workflow_id,
                specification=context.specification,
                constraints=context.ambient.constraints,
                feedback_history=(),
                dependency_artifacts=context.dependency_artifacts,
            ),
        )

    def _extract_import_rules(self, g_add_content: str) -> list[ImportRule]:
        """Extract import rules from g_add TestSuite artifact."""
        rules = []

        try:
            test_suite = TestSuite.model_validate_json(g_add_content)
        except Exception as e:
            logger.warning(f"[RulesExtractor] Could not parse g_add: {e}")
            return rules

        # Parse test names to extract rules
        # Patterns like: test_domain_no_infrastructure_imports
        #                test_application_no_infrastructure_imports
        for test in test_suite.tests:
            rule = self._parse_test_name(test.test_name, test.documentation_reference)
            if rule:
                rules.append(rule)

        # Deduplicate by source_layer
        seen = {}
        for rule in rules:
            if rule.source_layer not in seen:
                seen[rule.source_layer] = rule
            else:
                # Merge forbidden targets
                existing = seen[rule.source_layer]
                for target in rule.forbidden_targets:
                    if target not in existing.forbidden_targets:
                        existing.forbidden_targets.append(target)

        return list(seen.values())

    def _parse_test_name(self, test_name: str, doc_ref: str = "") -> ImportRule | None:
        """Parse a test name to extract the import rule it enforces."""
        # Common patterns:
        # test_domain_no_infrastructure_imports → domain cannot import infrastructure
        # test_domain_no_application_imports → domain cannot import application
        # test_application_no_infrastructure_imports → application cannot import infrastructure

        # Pattern: test_{source}_no_{target}_imports
        match = re.match(
            r"test_(\w+)_no_(\w+)_imports?",
            test_name,
            re.IGNORECASE,
        )
        if match:
            source = match.group(1).lower()
            target = match.group(2).lower()

            # Map common variations
            layer_map = {
                "infra": "infrastructure",
                "app": "application",
                "dom": "domain",
            }
            source = layer_map.get(source, source)
            target = layer_map.get(target, target)

            rationale = self._get_rationale(source, target, doc_ref)

            return ImportRule(
                source_layer=source,
                forbidden_targets=[target],
                rationale=rationale,
            )

        # Pattern: test_{source}_should_not_import_{target}
        match = re.match(
            r"test_(\w+)_should_not_import_(\w+)",
            test_name,
            re.IGNORECASE,
        )
        if match:
            source = match.group(1).lower()
            target = match.group(2).lower()
            return ImportRule(
                source_layer=source,
                forbidden_targets=[target],
                rationale=doc_ref or f"{source} layer should not depend on {target}",
            )

        return None

    def _get_rationale(self, source: str, target: str, doc_ref: str) -> str:
        """Generate rationale for an import rule."""
        if doc_ref:
            return doc_ref

        rationales = {
            (
                "domain",
                "infrastructure",
            ): "Domain must be pure with no external dependencies",
            (
                "domain",
                "application",
            ): "Domain must not depend on use cases or orchestration",
            (
                "application",
                "infrastructure",
            ): "Application depends on abstractions, not implementations",
        }
        return rationales.get(
            (source, target),
            f"{source} layer should not import from {target} layer",
        )

    def _extract_folder_structure(
        self, specification: str, source_root: str = "", package_name: str = ""
    ) -> list[FolderStructure]:
        """Extract folder structure from specification text."""
        folders = []

        # Look for Package Structure or Project Structure section
        # Pattern: path with trailing /
        # e.g., "src/taskmanager/domain/"

        # Find tree-like structure in specification
        lines = specification.split("\n")
        current_root = ""

        # Try to find source root from specification
        for line in lines:
            # Detect source root like "src/taskmanager/" or from layer paths
            root_match = re.search(r"(src/\w+/)", line)
            if root_match:
                current_root = root_match.group(1)
                break

            # Also try to find it from layer paths like "src/taskmanager/domain/"
            layer_path_match = re.search(
                r"(src/\w+)/(domain|application|infrastructure)/", line
            )
            if layer_path_match:
                current_root = layer_path_match.group(1) + "/"
                break

        # Use source_root from g_config if available and not found in spec
        if not current_root and source_root:
            current_root = source_root.rstrip("/") + "/"

        # Fallback to package name
        if not current_root and package_name:
            current_root = f"src/{package_name}/"

        for line in lines:
            # Detect layer folders: domain/, application/, infrastructure/
            for layer in ["domain", "application", "infrastructure"]:
                if f"{layer}/" in line.lower() or f"├── {layer}" in line.lower():
                    # Try to extract modules from subsequent lines
                    modules = self._find_modules_for_layer(lines, layer)
                    purpose = self._get_layer_purpose(layer)

                    path = (
                        f"{current_root}{layer}/" if current_root else f"src/{layer}/"
                    )

                    folders.append(
                        FolderStructure(
                            layer=layer,
                            path=path,
                            allowed_modules=modules,
                            purpose=purpose,
                        )
                    )
                    break

        # If we didn't find explicit structure, infer from common patterns
        if not folders:
            folders = self._default_folder_structure(
                specification, source_root, package_name
            )

        return folders

    def _find_modules_for_layer(self, lines: list[str], layer: str) -> list[str]:
        """Find module files for a given layer from the specification."""
        modules = []
        in_layer = False

        for line in lines:
            if f"{layer}/" in line.lower() or f"├── {layer}" in line.lower():
                in_layer = True
                continue

            if in_layer:
                # Look for .py files
                py_match = re.search(r"(\w+\.py)", line)
                if py_match:
                    modules.append(py_match.group(1))

                # Stop if we hit another layer or section
                if (
                    any(
                        x in line.lower()
                        for x in [
                            "application/",
                            "infrastructure/",
                            "domain/",
                            "tests/",
                        ]
                    )
                    and f"{layer}/" not in line.lower()
                ):
                    break

        # Default modules if none found
        if not modules:
            default_modules = {
                "domain": ["entities.py", "value_objects.py", "exceptions.py"],
                "application": ["ports.py", "use_cases.py", "services.py"],
                "infrastructure": ["repositories.py", "adapters.py"],
            }
            modules = default_modules.get(layer, ["__init__.py"])

        return modules

    def _get_layer_purpose(self, layer: str) -> str:
        """Get the purpose description for a layer."""
        purposes = {
            "domain": "Pure business logic with no external dependencies. Contains entities, value objects, and domain exceptions.",
            "application": "Use cases and orchestration. Defines ports (interfaces) and coordinates domain logic.",
            "infrastructure": "Concrete implementations of ports. Contains repositories, adapters, and external service integrations.",
        }
        return purposes.get(layer, "")

    def _default_folder_structure(
        self, specification: str, source_root: str = "", package_name: str = ""
    ) -> list[FolderStructure]:
        """Generate default folder structure if none found."""
        # Try to extract package name from specification or use provided values
        if source_root:
            # source_root like "src/taskmanager" -> use as base
            base = source_root.rstrip("/")
        elif package_name:
            base = f"src/{package_name}"
        else:
            package_match = re.search(r"src/(\w+)/", specification)
            package = package_match.group(1) if package_match else "app"
            base = f"src/{package}"

        return [
            FolderStructure(
                layer="domain",
                path=f"{base}/domain/",
                allowed_modules=["entities.py", "value_objects.py", "exceptions.py"],
                purpose="Pure business logic with no external dependencies",
            ),
            FolderStructure(
                layer="application",
                path=f"{base}/application/",
                allowed_modules=["ports.py", "use_cases.py", "services.py"],
                purpose="Use cases and orchestration logic",
            ),
            FolderStructure(
                layer="infrastructure",
                path=f"{base}/infrastructure/",
                allowed_modules=["repositories.py", "adapters.py"],
                purpose="Concrete implementations of ports",
            ),
        ]

    def _extract_layer_descriptions(self, specification: str) -> dict[str, str]:
        """Extract layer descriptions from specification."""
        descriptions = {}

        # Look for layer documentation sections
        layers = ["domain", "application", "infrastructure"]

        for layer in layers:
            # Pattern: "### Domain Layer" or "## domain/" followed by description
            pattern = rf"(?:###?\s*{layer}[^\n]*\n+)([^\n#]+)"
            match = re.search(pattern, specification, re.IGNORECASE)
            if match:
                descriptions[layer] = match.group(1).strip()[:200]
            else:
                descriptions[layer] = self._get_layer_purpose(layer)

        return descriptions
