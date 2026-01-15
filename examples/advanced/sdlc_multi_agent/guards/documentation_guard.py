"""
Documentation Guard: Validates DDD documentation completeness and structure.

Validates that all required DDD documentation files exist and have valid structure.
"""

import json
from pathlib import Path
from typing import Any

from ..interfaces import GuardResult, IGuard, WorkspaceManifest


class DocumentationGuard(IGuard):
    """Validate DDD documentation generation.

    Responsibilities:
    - Check all required files exist
    - Validate markdown structure
    - Extract metadata (entities, gates)

    Does NOT:
    - Call LLM (deterministic only)
    - Validate semantic correctness (future enhancement)
    - Store artifacts
    """

    REQUIRED_FILES = [
        "docs/domain_model.md",
        "docs/infrastructure_requirements.md",
        "docs/project_structure.md",
        "docs/ubiquitous_language.md",
    ]

    def validate(
        self, manifest: WorkspaceManifest, workspace: Path, context: dict[str, Any]
    ) -> GuardResult:
        """Validate DDD documentation.

        Args:
            manifest: Artifact content to validate
            workspace: Filesystem location (for verification)
            context: Dependencies and configuration

        Returns:
            GuardResult with verdict and feedback

        Validation checks:
        1. All required files present
        2. Each file has required sections
        3. No empty files
        """
        # Check 1: All required files exist
        file_paths = {f["path"] for f in manifest.files}
        missing_files = [f for f in self.REQUIRED_FILES if f not in file_paths]

        if missing_files:
            return GuardResult(
                passed=False,
                feedback=f"Missing required documentation files: {missing_files}",
                artifacts=None,
            )

        # Check 2: Validate structure of each file
        for required_file in self.REQUIRED_FILES:
            file_entry = next(f for f in manifest.files if f["path"] == required_file)
            content = file_entry["content"]

            validation = self._validate_file_structure(required_file, content)
            if not validation.passed:
                return validation

        # Check 3: Extract metadata (entities, gates)
        metadata = self._extract_metadata(manifest)

        return GuardResult(
            passed=True,
            feedback="",
            artifacts={
                "validated_files": self.REQUIRED_FILES,
                "entities": metadata.get("entities", []),
                "gates": metadata.get("gates", []),
            },
        )

    def _validate_file_structure(self, filename: str, content: str) -> GuardResult:
        """Validate structure of a specific documentation file.

        Args:
            filename: Name of the file
            content: File content

        Returns:
            GuardResult
        """
        if not content or len(content.strip()) < 50:
            return GuardResult(
                passed=False,
                feedback=f"{filename} is empty or too short (min 50 chars)",
            )

        if filename == "docs/domain_model.md":
            required_sections = ["## Entities", "## Value Objects"]
            for section in required_sections:
                if section not in content:
                    return GuardResult(
                        passed=False,
                        feedback=f"{filename} missing required section: {section}",
                    )

        elif filename == "docs/infrastructure_requirements.md":
            if "Gate" not in content:
                return GuardResult(
                    passed=False,
                    feedback=f"{filename} missing Gate definitions",
                )

        elif filename == "docs/project_structure.md":
            required_keywords = ["src/", "domain/"]
            if not any(keyword in content for keyword in required_keywords):
                return GuardResult(
                    passed=False,
                    feedback=f"{filename} missing directory structure (expecting src/domain/)",
                )

        elif filename == "docs/ubiquitous_language.md":
            if "##" not in content:
                return GuardResult(
                    passed=False,
                    feedback=f"{filename} missing term definitions (expecting markdown sections)",
                )

        return GuardResult(passed=True, feedback="")

    def _extract_metadata(self, manifest: WorkspaceManifest) -> dict[str, Any]:
        """Extract metadata from documentation.

        Args:
            manifest: Workspace manifest

        Returns:
            Dict with entities, gates, etc.
        """
        entities = []
        gates = []

        for file_entry in manifest.files:
            content = file_entry["content"]
            path = file_entry["path"]

            if "domain_model.md" in path:
                # Extract entity names (simple heuristic: lines starting with "### ")
                for line in content.split("\n"):
                    if line.startswith("### ") and any(
                        keyword in line.lower()
                        for keyword in ["entity", "aggregate", "root"]
                    ):
                        # Extract entity name
                        entity_name = (
                            line.replace("### ", "")
                            .split(":")[0]
                            .split("(")[0]
                            .strip()
                        )
                        if entity_name and entity_name not in entities:
                            entities.append(entity_name)

            if "infrastructure_requirements.md" in path:
                # Extract gate names (lines containing "Gate ")
                for line in content.split("\n"):
                    if "Gate " in line and any(
                        line.strip().startswith(prefix)
                        for prefix in ["**Gate", "### Gate", "## Gate"]
                    ):
                        gates.append(line.strip()[:80])  # First 80 chars

        return {
            "entities": entities,
            "gates": gates,
            "total_files": len(manifest.files),
        }
