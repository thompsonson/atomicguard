"""
Pydantic models for the Enhanced SDLC workflow.

These models define the schema for PydanticAI agents, ensuring
type-safe extraction of architecture gates, BDD scenarios, and test generation.

Additionally includes models for Extension support:
- Extension 07: IncrementalConfig for incremental execution settings
"""

from typing import Literal

from pydantic import BaseModel, Field

# =============================================================================
# Global Constraints (Ω) - Project Configuration
# =============================================================================


class ProjectConfig(BaseModel):
    """
    Structured global constraints (Ω) for SDLC workflow.

    Per the paper's Hierarchical Context Composition:
    ℰ (Ambient Environment) = ⟨ℛ, Ω⟩

    This model represents Ω - project-wide configuration that applies
    to ALL action pairs, extracted deterministically before the workflow starts.
    """

    source_root: str = Field(
        default="",
        description="Path to Python package root, e.g., 'src/myapp'",
    )
    package_name: str = Field(
        default="",
        description="Python package name, e.g., 'myapp'",
    )


# =============================================================================
# Rules Extraction: Structured constraints for code generation
# =============================================================================


class ImportRule(BaseModel):
    """Single import constraint rule extracted from architecture tests."""

    source_layer: str = Field(
        description="Layer that has the constraint, e.g., 'domain'"
    )
    forbidden_targets: list[str] = Field(
        description="Layers that cannot be imported, e.g., ['application', 'infrastructure']"
    )
    rationale: str = Field(
        default="",
        description="Why this constraint exists, e.g., 'Domain must be pure'",
    )


class FolderStructure(BaseModel):
    """Expected folder structure for a layer."""

    layer: str = Field(description="Layer name, e.g., 'domain'")
    path: str = Field(description="Relative path, e.g., 'src/taskmanager/domain/'")
    allowed_modules: list[str] = Field(
        default_factory=list,
        description="Expected modules, e.g., ['entities.py', 'value_objects.py']",
    )
    purpose: str = Field(
        default="", description="Layer purpose, e.g., 'Pure business logic'"
    )


class ArchitectureRules(BaseModel):
    """
    Extracted architecture rules from g_add tests and specification.

    Provides structured, actionable constraints for the coder agent.
    This is a deterministic extraction - no LLM needed.
    """

    import_rules: list[ImportRule] = Field(
        description="Import constraints extracted from architecture tests"
    )
    folder_structure: list[FolderStructure] = Field(
        description="Expected folder structure from specification"
    )
    dependency_direction: str = Field(
        default="infrastructure → application → domain",
        description="Direction of allowed dependencies",
    )
    layer_descriptions: dict[str, str] = Field(
        default_factory=dict,
        description="Layer name -> purpose mapping",
    )
    package_name: str = Field(
        default="",
        description="Python package name from g_config",
    )
    source_root: str = Field(
        default="",
        description="Source root path from g_config",
    )


# =============================================================================
# ADD Action Pair: Gate Extraction
# =============================================================================


class ArchitectureGate(BaseModel):
    """Single architecture gate extracted from documentation."""

    gate_id: str = Field(description="Unique identifier, e.g., 'Gate1', 'Gate10A'")
    description: str = Field(description="What this gate enforces")
    layer: Literal["domain", "application", "infrastructure"] = Field(
        description="Which architectural layer this gate applies to"
    )
    constraint_type: Literal["dependency", "naming", "containment", "injection"] = (
        Field(description="Type of architectural constraint")
    )
    source_section: str = Field(
        description="Reference to the documentation section where this gate is defined"
    )


class GatesExtractionResult(BaseModel):
    """
    Output of DocParserGenerator - extracted architecture gates.

    Note: source_root is NOT included here as it belongs to Ω (Global Constraints),
    not to the gates artifact (ℛ). See ProjectConfig for source_root.
    """

    gates: list[ArchitectureGate] = Field(
        description="List of architecture gates extracted from documentation"
    )
    ubiquitous_terms: dict[str, str] = Field(
        default_factory=dict,
        description="Domain terms and their definitions from the documentation",
    )
    layer_boundaries: list[str] = Field(
        default_factory=list,
        description="Layer boundary rules, e.g., 'domain cannot import infrastructure'",
    )


# =============================================================================
# ADD Action Pair: Test Generation
# =============================================================================


class ArchitectureTest(BaseModel):
    """Single pytest-arch test generated from a gate."""

    gate_id: str = Field(description="The gate this test enforces")
    test_name: str = Field(
        description="Pytest function name, e.g., 'test_gate_1_domain_no_infra_imports'"
    )
    test_code: str = Field(description="Complete Python test function code")
    imports_required: list[str] = Field(
        default_factory=list,
        description="Import statements needed for this test",
    )
    documentation_reference: str = Field(
        default="",
        description="Reference back to architecture documentation",
    )


class TestSuite(BaseModel):
    """Output of TestCodeGenerator - complete test module."""

    module_docstring: str = Field(
        description="Module-level docstring explaining the test suite"
    )
    imports: list[str] = Field(description="All import statements for the test module")
    fixtures: list[str] = Field(
        default_factory=list,
        description="Pytest fixture definitions if needed",
    )
    tests: list[ArchitectureTest] = Field(
        description="List of generated test functions"
    )


# =============================================================================
# ADD Action Pair: Artifact Packaging
# =============================================================================


class FileToWrite(BaseModel):
    """Single file in the artifact manifest."""

    path: str = Field(description="Relative path from workspace root")
    content: str = Field(description="File content to write")


class ArtifactManifest(BaseModel):
    """Output of FileWriterGenerator - manifest of generated files."""

    files: list[FileToWrite] = Field(description="List of files to write")
    test_count: int = Field(description="Number of tests generated")
    gates_covered: list[str] = Field(description="Gate IDs covered by generated tests")


# =============================================================================
# BDD Action Pair: Scenario Models
# =============================================================================


class BDDScenario(BaseModel):
    """Single BDD scenario in Gherkin format."""

    name: str = Field(description="Scenario name")
    feature: str = Field(description="Parent feature name")
    gherkin: str = Field(description="Full Gherkin scenario text")


class BDDScenariosResult(BaseModel):
    """Output of BDDGenerator - extracted BDD scenarios."""

    feature_name: str = Field(description="Name of the feature being tested")
    scenarios: list[BDDScenario] = Field(description="List of BDD scenarios")
    background: str | None = Field(
        default=None,
        description="Common setup steps shared across scenarios",
    )


# =============================================================================
# Coder Action Pair: Implementation Result
# =============================================================================


class ImplementationResult(BaseModel):
    """Output of CoderGenerator - generated implementation files."""

    files: list[FileToWrite] = Field(description="Implementation files to write")
    summary: str = Field(description="Brief summary of what was implemented")


# =============================================================================
# Extension 07: Incremental Execution Configuration
# =============================================================================


class IncrementalConfig(BaseModel):
    """
    Configuration for incremental execution (Extension 07).

    Controls how incremental execution behaves, including
    whether to use caching and how to handle changed configurations.
    """

    enabled: bool = Field(
        default=True,
        description="Enable incremental execution (skip unchanged steps)",
    )
    cache_artifacts: bool = Field(
        default=True,
        description="Cache accepted artifacts for reuse",
    )
    propagate_changes: bool = Field(
        default=True,
        description="Propagate upstream changes to downstream steps (Merkle propagation)",
    )


# =============================================================================
# Extension Flags for Workflow Configuration
# =============================================================================


class ExtensionFlags(BaseModel):
    """
    Flags indicating which extensions are enabled for this workflow.

    Used in workflow.json to configure extension behavior.
    """

    versioned_environment: bool = Field(
        default=True,
        description="Enable Extension 01: Versioned Environment (W_ref, config_ref)",
    )
    artifact_extraction: bool = Field(
        default=True,
        description="Enable Extension 02: Artifact Extraction (predicate queries)",
    )
    multi_agent: bool = Field(
        default=True,
        description="Enable Extension 03: Multi-Agent System (shared repository coordination)",
    )
    incremental_execution: bool = Field(
        default=True,
        description="Enable Extension 07: Incremental Execution (skip unchanged steps)",
    )
