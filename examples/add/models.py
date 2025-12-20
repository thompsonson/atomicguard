"""
Pydantic models for structured LLM output in ADD workflow.

These models define the schema for PydanticAI agents, ensuring
type-safe extraction of architecture gates and test generation.
"""

from typing import Literal

from pydantic import BaseModel, Field

# =============================================================================
# Action Pair 1: Gate Extraction
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
    """Output of DocParserGenerator - extracted architecture gates."""

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
# Action Pair 2: Test Generation
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
# Action Pair 3: Artifact Packaging
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
