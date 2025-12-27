"""Pydantic models for SDLC example.

Defines structured output types for BDD and Coder generators.
"""

from pydantic import BaseModel, Field

# =============================================================================
# BDD Generator Output Models
# =============================================================================


class GherkinStep(BaseModel):
    """A single step in a Gherkin scenario."""

    keyword: str = Field(description="Step keyword: Given, When, Then, And, But")
    text: str = Field(description="Step description text")


class BDDScenario(BaseModel):
    """A single BDD scenario in Gherkin format."""

    name: str = Field(description="Scenario name")
    given: list[str] = Field(description="Given steps (preconditions)")
    when: list[str] = Field(description="When steps (actions)")
    then: list[str] = Field(description="Then steps (expected outcomes)")
    tags: list[str] = Field(default_factory=list, description="Optional tags")


class BDDFeature(BaseModel):
    """A complete BDD feature with scenarios."""

    name: str = Field(description="Feature name")
    description: str = Field(description="Feature description")
    scenarios: list[BDDScenario] = Field(description="List of scenarios")


class BDDScenarios(BaseModel):
    """Output from BDDGenerator - a collection of BDD features."""

    features: list[BDDFeature] = Field(description="List of features")
    feature_file_content: str = Field(description="Complete .feature file content")


# =============================================================================
# Coder Generator Output Models
# =============================================================================


class ImplementationFile(BaseModel):
    """A single file to be written by the coder."""

    path: str = Field(description="Relative file path")
    content: str = Field(description="File content")


class ImplementationManifest(BaseModel):
    """Output from CoderGenerator - files implementing the feature."""

    files: list[ImplementationFile] = Field(description="Files to create/update")
    summary: str = Field(description="Brief summary of implementation")
    tests_expected_to_pass: list[str] = Field(
        default_factory=list,
        description="Test names expected to pass with this implementation",
    )


# =============================================================================
# Project Configuration (reused from ADD)
# =============================================================================


class ProjectConfig(BaseModel):
    """Project configuration extracted from documentation."""

    source_root: str = Field(description="Root directory for source code (e.g., 'src')")
    package_name: str = Field(description="Main package name (e.g., 'myapp')")
    test_framework: str = Field(default="pytest", description="Test framework to use")
    test_directory: str = Field(default="tests", description="Directory for test files")
