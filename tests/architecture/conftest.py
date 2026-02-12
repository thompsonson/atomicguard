"""Shared fixtures for architecture tests."""

import os

import pytest
from pytestarch import (
    EvaluableArchitecture,
    LayeredArchitecture,
    get_evaluable_architecture,
)


@pytest.fixture(scope="session")
def evaluable() -> EvaluableArchitecture:
    """Build evaluable architecture from src/atomicguard."""
    src_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "src")
    )
    project_path = os.path.join(src_dir, "atomicguard")
    return get_evaluable_architecture(src_dir, project_path)


@pytest.fixture(scope="session")
def layers() -> LayeredArchitecture:
    """Define the three DDD layers.

    PyTestArch resolves module names relative to the source root,
    so modules appear as 'src.atomicguard.domain', etc.
    """
    return (
        LayeredArchitecture()
        .layer("domain")
        .containing_modules(["src.atomicguard.domain"])
        .layer("application")
        .containing_modules(["src.atomicguard.application"])
        .layer("infrastructure")
        .containing_modules(["src.atomicguard.infrastructure"])
    )
