"""
Gate 10: Infrastructure Validation Tests.

TDD tests that validate architectural constraints:
- Gate 10A: Dependency Direction (domain must not import infrastructure)
- Gate 10B: Container Injects Abstractions (type hints use interfaces)
- Gate 10C: ACL Pattern Compliance (domain uses only interface methods)
- Gate 10F: Infrastructure Testability (services accept mocked dependencies)

These tests enforce clean architecture and prevent ACL violations.
"""

import importlib
import inspect
from unittest.mock import MagicMock

import pytest


class TestGate10A_DependencyDirection:
    """Gate 10A: Domain MUST NOT import infrastructure."""

    def test_domain_does_not_import_infrastructure(self):
        """Verify domain modules don't import from infrastructure."""
        domain_modules = [
            "atomicguard.domain.models",
            "atomicguard.domain.interfaces",
            "atomicguard.domain.extraction",
            "atomicguard.domain.workflow",
        ]

        for module_name in domain_modules:
            try:
                module = importlib.import_module(module_name)
                source = inspect.getsource(module)

                # Check for direct imports
                assert (
                    "from atomicguard.infrastructure" not in source
                ), f"{module_name} imports infrastructure directly"
                assert (
                    "import atomicguard.infrastructure" not in source
                ), f"{module_name} imports infrastructure directly"
            except ModuleNotFoundError:
                # Module doesn't exist yet - skip
                pytest.skip(f"Module {module_name} not found")

    def test_domain_does_not_access_implementation_details(self):
        """Verify domain doesn't access _private attributes of interfaces."""
        try:
            import atomicguard.domain.extraction

            source = inspect.getsource(atomicguard.domain.extraction)

            # This catches the _artifacts access in extraction.py
            assert (
                "._artifacts" not in source
            ), "extraction.py accesses private _artifacts - use interface method"
            assert (
                'hasattr(dag, "_' not in source
            ), "extraction.py probes for private attributes"
        except ModuleNotFoundError:
            pytest.skip("extraction module not found")


class TestGate10B_ContainerAbstractions:
    """Gate 10B: Container must inject interfaces, not concrete classes."""

    def test_workflow_resumer_accepts_interface_type(self):
        """Verify WorkflowResumer uses interface type hints."""
        try:
            from atomicguard.domain.workflow import WorkflowResumer

            # Check raw annotations (avoids forward reference resolution issues)
            annotations = WorkflowResumer.__init__.__annotations__
            checkpoint_dag_hint = annotations.get("checkpoint_dag", "")

            # The hint should reference the interface
            hint_str = str(checkpoint_dag_hint)
            assert (
                "CheckpointDAGInterface" in hint_str
            ), f"checkpoint_dag should be typed as CheckpointDAGInterface, got {hint_str}"
        except ModuleNotFoundError:
            pytest.skip("Required modules not found")

    def test_human_amendment_processor_accepts_interface_type(self):
        """Verify HumanAmendmentProcessor uses interface type hints."""
        try:
            from atomicguard.domain.workflow import HumanAmendmentProcessor

            # Check raw annotations (avoids forward reference resolution issues)
            annotations = HumanAmendmentProcessor.__init__.__annotations__
            checkpoint_dag_hint = annotations.get("checkpoint_dag", "")

            hint_str = str(checkpoint_dag_hint)
            assert (
                "CheckpointDAGInterface" in hint_str
            ), f"checkpoint_dag should be typed as CheckpointDAGInterface, got {hint_str}"
        except ModuleNotFoundError:
            pytest.skip("Required modules not found")


class TestGate10C_ACLCompliance:
    """Gate 10C: Domain must use interface methods, not reach through to impl."""

    def test_artifact_dag_interface_has_get_all_method(self):
        """ArtifactDAGInterface should have get_all() for extraction."""
        from atomicguard.domain.interfaces import ArtifactDAGInterface

        # Check that get_all is defined on the interface
        assert hasattr(
            ArtifactDAGInterface, "get_all"
        ), "ArtifactDAGInterface should have get_all() method"

        # Verify it's an abstract method
        assert getattr(
            ArtifactDAGInterface.get_all, "__isabstractmethod__", False
        ), "get_all() should be an abstract method"

    def test_extraction_uses_interface_methods_only(self):
        """extract() must only call methods defined on ArtifactDAGInterface."""
        try:
            import atomicguard.domain.extraction

            source = inspect.getsource(atomicguard.domain.extraction)

            # Specific check: must use dag.get_all(), not _artifacts access
            # Either get_all() is called, or we fail
            uses_get_all = "dag.get_all()" in source or ".get_all()" in source

            # Should not have implementation-specific access
            has_private_access = "._artifacts" in source or 'hasattr(dag, "_' in source

            assert not has_private_access, "extraction.py should not access private attributes - use interface methods"
            assert (
                uses_get_all
            ), "extraction.py should use dag.get_all() interface method"
        except ModuleNotFoundError:
            pytest.skip("extraction module not found")


class TestGate10D_AbstractionNaming:
    """Gate 10D: Naming consistency for interfaces and implementations."""

    def test_interface_names_end_with_interface(self):
        """Interface names should end with 'Interface' for clarity."""
        from atomicguard.domain import interfaces

        interface_classes = [
            name
            for name, obj in inspect.getmembers(interfaces, inspect.isclass)
            if inspect.isabstract(obj) and not name.startswith("_")
        ]

        for name in interface_classes:
            assert name.endswith(
                "Interface"
            ), f"Abstract class {name} should end with 'Interface'"


class TestGate10F_InfrastructureTestability:
    """Gate 10F: Services must accept mocked dependencies."""

    def test_workflow_resumer_accepts_mock_dag(self):
        """WorkflowResumer works with any CheckpointDAGInterface impl."""
        try:
            from atomicguard.domain.interfaces import CheckpointDAGInterface
            from atomicguard.domain.workflow import WorkflowResumer

            mock_dag = MagicMock(spec=CheckpointDAGInterface)
            resumer = WorkflowResumer(checkpoint_dag=mock_dag)

            # Should have stored the mock
            assert resumer._checkpoint_dag is mock_dag
        except ModuleNotFoundError:
            pytest.skip("Required modules not found")

    def test_human_amendment_processor_accepts_mock_dag(self):
        """HumanAmendmentProcessor works with any CheckpointDAGInterface impl."""
        try:
            from atomicguard.domain.interfaces import CheckpointDAGInterface
            from atomicguard.domain.workflow import HumanAmendmentProcessor

            mock_dag = MagicMock(spec=CheckpointDAGInterface)
            processor = HumanAmendmentProcessor(checkpoint_dag=mock_dag)

            assert processor._checkpoint_dag is mock_dag
        except ModuleNotFoundError:
            pytest.skip("Required modules not found")

    def test_in_memory_dag_implements_interface(self):
        """InMemoryArtifactDAG implements all ArtifactDAGInterface methods."""
        from atomicguard.domain.interfaces import ArtifactDAGInterface
        from atomicguard.infrastructure.persistence.memory import InMemoryArtifactDAG

        # Get abstract methods from interface
        abstract_methods = {
            name
            for name, method in inspect.getmembers(
                ArtifactDAGInterface, predicate=inspect.isfunction
            )
            if getattr(method, "__isabstractmethod__", False)
        }

        # Get methods from implementation
        impl_methods = {
            name
            for name, _ in inspect.getmembers(
                InMemoryArtifactDAG, predicate=inspect.isfunction
            )
        }

        # All abstract methods should be implemented
        missing = abstract_methods - impl_methods
        assert not missing, f"InMemoryArtifactDAG is missing methods: {missing}"

    def test_filesystem_dag_implements_interface(self):
        """FilesystemArtifactDAG implements all ArtifactDAGInterface methods."""
        from atomicguard.domain.interfaces import ArtifactDAGInterface
        from atomicguard.infrastructure.persistence.filesystem import (
            FilesystemArtifactDAG,
        )

        # Get abstract methods from interface
        abstract_methods = {
            name
            for name, method in inspect.getmembers(
                ArtifactDAGInterface, predicate=inspect.isfunction
            )
            if getattr(method, "__isabstractmethod__", False)
        }

        # Get methods from implementation
        impl_methods = {
            name
            for name, _ in inspect.getmembers(
                FilesystemArtifactDAG, predicate=inspect.isfunction
            )
        }

        # All abstract methods should be implemented
        missing = abstract_methods - impl_methods
        assert not missing, f"FilesystemArtifactDAG is missing methods: {missing}"
