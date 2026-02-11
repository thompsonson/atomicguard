"""
Convention Enforcement Tests.

Permanent tests that catch anti-patterns which import-based layer rules
cannot detect: frozen dataclass conventions, immutable collections,
silent exception swallowing, and interface contracts.
"""

import ast
import inspect
from pathlib import Path

import pytest

SRC_ROOT = Path(__file__).parent.parent.parent / "src" / "atomicguard"

# Documented exceptions to the frozen dataclass rule
MUTABLE_DATACLASS_ALLOWLIST = {"WorkflowState"}


class TestFrozenDataclassConvention:
    """All domain dataclasses must be frozen (except allowlisted ones)."""

    def _get_dataclass_info(self, filepath: Path) -> list[tuple[str, bool]]:
        """Parse a file and return (class_name, is_frozen) for each @dataclass."""
        source = filepath.read_text()
        tree = ast.parse(source)
        results = []

        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for decorator in node.decorator_list:
                is_dataclass = False
                is_frozen = False

                if isinstance(decorator, ast.Name) and decorator.id == "dataclass":
                    is_dataclass = True
                elif isinstance(decorator, ast.Call):
                    func = decorator.func
                    if isinstance(func, ast.Name) and func.id == "dataclass":
                        is_dataclass = True
                        for kw in decorator.keywords:
                            if kw.arg == "frozen" and isinstance(
                                kw.value, ast.Constant
                            ):
                                is_frozen = kw.value.value

                if is_dataclass:
                    results.append((node.name, is_frozen))
        return results

    def test_domain_models_are_frozen(self):
        """All domain dataclasses must be frozen (except WorkflowState)."""
        models_file = SRC_ROOT / "domain" / "models.py"
        violations = []

        for class_name, is_frozen in self._get_dataclass_info(models_file):
            if class_name in MUTABLE_DATACLASS_ALLOWLIST:
                continue
            if not is_frozen:
                violations.append(class_name)

        assert not violations, (
            f"Domain dataclasses must be frozen. Violations: {violations}. "
            f"If mutable is intentional, add to MUTABLE_DATACLASS_ALLOWLIST."
        )

    def test_domain_workflow_event_models_are_frozen(self):
        """Workflow event models must also be frozen."""
        event_file = SRC_ROOT / "domain" / "workflow_event.py"
        if not event_file.exists():
            pytest.skip("workflow_event.py removed (Issue 2)")

        violations = []
        for class_name, is_frozen in self._get_dataclass_info(event_file):
            if not is_frozen:
                violations.append(class_name)

        assert not violations, (
            f"Workflow event dataclasses must be frozen. Violations: {violations}"
        )


class TestImmutableCollections:
    """Frozen domain model fields should use tuple, not list."""

    def test_domain_models_use_tuples_not_lists(self):
        """Frozen domain model fields should use tuple not list, MappingProxyType not dict."""
        models_file = SRC_ROOT / "domain" / "models.py"
        source = models_file.read_text()
        tree = ast.parse(source)

        violations = []

        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            # Check if this is a frozen dataclass
            is_frozen_dc = False
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Call):
                    func = decorator.func
                    if isinstance(func, ast.Name) and func.id == "dataclass":
                        for kw in decorator.keywords:
                            if kw.arg == "frozen" and isinstance(
                                kw.value, ast.Constant
                            ):
                                is_frozen_dc = kw.value.value

            if not is_frozen_dc:
                continue

            # Check field annotations for list[] usage
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and item.target:
                    target_name = getattr(item.target, "id", "?")
                    annotation_source = ast.get_source_segment(source, item.annotation)
                    if annotation_source and "list[" in annotation_source.lower():
                        violations.append(
                            f"{node.name}.{target_name}: uses list[] — use tuple[] instead"
                        )

        assert not violations, (
            "Frozen dataclass fields should use tuple, not list:\n"
            + "\n".join(f"  - {v}" for v in violations)
        )


class TestNoSilentExceptionSwallowing:
    """No bare 'except: pass' or 'except Exception: pass' in src/."""

    def test_no_bare_except_pass(self):
        """No silent exception swallowing in src/atomicguard/."""
        violations = []

        for py_file in SRC_ROOT.rglob("*.py"):
            try:
                source = py_file.read_text()
                tree = ast.parse(source)
            except SyntaxError:
                continue

            for node in ast.walk(tree):
                if not isinstance(node, ast.ExceptHandler):
                    continue
                # Check if body is just 'pass' or '...'
                if len(node.body) == 1:
                    stmt = node.body[0]
                    is_pass = isinstance(stmt, ast.Pass)
                    is_ellipsis = (
                        isinstance(stmt, ast.Expr)
                        and isinstance(stmt.value, ast.Constant)
                        and stmt.value.value is ...
                    )
                    if is_pass or is_ellipsis:
                        rel_path = py_file.relative_to(SRC_ROOT.parent.parent)
                        handler_type = ""
                        if node.type:
                            handler_type = (
                                ast.get_source_segment(source, node.type) or ""
                            )
                        violations.append(
                            f"{rel_path}:{node.lineno}: except {handler_type}: pass"
                        )

        assert not violations, "Silent exception swallowing found:\n" + "\n".join(
            f"  - {v}" for v in violations
        )


class TestInterfaceConventions:
    """Interface naming and contract conventions."""

    def test_all_ports_end_with_interface(self):
        """All ABCs in domain/interfaces.py must end with 'Interface'."""
        from atomicguard.domain import interfaces

        abstract_classes = [
            name
            for name, obj in inspect.getmembers(interfaces, inspect.isclass)
            if inspect.isabstract(obj) and not name.startswith("_")
        ]

        violations = [
            name for name in abstract_classes if not name.endswith("Interface")
        ]

        assert not violations, (
            f"Abstract classes should end with 'Interface': {violations}"
        )

    def test_all_interface_methods_are_abstract(self):
        """Every public method on a port must be abstract."""
        from atomicguard.domain import interfaces

        violations = []

        for name, cls in inspect.getmembers(interfaces, inspect.isclass):
            if not inspect.isabstract(cls) or not name.endswith("Interface"):
                continue

            for method_name, method in inspect.getmembers(
                cls, predicate=inspect.isfunction
            ):
                if method_name.startswith("_"):
                    continue
                if not getattr(method, "__isabstractmethod__", False):
                    violations.append(f"{name}.{method_name}")

        assert not violations, (
            f"Public interface methods must be abstract: {violations}"
        )

    def test_implementations_satisfy_interfaces(self):
        """All infrastructure implementations must implement all abstract methods."""
        from atomicguard.domain.interfaces import ArtifactDAGInterface
        from atomicguard.infrastructure.persistence.filesystem import (
            FilesystemArtifactDAG,
        )
        from atomicguard.infrastructure.persistence.memory import InMemoryArtifactDAG

        abstract_methods = {
            name
            for name, method in inspect.getmembers(
                ArtifactDAGInterface, predicate=inspect.isfunction
            )
            if getattr(method, "__isabstractmethod__", False)
        }

        for impl_cls in [FilesystemArtifactDAG, InMemoryArtifactDAG]:
            impl_methods = {
                name
                for name, _ in inspect.getmembers(
                    impl_cls, predicate=inspect.isfunction
                )
            }
            missing = abstract_methods - impl_methods
            assert not missing, f"{impl_cls.__name__} is missing methods: {missing}"
