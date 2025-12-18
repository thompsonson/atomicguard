# Dual-State Agent: Guard Library

## Theoretical Guard Catalog

From paper Appendix C. Guards organized by SDLC phase.

| ID | Transition | Predicates |
|----|------------|------------|
| G₁ | INTENT → DOMAIN_MODEL | entities_identified ∧ value_objects_identified ∧ invariants_documented |
| G₄ | SKELETON → ARCH_TESTS | pytestarch_syntax_valid ∧ all_gates_have_tests |
| G₇ | FILE_REQUEST → FILE_VALID | path_in_documented_structure ∧ layer_boundaries_enforced |
| G₈ | CODE_GEN → SYNTAX_VALID | ast_parse_succeeds ∧ imports_resolve |
| G₉ | SYNTAX → TYPE_VALID | mypy_check_passes ∧ type_annotations_present |
| G₁₀ | TYPE → FUNCTIONALLY_CORRECT | unit_tests_pass ∧ coverage ≥ threshold |
| G₁₁ | IMPL → ARCH_VALID | domain_never_imports_infrastructure ∧ no_circular_deps |
| G₁₂ | ARCH → DI_VALID | container_registers_interfaces_only |
| G₁₅ | BDD → QUALITY_GATES | code_formatted ∧ linter_score ≥ threshold ∧ security_clean |

---

## Python Implementations

### Core Guards (PoC)

```python
import ast
import subprocess
from pathlib import Path
from typing import Set

class SyntaxGuard(GuardInterface):
    """G₈: Validates Python AST parsing."""

    def validate(self, artifact: Artifact, **deps: Artifact) -> GuardResult:
        try:
            ast.parse(artifact.content)
            return GuardResult(passed=True)
        except SyntaxError as e:
            return GuardResult(
                passed=False,
                feedback=f"Line {e.lineno}: {e.msg}"
            )


class ImportsResolveGuard(GuardInterface):
    """G₈ (partial): Validates all imports can resolve."""

    def __init__(self, allowed_modules: Set[str] = None):
        self._allowed = allowed_modules or set()

    def validate(self, artifact: Artifact, **deps: Artifact) -> GuardResult:
        try:
            tree = ast.parse(artifact.content)
        except SyntaxError as e:
            return GuardResult(passed=False, feedback=f"Syntax error: {e}")

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if self._allowed and alias.name.split('.')[0] not in self._allowed:
                        return GuardResult(
                            passed=False,
                            feedback=f"Unresolved import: {alias.name}"
                        )
            elif isinstance(node, ast.ImportFrom):
                if node.module and self._allowed:
                    if node.module.split('.')[0] not in self._allowed:
                        return GuardResult(
                            passed=False,
                            feedback=f"Unresolved import: from {node.module}"
                        )

        return GuardResult(passed=True)


class TestGuard(GuardInterface):
    """G₁₀: Validates via test execution."""

    def validate(self, artifact: Artifact, **deps: Artifact) -> GuardResult:
        test_artifact = deps.get('test')
        test_code = test_artifact.content if test_artifact else None

        if not test_code:
            return GuardResult(passed=False, feedback="No test artifact provided")

        namespace = {}
        try:
            exec(artifact.content, namespace)
            exec(test_code, namespace)
            return GuardResult(passed=True)
        except AssertionError as e:
            return GuardResult(passed=False, feedback=f"Test failed: {e}")
        except Exception as e:
            return GuardResult(passed=False, feedback=f"{type(e).__name__}: {e}")
```

### DDD Guards

```python
class ArchitectureBoundaryGuard(GuardInterface):
    """
    G₁₁: domain_never_imports_infrastructure
    Validates Clean Architecture dependency rule.
    """

    FORBIDDEN_IN_DOMAIN = {'boto3', 'sqlalchemy', 'requests', 'django', 'flask', 'fastapi'}

    def validate(self, artifact: Artifact, **deps: Artifact) -> GuardResult:
        try:
            tree = ast.parse(artifact.content)
        except SyntaxError as e:
            return GuardResult(passed=False, feedback=f"Syntax error: {e}")

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in self.FORBIDDEN_IN_DOMAIN:
                        return GuardResult(
                            passed=False,
                            feedback=f"Domain imports infrastructure: {alias.name}"
                        )
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    root = node.module.split('.')[0]
                    if root in self.FORBIDDEN_IN_DOMAIN:
                        return GuardResult(
                            passed=False,
                            feedback=f"Domain imports infrastructure: {node.module}"
                        )
                    if 'infrastructure' in node.module:
                        return GuardResult(
                            passed=False,
                            feedback=f"Domain imports infrastructure layer: {node.module}"
                        )

        return GuardResult(passed=True)


class ACLInterfaceGuard(GuardInterface):
    """
    G₁₂ (partial): Validates ACL defines abstract interfaces.
    """

    def validate(self, artifact: Artifact, **deps: Artifact) -> GuardResult:
        try:
            tree = ast.parse(artifact.content)
        except SyntaxError as e:
            return GuardResult(passed=False, feedback=f"Syntax error: {e}")

        has_abc_import = False
        interfaces = []

        for node in ast.walk(tree):
            # Check for ABC import
            if isinstance(node, ast.ImportFrom):
                if node.module == 'abc':
                    has_abc_import = True

            # Check for abstract classes
            if isinstance(node, ast.ClassDef):
                for base in node.bases:
                    if isinstance(base, ast.Name) and base.id == 'ABC':
                        interfaces.append(node.name)

        if not has_abc_import:
            return GuardResult(
                passed=False,
                feedback="ACL must import from abc module"
            )

        if not interfaces:
            return GuardResult(
                passed=False,
                feedback="ACL must define abstract interfaces (ABC subclasses)"
            )

        return GuardResult(passed=True)


class DIContainerGuard(GuardInterface):
    """
    G₁₂: container_registers_interfaces_only
    Validates DI binds abstractions to concretions.
    """

    def validate(self, artifact: Artifact, **deps: Artifact) -> GuardResult:
        try:
            tree = ast.parse(artifact.content)
        except SyntaxError as e:
            return GuardResult(passed=False, feedback=f"Syntax error: {e}")

        registrations = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr == 'register':
                    if len(node.args) >= 2:
                        key = node.args[0]
                        if isinstance(key, ast.Name):
                            if not key.id.startswith('I') and not key.id.endswith('Interface'):
                                return GuardResult(
                                    passed=False,
                                    feedback=f"DI key '{key.id}' is not an interface (must start with 'I' or end with 'Interface')"
                                )
                            registrations.append(key.id)

        if not registrations:
            return GuardResult(
                passed=False,
                feedback="No service registrations found"
            )

        return GuardResult(passed=True)
```

### Quality Guards

```python
class MypyGuard(GuardInterface):
    """G₉: mypy_check_passes"""

    def validate(self, artifact: Artifact, **deps: Artifact) -> GuardResult:
        # Write to temp file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(artifact.content)
            tmp_path = f.name

        try:
            result = subprocess.run(
                ['mypy', '--strict', tmp_path],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                return GuardResult(passed=True)
            else:
                return GuardResult(
                    passed=False,
                    feedback=result.stdout or result.stderr
                )
        except FileNotFoundError:
            return GuardResult(passed=False, feedback="mypy not installed")
        except subprocess.TimeoutExpired:
            return GuardResult(passed=False, feedback="mypy timeout")
        finally:
            Path(tmp_path).unlink(missing_ok=True)


class RuffGuard(GuardInterface):
    """G₁₅ (partial): code_formatted ∧ linter_score"""

    def validate(self, artifact: Artifact, **deps: Artifact) -> GuardResult:
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(artifact.content)
            tmp_path = f.name

        try:
            result = subprocess.run(
                ['ruff', 'check', tmp_path],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                return GuardResult(passed=True)
            else:
                return GuardResult(
                    passed=False,
                    feedback=result.stdout or result.stderr
                )
        except FileNotFoundError:
            return GuardResult(passed=False, feedback="ruff not installed")
        finally:
            Path(tmp_path).unlink(missing_ok=True)
```

### Composite Guards

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

class CompositeGuard(GuardInterface):
    """Combines multiple guards with AND semantics (sequential)."""

    def __init__(self, guards: List[GuardInterface]):
        self._guards = guards

    def validate(self, artifact: Artifact, **deps: Artifact) -> GuardResult:
        for guard in self._guards:
            result = guard.validate(artifact, **deps)
            if not result.passed:
                return result  # Fail fast
        return GuardResult(passed=True)


class ParallelGuard(GuardInterface):
    """
    Combines multiple guards with AND semantics (parallel execution).
    Use for independent guards (e.g., Syntax + Ruff + Mypy).
    """

    def __init__(self, guards: List[GuardInterface], max_workers: int = 4):
        self._guards = guards
        self._max_workers = max_workers

    def validate(self, artifact: Artifact, **deps: Artifact) -> GuardResult:
        failures = []

        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = {
                executor.submit(g.validate, artifact, **deps): g
                for g in self._guards
            }

            for future in as_completed(futures):
                result = future.result()
                if not result.passed:
                    failures.append(result.feedback)

        if failures:
            return GuardResult(
                passed=False,
                feedback="\n---\n".join(failures)
            )
        return GuardResult(passed=True)


# G₂₀: PRODUCTION_READY composite (parallel)
def create_production_guard() -> ParallelGuard:
    return ParallelGuard([
        SyntaxGuard(),           # Fast, independent
        ArchitectureBoundaryGuard(),  # AST-based, independent
        MypyGuard(),             # Subprocess, slow
        RuffGuard()              # Subprocess, slow
    ])
```

---

## Guard Parallelization Strategy

| Guard Type | Parallel-Safe | Notes |
|------------|---------------|-------|
| AST-based (Syntax, ACL) | ✓ | Pure functions, no side effects |
| Subprocess (Mypy, Ruff) | ✓ | Separate processes, temp files |
| Test execution | ✗ | Shared namespace, use sequential |
| File I/O guards | Depends | Check for write conflicts |

**Rule**: Use `ParallelGuard` for independent checks, `CompositeGuard` when order matters or state is shared.

---

## Guard Selection by Task

| Task Type | Recommended Guards |
|-----------|-------------------|
| Code generation | SyntaxGuard → TestGuard |
| Domain modeling | ACLInterfaceGuard, ArchitectureBoundaryGuard |
| DI configuration | DIContainerGuard |
| Pre-commit | CompositeGuard(Syntax, Mypy, Ruff) |
| Production gate | create_production_guard() |
