"""
Test execution guards.

Guards that validate artifacts by running tests against them.
"""

import multiprocessing
import sys
from typing import Any

from atomicguard.domain.interfaces import GuardInterface
from atomicguard.domain.models import Artifact, GuardResult


class TestGuard(GuardInterface):
    """
    Validates artifact via test execution in the same process.

    Simple guard that executes test code against artifact content.
    For isolation, use DynamicTestGuard instead.
    """

    def __init__(self, test_code: str | None = None):
        """
        Args:
            test_code: Static test code to run (if not using dependencies)
        """
        self._static_test_code = test_code

    def validate(self, artifact: Artifact, **deps: Any) -> GuardResult:
        """
        Execute test code against artifact.

        Args:
            artifact: The implementation artifact to test
            **deps: May include 'test' artifact with test code

        Returns:
            GuardResult with test outcome
        """
        test_artifact = deps.get("test")
        test_code = test_artifact.content if test_artifact else self._static_test_code

        if not test_code:
            return GuardResult(passed=False, feedback="No test code provided")

        namespace: dict[str, Any] = {}
        try:
            exec(artifact.content, namespace)
            exec(test_code, namespace)
            return GuardResult(passed=True)
        except AssertionError as e:
            return GuardResult(passed=False, feedback=f"Test failed: {e}")
        except Exception as e:
            return GuardResult(passed=False, feedback=f"{type(e).__name__}: {e}")


class DynamicTestGuard(GuardInterface):
    """
    Runs test code against implementation in isolated subprocess.

    Expects 'test' dependency containing the test artifact.
    Executes tests and returns pass/fail with detailed feedback.

    Uses multiprocessing for isolation to prevent test code from
    affecting the parent process.
    """

    def __init__(self, timeout: float = 60.0):
        """
        Args:
            timeout: Maximum time in seconds to wait for test execution
        """
        self.timeout = timeout

    def validate(self, artifact: Artifact, **deps: Any) -> GuardResult:
        """
        Run tests in isolated subprocess.

        Args:
            artifact: The implementation artifact to test
            **deps: Must include 'test' artifact with test code

        Returns:
            GuardResult with test outcome
        """
        test_artifact = deps.get("test")
        if not test_artifact:
            return GuardResult(
                passed=False, feedback="No test artifact in dependencies"
            )

        q: multiprocessing.Queue = multiprocessing.Queue()
        p = multiprocessing.Process(
            target=self._run_tests, args=(artifact, test_artifact, q)
        )
        p.start()
        p.join(self.timeout)

        if p.is_alive():
            p.terminate()
            p.join()
            return GuardResult(
                passed=False,
                feedback=f"Timeout: Test execution exceeded {self.timeout}s",
            )

        if not q.empty():
            passed, msg = q.get()
            return GuardResult(passed=passed, feedback=msg)
        return GuardResult(passed=False, feedback="Test execution crashed")

    def _run_tests(
        self, impl_artifact: Artifact, test_artifact: Artifact, q: Any
    ) -> None:
        """
        Execute tests using pytest in an isolated temp directory.

        This method runs in a forked process for isolation.
        Supports pytest classes, fixtures, and parameterized tests.
        """
        import os
        import tempfile
        from io import StringIO

        impl_code = impl_artifact.content
        test_code = test_artifact.content

        if not impl_code:
            q.put((False, "No implementation code"))
            return

        if not test_code:
            q.put((False, "No test code"))
            return

        with tempfile.TemporaryDirectory() as tmpdir:
            # Write implementation as importable module
            impl_path = os.path.join(tmpdir, "implementation.py")
            with open(impl_path, "w") as f:
                f.write(impl_code)

            # Write test file
            test_path = os.path.join(tmpdir, "test_generated.py")
            with open(test_path, "w") as f:
                f.write(test_code)

            # Add tmpdir to sys.path for imports
            sys.path.insert(0, tmpdir)

            try:
                import pytest

                # Capture pytest output
                captured_output = StringIO()

                class OutputCapture:
                    """Pytest plugin to capture failure output."""

                    @pytest.hookimpl(hookwrapper=True)
                    def pytest_runtest_logreport(self, report: Any) -> Any:
                        yield
                        if report.failed:
                            captured_output.write(
                                f"{report.nodeid}: {report.longreprtext}\n"
                            )

                # Run pytest
                exit_code = pytest.main(
                    [
                        test_path,
                        "-v",
                        "--tb=short",
                        "-q",
                        "--no-header",
                    ],
                    plugins=[OutputCapture()],
                )

                if exit_code == pytest.ExitCode.OK:
                    q.put((True, "All tests passed"))
                elif exit_code == pytest.ExitCode.NO_TESTS_COLLECTED:
                    q.put((False, "No tests collected by pytest"))
                else:
                    output = captured_output.getvalue()
                    if output:
                        q.put((False, f"Test failures:\n{output}"))
                    else:
                        q.put((False, f"pytest exited with code {exit_code}"))

            except SyntaxError as e:
                q.put((False, f"Syntax error: {e}"))
            except Exception as e:
                q.put((False, f"pytest execution error: {type(e).__name__}: {e}"))
            finally:
                # Clean up sys.path
                if tmpdir in sys.path:
                    sys.path.remove(tmpdir)
                # Clean up implementation module if loaded
                if "implementation" in sys.modules:
                    del sys.modules["implementation"]
