"""Test guard for validating implementation against tests."""

import subprocess
import tempfile
from pathlib import Path
from typing import Tuple


class TestGuard:
    """Validates that implementation passes all generated tests."""

    def validate(self, impl_code: str, test_code: str) -> Tuple[bool, str]:
        """Run tests against implementation.

        Args:
            impl_code: Implementation code to test
            test_code: Test code to run

        Returns:
            Tuple of (all_tests_passed, feedback_message)
        """
        # Create temporary directory for test execution
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Write implementation to solution.py
            solution_path = tmpdir_path / "solution.py"
            solution_path.write_text(impl_code, encoding="utf-8")

            # Write tests to test_solution.py
            test_path = tmpdir_path / "test_solution.py"
            test_path.write_text(test_code, encoding="utf-8")

            # Run pytest
            try:
                result = subprocess.run(
                    ["pytest", str(test_path), "-v", "--tb=short"],
                    cwd=tmpdir,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )

                # Check if all tests passed
                all_passed = result.returncode == 0

                if all_passed:
                    # Count passed tests
                    output = result.stdout
                    if "passed" in output:
                        return True, f"All tests passed\n{output}"
                    return True, "Tests passed"
                else:
                    # Extract failure information
                    feedback = "Tests failed:\n"
                    feedback += result.stdout
                    if result.stderr:
                        feedback += f"\nErrors:\n{result.stderr}"
                    return False, feedback

            except subprocess.TimeoutExpired:
                return False, "Test execution timed out (30s limit)"
            except FileNotFoundError:
                return False, "pytest not found - ensure pytest is installed"
            except Exception as e:
                return False, f"Test execution error: {str(e)}"

    def validate_with_coverage(
        self, impl_code: str, test_code: str
    ) -> Tuple[bool, str, float]:
        """Run tests and measure code coverage.

        Args:
            impl_code: Implementation code to test
            test_code: Test code to run

        Returns:
            Tuple of (all_tests_passed, feedback_message, coverage_percentage)
        """
        # Create temporary directory for test execution
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Write implementation to solution.py
            solution_path = tmpdir_path / "solution.py"
            solution_path.write_text(impl_code, encoding="utf-8")

            # Write tests to test_solution.py
            test_path = tmpdir_path / "test_solution.py"
            test_path.write_text(test_code, encoding="utf-8")

            # Run pytest with coverage
            try:
                result = subprocess.run(
                    [
                        "pytest",
                        str(test_path),
                        "--cov=solution",
                        "--cov-report=term-missing",
                        "-v",
                    ],
                    cwd=tmpdir,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )

                all_passed = result.returncode == 0

                # Extract coverage percentage
                coverage = 0.0
                output = result.stdout
                for line in output.split("\n"):
                    if "solution.py" in line and "%" in line:
                        # Parse coverage percentage
                        parts = line.split()
                        for part in parts:
                            if "%" in part:
                                coverage = float(part.rstrip("%"))
                                break

                if all_passed:
                    return True, f"All tests passed (Coverage: {coverage}%)\n{output}", coverage
                else:
                    feedback = f"Tests failed (Coverage: {coverage}%):\n{output}"
                    if result.stderr:
                        feedback += f"\nErrors:\n{result.stderr}"
                    return False, feedback, coverage

            except subprocess.TimeoutExpired:
                return False, "Test execution timed out (30s limit)", 0.0
            except Exception as e:
                return False, f"Test execution error: {str(e)}", 0.0
