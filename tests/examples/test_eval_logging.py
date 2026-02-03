"""Tests for per-instance evaluation logging."""

from pathlib import Path

from examples.swe_bench_ablation.evaluation import (
    EvalResult,
    _sanitize_instance_id,
    format_instance_log,
    write_eval_logs,
    write_instance_log,
)

# ---------------------------------------------------------------------------
# _sanitize_instance_id
# ---------------------------------------------------------------------------


class TestSanitizeInstanceId:
    def test_forward_slash(self):
        assert _sanitize_instance_id("astropy/astropy-12907") == "astropy__astropy-12907"

    def test_backslash(self):
        assert _sanitize_instance_id("a\\b") == "a__b"

    def test_no_slashes(self):
        assert _sanitize_instance_id("django__django-1234") == "django__django-1234"

    def test_multiple_slashes(self):
        assert _sanitize_instance_id("a/b/c") == "a__b__c"


# ---------------------------------------------------------------------------
# format_instance_log
# ---------------------------------------------------------------------------


class TestFormatInstanceLog:
    def test_minimal_resolved(self):
        result = EvalResult(instance_id="repo__issue-1", resolved=True)
        log = format_instance_log(result)
        assert "Instance: repo__issue-1" in log
        assert "Resolved: True" in log
        assert "Wall time: 0.0s" in log

    def test_with_error(self):
        result = EvalResult(
            instance_id="repo__issue-2",
            resolved=False,
            error="Docker build failed",
        )
        log = format_instance_log(result)
        assert "Error: Docker build failed" in log
        assert "Resolved: False" in log

    def test_fail_to_pass_section(self):
        result = EvalResult(
            instance_id="x",
            resolved=True,
            fail_to_pass_results={
                "test_a": True,
                "test_b": False,
            },
        )
        log = format_instance_log(result)
        assert "Fail-to-Pass Tests (1/2 passed):" in log
        assert "[PASS] test_a" in log
        assert "[FAIL] test_b" in log

    def test_pass_to_pass_section(self):
        result = EvalResult(
            instance_id="x",
            resolved=False,
            pass_to_pass_results={
                "test_c": True,
                "test_d": True,
            },
        )
        log = format_instance_log(result)
        assert "Pass-to-Pass Tests (2/2 passed):" in log
        assert "[PASS] test_c" in log
        assert "[PASS] test_d" in log

    def test_execution_log_section(self):
        result = EvalResult(
            instance_id="x",
            resolved=True,
            log="running pytest...\n2 passed",
        )
        log = format_instance_log(result)
        assert "--- Execution Log ---" in log
        assert "running pytest..." in log
        assert "2 passed" in log

    def test_wall_time(self):
        result = EvalResult(
            instance_id="x",
            resolved=True,
            wall_time_seconds=42.7,
        )
        log = format_instance_log(result)
        assert "Wall time: 42.7s" in log

    def test_all_sections_together(self):
        result = EvalResult(
            instance_id="repo/issue-5",
            resolved=False,
            fail_to_pass_results={"test_fix": False},
            pass_to_pass_results={"test_existing": True},
            error="patch apply failed",
            log="git apply returned exit code 1",
            wall_time_seconds=3.2,
        )
        log = format_instance_log(result)
        assert "Instance: repo/issue-5" in log
        assert "Error: patch apply failed" in log
        assert "Fail-to-Pass Tests" in log
        assert "Pass-to-Pass Tests" in log
        assert "--- Execution Log ---" in log


# ---------------------------------------------------------------------------
# write_instance_log
# ---------------------------------------------------------------------------


class TestWriteInstanceLog:
    def test_creates_file(self, tmp_path: Path):
        result = EvalResult(instance_id="django__django-1234", resolved=True)
        path = write_instance_log(result, tmp_path)

        assert path.exists()
        assert path.name == "django__django-1234.log"
        content = path.read_text()
        assert "Instance: django__django-1234" in content

    def test_sanitizes_slashes(self, tmp_path: Path):
        result = EvalResult(instance_id="org/repo-99", resolved=False)
        path = write_instance_log(result, tmp_path)

        assert path.name == "org__repo-99.log"

    def test_creates_directory(self, tmp_path: Path):
        log_dir = tmp_path / "deep" / "nested"
        result = EvalResult(instance_id="x", resolved=True)
        path = write_instance_log(result, log_dir)

        assert log_dir.is_dir()
        assert path.exists()


# ---------------------------------------------------------------------------
# write_eval_logs
# ---------------------------------------------------------------------------


class TestWriteEvalLogs:
    def _make_results(self) -> dict[str, EvalResult]:
        return {
            "repo__issue-1": EvalResult(
                instance_id="repo__issue-1",
                resolved=True,
                fail_to_pass_results={"test_a": True},
                wall_time_seconds=10.5,
            ),
            "repo__issue-2": EvalResult(
                instance_id="repo__issue-2",
                resolved=False,
                error="timeout",
                wall_time_seconds=600.0,
            ),
            "repo__issue-3": EvalResult(
                instance_id="repo__issue-3",
                resolved=False,
                wall_time_seconds=5.0,
            ),
        }

    def test_creates_per_instance_files(self, tmp_path: Path):
        results = self._make_results()
        log_dir = write_eval_logs(results, tmp_path, run_id="run1")

        assert log_dir == tmp_path / "eval_logs" / "run1"
        assert (log_dir / "repo__issue-1.log").exists()
        assert (log_dir / "repo__issue-2.log").exists()
        assert (log_dir / "repo__issue-3.log").exists()

    def test_creates_summary(self, tmp_path: Path):
        results = self._make_results()
        log_dir = write_eval_logs(results, tmp_path, run_id="run1")

        summary = (log_dir / "_summary.log").read_text()
        assert "Total instances: 3" in summary
        assert "Resolved: 1/3 (33.3%)" in summary
        assert "Errors: 1" in summary

    def test_summary_per_instance_lines(self, tmp_path: Path):
        results = self._make_results()
        log_dir = write_eval_logs(results, tmp_path, run_id="run1")

        summary = (log_dir / "_summary.log").read_text()
        assert "repo__issue-1: RESOLVED (10.5s)" in summary
        assert "repo__issue-2: ERROR (timeout)" in summary
        assert "repo__issue-3: FAILED (5.0s)" in summary

    def test_empty_results(self, tmp_path: Path):
        log_dir = write_eval_logs({}, tmp_path, run_id="empty")

        summary = (log_dir / "_summary.log").read_text()
        assert "Total instances: 0" in summary
        assert "Resolved: 0/0" in summary

    def test_default_run_id(self, tmp_path: Path):
        log_dir = write_eval_logs({}, tmp_path)
        assert log_dir == tmp_path / "eval_logs" / "experiment_7_2"

    def test_instance_log_content_matches(self, tmp_path: Path):
        results = {
            "x": EvalResult(
                instance_id="x",
                resolved=True,
                log="some output",
            ),
        }
        log_dir = write_eval_logs(results, tmp_path, run_id="r")

        content = (log_dir / "x.log").read_text()
        assert "--- Execution Log ---" in content
        assert "some output" in content
