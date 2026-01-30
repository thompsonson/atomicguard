"""Tests for evaluation dataset adapters (SWE-PolyBench, SWE-bench)."""

from __future__ import annotations

import json
from pathlib import Path

from examples.advanced.g_plan_benchmark.evaluation.adapters import (
    _polybench_instance_to_problem,
    _swebench_instance_to_problem,
    load_swe_bench,
    load_swe_polybench,
)
from examples.advanced.g_plan_benchmark.evaluation.problem import (
    TYPE_TO_STRATEGY,
    ProblemSet,
)


# ---------------------------------------------------------------------------
# SWE-PolyBench instance conversion
# ---------------------------------------------------------------------------


class TestPolyBenchInstanceConversion:
    """Test _polybench_instance_to_problem mapping."""

    def _make_instance(self, **overrides) -> dict:
        base = {
            "instance_id": "repo__name-123",
            "problem_statement": "Fix the crash when parsing empty input",
            "task_category": "Bug Fix",
            "language": "Python",
            "repo": "org/repo",
            "base_commit": "abc123",
        }
        base.update(overrides)
        return base

    def test_basic_mapping(self):
        inst = self._make_instance()
        p = _polybench_instance_to_problem(inst)
        assert p.problem_id == "repo__name-123"
        assert p.description == "Fix the crash when parsing empty input"
        assert p.expected_type == "bug_fix"
        assert p.language == "python"
        assert p.source == "swe-polybench"

    def test_expected_strategy_set_from_type(self):
        p = _polybench_instance_to_problem(self._make_instance(task_category="Bug Fix"))
        assert p.expected_strategy == "S1_locate_and_fix"

        p = _polybench_instance_to_problem(self._make_instance(task_category="Feature"))
        assert p.expected_strategy == "S2_tdd_feature"

        p = _polybench_instance_to_problem(
            self._make_instance(task_category="Refactoring")
        )
        assert p.expected_strategy == "S3_refactor_safely"

    def test_language_normalization(self):
        for raw, expected in [
            ("Python", "python"),
            ("Java", "java"),
            ("JavaScript", "javascript"),
            ("TypeScript", "typescript"),
        ]:
            p = _polybench_instance_to_problem(self._make_instance(language=raw))
            assert p.language == expected

    def test_unknown_category_maps_to_unknown(self):
        p = _polybench_instance_to_problem(
            self._make_instance(task_category="SomethingElse")
        )
        assert p.expected_type == "unknown"
        assert p.expected_strategy == ""

    def test_metadata_preserves_repo_fields(self):
        inst = self._make_instance(
            repo="org/repo",
            base_commit="abc",
            patch="diff --git a/...",
            test_patch="diff --git b/...",
        )
        p = _polybench_instance_to_problem(inst)
        assert p.metadata["repo"] == "org/repo"
        assert p.metadata["base_commit"] == "abc"
        assert p.metadata["patch"] == "diff --git a/..."

    def test_missing_problem_statement_raises(self):
        inst = self._make_instance(problem_statement="")
        try:
            _polybench_instance_to_problem(inst)
            assert False, "Should have raised"
        except ValueError as e:
            assert "problem_statement" in str(e)

    def test_fallback_id_from_repo(self):
        inst = self._make_instance()
        del inst["instance_id"]
        p = _polybench_instance_to_problem(inst)
        assert p.problem_id == "org/repo"

    def test_lowercase_category_accepted(self):
        p = _polybench_instance_to_problem(self._make_instance(task_category="bug_fix"))
        assert p.expected_type == "bug_fix"

    def test_difficulty_defaults_to_unknown(self):
        p = _polybench_instance_to_problem(self._make_instance())
        assert p.difficulty == "unknown"


# ---------------------------------------------------------------------------
# SWE-bench instance conversion
# ---------------------------------------------------------------------------


class TestSWEBenchInstanceConversion:
    """Test _swebench_instance_to_problem mapping."""

    def _make_instance(self, **overrides) -> dict:
        base = {
            "instance_id": "django__django-12345",
            "problem_statement": "QuerySet.union() crashes on empty queryset",
            "difficulty": "Medium",
            "repo": "django/django",
            "base_commit": "def456",
        }
        base.update(overrides)
        return base

    def test_basic_mapping(self):
        p = _swebench_instance_to_problem(self._make_instance())
        assert p.problem_id == "django__django-12345"
        assert "QuerySet.union()" in p.description
        assert p.language == "python"
        assert p.source == "swe-bench"

    def test_type_always_unknown(self):
        """SWE-bench has no task_category — type should always be unknown."""
        p = _swebench_instance_to_problem(self._make_instance())
        assert p.expected_type == "unknown"
        assert p.expected_strategy == ""

    def test_difficulty_mapping(self):
        for raw, expected in [
            ("Easy", "easy"),
            ("Medium", "medium"),
            ("Hard", "hard"),
            ("Very Hard", "hard"),  # collapsed to 3-level
        ]:
            p = _swebench_instance_to_problem(self._make_instance(difficulty=raw))
            assert p.difficulty == expected, f"Failed for {raw}"

    def test_unknown_difficulty(self):
        p = _swebench_instance_to_problem(self._make_instance(difficulty="???"))
        assert p.difficulty == "unknown"

    def test_metadata_preserves_fields(self):
        inst = self._make_instance(
            FAIL_TO_PASS="test_union_empty",
            PASS_TO_PASS="test_union_basic",
            hints_text="Check the SQL generation",
        )
        p = _swebench_instance_to_problem(inst)
        assert p.metadata["FAIL_TO_PASS"] == "test_union_empty"
        assert p.metadata["hints_text"] == "Check the SQL generation"

    def test_missing_problem_statement_raises(self):
        inst = self._make_instance(problem_statement="")
        try:
            _swebench_instance_to_problem(inst)
            assert False, "Should have raised"
        except ValueError as e:
            assert "problem_statement" in str(e)


# ---------------------------------------------------------------------------
# Local file loading
# ---------------------------------------------------------------------------


class TestLocalFileLoading:
    """Test loading adapters from local JSON/JSONL files."""

    def test_polybench_from_json_file(self, tmp_path: Path):
        data = [
            {
                "instance_id": "PB-1",
                "problem_statement": "Fix null pointer",
                "task_category": "Bug Fix",
                "language": "Java",
            },
            {
                "instance_id": "PB-2",
                "problem_statement": "Add search feature",
                "task_category": "Feature",
                "language": "Python",
            },
        ]
        f = tmp_path / "problems.json"
        f.write_text(json.dumps(data))

        ps = load_swe_polybench(path=f)
        assert len(ps) == 2
        assert ps["PB-1"].expected_type == "bug_fix"
        assert ps["PB-2"].expected_type == "feature"

    def test_polybench_from_jsonl_file(self, tmp_path: Path):
        lines = [
            json.dumps({
                "instance_id": "PB-1",
                "problem_statement": "Fix crash",
                "task_category": "Bug Fix",
                "language": "Python",
            }),
            json.dumps({
                "instance_id": "PB-2",
                "problem_statement": "Add dark mode",
                "task_category": "Feature",
                "language": "TypeScript",
            }),
        ]
        f = tmp_path / "problems.jsonl"
        f.write_text("\n".join(lines) + "\n")

        ps = load_swe_polybench(path=f)
        assert len(ps) == 2
        assert ps["PB-1"].language == "python"
        assert ps["PB-2"].language == "typescript"

    def test_swebench_from_json_file(self, tmp_path: Path):
        data = {
            "instances": [
                {
                    "instance_id": "django__django-1",
                    "problem_statement": "Fix ORM bug",
                    "difficulty": "Easy",
                },
            ]
        }
        f = tmp_path / "swebench.json"
        f.write_text(json.dumps(data))

        ps = load_swe_bench(path=f)
        assert len(ps) == 1
        assert ps["django__django-1"].difficulty == "easy"
        assert ps["django__django-1"].language == "python"

    def test_load_from_directory(self, tmp_path: Path):
        for i in range(3):
            p = {
                "instance_id": f"PB-{i}",
                "problem_statement": f"Problem {i}",
                "task_category": "Bug Fix",
                "language": "Python",
            }
            (tmp_path / f"problem_{i}.json").write_text(json.dumps(p))

        ps = load_swe_polybench(path=tmp_path)
        assert len(ps) == 3

    def test_limit_parameter(self, tmp_path: Path):
        data = [
            {
                "instance_id": f"PB-{i}",
                "problem_statement": f"Problem {i}",
                "task_category": "Bug Fix",
                "language": "Python",
            }
            for i in range(10)
        ]
        f = tmp_path / "problems.json"
        f.write_text(json.dumps(data))

        ps = load_swe_polybench(path=f, limit=3)
        assert len(ps) == 3
        assert ps["PB-0"].problem_id == "PB-0"
        assert ps["PB-2"].problem_id == "PB-2"

    def test_empty_directory_raises(self, tmp_path: Path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        try:
            load_swe_polybench(path=empty_dir)
            assert False, "Should have raised"
        except ValueError as e:
            assert "No instances" in str(e)

    def test_unsupported_file_type_raises(self, tmp_path: Path):
        f = tmp_path / "data.csv"
        f.write_text("a,b\n1,2\n")
        try:
            load_swe_polybench(path=f)
            assert False, "Should have raised"
        except ValueError as e:
            assert ".csv" in str(e)


# ---------------------------------------------------------------------------
# ProblemSet integration
# ---------------------------------------------------------------------------


class TestAdapterProblemSetIntegration:
    """Verify adapter output integrates with ProblemSet methods."""

    def _make_polybench_file(self, tmp_path: Path) -> Path:
        data = [
            {
                "instance_id": "PB-BUG-1",
                "problem_statement": "Fix crash on empty input",
                "task_category": "Bug Fix",
                "language": "Python",
            },
            {
                "instance_id": "PB-FEAT-1",
                "problem_statement": "Add search feature",
                "task_category": "Feature",
                "language": "Java",
            },
            {
                "instance_id": "PB-REF-1",
                "problem_statement": "Extract helper class",
                "task_category": "Refactoring",
                "language": "Python",
            },
        ]
        f = tmp_path / "problems.json"
        f.write_text(json.dumps(data))
        return f

    def test_filter_by_type(self, tmp_path: Path):
        ps = load_swe_polybench(path=self._make_polybench_file(tmp_path))
        bugs = ps.filter_by_type("bug_fix")
        assert len(bugs) == 1
        assert bugs.problems[0].problem_id == "PB-BUG-1"

    def test_filter_by_language(self, tmp_path: Path):
        ps = load_swe_polybench(path=self._make_polybench_file(tmp_path))
        java = ps.filter_by_language("java")
        assert len(java) == 1
        assert java.problems[0].problem_id == "PB-FEAT-1"

    def test_to_dict_roundtrip(self, tmp_path: Path):
        ps = load_swe_polybench(path=self._make_polybench_file(tmp_path))
        d = ps.to_dict()
        ps2 = ProblemSet([
            __import__(
                "examples.advanced.g_plan_benchmark.evaluation.problem",
                fromlist=["Problem"],
            ).Problem.from_dict(p)
            for p in d["problems"]
        ])
        assert len(ps2) == 3
        assert ps2["PB-BUG-1"].expected_type == "bug_fix"

    def test_strategy_alignment_data_present(self, tmp_path: Path):
        """Adapter output has expected_strategy for all known types."""
        ps = load_swe_polybench(path=self._make_polybench_file(tmp_path))
        for problem in ps:
            if problem.expected_type in TYPE_TO_STRATEGY:
                assert problem.expected_strategy != "", (
                    f"{problem.problem_id} has type {problem.expected_type} "
                    f"but no expected_strategy"
                )
