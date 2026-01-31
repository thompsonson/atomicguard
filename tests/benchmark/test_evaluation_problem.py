"""Tests for evaluation problem model and loader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from examples.advanced.g_plan_benchmark.evaluation.problem import (
    TYPE_TO_STRATEGY,
    Problem,
    ProblemSet,
)


# =============================================================================
# Problem dataclass
# =============================================================================


class TestProblem:
    def test_minimal_construction(self):
        p = Problem(problem_id="P1", description="Fix the bug")
        assert p.problem_id == "P1"
        assert p.description == "Fix the bug"
        assert p.expected_type == "unknown"
        assert p.language == "unknown"
        assert p.difficulty == "unknown"
        assert p.source == "catalog"
        assert p.metadata == {}

    def test_full_construction(self):
        p = Problem(
            problem_id="SWE-001",
            description="KeyError in parse_config()",
            expected_type="bug_fix",
            language="python",
            expected_strategy="S1_locate_and_fix",
            difficulty="easy",
            source="swe-polybench",
            metadata={"repo": "owner/repo"},
        )
        assert p.expected_type == "bug_fix"
        assert p.expected_strategy == "S1_locate_and_fix"
        assert p.metadata["repo"] == "owner/repo"

    def test_to_dict_roundtrip(self):
        p = Problem(
            problem_id="P1",
            description="Add feature X",
            expected_type="feature",
            language="java",
            expected_strategy="S2_tdd_feature",
            difficulty="medium",
        )
        d = p.to_dict()
        p2 = Problem.from_dict(d)
        assert p2.problem_id == p.problem_id
        assert p2.description == p.description
        assert p2.expected_type == p.expected_type
        assert p2.language == p.language
        assert p2.expected_strategy == p.expected_strategy
        assert p2.difficulty == p.difficulty

    def test_from_dict_defaults(self):
        d = {"problem_id": "P1", "description": "desc"}
        p = Problem.from_dict(d)
        assert p.expected_type == "unknown"
        assert p.language == "unknown"
        assert p.expected_strategy == ""
        assert p.difficulty == "unknown"
        assert p.source == "catalog"

    def test_frozen(self):
        p = Problem(problem_id="P1", description="desc")
        with pytest.raises(AttributeError):
            p.problem_id = "P2"  # type: ignore[misc]


# =============================================================================
# TYPE_TO_STRATEGY mapping
# =============================================================================


class TestTypeToStrategy:
    def test_bug_fix_maps_to_s1(self):
        assert TYPE_TO_STRATEGY["bug_fix"] == "S1_locate_and_fix"

    def test_feature_maps_to_s2(self):
        assert TYPE_TO_STRATEGY["feature"] == "S2_tdd_feature"

    def test_refactoring_maps_to_s3(self):
        assert TYPE_TO_STRATEGY["refactoring"] == "S3_refactor_safely"

    def test_performance_maps_to_s4(self):
        assert TYPE_TO_STRATEGY["performance"] == "S4_profile_and_optimize"


# =============================================================================
# ProblemSet
# =============================================================================


def _make_problems() -> list[Problem]:
    return [
        Problem(problem_id="BUG-1", description="Fix crash", expected_type="bug_fix", language="python"),
        Problem(problem_id="FEAT-1", description="Add search", expected_type="feature", language="java"),
        Problem(problem_id="REF-1", description="Extract module", expected_type="refactoring", language="python"),
        Problem(problem_id="PERF-1", description="Slow query", expected_type="performance", language="python"),
    ]


class TestProblemSet:
    def test_len(self):
        ps = ProblemSet(_make_problems())
        assert len(ps) == 4

    def test_iter(self):
        ps = ProblemSet(_make_problems())
        ids = [p.problem_id for p in ps]
        assert ids == ["BUG-1", "FEAT-1", "REF-1", "PERF-1"]

    def test_getitem(self):
        ps = ProblemSet(_make_problems())
        assert ps["FEAT-1"].description == "Add search"

    def test_getitem_missing_raises(self):
        ps = ProblemSet(_make_problems())
        with pytest.raises(KeyError):
            ps["NONEXISTENT"]

    def test_filter_by_type(self):
        ps = ProblemSet(_make_problems())
        bugs = ps.filter_by_type("bug_fix")
        assert len(bugs) == 1
        assert bugs.problems[0].problem_id == "BUG-1"

    def test_filter_by_language(self):
        ps = ProblemSet(_make_problems())
        py = ps.filter_by_language("python")
        assert len(py) == 3

    def test_filter_by_difficulty(self):
        problems = [
            Problem(problem_id="P1", description="d", difficulty="easy"),
            Problem(problem_id="P2", description="d", difficulty="hard"),
        ]
        ps = ProblemSet(problems)
        easy = ps.filter_by_difficulty("easy")
        assert len(easy) == 1

    def test_to_dict(self):
        ps = ProblemSet(_make_problems())
        d = ps.to_dict()
        assert "problems" in d
        assert len(d["problems"]) == 4
        assert d["problems"][0]["problem_id"] == "BUG-1"


# =============================================================================
# ProblemSet loading
# =============================================================================


class TestProblemSetLoading:
    def test_load_from_json_file_with_problems_key(self, tmp_path):
        data = {
            "problems": [
                {"problem_id": "P1", "description": "d1"},
                {"problem_id": "P2", "description": "d2"},
            ]
        }
        path = tmp_path / "problems.json"
        path.write_text(json.dumps(data))
        ps = ProblemSet.load(path)
        assert len(ps) == 2

    def test_load_from_json_file_as_array(self, tmp_path):
        data = [
            {"problem_id": "P1", "description": "d1"},
            {"problem_id": "P2", "description": "d2"},
        ]
        path = tmp_path / "problems.json"
        path.write_text(json.dumps(data))
        ps = ProblemSet.load(path)
        assert len(ps) == 2

    def test_load_from_directory(self, tmp_path):
        for i in range(3):
            p = {"problem_id": f"P{i}", "description": f"desc {i}"}
            (tmp_path / f"P{i}.json").write_text(json.dumps(p))
        ps = ProblemSet.load(tmp_path)
        assert len(ps) == 3

    def test_load_directory_sorted(self, tmp_path):
        for name in ["C.json", "A.json", "B.json"]:
            p = {"problem_id": name[0], "description": "d"}
            (tmp_path / name).write_text(json.dumps(p))
        ps = ProblemSet.load(tmp_path)
        ids = [p.problem_id for p in ps]
        assert ids == ["A", "B", "C"]

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            ProblemSet.load("/nonexistent/path")

    def test_load_empty_directory_raises(self, tmp_path):
        with pytest.raises(ValueError, match="No .json files"):
            ProblemSet.load(tmp_path)

    def test_load_invalid_format_raises(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"not_problems": []}))
        with pytest.raises(ValueError, match="Expected"):
            ProblemSet.load(path)
