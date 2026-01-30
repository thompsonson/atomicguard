"""Problem representation and loading for the evaluation harness."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Problem:
    """A single benchmark problem with metadata for scoring.

    Attributes:
        problem_id: Unique identifier (e.g. "SWE-001", "BUG-simple-keyerror").
        description: The problem statement — fed as specification to the pipeline.
        expected_type: Ground-truth problem category for strategy alignment scoring.
            One of: bug_fix, feature, refactoring, performance, unknown.
        language: Programming language (python, java, javascript, typescript, unknown).
        expected_strategy: Expected S1-S5 strategy ID, if known.
        difficulty: Problem difficulty (easy, medium, hard, unknown).
        source: Origin dataset (e.g. "catalog", "swe-polybench", "swe-bench").
        metadata: Arbitrary extra fields from the source dataset.
    """

    problem_id: str
    description: str
    expected_type: str = "unknown"
    language: str = "unknown"
    expected_strategy: str = ""
    difficulty: str = "unknown"
    source: str = "catalog"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "problem_id": self.problem_id,
            "description": self.description,
            "expected_type": self.expected_type,
            "language": self.language,
            "expected_strategy": self.expected_strategy,
            "difficulty": self.difficulty,
            "source": self.source,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Problem:
        """Deserialize from a dict."""
        return cls(
            problem_id=data["problem_id"],
            description=data["description"],
            expected_type=data.get("expected_type", "unknown"),
            language=data.get("language", "unknown"),
            expected_strategy=data.get("expected_strategy", ""),
            difficulty=data.get("difficulty", "unknown"),
            source=data.get("source", "catalog"),
            metadata=data.get("metadata", {}),
        )


# Strategy mapping from problem type to expected strategy ID.
TYPE_TO_STRATEGY: dict[str, str] = {
    "bug_fix": "S1_locate_and_fix",
    "feature": "S2_tdd_feature",
    "refactoring": "S3_refactor_safely",
    "performance": "S4_profile_and_optimize",
}


class ProblemSet:
    """A collection of problems loaded from a JSON file or directory.

    File format (single file):
        {
            "problems": [
                { "problem_id": "...", "description": "...", ... },
                ...
            ]
        }

    Directory format:
        problems/
        ├── BUG-001.json    # each file is one Problem dict
        ├── FEAT-001.json
        └── ...
    """

    def __init__(self, problems: list[Problem]) -> None:
        self._problems = list(problems)
        self._by_id = {p.problem_id: p for p in self._problems}

    @property
    def problems(self) -> list[Problem]:
        return list(self._problems)

    def __len__(self) -> int:
        return len(self._problems)

    def __iter__(self):
        return iter(self._problems)

    def __getitem__(self, problem_id: str) -> Problem:
        return self._by_id[problem_id]

    def filter_by_type(self, expected_type: str) -> ProblemSet:
        """Return a subset matching the given problem type."""
        return ProblemSet([p for p in self._problems if p.expected_type == expected_type])

    def filter_by_language(self, language: str) -> ProblemSet:
        """Return a subset matching the given language."""
        return ProblemSet([p for p in self._problems if p.language == language])

    def filter_by_difficulty(self, difficulty: str) -> ProblemSet:
        """Return a subset matching the given difficulty."""
        return ProblemSet([p for p in self._problems if p.difficulty == difficulty])

    @classmethod
    def load(cls, path: str | Path) -> ProblemSet:
        """Load problems from a JSON file or a directory of JSON files."""
        path = Path(path)
        if path.is_file():
            return cls._load_file(path)
        if path.is_dir():
            return cls._load_directory(path)
        raise FileNotFoundError(f"Problem source not found: {path}")

    @classmethod
    def _load_file(cls, path: Path) -> ProblemSet:
        """Load from a single JSON file containing a 'problems' array."""
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            problems = [Problem.from_dict(d) for d in data]
        elif isinstance(data, dict) and "problems" in data:
            problems = [Problem.from_dict(d) for d in data["problems"]]
        else:
            raise ValueError(
                f"Expected a JSON array or object with 'problems' key, got: {type(data)}"
            )
        return cls(problems)

    @classmethod
    def _load_directory(cls, path: Path) -> ProblemSet:
        """Load from a directory where each .json file is one problem."""
        problems: list[Problem] = []
        for json_file in sorted(path.glob("*.json")):
            with open(json_file) as f:
                data = json.load(f)
            problems.append(Problem.from_dict(data))
        if not problems:
            raise ValueError(f"No .json files found in {path}")
        return cls(problems)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the problem set."""
        return {"problems": [p.to_dict() for p in self._problems]}
