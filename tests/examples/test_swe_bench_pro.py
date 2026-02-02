"""Tests for the swe_bench_pro example package.

Covers language config, guards, generators (prompt construction), and
evaluation helpers.  No network, Docker, or HuggingFace access required.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from examples.swe_bench_pro.language import (
    LANGUAGE_CONFIGS,
    LanguageConfig,
    _check_basic_braces,
    _check_python_syntax,
    detect_test_functions,
    get_language_config,
)


# =========================================================================
# language.py
# =========================================================================


class TestLanguageConfigs:
    def test_all_four_languages_present(self):
        assert set(LANGUAGE_CONFIGS) == {"python", "go", "javascript", "typescript"}

    def test_python_has_syntax_checker(self):
        assert LANGUAGE_CONFIGS["python"].syntax_check_fn is not None

    def test_get_unknown_raises_value_error(self):
        with pytest.raises(ValueError, match="Unsupported language"):
            get_language_config("rust")

    @pytest.mark.parametrize("lang", ["python", "go", "javascript", "typescript"])
    def test_get_known_language(self, lang):
        cfg = get_language_config(lang)
        assert cfg.name == lang

    def test_case_insensitive_lookup(self):
        cfg = get_language_config("Python")
        assert cfg.name == "python"


class TestPythonSyntaxCheck:
    def test_valid_code(self):
        ok, msg = _check_python_syntax("x = 1 + 2")
        assert ok is True
        assert msg == ""

    def test_invalid_code(self):
        ok, msg = _check_python_syntax("def f(\n  return")
        assert ok is False
        assert "line" in msg


class TestBraceBalance:
    def test_balanced(self):
        ok, msg = _check_basic_braces("func main() { fmt.Println() }")
        assert ok is True

    def test_unbalanced(self):
        ok, msg = _check_basic_braces("func main() { fmt.Println()")
        assert ok is False
        assert "Unbalanced" in msg


class TestDetectTestFunctions:
    def test_python_test(self):
        assert detect_test_functions("def test_foo(): pass", "python") is True

    def test_python_no_test(self):
        assert detect_test_functions("def foo(): pass", "python") is False

    def test_go_test(self):
        code = 'func TestAdd(t *testing.T) { t.Log("ok") }'
        assert detect_test_functions(code, "go") is True

    def test_js_test(self):
        code = 'test("adds 1 + 2", () => { expect(add(1,2)).toBe(3); })'
        assert detect_test_functions(code, "javascript") is True

    def test_js_describe(self):
        code = 'describe("math", () => { it("works", () => {}); })'
        assert detect_test_functions(code, "javascript") is True


# =========================================================================
# guards/multilang_test_syntax.py
# =========================================================================


class TestMultiLangTestSyntaxGuard:
    @pytest.fixture()
    def _make_artifact(self):
        """Factory for minimal Artifact-like objects."""
        from atomicguard.domain.models import (
            Artifact,
            ArtifactStatus,
            ContextSnapshot,
        )

        def _factory(content: str) -> Artifact:
            ctx = ContextSnapshot(
                workflow_id="w",
                specification="s",
                constraints="",
                feedback_history=(),
                dependency_artifacts=(),
            )
            return Artifact(
                artifact_id="a",
                workflow_id="w",
                content=content,
                previous_attempt_id=None,
                parent_action_pair_id=None,
                action_pair_id="ap",
                created_at="2025-01-01T00:00:00Z",
                attempt_number=1,
                status=ArtifactStatus.PENDING,
                guard_result=None,
                context=ctx,
            )

        return _factory

    def test_python_valid(self, _make_artifact):
        from examples.swe_bench_pro.guards import MultiLangTestSyntaxGuard

        guard = MultiLangTestSyntaxGuard(
            language_config=get_language_config("python"),
        )
        art = _make_artifact("def test_foo():\n    assert True")
        result = guard.validate(art)
        assert result.passed is True

    def test_python_syntax_error(self, _make_artifact):
        from examples.swe_bench_pro.guards import MultiLangTestSyntaxGuard

        guard = MultiLangTestSyntaxGuard(
            language_config=get_language_config("python"),
        )
        art = _make_artifact("def test_foo(\n    assert True")
        result = guard.validate(art)
        assert result.passed is False
        assert "Syntax" in result.feedback

    def test_python_no_test_function(self, _make_artifact):
        from examples.swe_bench_pro.guards import MultiLangTestSyntaxGuard

        guard = MultiLangTestSyntaxGuard(
            language_config=get_language_config("python"),
        )
        art = _make_artifact("x = 1 + 2")
        result = guard.validate(art)
        assert result.passed is False
        assert "test patterns" in result.feedback.lower() or "pytest" in result.feedback

    def test_go_valid(self, _make_artifact):
        from examples.swe_bench_pro.guards import MultiLangTestSyntaxGuard

        guard = MultiLangTestSyntaxGuard(
            language_config=get_language_config("go"),
        )
        code = 'func TestAdd(t *testing.T) { t.Log("ok") }'
        art = _make_artifact(code)
        result = guard.validate(art)
        assert result.passed is True

    def test_go_unbalanced_braces(self, _make_artifact):
        from examples.swe_bench_pro.guards import MultiLangTestSyntaxGuard

        guard = MultiLangTestSyntaxGuard(
            language_config=get_language_config("go"),
        )
        art = _make_artifact("func TestAdd(t *testing.T) { t.Log(")
        result = guard.validate(art)
        assert result.passed is False
        assert "Unbalanced" in result.feedback

    def test_js_valid(self, _make_artifact):
        from examples.swe_bench_pro.guards import MultiLangTestSyntaxGuard

        guard = MultiLangTestSyntaxGuard(
            language_config=get_language_config("javascript"),
        )
        code = 'test("adds", () => { expect(1+1).toBe(2); });'
        art = _make_artifact(code)
        result = guard.validate(art)
        assert result.passed is True

    def test_empty_code(self, _make_artifact):
        from examples.swe_bench_pro.guards import MultiLangTestSyntaxGuard

        guard = MultiLangTestSyntaxGuard(
            language_config=get_language_config("python"),
        )
        art = _make_artifact("")
        result = guard.validate(art)
        assert result.passed is False
        assert "Empty" in result.feedback

    def test_error_marker(self, _make_artifact):
        from examples.swe_bench_pro.guards import MultiLangTestSyntaxGuard

        guard = MultiLangTestSyntaxGuard(
            language_config=get_language_config("go"),
        )
        art = _make_artifact("// Error: LLM call failed")
        result = guard.validate(art)
        assert result.passed is False


# =========================================================================
# evaluation.py – prepare_predictions
# =========================================================================


class TestPreparePredictions:
    def test_writes_json_per_arm(self, tmp_path):
        from examples.swe_bench_ablation.experiment_runner import ArmResult
        from examples.swe_bench_pro.evaluation import prepare_predictions

        results = [
            ArmResult(
                instance_id="repo__1",
                arm="02_singleshot",
                workflow_status="success",
                patch_content="diff --git a/f.py",
            ),
            ArmResult(
                instance_id="repo__2",
                arm="02_singleshot",
                workflow_status="error",
                patch_content="",
            ),
            ArmResult(
                instance_id="repo__3",
                arm="03_s1_direct",
                workflow_status="success",
                patch_content="diff --git a/g.go",
            ),
        ]

        files = prepare_predictions(results, str(tmp_path))

        assert "02_singleshot" in files
        assert "03_s1_direct" in files

        ss = json.loads(files["02_singleshot"].read_text())
        assert len(ss) == 1  # error result excluded
        assert ss[0]["instance_id"] == "repo__1"
        assert ss[0]["patch"] == "diff --git a/f.py"
        assert "prefix" in ss[0]

    def test_empty_results(self, tmp_path):
        from examples.swe_bench_pro.evaluation import prepare_predictions

        files = prepare_predictions([], str(tmp_path))
        assert files == {}


# =========================================================================
# evaluation.py – load_evaluation_results
# =========================================================================


class TestLoadEvalResults:
    def test_loads_from_eval_output(self, tmp_path):
        from examples.swe_bench_pro.evaluation import load_evaluation_results

        out = tmp_path / "eval_output"
        out.mkdir()
        data = {"repo__1": True, "repo__2": False}
        (out / "eval_results.json").write_text(json.dumps(data))

        resolved = load_evaluation_results(str(tmp_path), write_logs=False)
        assert resolved == {"repo__1": True, "repo__2": False}

    def test_writes_logs_when_enabled(self, tmp_path):
        from examples.swe_bench_pro.evaluation import load_evaluation_results

        out = tmp_path / "eval_output"
        out.mkdir()
        (out / "eval_results.json").write_text(json.dumps({"repo__1": True}))

        resolved = load_evaluation_results(str(tmp_path), run_id="r1", write_logs=True)
        assert resolved == {"repo__1": True}

        log_dir = tmp_path / "eval_logs" / "r1"
        assert log_dir.exists()
        assert (log_dir / "repo__1.log").exists()

    def test_missing_file_returns_empty(self, tmp_path):
        from examples.swe_bench_pro.evaluation import load_evaluation_results

        resolved = load_evaluation_results(str(tmp_path), write_logs=False)
        assert resolved == {}


# =========================================================================
# experiment_runner.py – build helpers
# =========================================================================


class TestWorkflowBuildHelpers:
    def test_load_workflow_config(self):
        from examples.swe_bench_pro.experiment_runner import load_workflow_config

        config = load_workflow_config("02_singleshot")
        assert "action_pairs" in config

    def test_load_prompts(self):
        from examples.swe_bench_pro.experiment_runner import load_prompts

        prompts = load_prompts()
        assert "ap_patch" in prompts
        assert "ap_singleshot" in prompts
        # Verify language-neutral text
        assert "VALID PYTHON" not in prompts["ap_patch"].constraints
        assert "VALID CODE" in prompts["ap_patch"].constraints

    def test_get_registries_python(self):
        from examples.swe_bench_pro.experiment_runner import (
            _get_generator_registry,
            _get_guard_registry,
        )

        lang = get_language_config("python")
        gen_reg = _get_generator_registry(lang)
        guard_reg = _get_guard_registry(lang)

        # For Python, should use the base classes
        from examples.swe_bench_ablation.generators import PatchGenerator, TestGenerator
        from examples.swe_bench_ablation.guards import TestSyntaxGuard

        assert gen_reg["PatchGenerator"] is PatchGenerator
        assert gen_reg["TestGenerator"] is TestGenerator
        assert guard_reg["test_syntax"] is TestSyntaxGuard

    def test_get_registries_go(self):
        from examples.swe_bench_pro.experiment_runner import (
            _get_generator_registry,
            _get_guard_registry,
        )
        from examples.swe_bench_pro.generators import (
            MultiLangPatchGenerator,
            MultiLangTestGenerator,
        )
        from examples.swe_bench_pro.guards import MultiLangTestSyntaxGuard

        lang = get_language_config("go")
        gen_reg = _get_generator_registry(lang)
        guard_reg = _get_guard_registry(lang)

        assert gen_reg["PatchGenerator"] is MultiLangPatchGenerator
        assert gen_reg["TestGenerator"] is MultiLangTestGenerator
        assert guard_reg["test_syntax"] is MultiLangTestSyntaxGuard


# =========================================================================
# dataset.py – SWEBenchProInstance
# =========================================================================


class TestSWEBenchProInstance:
    def test_frozen(self):
        from examples.swe_bench_pro.dataset import SWEBenchProInstance

        inst = SWEBenchProInstance(
            instance_id="repo__1",
            repo="org/repo",
            base_commit="abc123",
            problem_statement="bug",
            patch="diff",
            test_patch="diff",
            repo_language="go",
        )
        with pytest.raises(AttributeError):
            inst.repo_language = "python"  # type: ignore[misc]

    def test_defaults(self):
        from examples.swe_bench_pro.dataset import SWEBenchProInstance

        inst = SWEBenchProInstance(
            instance_id="x",
            repo="r",
            base_commit="c",
            problem_statement="p",
            patch="d",
            test_patch="t",
        )
        assert inst.repo_language == ""
        assert inst.requirements == ""
        assert inst.interface == ""
        assert inst.fail_to_pass == []
        assert inst.pass_to_pass == []
