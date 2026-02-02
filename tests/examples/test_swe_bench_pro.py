"""Tests for the swe_bench_pro example package.

Covers language config, guards, generators (prompt construction), and
evaluation helpers.  No network, Docker, or HuggingFace access required.
"""

import json
import logging
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

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

    def test_dict_value_with_resolved_key(self, tmp_path):
        from examples.swe_bench_pro.evaluation import load_evaluation_results

        out = tmp_path / "eval_output"
        out.mkdir()
        data = {"repo__1": {"resolved": True, "extra": "info"}}
        (out / "eval_results.json").write_text(json.dumps(data))

        resolved = load_evaluation_results(str(tmp_path), write_logs=False)
        assert resolved == {"repo__1": True}

    def test_dict_value_missing_resolved_key_raises(self, tmp_path):
        from examples.swe_bench_pro.evaluation import load_evaluation_results

        out = tmp_path / "eval_output"
        out.mkdir()
        data = {"repo__1": {"status": "ok"}}
        (out / "eval_results.json").write_text(json.dumps(data))

        with pytest.raises(KeyError, match="resolved"):
            load_evaluation_results(str(tmp_path), write_logs=False)

    def test_unexpected_value_type_raises(self, tmp_path):
        from examples.swe_bench_pro.evaluation import load_evaluation_results

        out = tmp_path / "eval_output"
        out.mkdir()
        data = {"repo__1": 42}
        (out / "eval_results.json").write_text(json.dumps(data))

        with pytest.raises(TypeError, match="Unexpected value type"):
            load_evaluation_results(str(tmp_path), write_logs=False)

    def test_non_dict_top_level_raises(self, tmp_path):
        from examples.swe_bench_pro.evaluation import load_evaluation_results

        out = tmp_path / "eval_output"
        out.mkdir()
        (out / "eval_results.json").write_text(json.dumps([1, 2, 3]))

        with pytest.raises(ValueError, match="Expected dict"):
            load_evaluation_results(str(tmp_path), write_logs=False)


# =========================================================================
# evaluation.py – ensure_eval_repo
# =========================================================================


class TestEnsureEvalRepo:
    def test_update_path_raises_on_git_failure(self, tmp_path):
        """When the repo dir exists but git commands fail, RuntimeError is raised."""
        import subprocess

        from examples.swe_bench_pro.evaluation import ensure_eval_repo

        # Create a fake repo dir (not a real git repo)
        repo_dir = tmp_path / "SWE-bench_Pro-os"
        repo_dir.mkdir()

        with pytest.raises(RuntimeError, match="Failed to update eval repo"):
            ensure_eval_repo(cache_dir=str(tmp_path), commit="main")


# =========================================================================
# experiment_runner.py – build helpers
# =========================================================================


class TestWorkflowBuildHelpers:
    def test_load_workflow_config(self):
        from examples.swe_bench_pro.experiment_runner import load_workflow_config

        config = load_workflow_config("02_singleshot")
        assert "action_pairs" in config

    def test_load_workflow_config_missing_raises(self):
        from examples.swe_bench_pro.experiment_runner import load_workflow_config

        with pytest.raises(FileNotFoundError, match="Workflow not found"):
            load_workflow_config("nonexistent_workflow_99")

    def test_load_prompts_exists(self):
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


# =========================================================================
# generators/multilang_test.py – _extract_code
# =========================================================================


class TestExtractCode:
    def _make_generator(self, lang: str = "go"):
        from examples.swe_bench_pro.generators.multilang_test import (
            MultiLangTestGenerator,
        )

        return MultiLangTestGenerator(
            language_config=get_language_config(lang),
            model="test",
            base_url="http://localhost",
            api_key="test",
        )

    def test_extracts_language_specific_code_fence(self):
        gen = self._make_generator("go")
        content = 'Some text\n```go\nfunc TestFoo(t *testing.T) {}\n```\nMore text'
        result = gen._extract_code(content)
        assert "func TestFoo" in result

    def test_extracts_generic_code_fence(self):
        gen = self._make_generator("go")
        content = 'Some text\n```\nfunc TestFoo(t *testing.T) {}\n```\nMore text'
        result = gen._extract_code(content)
        assert "func TestFoo" in result

    def test_logs_warning_on_raw_pattern_match(self, caplog):
        gen = self._make_generator("go")
        content = 'func TestFoo(t *testing.T) { t.Log("hi") }'
        with caplog.at_level(logging.WARNING, logger="swe_bench_pro.generators"):
            result = gen._extract_code(content)
        assert "func TestFoo" in result
        assert "No code fence" in caplog.text

    def test_logs_warning_on_no_extraction(self, caplog):
        gen = self._make_generator("go")
        content = "This is just some random text with no code."
        with caplog.at_level(logging.WARNING, logger="swe_bench_pro.generators"):
            result = gen._extract_code(content)
        assert "Could not extract" in result
        assert "No code fences or test patterns found" in caplog.text


# =========================================================================
# experiment_runner.py – run_all parallel execution
# =========================================================================


class TestRunAllParallel:
    """Tests for parallel execution in ``SWEBenchProRunner.run_all``."""

    def _make_instances(self, n: int = 3):
        from examples.swe_bench_pro.dataset import SWEBenchProInstance

        return [
            SWEBenchProInstance(
                instance_id=f"org__repo__{i}",
                repo="org/repo",
                base_commit="abc123",
                problem_statement=f"bug {i}",
                patch="diff",
                test_patch="diff",
                repo_language="python",
            )
            for i in range(n)
        ]

    def test_max_workers_zero_raises(self, tmp_path):
        from examples.swe_bench_pro.experiment_runner import SWEBenchProRunner

        runner = SWEBenchProRunner(output_dir=str(tmp_path / "out"))
        with pytest.raises(ValueError, match="max_workers must be >= 1"):
            runner.run_all(arms=["02_singleshot"], max_workers=0)

    @patch("examples.swe_bench_pro.experiment_runner.load_swe_bench_pro")
    def test_sequential_execution(self, mock_load, tmp_path):
        from examples.swe_bench_ablation.experiment_runner import ArmResult
        from examples.swe_bench_pro.experiment_runner import SWEBenchProRunner

        instances = self._make_instances(2)
        mock_load.return_value = instances

        runner = SWEBenchProRunner(output_dir=str(tmp_path / "out"))
        call_order = []

        def fake_run_instance(instance, arm):
            call_order.append(instance.instance_id)
            return ArmResult(
                instance_id=instance.instance_id,
                arm=arm,
                workflow_status="success",
                patch_content="diff",
            )

        runner.run_instance = fake_run_instance

        results = runner.run_all(arms=["02_singleshot"], max_workers=1)

        assert len(results) == 2
        assert call_order == ["org__repo__0", "org__repo__1"]
        # Verify JSONL was written
        results_path = tmp_path / "out" / "results.jsonl"
        assert results_path.exists()
        lines = results_path.read_text().strip().split("\n")
        assert len(lines) == 2

    @patch("examples.swe_bench_pro.experiment_runner.load_swe_bench_pro")
    def test_parallel_execution(self, mock_load, tmp_path):
        from examples.swe_bench_ablation.experiment_runner import ArmResult
        from examples.swe_bench_pro.experiment_runner import SWEBenchProRunner

        instances = self._make_instances(4)
        mock_load.return_value = instances

        runner = SWEBenchProRunner(output_dir=str(tmp_path / "out"))

        def fake_run_instance(instance, arm):
            return ArmResult(
                instance_id=instance.instance_id,
                arm=arm,
                workflow_status="success",
                patch_content="diff",
            )

        runner.run_instance = fake_run_instance

        results = runner.run_all(arms=["02_singleshot"], max_workers=3)

        assert len(results) == 4
        # All instances should have results
        result_ids = {r.instance_id for r in results}
        assert result_ids == {f"org__repo__{i}" for i in range(4)}
        # JSONL should have 4 lines
        lines = (tmp_path / "out" / "results.jsonl").read_text().strip().split("\n")
        assert len(lines) == 4

    @patch("examples.swe_bench_pro.experiment_runner.load_swe_bench_pro")
    def test_parallel_with_resume(self, mock_load, tmp_path):
        from examples.swe_bench_ablation.experiment_runner import ArmResult
        from examples.swe_bench_pro.experiment_runner import SWEBenchProRunner

        instances = self._make_instances(3)
        mock_load.return_value = instances

        # Pre-populate results for first instance
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        existing = ArmResult(
            instance_id="org__repo__0",
            arm="02_singleshot",
            workflow_status="success",
            patch_content="existing",
        )
        from dataclasses import asdict

        (out_dir / "results.jsonl").write_text(
            json.dumps(asdict(existing)) + "\n"
        )

        runner = SWEBenchProRunner(output_dir=str(out_dir))

        def fake_run_instance(instance, arm):
            return ArmResult(
                instance_id=instance.instance_id,
                arm=arm,
                workflow_status="success",
                patch_content="new",
            )

        runner.run_instance = fake_run_instance

        results = runner.run_all(
            arms=["02_singleshot"],
            max_workers=2,
            resume_from=str(out_dir),
        )

        # Should have 3 results: 1 existing + 2 new
        assert len(results) == 3
        # The existing one should still be there
        existing_results = [r for r in results if r.instance_id == "org__repo__0"]
        assert len(existing_results) == 1
        assert existing_results[0].patch_content == "existing"

    @patch("examples.swe_bench_pro.experiment_runner.load_swe_bench_pro")
    def test_concurrent_jsonl_writes_are_valid(self, mock_load, tmp_path):
        """Force overlapping writes and verify every JSONL line is valid JSON."""
        from examples.swe_bench_ablation.experiment_runner import ArmResult
        from examples.swe_bench_pro.experiment_runner import SWEBenchProRunner

        n_instances = 20
        instances = self._make_instances(n_instances)
        mock_load.return_value = instances

        runner = SWEBenchProRunner(output_dir=str(tmp_path / "out"))
        barrier = threading.Barrier(min(4, n_instances), timeout=5)

        def slow_run_instance(instance, arm):
            # Force threads to pile up, then release together.
            try:
                barrier.wait()
            except threading.BrokenBarrierError:
                pass
            return ArmResult(
                instance_id=instance.instance_id,
                arm=arm,
                workflow_status="success",
                patch_content=f"diff for {instance.instance_id}",
            )

        runner.run_instance = slow_run_instance

        results = runner.run_all(arms=["02_singleshot"], max_workers=4)

        assert len(results) == n_instances

        # Every line in the JSONL must be independently parseable JSON.
        jsonl_path = tmp_path / "out" / "results.jsonl"
        lines = jsonl_path.read_text().strip().split("\n")
        assert len(lines) == n_instances
        parsed_ids = set()
        for i, line in enumerate(lines):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                pytest.fail(f"JSONL line {i} is not valid JSON: {line!r}")
            parsed_ids.add(obj["instance_id"])
        # No duplicates, no missing.
        assert parsed_ids == {f"org__repo__{i}" for i in range(n_instances)}

    @patch("examples.swe_bench_pro.experiment_runner.load_swe_bench_pro")
    def test_concurrent_results_no_duplicates(self, mock_load, tmp_path):
        """With many workers, results list must have no duplicates or drops."""
        from examples.swe_bench_ablation.experiment_runner import ArmResult
        from examples.swe_bench_pro.experiment_runner import SWEBenchProRunner

        n_instances = 15
        arms = ["02_singleshot", "03_s1_direct"]
        instances = self._make_instances(n_instances)
        mock_load.return_value = instances

        runner = SWEBenchProRunner(output_dir=str(tmp_path / "out"))

        def fake_run_instance(instance, arm):
            # Small jitter to provoke interleaving.
            time.sleep(0.001)
            return ArmResult(
                instance_id=instance.instance_id,
                arm=arm,
                workflow_status="success",
                patch_content="diff",
            )

        runner.run_instance = fake_run_instance

        results = runner.run_all(arms=arms, max_workers=6)

        expected_keys = {
            (f"org__repo__{i}", arm) for i in range(n_instances) for arm in arms
        }
        actual_keys = {(r.instance_id, r.arm) for r in results}
        assert actual_keys == expected_keys, (
            f"Missing: {expected_keys - actual_keys}, "
            f"Extra: {actual_keys - expected_keys}"
        )

    @patch("examples.swe_bench_pro.experiment_runner.load_swe_bench_pro")
    def test_concurrent_progress_counter(self, mock_load, tmp_path, caplog):
        """finished_count in log messages must reach total_runs exactly."""
        from examples.swe_bench_ablation.experiment_runner import ArmResult
        from examples.swe_bench_pro.experiment_runner import SWEBenchProRunner

        n_instances = 8
        instances = self._make_instances(n_instances)
        mock_load.return_value = instances

        runner = SWEBenchProRunner(output_dir=str(tmp_path / "out"))

        def fake_run_instance(instance, arm):
            time.sleep(0.001)
            return ArmResult(
                instance_id=instance.instance_id,
                arm=arm,
                workflow_status="success",
                patch_content="diff",
            )

        runner.run_instance = fake_run_instance

        with caplog.at_level(logging.INFO, logger="swe_bench_pro"):
            results = runner.run_all(arms=["02_singleshot"], max_workers=4)

        assert len(results) == n_instances

        # Extract "Finished N/M" from log messages.
        import re

        finished_nums = []
        for record in caplog.records:
            m = re.search(r"Finished (\d+)/(\d+):", record.getMessage())
            if m:
                finished_nums.append(int(m.group(1)))
                assert int(m.group(2)) == n_instances
        # Must see all counts from 1..n_instances (order may vary).
        assert sorted(finished_nums) == list(range(1, n_instances + 1))
