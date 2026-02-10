"""Tests for the swe_bench_pro example package.

Covers language config, guards, generators (prompt construction), and
evaluation helpers.  No network, Docker, or HuggingFace access required.
"""

import contextlib
import json
import logging
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from examples.swe_bench_pro.language import (
    LANGUAGE_CONFIGS,
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
        assert set(LANGUAGE_CONFIGS) == {
            "python",
            "go",
            "javascript",
            "typescript",
        }

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
                patch_content="diff --git a/f.py",
            ),
            ArmResult(
                instance_id="repo__2",
                arm="02_singleshot",
                error="Some error",
                patch_content="",
            ),
            ArmResult(
                instance_id="repo__3",
                arm="03_s1_direct",
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

    def test_build_workflow_passes_repo_root_to_patch_generator(self, tmp_path):
        """``build_workflow`` must inject ``repo_root`` into PatchGenerator instances."""
        from examples.swe_bench_pro.experiment_runner import (
            build_workflow,
            load_prompts,
            load_workflow_config,
        )

        from atomicguard.infrastructure.persistence.filesystem import (
            FilesystemArtifactDAG,
        )

        config = load_workflow_config("02_singleshot")
        prompts = load_prompts()
        lang = get_language_config("python")
        dag = FilesystemArtifactDAG(str(tmp_path / "dag"))

        repo_root = str(tmp_path / "fake_repo")

        wf = build_workflow(
            config=config,
            prompts=prompts,
            lang_config=lang,
            model="test",
            base_url="http://localhost",
            artifact_dag=dag,
            repo_root=repo_root,
            api_key="test",
        )

        # The singleshot step uses a PatchGenerator; verify it received repo_root.
        found_patch_gen = False
        for step in wf._steps:
            gen = step.action_pair.generator
            if hasattr(gen, "_repo_root"):
                found_patch_gen = True
                assert gen._repo_root == repo_root, (
                    f"Generator {type(gen).__name__} did not receive repo_root"
                )
        assert found_patch_gen, "No PatchGenerator found in workflow steps"


# =========================================================================
# PatchGenerator – repo_root via constructor
# =========================================================================


class TestPatchGeneratorRepoRoot:
    """Verify that PatchGenerator uses repo_root from its constructor.

    The ``_build_prompt()`` and ``_process_output()`` methods resolve
    ``repo_root`` from the context first, then fall back to
    ``self._repo_root`` (set via the constructor).
    """

    def test_constructor_stores_repo_root(self, tmp_path):
        """Constructor ``repo_root`` kwarg is stored on the instance."""
        from examples.swe_bench_ablation.generators import PatchGenerator

        gen = PatchGenerator(
            model="test",
            base_url="http://localhost",
            api_key="test",
            repo_root=str(tmp_path),
        )
        assert gen._repo_root == str(tmp_path)

    def test_constructor_repo_root_defaults_to_none(self):
        from examples.swe_bench_ablation.generators import PatchGenerator

        gen = PatchGenerator(
            model="test",
            base_url="http://localhost",
            api_key="test",
        )
        assert gen._repo_root is None

    def test_process_output_produces_patch_key_with_repo_root(self, tmp_path):
        """_process_output produces a 'patch' key when given a valid repo_root."""
        from examples.swe_bench_ablation.generators import PatchGenerator
        from examples.swe_bench_ablation.models import Patch, SearchReplaceEdit

        from atomicguard.domain.models import AmbientEnvironment, Context

        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "hello.py").write_text("print('hello')\n")

        gen = PatchGenerator(
            model="test",
            base_url="http://localhost",
            api_key="test",
            repo_root=str(repo),
        )

        patch_output = Patch(
            edits=[
                SearchReplaceEdit(
                    file="hello.py",
                    search="print('hello')",
                    replace="print('world')",
                )
            ],
            reasoning="Changed greeting",
        )

        mock_repo = MagicMock(spec=[])  # no metadata attribute
        ambient = AmbientEnvironment(repository=mock_repo, constraints="")
        ctx = Context(ambient=ambient, specification="bug")

        result = gen._process_output(patch_output, ctx)
        data = json.loads(result)

        assert "patch" in data, f"Expected 'patch' key, got keys: {list(data.keys())}"
        assert "hello.py" in data["patch"]
        assert "-print('hello')" in data["patch"]
        assert "+print('world')" in data["patch"]

    def test_process_output_without_repo_root_has_no_patch(self):
        """Without repo_root, _process_output returns raw edits (no 'patch' key)."""
        from examples.swe_bench_ablation.generators import PatchGenerator
        from examples.swe_bench_ablation.models import Patch, SearchReplaceEdit

        from atomicguard.domain.models import AmbientEnvironment, Context

        gen = PatchGenerator(
            model="test",
            base_url="http://localhost",
            api_key="test",
        )

        patch_output = Patch(
            edits=[
                SearchReplaceEdit(
                    file="hello.py",
                    search="print('hello')",
                    replace="print('world')",
                )
            ],
            reasoning="Changed greeting",
        )

        mock_repo = MagicMock(spec=[])  # no metadata attribute
        ambient = AmbientEnvironment(repository=mock_repo, constraints="")
        ctx = Context(ambient=ambient, specification="bug")

        result = gen._process_output(patch_output, ctx)
        data = json.loads(result)

        assert "patch" not in data
        assert "edits" in data


# =========================================================================
# PatchGenerator – _list_repo_files
# =========================================================================


class TestListRepoFiles:
    """Verify _list_repo_files discovers source files correctly."""

    def _make_gen(self):
        from examples.swe_bench_ablation.generators import PatchGenerator

        return PatchGenerator(
            model="test",
            base_url="http://localhost",
            api_key="test",
        )

    def test_finds_python_files(self, tmp_path):
        (tmp_path / "main.py").write_text("x = 1")
        (tmp_path / "lib").mkdir()
        (tmp_path / "lib" / "util.py").write_text("y = 2")
        (tmp_path / "README.md").write_text("docs")

        gen = self._make_gen()
        files = gen._list_repo_files(str(tmp_path))
        assert "main.py" in files
        assert str(Path("lib") / "util.py") in files
        assert "README.md" not in files

    def test_skips_pycache_and_git(self, tmp_path):
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "cached.py").write_text("")
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "config.py").write_text("")
        (tmp_path / "real.py").write_text("code")

        gen = self._make_gen()
        files = gen._list_repo_files(str(tmp_path))
        assert files == ["real.py"]

    def test_respects_max_files(self, tmp_path):
        for i in range(20):
            (tmp_path / f"f{i:02d}.py").write_text("")

        gen = self._make_gen()
        files = gen._list_repo_files(str(tmp_path), max_files=5)
        assert len(files) == 5

    def test_custom_extensions(self, tmp_path):
        (tmp_path / "main.go").write_text("package main")
        (tmp_path / "util.py").write_text("x = 1")

        gen = self._make_gen()
        files = gen._list_repo_files(str(tmp_path), extensions=(".go",))
        assert "main.go" in files
        assert "util.py" not in files


# =========================================================================
# experiment_runner.py – _list_repo_files (standalone)
# =========================================================================


class TestListRepoFilesStandalone:
    """Verify the module-level _list_repo_files in experiment_runner."""

    def test_finds_python_files(self, tmp_path):
        from examples.swe_bench_pro.experiment_runner import _list_repo_files

        (tmp_path / "main.py").write_text("x = 1")
        (tmp_path / "lib").mkdir()
        (tmp_path / "lib" / "util.py").write_text("y = 2")
        (tmp_path / "README.md").write_text("docs")

        files = _list_repo_files(str(tmp_path))
        assert "main.py" in files
        assert str(Path("lib") / "util.py") in files
        assert "README.md" not in files

    def test_skips_pycache_and_git(self, tmp_path):
        from examples.swe_bench_pro.experiment_runner import _list_repo_files

        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "cached.py").write_text("")
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "config.py").write_text("")
        (tmp_path / "real.py").write_text("code")

        files = _list_repo_files(str(tmp_path))
        assert files == ["real.py"]

    def test_returns_all_files(self, tmp_path):
        """_list_repo_files returns all matching files (no truncation)."""
        from examples.swe_bench_pro.experiment_runner import _list_repo_files

        for i in range(20):
            (tmp_path / f"f{i:02d}.py").write_text("")

        files = _list_repo_files(str(tmp_path))
        assert len(files) == 20

    def test_custom_extensions(self, tmp_path):
        from examples.swe_bench_pro.experiment_runner import _list_repo_files

        (tmp_path / "main.go").write_text("package main")
        (tmp_path / "util.py").write_text("x = 1")

        files = _list_repo_files(str(tmp_path), extensions=(".go",))
        assert "main.go" in files
        assert "util.py" not in files

    def test_empty_repo(self, tmp_path):
        from examples.swe_bench_pro.experiment_runner import _list_repo_files

        files = _list_repo_files(str(tmp_path))
        assert files == []


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
            patch_content="existing",
        )
        from dataclasses import asdict

        (out_dir / "results.jsonl").write_text(json.dumps(asdict(existing)) + "\n")

        runner = SWEBenchProRunner(output_dir=str(out_dir))

        def fake_run_instance(instance, arm):
            return ArmResult(
                instance_id=instance.instance_id,
                arm=arm,
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
            with contextlib.suppress(threading.BrokenBarrierError):
                barrier.wait()
            return ArmResult(
                instance_id=instance.instance_id,
                arm=arm,
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


# =========================================================================
# AnalysisGuard – file existence validation
# =========================================================================


class TestAnalysisGuardFileValidation:
    """Verify AnalysisGuard rejects hallucinated file paths."""

    @pytest.fixture()
    def _make_artifact(self):
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

    def _analysis_json(self, files: list[str]) -> str:
        return json.dumps(
            {
                "bug_type": "logic",
                "root_cause_hypothesis": "Something is wrong",
                "fix_approach": "Fix it",
                "files": files,
                "confidence": "medium",
            }
        )

    def test_valid_files_with_repo_root(self, tmp_path, _make_artifact):
        from examples.swe_bench_ablation.guards.analysis_guard import AnalysisGuard

        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("x = 1")

        guard = AnalysisGuard(repo_root=str(tmp_path))
        art = _make_artifact(self._analysis_json(["src/main.py"]))
        result = guard.validate(art)
        assert result.passed is True

    def test_hallucinated_files_rejected(self, tmp_path, _make_artifact):
        from examples.swe_bench_ablation.guards.analysis_guard import AnalysisGuard

        guard = AnalysisGuard(repo_root=str(tmp_path))
        art = _make_artifact(self._analysis_json(["tests/utils/test_log.py"]))
        result = guard.validate(art)
        assert result.passed is False
        assert "Files not found in repository" in result.feedback
        assert "tests/utils/test_log.py" in result.feedback

    def test_mixed_valid_and_hallucinated_rejected(self, tmp_path, _make_artifact):
        from examples.swe_bench_ablation.guards.analysis_guard import AnalysisGuard

        (tmp_path / "real.py").write_text("x = 1")

        guard = AnalysisGuard(repo_root=str(tmp_path))
        art = _make_artifact(self._analysis_json(["real.py", "fake.py"]))
        result = guard.validate(art)
        assert result.passed is False
        assert "fake.py" in result.feedback

    def test_no_repo_root_skips_file_check(self, _make_artifact):
        from examples.swe_bench_ablation.guards.analysis_guard import AnalysisGuard

        guard = AnalysisGuard()
        art = _make_artifact(self._analysis_json(["nonexistent/file.py"]))
        result = guard.validate(art)
        assert result.passed is True

    def test_many_missing_files_truncated(self, tmp_path, _make_artifact):
        from examples.swe_bench_ablation.guards.analysis_guard import AnalysisGuard

        guard = AnalysisGuard(repo_root=str(tmp_path))
        files = [f"missing_{i}.py" for i in range(5)]
        art = _make_artifact(self._analysis_json(files))
        result = guard.validate(art)
        assert result.passed is False
        assert "and 2 more" in result.feedback


# =========================================================================
# PatchGuard – empty diff / identical edit rejection
# =========================================================================


class TestPatchGuardEmptyDiff:
    """Verify PatchGuard rejects patches with no actual changes."""

    @pytest.fixture()
    def _make_artifact(self):
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

    def test_patch_with_changes_passes(self, _make_artifact):
        from examples.swe_bench_ablation.guards.patch_guard import PatchGuard

        patch = (
            "--- a/file.py\n"
            "+++ b/file.py\n"
            "@@ -1,3 +1,3 @@\n"
            " x = 1\n"
            "-y = 2\n"
            "+y = 3\n"
            " z = 4\n"
        )
        guard = PatchGuard(require_git_apply=False, require_syntax_valid=False)
        art = _make_artifact(json.dumps({"patch": patch}))
        result = guard.validate(art)
        assert result.passed is True

    def test_patch_with_no_changes_rejected(self, _make_artifact):
        from examples.swe_bench_ablation.guards.patch_guard import PatchGuard

        patch = (
            "--- a/file.py\n+++ b/file.py\n@@ -1,3 +1,3 @@\n x = 1\n y = 2\n z = 4\n"
        )
        guard = PatchGuard(require_git_apply=False, require_syntax_valid=False)
        art = _make_artifact(json.dumps({"patch": patch}))
        result = guard.validate(art)
        assert result.passed is False
        assert "no actual changes" in result.feedback.lower()

    def test_edits_with_identical_search_replace_rejected(self, _make_artifact):
        from examples.swe_bench_ablation.guards.patch_guard import PatchGuard

        edits = [{"file": "f.py", "search": "x = 1", "replace": "x = 1"}]
        guard = PatchGuard(require_git_apply=False, require_syntax_valid=False)
        art = _make_artifact(json.dumps({"edits": edits}))
        result = guard.validate(art)
        assert result.passed is False
        assert "identical search and replace" in result.feedback.lower()

    def test_edits_with_real_changes_passes(self, _make_artifact):
        from examples.swe_bench_ablation.guards.patch_guard import PatchGuard

        edits = [{"file": "f.py", "search": "x = 1", "replace": "x = 2"}]
        guard = PatchGuard(require_git_apply=False, require_syntax_valid=False)
        art = _make_artifact(json.dumps({"edits": edits}))
        result = guard.validate(art)
        assert result.passed is True


# =========================================================================
# PatchGenerator – code_block_tag param
# =========================================================================


class TestPatchGeneratorLanguageParams:
    """Verify configurable code_block_tag."""

    def _make_gen(self, **kwargs):
        from examples.swe_bench_ablation.generators import PatchGenerator

        defaults = {"model": "test", "base_url": "http://localhost", "api_key": "test"}
        defaults.update(kwargs)
        return PatchGenerator(**defaults)

    def _make_context(self, specification: str = "bug"):
        from atomicguard.domain.models import AmbientEnvironment, Context

        ambient = AmbientEnvironment(repository=MagicMock(spec=[]), constraints="")
        return Context(ambient=ambient, specification=specification)

    def test_default_code_block_tag_is_python(self):
        gen = self._make_gen()
        assert gen._code_block_tag == "python"

    def test_custom_code_block_tag(self):
        gen = self._make_gen(code_block_tag="go")
        assert gen._code_block_tag == "go"


# =========================================================================
# PatchGenerator – singleshot file content fallback
# =========================================================================


class TestPatchGeneratorSingleshotFileContent:
    """Verify PatchGenerator includes referenced file content for singleshot."""

    def _make_gen(self, **kwargs):
        from examples.swe_bench_ablation.generators import PatchGenerator

        defaults = {"model": "test", "base_url": "http://localhost", "api_key": "test"}
        defaults.update(kwargs)
        return PatchGenerator(**defaults)

    def _make_context(self, specification: str):
        from atomicguard.domain.models import AmbientEnvironment, Context

        ambient = AmbientEnvironment(repository=MagicMock(spec=[]), constraints="")
        return Context(ambient=ambient, specification=specification)

    def test_singleshot_includes_referenced_files(self, tmp_path):
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("def hello():\n    return 'hi'\n")
        (tmp_path / "src" / "other.py").write_text("x = 1\n")

        gen = self._make_gen(repo_root=str(tmp_path))
        spec = "There is a bug in src/main.py where hello() returns wrong value"
        ctx = self._make_context(spec)
        prompt = gen._build_prompt(ctx, None)

        assert "## Current File Content" in prompt
        assert "src/main.py" in prompt
        assert "def hello():" in prompt

    def test_singleshot_no_unreferenced_files(self, tmp_path):
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("def hello():\n    return 'hi'\n")
        (tmp_path / "src" / "other.py").write_text("x = 1\n")

        gen = self._make_gen(repo_root=str(tmp_path))
        spec = "There is a bug in src/main.py"
        ctx = self._make_context(spec)
        prompt = gen._build_prompt(ctx, None)

        assert "src/main.py" in prompt
        # other.py is not mentioned in spec, so its content should not appear
        assert "x = 1" not in prompt

    def test_singleshot_limits_to_five_files(self, tmp_path):
        (tmp_path / "src").mkdir()
        file_names = []
        for i in range(8):
            name = f"f{i}.py"
            (tmp_path / "src" / name).write_text(f"content_{i}\n")
            file_names.append(f"src/{name}")

        gen = self._make_gen(repo_root=str(tmp_path))
        # Reference all 8 files in the spec
        spec = "Bug affects: " + ", ".join(file_names)
        ctx = self._make_context(spec)
        prompt = gen._build_prompt(ctx, None)

        # Count how many file content sections appear
        count = prompt.count("### src/f")
        assert count == 5


# =========================================================================
# MultiLangPatchGenerator – singleshot file content fallback
# =========================================================================


class TestMultiLangPatchGeneratorSingleshotFileContent:
    """Verify MultiLangPatchGenerator includes referenced file content for singleshot."""

    def test_singleshot_includes_referenced_files(self, tmp_path):
        from examples.swe_bench_pro.generators.multilang_patch import (
            MultiLangPatchGenerator,
        )

        lang = get_language_config("go")
        (tmp_path / "cmd").mkdir()
        (tmp_path / "cmd" / "main.go").write_text("package main\n")

        gen = MultiLangPatchGenerator(
            language_config=lang,
            model="test",
            base_url="http://localhost",
            api_key="test",
            repo_root=str(tmp_path),
        )

        from atomicguard.domain.models import AmbientEnvironment, Context

        ambient = AmbientEnvironment(repository=MagicMock(spec=[]), constraints="")
        spec = "Bug in cmd/main.go"
        ctx = Context(ambient=ambient, specification=spec)
        prompt = gen._build_prompt(ctx, None)

        assert "## Current File Content" in prompt
        assert "cmd/main.go" in prompt
        assert "```go" in prompt
        assert "package main" in prompt


# =========================================================================
# PatchGuard – file existence validation
# =========================================================================


class TestPatchGuardFileExistence:
    """Verify PatchGuard rejects edits targeting non-existent files."""

    @pytest.fixture()
    def _make_artifact(self):
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

    def test_edit_to_existing_file_passes(self, tmp_path, _make_artifact):
        from examples.swe_bench_ablation.guards.patch_guard import PatchGuard

        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("x = 1\n")

        edits = [{"file": "src/main.py", "search": "x = 1", "replace": "x = 2"}]
        guard = PatchGuard(
            require_git_apply=False,
            require_syntax_valid=False,
            repo_root=str(tmp_path),
        )
        art = _make_artifact(json.dumps({"edits": edits}))
        result = guard.validate(art)
        assert result.passed is True

    def test_edit_to_missing_file_rejected(self, tmp_path, _make_artifact):
        from examples.swe_bench_ablation.guards.patch_guard import PatchGuard

        edits = [{"file": "nonexistent.py", "search": "x = 1", "replace": "x = 2"}]
        guard = PatchGuard(
            require_git_apply=False,
            require_syntax_valid=False,
            repo_root=str(tmp_path),
        )
        art = _make_artifact(json.dumps({"edits": edits}))
        result = guard.validate(art)
        assert result.passed is False
        assert "Files not found in repository" in result.feedback
        assert "nonexistent.py" in result.feedback

    def test_multiple_missing_files_truncated(self, tmp_path, _make_artifact):
        from examples.swe_bench_ablation.guards.patch_guard import PatchGuard

        edits = [
            {"file": f"missing_{i}.py", "search": "x", "replace": "y"} for i in range(5)
        ]
        guard = PatchGuard(
            require_git_apply=False,
            require_syntax_valid=False,
            repo_root=str(tmp_path),
        )
        art = _make_artifact(json.dumps({"edits": edits}))
        result = guard.validate(art)
        assert result.passed is False
        assert "and 2 more" in result.feedback

    def test_no_repo_root_skips_check(self, _make_artifact):
        from examples.swe_bench_ablation.guards.patch_guard import PatchGuard

        edits = [{"file": "nonexistent.py", "search": "x = 1", "replace": "x = 2"}]
        guard = PatchGuard(
            require_git_apply=False,
            require_syntax_valid=False,
            repo_root=None,
        )
        art = _make_artifact(json.dumps({"edits": edits}))
        result = guard.validate(art)
        # Without repo_root, file existence check is skipped, so it passes
        assert result.passed is True


# =========================================================================
# PatchGenerator – Test file detection
# =========================================================================


class TestPatchGeneratorTestFileDetection:
    """Verify _is_test_file correctly identifies test files."""

    def _make_gen(self, **kwargs):
        from examples.swe_bench_ablation.generators import PatchGenerator

        defaults = {"model": "test", "base_url": "http://localhost", "api_key": "test"}
        defaults.update(kwargs)
        return PatchGenerator(**defaults)

    @pytest.mark.parametrize(
        "path,expected",
        [
            # Should be detected as test files
            ("tests/test_foo.py", True),
            ("test/unit/test_bar.py", True),
            ("tests/unit/utils/test_log.py", True),
            ("test_module.py", True),
            ("src/module_test.py", True),
            ("foo/bar/test_baz.py", True),
            # Should NOT be detected as test files
            ("src/utils.py", False),
            ("qutebrowser/utils/urlutils.py", False),
            ("lib/testing_helpers.py", False),  # 'testing' != 'test' or 'tests'
            ("contest/module.py", False),  # 'contest' contains 'test' but not /test
            ("src/latest.py", False),
        ],
    )
    def test_is_test_file(self, path, expected):
        gen = self._make_gen()
        assert gen._is_test_file(path) is expected


# =========================================================================
# PatchGenerator – Utility file discovery
# =========================================================================


class TestPatchGeneratorUtilityDiscovery:
    """Verify _discover_utility_files and _extract_function_signatures."""

    def _make_gen(self, **kwargs):
        from examples.swe_bench_ablation.generators import PatchGenerator

        defaults = {"model": "test", "base_url": "http://localhost", "api_key": "test"}
        defaults.update(kwargs)
        return PatchGenerator(**defaults)

    def test_discover_utility_files(self, tmp_path):
        # Create a mock repository with utility files
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "utils.py").write_text("def helper(): pass")
        (tmp_path / "src" / "helpers").mkdir()
        (tmp_path / "src" / "helpers" / "common.py").write_text("def common_func(): pass")
        (tmp_path / "src" / "main.py").write_text("def main(): pass")
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_utils.py").write_text("def test_helper(): pass")

        gen = self._make_gen()
        utility_files = gen._discover_utility_files(str(tmp_path))

        # Should find utils.py and helpers/common.py but not test_utils.py
        assert "src/utils.py" in utility_files
        assert "src/helpers/common.py" in utility_files
        assert "src/main.py" not in utility_files
        assert "tests/test_utils.py" not in utility_files

    def test_extract_function_signatures(self, tmp_path):
        code = '''
def simple_func():
    pass

def func_with_args(a, b, c):
    return a + b + c

def func_with_types(x: int, y: str) -> bool:
    return True

class MyClass:
    def method(self):
        pass
'''
        (tmp_path / "utils.py").write_text(code)

        gen = self._make_gen()
        signatures = gen._extract_function_signatures(str(tmp_path), "utils.py")

        # Only top-level functions are extracted (regex uses ^ anchor)
        assert "def simple_func():" in signatures
        assert "def func_with_args(a, b, c):" in signatures
        assert "def func_with_types(x: int, y: str) -> bool:" in signatures
        # Methods inside classes are NOT extracted (indented, so don't match ^)
        assert "def method(self):" not in signatures


# =========================================================================
# PatchGenerator – TDD prompt includes guidance note
# =========================================================================


class TestPatchGeneratorTestGuidance:
    """Verify PatchGenerator includes guidance when test code is present."""

    def _make_gen(self, **kwargs):
        from examples.swe_bench_ablation.generators import PatchGenerator

        defaults = {"model": "test", "base_url": "http://localhost", "api_key": "test"}
        defaults.update(kwargs)
        return PatchGenerator(**defaults)

    def _make_context_with_test(self, test_code: str):
        from atomicguard.domain.models import AmbientEnvironment, Context

        # Create a mock repository that returns an artifact with test code
        repo = MagicMock(spec=["get_artifact"])
        artifact = MagicMock()
        artifact.content = test_code
        repo.get_artifact.return_value = artifact

        ambient = AmbientEnvironment(repository=repo, constraints="")
        ctx = Context(
            ambient=ambient,
            specification="There is a bug",
            dependency_artifacts=(("test", "test-artifact-id"),),
        )
        return ctx

    def test_tdd_prompt_includes_guidance_note(self):
        gen = self._make_gen()
        ctx = self._make_context_with_test("def test_foo(): assert True")
        prompt = gen._build_prompt(ctx, None)

        assert "REFERENCE ONLY" in prompt

    def test_tdd_prompt_warns_not_to_patch_tests(self):
        gen = self._make_gen()
        ctx = self._make_context_with_test("def test_foo(): assert True")
        prompt = gen._build_prompt(ctx, None)

        assert "DO NOT MODIFY" in prompt
        assert "Do NOT create or modify any test files" in prompt


# =========================================================================
# MultiLangPatchGenerator – TDD prompt includes guidance note
# =========================================================================


class TestMultiLangPatchGeneratorTestGuidance:
    """Verify MultiLangPatchGenerator includes guidance when test code is present."""

    def _make_context_with_test(self, test_code: str):
        from atomicguard.domain.models import AmbientEnvironment, Context

        repo = MagicMock(spec=["get_artifact"])
        artifact = MagicMock()
        artifact.content = test_code
        repo.get_artifact.return_value = artifact

        ambient = AmbientEnvironment(repository=repo, constraints="")
        ctx = Context(
            ambient=ambient,
            specification="There is a bug",
            dependency_artifacts=(("test", "test-artifact-id"),),
        )
        return ctx

    def test_tdd_prompt_includes_guidance_note(self):
        from examples.swe_bench_pro.generators.multilang_patch import (
            MultiLangPatchGenerator,
        )

        lang = get_language_config("go")
        gen = MultiLangPatchGenerator(
            language_config=lang,
            model="test",
            base_url="http://localhost",
            api_key="test",
        )
        ctx = self._make_context_with_test('func TestFoo(t *testing.T) { t.Log("ok") }')
        prompt = gen._build_prompt(ctx, None)

        assert "REFERENCE ONLY" in prompt

    def test_tdd_prompt_warns_not_to_patch_tests(self):
        from examples.swe_bench_pro.generators.multilang_patch import (
            MultiLangPatchGenerator,
        )

        lang = get_language_config("go")
        gen = MultiLangPatchGenerator(
            language_config=lang,
            model="test",
            base_url="http://localhost",
            api_key="test",
        )
        ctx = self._make_context_with_test('func TestFoo(t *testing.T) { t.Log("ok") }')
        prompt = gen._build_prompt(ctx, None)

        assert "DO NOT MODIFY" in prompt
        assert "Do NOT create or modify any test files" in prompt


# =========================================================================
# PydanticAIGenerator base class
# =========================================================================


class TestPydanticAIGeneratorBase:
    """Verify PydanticAIGenerator base class behaviour."""

    def test_ollama_provider_detection(self):
        """Ollama provider creates an OllamaProvider-backed model."""
        from examples.swe_bench_ablation.generators import AnalysisGenerator

        gen = AnalysisGenerator(
            model="test",
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            provider="ollama",
        )
        assert gen._agent.model.system == "ollama"

    def test_openai_provider_detection(self):
        """OpenAI provider creates an OpenAIProvider-backed model."""
        from examples.swe_bench_ablation.generators import AnalysisGenerator

        gen = AnalysisGenerator(
            model="test",
            base_url="http://example.com/v1",
            api_key="sk-test",
            provider="openai",
        )
        assert gen._agent.model.system == "openai"

    def test_openrouter_provider_detection(self):
        """OpenRouter provider creates an OpenRouterProvider-backed model."""
        from examples.swe_bench_ablation.generators import AnalysisGenerator

        gen = AnalysisGenerator(
            model="test",
            base_url="",
            api_key="sk-or-test",
            provider="openrouter",
        )
        assert gen._agent.model.system == "openrouter"

    def test_temperature_passthrough(self):
        """Temperature is stored and can be customised."""
        from examples.swe_bench_ablation.generators import AnalysisGenerator

        gen = AnalysisGenerator(
            model="test",
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            provider="ollama",
            temperature=0.7,
        )
        assert gen._temperature == 0.7

    def test_default_temperature_analysis(self):
        """AnalysisGenerator defaults to temperature=0.2."""
        from examples.swe_bench_ablation.generators import AnalysisGenerator

        gen = AnalysisGenerator(
            model="test",
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            provider="ollama",
        )
        assert gen._temperature == 0.2

    def test_default_temperature_patch(self):
        """PatchGenerator defaults to temperature=0.3."""
        from examples.swe_bench_ablation.generators import PatchGenerator

        gen = PatchGenerator(
            model="test",
            base_url="http://localhost",
            api_key="test",
            provider="ollama",
        )
        assert gen._temperature == 0.3

    def test_huggingface_provider_detection(self):
        """Explicit huggingface provider creates a HuggingFaceModel."""
        from examples.swe_bench_ablation.generators import AnalysisGenerator
        from pydantic_ai.models.huggingface import HuggingFaceModel

        gen = AnalysisGenerator(
            model="Qwen/Qwen2.5-Coder-32B-Instruct",
            base_url="",
            api_key="hf_test123",
            provider="huggingface",
        )
        assert isinstance(gen._agent.model, HuggingFaceModel)

    def test_unknown_provider_raises(self):
        """An unknown provider value raises ValueError."""
        from examples.swe_bench_ablation.generators import AnalysisGenerator

        with pytest.raises(ValueError, match="Unknown provider"):
            AnalysisGenerator(
                model="test",
                base_url="",
                api_key="test",
                provider="bedrock",
            )


# =========================================================================
# AnalysisGenerator – output_type
# =========================================================================


class TestAnalysisGeneratorPydanticAI:
    """Verify AnalysisGenerator uses PydanticAI structured output."""

    def test_output_type_is_analysis(self):
        from examples.swe_bench_ablation.generators import AnalysisGenerator
        from examples.swe_bench_ablation.models import Analysis

        assert AnalysisGenerator.output_type is Analysis

    def test_inherits_from_pydantic_ai_generator(self):
        from examples.base.generators import PydanticAIGenerator
        from examples.swe_bench_ablation.generators import AnalysisGenerator

        assert issubclass(AnalysisGenerator, PydanticAIGenerator)

    def test_localization_output_type(self):
        from examples.swe_bench_ablation.generators import LocalizationGenerator
        from examples.swe_bench_ablation.models import Localization

        assert LocalizationGenerator.output_type is Localization

    def test_patch_output_type(self):
        from examples.swe_bench_ablation.generators import PatchGenerator
        from examples.swe_bench_ablation.models import Patch

        assert PatchGenerator.output_type is Patch


# =========================================================================
# PatchGenerator – _resolve_repo_root
# =========================================================================


class TestPatchGeneratorResolveRepoRoot:
    """Verify _resolve_repo_root resolves from context then constructor."""

    def test_context_metadata_takes_priority(self, tmp_path):
        from examples.swe_bench_ablation.generators import PatchGenerator

        from atomicguard.domain.models import AmbientEnvironment, Context

        gen = PatchGenerator(
            model="test",
            base_url="http://localhost",
            api_key="test",
            repo_root="/fallback",
        )

        repo = MagicMock()
        repo.metadata = {"repo_root": str(tmp_path)}
        ambient = AmbientEnvironment(repository=repo, constraints="")
        ctx = Context(ambient=ambient, specification="bug")

        assert gen._resolve_repo_root(ctx) == str(tmp_path)

    def test_falls_back_to_constructor(self):
        from examples.swe_bench_ablation.generators import PatchGenerator

        from atomicguard.domain.models import AmbientEnvironment, Context

        gen = PatchGenerator(
            model="test",
            base_url="http://localhost",
            api_key="test",
            repo_root="/constructor",
        )

        ambient = AmbientEnvironment(repository=MagicMock(spec=[]), constraints="")
        ctx = Context(ambient=ambient, specification="bug")

        assert gen._resolve_repo_root(ctx) == "/constructor"


# =========================================================================
# PydanticAIGenerator.generate() – happy path
# =========================================================================


class TestGenerateMethodHappyPath:
    """Verify the generate() method produces correct artifacts on success."""

    def _make_gen(self, cls_name="AnalysisGenerator", **kwargs):
        from examples.swe_bench_ablation.generators import (
            AnalysisGenerator,
            PatchGenerator,
        )

        cls = AnalysisGenerator if cls_name == "AnalysisGenerator" else PatchGenerator
        defaults = {
            "model": "test",
            "base_url": "http://localhost:11434/v1",
            "api_key": "ollama",
        }
        defaults.update(kwargs)
        return cls(**defaults)

    def _make_context(self, specification="bug", deps=()):
        from atomicguard.domain.models import AmbientEnvironment, Context

        ambient = AmbientEnvironment(repository=MagicMock(spec=[]), constraints="c")
        return Context(
            ambient=ambient,
            specification=specification,
            dependency_artifacts=deps,
        )

    def _mock_agent(self, output):
        mock_agent = MagicMock()
        mock_result = MagicMock(spec=[])  # no usage attribute
        mock_result.output = output
        mock_agent.run_sync.return_value = mock_result
        return mock_agent

    def test_generate_returns_artifact_on_success(self):
        from examples.swe_bench_ablation.models import Analysis

        from atomicguard.domain.models import ArtifactStatus

        gen = self._make_gen()
        gen._agent = self._mock_agent(
            Analysis(
                bug_type="logic",
                root_cause_hypothesis="wrong branch",
                files=["f.py"],
                fix_approach="fix branch",
            )
        )

        ctx = self._make_context()
        art = gen.generate(ctx)

        assert art.status == ArtifactStatus.PENDING
        data = json.loads(art.content)
        assert data["bug_type"] == "logic"
        assert data["root_cause_hypothesis"] == "wrong branch"

    def test_generate_passes_prompt_and_system_to_agent(self):
        from examples.swe_bench_ablation.models import Analysis

        from atomicguard.domain.prompts import PromptTemplate

        gen = self._make_gen()
        mock_agent = self._mock_agent(
            Analysis(
                bug_type="logic",
                root_cause_hypothesis="x",
                files=["f.py"],
                fix_approach="y",
            )
        )
        gen._agent = mock_agent

        tmpl = PromptTemplate(role="Bug analyst", constraints="", task="Analyze this")
        ctx = self._make_context(specification="segfault in foo()")
        gen.generate(ctx, template=tmpl)

        call_args = mock_agent.run_sync.call_args
        prompt_arg = call_args[0][0]
        assert "segfault in foo()" in prompt_arg
        assert "Analyze this" in prompt_arg
        assert call_args[1]["instructions"] == "Bug analyst"

    def test_generate_increments_attempt_counter(self):
        from examples.swe_bench_ablation.models import Analysis

        gen = self._make_gen()
        gen._agent = self._mock_agent(
            Analysis(
                bug_type="logic",
                root_cause_hypothesis="x",
                files=["f.py"],
                fix_approach="y",
            )
        )

        ctx = self._make_context()
        art1 = gen.generate(ctx)
        art2 = gen.generate(ctx)

        assert art1.attempt_number == 1
        assert art2.attempt_number == 2

    def test_generate_captures_context_snapshot(self):
        from examples.swe_bench_ablation.models import Analysis

        gen = self._make_gen()
        gen._agent = self._mock_agent(
            Analysis(
                bug_type="logic",
                root_cause_hypothesis="x",
                files=["f.py"],
                fix_approach="y",
            )
        )

        ctx = self._make_context(
            specification="spec text",
            deps=(("analysis", "art-1"),),
        )
        art = gen.generate(ctx, workflow_id="wf-1")

        assert art.context.specification == "spec text"
        assert art.context.workflow_id == "wf-1"
        assert art.context.constraints == "c"
        assert art.context.dependency_artifacts == (("analysis", "art-1"),)

    def test_generate_patch_calls_process_output(self, tmp_path):
        from examples.swe_bench_ablation.models import Patch, SearchReplaceEdit

        (tmp_path / "hello.py").write_text("print('hello')\n")

        gen = self._make_gen(cls_name="PatchGenerator", repo_root=str(tmp_path))
        gen._agent = self._mock_agent(
            Patch(
                edits=[
                    SearchReplaceEdit(
                        file="hello.py",
                        search="print('hello')",
                        replace="print('world')",
                    )
                ],
                reasoning="Changed greeting",
            )
        )

        ctx = self._make_context()
        art = gen.generate(ctx)
        data = json.loads(art.content)
        assert "patch" in data


# =========================================================================
# PydanticAIGenerator.generate() – error handling
# =========================================================================


class TestGenerateMethodErrorPaths:
    """Verify generate() handles errors correctly."""

    def _make_gen(self):
        from examples.swe_bench_ablation.generators import AnalysisGenerator

        return AnalysisGenerator(
            model="test",
            base_url="http://localhost:11434/v1",
            api_key="ollama",
        )

    def _make_context(self):
        from atomicguard.domain.models import AmbientEnvironment, Context

        ambient = AmbientEnvironment(repository=MagicMock(spec=[]), constraints="")
        return Context(ambient=ambient, specification="bug")

    def test_generate_validation_error_returns_error_artifact(self):
        from pydantic_ai.exceptions import UnexpectedModelBehavior

        gen = self._make_gen()
        mock_agent = MagicMock()
        mock_agent.run_sync.side_effect = UnexpectedModelBehavior(
            "1 validation error for Analysis", body='{"bad": "data"}'
        )
        gen._agent = mock_agent

        art = gen.generate(self._make_context())
        # Content is now the raw body from the LLM (may be re-formatted by PydanticAI)
        assert "bad" in art.content
        assert "data" in art.content
        # Error metadata is set
        assert "generator_error" in art.metadata
        assert "Output validation failed:" in art.metadata["generator_error"]
        assert art.metadata["generator_error_kind"] == "validation"

    def test_generate_validation_error_stores_raw_body(self):
        from pydantic_ai.exceptions import UnexpectedModelBehavior

        gen = self._make_gen()
        mock_agent = MagicMock()
        raw_body = "Here is my analysis: the bug is in line 42"
        mock_agent.run_sync.side_effect = UnexpectedModelBehavior(
            "Exceeded maximum retries (0) for output validation", body=raw_body
        )
        gen._agent = mock_agent

        art = gen.generate(self._make_context())
        assert art.content == raw_body
        assert "generator_error" in art.metadata
        assert "tool call" in art.metadata["generator_error"]
        assert "Raw response preview:" in art.metadata["generator_error"]

    def test_generate_validation_error_without_body(self):
        from pydantic_ai.exceptions import UnexpectedModelBehavior

        gen = self._make_gen()
        mock_agent = MagicMock()
        mock_agent.run_sync.side_effect = UnexpectedModelBehavior(
            "1 validation error for Analysis"
        )
        gen._agent = mock_agent

        art = gen.generate(self._make_context())
        # Without a body, content falls back to str(e) which is just the message
        assert "1 validation error for Analysis" in art.content
        assert art.metadata["generator_error_kind"] == "validation"
        # No body means no "Raw response preview" in the error
        assert "Raw response preview:" not in art.metadata["generator_error"]

    def test_generate_json_error_returns_error_with_body(self):
        from pydantic_ai.exceptions import UnexpectedModelBehavior

        gen = self._make_gen()
        mock_agent = MagicMock()
        mock_agent.run_sync.side_effect = UnexpectedModelBehavior(
            "invalid json in response", body="not valid json {"
        )
        gen._agent = mock_agent

        art = gen.generate(self._make_context())
        assert art.content == "not valid json {"
        assert art.metadata["generator_error_kind"] == "validation"

    def test_generate_generic_error_returns_error_artifact(self):
        gen = self._make_gen()
        mock_agent = MagicMock()
        mock_agent.run_sync.side_effect = RuntimeError("network timeout")
        gen._agent = mock_agent

        art = gen.generate(self._make_context())
        assert art.content == "network timeout"
        assert art.metadata["generator_error"] == "Generation failed: network timeout"
        assert art.metadata["generator_error_kind"] == "infrastructure"

    def test_generate_error_still_increments_counter(self):
        gen = self._make_gen()
        mock_agent = MagicMock()
        mock_agent.run_sync.side_effect = RuntimeError("fail")
        gen._agent = mock_agent

        ctx = self._make_context()
        art1 = gen.generate(ctx)
        art2 = gen.generate(ctx)

        assert art1.attempt_number == 1
        assert art2.attempt_number == 2


# =========================================================================
# PydanticAIGenerator._format_validation_error()
# =========================================================================


class TestFormatValidationError:
    """Verify _format_validation_error uses error.message and error.body."""

    def _make_gen(self):
        from examples.swe_bench_ablation.generators import AnalysisGenerator

        return AnalysisGenerator(
            model="test",
            base_url="http://localhost:11434/v1",
            api_key="ollama",
        )

    def test_with_body_includes_preview_and_tool_hint(self):
        from pydantic_ai.exceptions import UnexpectedModelBehavior

        gen = self._make_gen()
        error = UnexpectedModelBehavior(
            "Exceeded maximum retries (0) for output validation",
            body="Here is my analysis...",
        )
        result = gen._format_validation_error(error)
        assert result.startswith("Output validation failed:")
        assert "tool call" in result
        assert "Raw response preview:" in result
        assert "Here is my analysis..." in result

    def test_without_body_returns_simple_message(self):
        from pydantic_ai.exceptions import UnexpectedModelBehavior

        gen = self._make_gen()
        error = UnexpectedModelBehavior("1 Validation error for Analysis")
        result = gen._format_validation_error(error)
        assert result.startswith("Output validation failed:")
        assert "1 Validation error for Analysis" in result
        assert "Raw response preview:" not in result

    def test_without_body_with_cause_chain(self):
        """When body is None, validation details are extracted from __cause__."""
        from pydantic_ai.exceptions import UnexpectedModelBehavior

        gen = self._make_gen()
        # Simulate the real chain: UnexpectedModelBehavior → cause
        cause = ValueError("field 'bug_type' is required")
        error = UnexpectedModelBehavior(
            "Exceeded maximum retries (0) for output validation"
        )
        error.__cause__ = cause
        result = gen._format_validation_error(error)
        assert "Validation details:" in result
        assert "bug_type" in result
        assert "tool" in result

    def test_without_body_with_tool_retry_cause(self):
        """ToolRetryError cause — extracts .tool_retry.content."""
        from pydantic_ai.exceptions import ToolRetryError, UnexpectedModelBehavior

        gen = self._make_gen()
        # Build a mock ToolRetryError with a tool_retry that has content
        retry_part = MagicMock()
        retry_part.content = [
            {"type": "missing", "loc": ["bug_type"], "msg": "Field required"}
        ]
        cause = ToolRetryError(retry_part)
        error = UnexpectedModelBehavior(
            "Exceeded maximum retries (0) for output validation"
        )
        error.__cause__ = cause
        result = gen._format_validation_error(error)
        assert "Validation details:" in result
        assert "bug_type" in result

    def test_body_truncated_at_500_chars(self):
        from pydantic_ai.exceptions import UnexpectedModelBehavior

        gen = self._make_gen()
        long_body = "x" * 1000
        error = UnexpectedModelBehavior("validation failed", body=long_body)
        result = gen._format_validation_error(error)
        # The preview should contain exactly 500 chars of body
        assert "x" * 500 in result
        assert "x" * 501 not in result

    def test_fallback_for_plain_exception(self):
        gen = self._make_gen()
        # Plain Exception has no .message or .body attributes;
        # method falls back to str(error) via getattr
        result = gen._format_validation_error(Exception("something unexpected"))
        assert result.startswith("Output validation failed:")
        assert "something unexpected" in result


# =========================================================================
# PydanticAIGenerator._get_system_prompt()
# =========================================================================


class TestGetSystemPrompt:
    """Verify _get_system_prompt selects the right source."""

    def _make_gen(self):
        from examples.swe_bench_ablation.generators import AnalysisGenerator

        return AnalysisGenerator(
            model="test",
            base_url="http://localhost:11434/v1",
            api_key="ollama",
        )

    def test_returns_template_role(self):
        from atomicguard.domain.prompts import PromptTemplate

        gen = self._make_gen()
        tmpl = PromptTemplate(role="Bug analyst", constraints="", task="")
        assert gen._get_system_prompt(tmpl) == "Bug analyst"

    def test_returns_default_when_no_template(self):
        gen = self._make_gen()
        assert gen._get_system_prompt(None) == "You are a helpful assistant."


# =========================================================================
# PatchGenerator – dependency extraction helpers
# =========================================================================


class TestDependencyExtraction:
    """Verify _get_analysis, _get_localization, _get_test_code."""

    def _make_gen(self):
        from examples.swe_bench_ablation.generators import PatchGenerator

        return PatchGenerator(
            model="test",
            base_url="http://localhost",
            api_key="test",
        )

    def _make_context(self, deps, artifact_map):
        """Build a Context whose repo returns artifacts from *artifact_map*.

        Args:
            deps: Tuple of (dep_id, artifact_id) pairs.
            artifact_map: Dict mapping artifact_id to content strings.
        """
        from atomicguard.domain.models import AmbientEnvironment, Context

        repo = MagicMock(spec=["get_artifact"])

        def _get(aid):
            content = artifact_map.get(aid)
            if content is None:
                return None
            art = MagicMock()
            art.content = content
            return art

        repo.get_artifact.side_effect = _get

        ambient = AmbientEnvironment(repository=repo, constraints="")
        return Context(ambient=ambient, specification="bug", dependency_artifacts=deps)

    # -- _get_analysis -------------------------------------------------

    def test_get_analysis_valid_returns_model(self):
        from examples.swe_bench_ablation.models import Analysis

        gen = self._make_gen()
        analysis_json = json.dumps(
            {
                "bug_type": "logic",
                "root_cause_hypothesis": "wrong branch",
                "files": ["f.py"],
                "fix_approach": "fix branch",
            }
        )
        ctx = self._make_context(
            deps=(("analysis", "a1"),),
            artifact_map={"a1": analysis_json},
        )
        result = gen._get_analysis(ctx)
        assert isinstance(result, Analysis)
        assert result.bug_type.value == "logic"

    def test_get_analysis_invalid_json_returns_none(self):
        gen = self._make_gen()
        ctx = self._make_context(
            deps=(("analysis", "a1"),),
            artifact_map={"a1": "not json"},
        )
        assert gen._get_analysis(ctx) is None

    def test_get_analysis_no_matching_dep_returns_none(self):
        gen = self._make_gen()
        ctx = self._make_context(
            deps=(("patch", "a1"),),
            artifact_map={"a1": "{}"},
        )
        assert gen._get_analysis(ctx) is None

    # -- _get_localization ---------------------------------------------

    def test_get_localization_valid_returns_model(self):
        from examples.swe_bench_ablation.models import Localization

        gen = self._make_gen()
        loc_json = json.dumps(
            {
                "files": ["src/main.py"],
                "functions": [{"name": "foo", "file": "src/main.py"}],
                "reasoning": "because",
            }
        )
        ctx = self._make_context(
            deps=(("localize", "l1"),),
            artifact_map={"l1": loc_json},
        )
        result = gen._get_localization(ctx)
        assert isinstance(result, Localization)
        assert result.files == ["src/main.py"]

    def test_get_localization_invalid_json_returns_none(self):
        gen = self._make_gen()
        ctx = self._make_context(
            deps=(("localize", "l1"),),
            artifact_map={"l1": "not valid"},
        )
        assert gen._get_localization(ctx) is None

    def test_get_localization_missing_artifact_returns_none(self):
        gen = self._make_gen()
        ctx = self._make_context(
            deps=(("localize", "l1"),),
            artifact_map={},  # get_artifact returns None
        )
        assert gen._get_localization(ctx) is None

    # -- _get_test_code ------------------------------------------------

    def test_get_test_code_valid_returns_content(self):
        gen = self._make_gen()
        ctx = self._make_context(
            deps=(("test", "t1"),),
            artifact_map={"t1": "def test_foo(): assert True"},
        )
        assert gen._get_test_code(ctx) == "def test_foo(): assert True"

    def test_get_test_code_empty_content_returns_none(self):
        gen = self._make_gen()
        ctx = self._make_context(
            deps=(("test", "t1"),),
            artifact_map={"t1": "   \n  "},
        )
        assert gen._get_test_code(ctx) is None

    def test_get_test_code_no_test_dep_returns_none(self):
        gen = self._make_gen()
        ctx = self._make_context(
            deps=(("analysis", "a1"),),
            artifact_map={"a1": "def test_foo(): assert True"},
        )
        assert gen._get_test_code(ctx) is None


# =========================================================================
# PatchGenerator._create_unified_diff() – edge cases
# =========================================================================


class TestCreateUnifiedDiffEdgeCases:
    """Verify _create_unified_diff handles edge cases."""

    def _make_gen(self):
        from examples.swe_bench_ablation.generators import PatchGenerator

        return PatchGenerator(
            model="test",
            base_url="http://localhost",
            api_key="test",
        )

    def test_file_not_found_skipped(self, tmp_path):
        from examples.swe_bench_ablation.models import SearchReplaceEdit

        gen = self._make_gen()
        edits = [SearchReplaceEdit(file="nonexistent.py", search="x", replace="y")]
        assert gen._create_unified_diff(edits, str(tmp_path)) == ""

    def test_search_string_not_found_skipped(self, tmp_path):
        from examples.swe_bench_ablation.models import SearchReplaceEdit

        (tmp_path / "real.py").write_text("print('hello')\n")
        gen = self._make_gen()
        edits = [SearchReplaceEdit(file="real.py", search="nonexistent", replace="y")]
        assert gen._create_unified_diff(edits, str(tmp_path)) == ""

    def test_all_edits_fail_returns_empty(self, tmp_path):
        from examples.swe_bench_ablation.models import SearchReplaceEdit

        (tmp_path / "a.py").write_text("x = 1\n")
        gen = self._make_gen()
        edits = [
            SearchReplaceEdit(file="a.py", search="not found", replace="y"),
            SearchReplaceEdit(file="missing.py", search="x", replace="y"),
        ]
        assert gen._create_unified_diff(edits, str(tmp_path)) == ""

    def test_multiple_edits_different_files(self, tmp_path):
        from examples.swe_bench_ablation.models import SearchReplaceEdit

        (tmp_path / "a.py").write_text("x = 1\n")
        (tmp_path / "b.py").write_text("y = 2\n")
        gen = self._make_gen()
        edits = [
            SearchReplaceEdit(file="a.py", search="x = 1", replace="x = 10"),
            SearchReplaceEdit(file="b.py", search="y = 2", replace="y = 20"),
        ]
        diff = gen._create_unified_diff(edits, str(tmp_path))
        assert "a/a.py" in diff
        assert "a/b.py" in diff


# =========================================================================
# PatchGenerator._read_file() – edge cases
# =========================================================================


class TestReadFileEdgeCases:
    """Verify _read_file handles edge cases."""

    def _make_gen(self, **kwargs):
        from examples.swe_bench_ablation.generators import PatchGenerator

        defaults = {"model": "test", "base_url": "http://localhost", "api_key": "test"}
        defaults.update(kwargs)
        return PatchGenerator(**defaults)

    def test_long_file_not_truncated(self, tmp_path):
        """Files are returned in full, not truncated."""
        content = "\n".join(f"line_{i}" for i in range(500))
        (tmp_path / "big.py").write_text(content)
        gen = self._make_gen()  # No max_file_lines by default

        result, error = gen._read_file(str(tmp_path), "big.py")
        assert result is not None
        assert error is None
        assert "TRUNCATED" not in result
        assert "line_250" in result  # Middle content present

    def test_file_exceeds_limit_returns_error(self, tmp_path):
        """When max_file_lines set, exceeding it returns error."""
        content = "\n".join(f"line_{i}" for i in range(200))
        (tmp_path / "big.py").write_text(content)
        gen = self._make_gen(max_file_lines=100)

        result, error = gen._read_file(str(tmp_path), "big.py")
        assert result is None
        assert error is not None
        assert "exceeding limit" in error

    def test_file_not_found_returns_none(self, tmp_path):
        gen = self._make_gen()
        result, error = gen._read_file(str(tmp_path), "nonexistent.py")
        assert result is None
        assert error is None

    def test_read_error_returns_none(self, tmp_path):
        (tmp_path / "bad.py").write_text("x = 1")
        gen = self._make_gen()
        with patch.object(Path, "read_text", side_effect=PermissionError("denied")):
            result, error = gen._read_file(str(tmp_path), "bad.py")
            assert result is None
            assert error is None


# =========================================================================
# PydanticAIGenerator._process_output() – default implementation
# =========================================================================


class TestDefaultProcessOutput:
    """Verify the default _process_output serialises models to JSON."""

    def _make_context(self):
        from atomicguard.domain.models import AmbientEnvironment, Context

        ambient = AmbientEnvironment(repository=MagicMock(spec=[]), constraints="")
        return Context(ambient=ambient, specification="bug")

    def test_analysis_process_output_returns_json(self):
        from examples.swe_bench_ablation.generators import AnalysisGenerator
        from examples.swe_bench_ablation.models import Analysis

        gen = AnalysisGenerator(
            model="test",
            base_url="http://localhost:11434/v1",
            api_key="ollama",
        )
        output = Analysis(
            bug_type="logic",
            root_cause_hypothesis="wrong branch",
            files=["f.py"],
            fix_approach="fix it",
        )
        result = gen._process_output(output, self._make_context())
        data = json.loads(result)
        assert data["bug_type"] == "logic"
        assert data["files"] == ["f.py"]

    def test_localization_process_output_returns_json(self):
        from examples.swe_bench_ablation.generators import LocalizationGenerator
        from examples.swe_bench_ablation.models import Localization

        gen = LocalizationGenerator(
            model="test",
            base_url="http://localhost:11434/v1",
            api_key="ollama",
        )
        output = Localization(
            files=["src/main.py"],
            reasoning="because",
        )
        result = gen._process_output(output, self._make_context())
        data = json.loads(result)
        assert data["files"] == ["src/main.py"]
        assert data["reasoning"] == "because"

    def test_process_output_preserves_all_fields(self):
        from examples.swe_bench_ablation.generators import AnalysisGenerator
        from examples.swe_bench_ablation.models import Analysis

        gen = AnalysisGenerator(
            model="test",
            base_url="http://localhost:11434/v1",
            api_key="ollama",
        )
        output = Analysis(
            bug_type="logic",
            root_cause_hypothesis="wrong branch",
            affected_components=["mod.cls"],
            files=["f.py"],
            fix_approach="fix it",
            confidence="high",
        )
        result = gen._process_output(output, self._make_context())
        data = json.loads(result)
        assert data["affected_components"] == ["mod.cls"]
        assert data["confidence"] == "high"
        assert data["bug_type"] == "logic"
        assert data["root_cause_hypothesis"] == "wrong branch"
        assert data["fix_approach"] == "fix it"


# =========================================================================
# ActionPair – generator_error guard skip
# =========================================================================


class TestActionPairGeneratorErrorSkip:
    """Verify ActionPair skips the domain guard when generator_error is set."""

    def _make_artifact(self, metadata=None):
        from types import MappingProxyType

        from atomicguard.domain.models import (
            Artifact,
            ArtifactStatus,
            ContextSnapshot,
        )

        return Artifact(
            artifact_id="art-1",
            workflow_id="wf-1",
            content="some content",
            previous_attempt_id=None,
            parent_action_pair_id=None,
            action_pair_id="ap-1",
            created_at="2025-01-01T00:00:00",
            attempt_number=1,
            status=ArtifactStatus.PENDING,
            guard_result=None,
            context=ContextSnapshot(
                workflow_id="wf-1",
                specification="spec",
                constraints="",
                feedback_history=(),
            ),
            metadata=MappingProxyType(metadata or {}),
        )

    def test_action_pair_skips_guard_on_generator_error(self):
        from atomicguard.application.action_pair import ActionPair
        from atomicguard.domain.prompts import PromptTemplate

        error_msg = "Output validation failed: bad format"
        artifact = self._make_artifact(
            metadata={
                "generator_error": error_msg,
                "generator_error_kind": "validation",
            }
        )

        mock_generator = MagicMock()
        mock_generator.generate.return_value = artifact

        mock_guard = MagicMock()

        template = PromptTemplate(role="", constraints="", task="")
        ap = ActionPair(mock_generator, mock_guard, template)

        ctx = MagicMock()
        _, result = ap.execute(ctx, action_pair_id="ap-1", workflow_id="wf-1")

        # Guard should NOT have been called
        mock_guard.validate.assert_not_called()

        # Result should be a synthetic rejection with the generator error
        assert result.passed is False
        assert result.feedback == error_msg
        assert result.guard_name == "GeneratorValidation"

    def test_action_pair_calls_guard_on_normal_artifact(self):
        from atomicguard.application.action_pair import ActionPair
        from atomicguard.domain.models import GuardResult
        from atomicguard.domain.prompts import PromptTemplate

        artifact = self._make_artifact(metadata={})

        mock_generator = MagicMock()
        mock_generator.generate.return_value = artifact

        mock_guard = MagicMock()
        mock_guard.validate.return_value = GuardResult(
            passed=True, feedback="", guard_name="TestGuard"
        )

        template = PromptTemplate(role="", constraints="", task="")
        ap = ActionPair(mock_generator, mock_guard, template)

        ctx = MagicMock()
        _, result = ap.execute(ctx, action_pair_id="ap-1", workflow_id="wf-1")

        # Guard SHOULD have been called
        mock_guard.validate.assert_called_once()
        assert result.passed is True

    def test_action_pair_generator_error_feedback_flows_to_retry(self):
        """End-to-end: the PydanticAI error reaches feedback_history."""
        from atomicguard.application.action_pair import ActionPair
        from atomicguard.domain.prompts import PromptTemplate

        error_msg = (
            "Output validation failed: Exceeded maximum retries (0)\n"
            "Your response could not be parsed as a tool call. "
            "You must use the provided tool to structure your output.\n"
            "Raw response preview: Here is my analysis..."
        )
        artifact = self._make_artifact(
            metadata={
                "generator_error": error_msg,
                "generator_error_kind": "validation",
            }
        )

        mock_generator = MagicMock()
        mock_generator.generate.return_value = artifact

        mock_guard = MagicMock()

        template = PromptTemplate(role="", constraints="", task="")
        ap = ActionPair(mock_generator, mock_guard, template)

        ctx = MagicMock()
        art, result = ap.execute(ctx, action_pair_id="ap-1", workflow_id="wf-1")

        # The feedback that would go into feedback_history for retry
        assert not result.passed
        assert "tool call" in result.feedback
        assert "Raw response preview:" in result.feedback
        # Guard was bypassed
        mock_guard.validate.assert_not_called()


# =========================================================================
# TDD Verification Guards
# =========================================================================


class TestQuickTestRunner:
    """Unit tests for QuickTestRunner without Docker."""

    @pytest.fixture()
    def mock_instance(self):
        from examples.swe_bench_pro.dataset import SWEBenchProInstance

        return SWEBenchProInstance(
            instance_id="django__django-1234",
            repo="django/django",
            base_commit="abc123",
            problem_statement="Bug in ORM",
            patch="",
            test_patch="",
            repo_language="python",
        )

    def test_get_docker_image(self, mock_instance):
        from examples.swe_bench_pro.guards import QuickTestRunner

        runner = QuickTestRunner(instance=mock_instance)
        # Official SWE-bench Pro format: {repo_base}.{repo_name}-{uid}
        assert runner._get_docker_image() == "jefzda/sweap-images:django.django-django__django-1234"

    def test_get_test_filename_python(self, mock_instance):
        from examples.swe_bench_pro.guards import QuickTestRunner

        runner = QuickTestRunner(instance=mock_instance)
        assert runner._get_test_filename() == "test_atomicguard.py"

    def test_get_test_filename_go(self):
        from examples.swe_bench_pro.dataset import SWEBenchProInstance
        from examples.swe_bench_pro.guards import QuickTestRunner

        go_instance = SWEBenchProInstance(
            instance_id="golang__go-1234",
            repo="golang/go",
            base_commit="abc123",
            problem_statement="Bug",
            patch="",
            test_patch="",
            repo_language="go",
        )
        runner = QuickTestRunner(instance=go_instance)
        assert runner._get_test_filename() == "atomicguard_test.go"

    def test_create_test_script_without_patch(self, mock_instance):
        from examples.swe_bench_pro.guards import QuickTestRunner

        runner = QuickTestRunner(instance=mock_instance)
        script = runner._create_test_script("def test_foo(): pass", patch_diff=None)

        assert "git checkout -f abc123" in script
        assert "git clean -fdx" in script
        assert "def test_foo(): pass" in script
        assert "git apply" not in script

    def test_create_test_script_with_patch(self, mock_instance):
        from examples.swe_bench_pro.guards import QuickTestRunner

        runner = QuickTestRunner(instance=mock_instance)
        script = runner._create_test_script(
            "def test_foo(): pass",
            patch_diff="--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-old\n+new",
        )

        assert "git apply" in script
        assert "--- a/foo.py" in script


class TestTestRedGuard:
    """Unit tests for TestRedGuard with mocked Docker."""

    @pytest.fixture()
    def mock_instance(self):
        from examples.swe_bench_pro.dataset import SWEBenchProInstance

        return SWEBenchProInstance(
            instance_id="django__django-1234",
            repo="django/django",
            base_commit="abc123",
            problem_statement="Bug in ORM",
            patch="",
            test_patch="",
            repo_language="python",
        )

    @pytest.fixture()
    def make_artifact(self):
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

    def test_empty_test_fails(self, mock_instance, make_artifact):
        from examples.swe_bench_pro.guards import TestRedGuard

        guard = TestRedGuard(instance=mock_instance)
        artifact = make_artifact("")
        result = guard.validate(artifact)

        assert result.passed is False
        assert "empty" in result.feedback.lower()

    def test_fatal_when_docker_unavailable(self, mock_instance, make_artifact):
        from examples.swe_bench_pro.guards import TestRedGuard

        guard = TestRedGuard(instance=mock_instance)
        # Mock ensure_image_available to return unavailable
        guard._runner.ensure_image_available = lambda: (False, "Docker image not available")

        artifact = make_artifact("def test_foo(): assert False")
        result = guard.validate(artifact)

        # Should fail with fatal=True when Docker unavailable (⊥fatal)
        assert result.passed is False
        assert result.fatal is True
        assert "FATAL" in result.feedback
        assert "Docker image not available" in result.feedback

    def test_fails_when_test_passes_on_buggy_code(self, mock_instance, make_artifact):
        from examples.swe_bench_pro.guards import TestRedGuard
        from examples.swe_bench_pro.guards.quick_test_runner import QuickTestResult

        guard = TestRedGuard(instance=mock_instance)
        guard._runner.ensure_image_available = lambda: (True, "Image available")
        guard._runner.run_test = lambda *_args, **_kwargs: QuickTestResult(
            status="PASSED",
            output="test passed",
            exit_code=0,
            duration_seconds=1.0,
        )

        artifact = make_artifact("def test_foo(): assert True")
        result = guard.validate(artifact)

        assert result.passed is False
        assert "PASSES on buggy code" in result.feedback

    def test_passes_when_test_fails_on_buggy_code(self, mock_instance, make_artifact):
        from examples.swe_bench_pro.guards import TestRedGuard
        from examples.swe_bench_pro.guards.quick_test_runner import QuickTestResult

        guard = TestRedGuard(instance=mock_instance)
        guard._runner.ensure_image_available = lambda: (True, "Image available")
        guard._runner.run_test = lambda *_args, **_kwargs: QuickTestResult(
            status="FAILED",
            output="AssertionError: expected 1 got 0",
            exit_code=1,
            duration_seconds=1.0,
        )

        artifact = make_artifact("def test_foo(): assert bug()")
        result = guard.validate(artifact)

        assert result.passed is True
        assert "correctly FAILS" in result.feedback


class TestTestGreenGuard:
    """Unit tests for TestGreenGuard with mocked Docker."""

    @pytest.fixture()
    def mock_instance(self):
        from examples.swe_bench_pro.dataset import SWEBenchProInstance

        return SWEBenchProInstance(
            instance_id="django__django-1234",
            repo="django/django",
            base_commit="abc123",
            problem_statement="Bug in ORM",
            patch="",
            test_patch="",
            repo_language="python",
        )

    @pytest.fixture()
    def make_artifact(self):
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

    def test_fails_without_test_dependency(self, mock_instance, make_artifact):
        from examples.swe_bench_pro.guards import TestGreenGuard

        guard = TestGreenGuard(instance=mock_instance)
        patch_artifact = make_artifact('{"patch": "diff --git ..."}')
        result = guard.validate(patch_artifact)

        assert result.passed is False
        assert "No test artifact" in result.feedback

    def test_fatal_when_docker_unavailable(self, mock_instance, make_artifact):
        from examples.swe_bench_pro.guards import TestGreenGuard

        guard = TestGreenGuard(instance=mock_instance)
        guard._runner.ensure_image_available = lambda: (False, "Docker image not available")

        patch_artifact = make_artifact('{"patch": "diff --git a/foo.py ..."}')
        test_artifact = make_artifact("def test_foo(): assert fixed()")

        result = guard.validate(patch_artifact, ap_gen_test=test_artifact)

        # Should fail with fatal=True when Docker unavailable (⊥fatal)
        assert result.passed is False
        assert result.fatal is True
        assert "FATAL" in result.feedback
        assert "Docker image not available" in result.feedback

    def test_passes_when_test_passes_after_patch(self, mock_instance, make_artifact):
        from examples.swe_bench_pro.guards import TestGreenGuard
        from examples.swe_bench_pro.guards.quick_test_runner import QuickTestResult

        guard = TestGreenGuard(instance=mock_instance)
        guard._runner.ensure_image_available = lambda: (True, "Image available")
        guard._runner.run_test = lambda *_args, **_kwargs: QuickTestResult(
            status="PASSED",
            output="test passed",
            exit_code=0,
            duration_seconds=1.0,
        )

        patch_artifact = make_artifact('{"patch": "diff --git a/foo.py ..."}')
        test_artifact = make_artifact("def test_foo(): assert fixed()")

        result = guard.validate(patch_artifact, ap_gen_test=test_artifact)

        assert result.passed is True
        assert "PASSES after patch" in result.feedback

    def test_fails_when_test_still_fails_after_patch(self, mock_instance, make_artifact):
        from examples.swe_bench_pro.guards import TestGreenGuard
        from examples.swe_bench_pro.guards.quick_test_runner import QuickTestResult

        guard = TestGreenGuard(instance=mock_instance)
        guard._runner.ensure_image_available = lambda: (True, "Image available")
        guard._runner.run_test = lambda *_args, **_kwargs: QuickTestResult(
            status="FAILED",
            output="AssertionError",
            exit_code=1,
            duration_seconds=1.0,
        )

        patch_artifact = make_artifact('{"patch": "diff --git a/foo.py ..."}')
        test_artifact = make_artifact("def test_foo(): assert fixed()")

        result = guard.validate(patch_artifact, ap_gen_test=test_artifact)

        assert result.passed is False
        assert "still FAILS" in result.feedback


class TestTDDVerifiedWorkflowConfig:
    """Tests for the TDD-verified workflow configuration with CompositeGuards."""

    def test_workflow_loads(self):
        from examples.swe_bench_pro.experiment_runner import load_workflow_config

        config = load_workflow_config("05_s1_tdd_verified")
        assert config["name"] == "S1 TDD Verified"
        assert config["rmax"] == 6

    def test_workflow_has_three_action_pairs(self):
        """New workflow has 3 action pairs instead of 6 (using CompositeGuards)."""
        from examples.swe_bench_pro.experiment_runner import load_workflow_config

        config = load_workflow_config("05_s1_tdd_verified")
        expected = {
            "ap_analysis",
            "ap_gen_test",
            "ap_gen_patch",
        }
        assert set(config["action_pairs"].keys()) == expected

    def test_gen_test_uses_composite_guard(self):
        """ap_gen_test should use composite guard with test_syntax and test_red."""
        from examples.swe_bench_pro.experiment_runner import load_workflow_config

        config = load_workflow_config("05_s1_tdd_verified")
        ap = config["action_pairs"]["ap_gen_test"]
        assert ap["generator"] == "TestGenerator"
        assert ap["guard"] == "composite"
        assert ap["guards"] == ["test_syntax", "test_red"]

    def test_gen_patch_uses_composite_guard(self):
        """ap_gen_patch should use composite guard with patch, lint, test_green, full_eval."""
        from examples.swe_bench_pro.experiment_runner import load_workflow_config

        config = load_workflow_config("05_s1_tdd_verified")
        ap = config["action_pairs"]["ap_gen_patch"]
        assert ap["generator"] == "PatchGenerator"
        assert ap["guard"] == "composite"
        assert ap["guards"] == ["patch", "lint", "test_green", "full_eval"]

    def test_arm_map_includes_verified(self):
        from examples.swe_bench_pro.demo import _ARM_MAP

        assert "s1_tdd_verified" in _ARM_MAP
        assert _ARM_MAP["s1_tdd_verified"] == "05_s1_tdd_verified"


class TestCompositeGuardWorkflow:
    """Tests for CompositeGuard support in build_workflow()."""

    def test_build_workflow_with_composite_guard(self, tmp_path):
        """build_workflow should correctly parse composite guard config."""
        from examples.swe_bench_pro.dataset import SWEBenchProInstance
        from examples.swe_bench_pro.experiment_runner import (
            build_workflow,
            load_prompts,
            load_workflow_config,
        )

        from atomicguard import CompositeGuard
        from atomicguard.infrastructure.persistence.filesystem import (
            FilesystemArtifactDAG,
        )

        config = load_workflow_config("05_s1_tdd_verified")
        prompts = load_prompts()
        lang = get_language_config("python")
        dag = FilesystemArtifactDAG(str(tmp_path / "dag"))

        mock_instance = SWEBenchProInstance(
            instance_id="test__test__1",
            repo="test/test",
            base_commit="abc123",
            problem_statement="Test bug",
            patch="diff",
            test_patch="diff",
            repo_language="python",
        )

        wf = build_workflow(
            config=config,
            prompts=prompts,
            lang_config=lang,
            model="test",
            base_url="http://localhost",
            artifact_dag=dag,
            repo_root=str(tmp_path),
            api_key="test",
            instance=mock_instance,
        )

        # Find ap_gen_test step and verify it has a CompositeGuard
        gen_test_step = None
        gen_patch_step = None
        for step in wf._steps:
            if step.guard_id == "ap_gen_test":
                gen_test_step = step
            elif step.guard_id == "ap_gen_patch":
                gen_patch_step = step

        assert gen_test_step is not None, "ap_gen_test step not found"
        assert isinstance(gen_test_step.action_pair.guard, CompositeGuard)
        assert len(gen_test_step.action_pair.guard.guards) == 2

        assert gen_patch_step is not None, "ap_gen_patch step not found"
        assert isinstance(gen_patch_step.action_pair.guard, CompositeGuard)
        assert len(gen_patch_step.action_pair.guard.guards) == 4

    def test_composite_guard_requires_guards_array(self, tmp_path):
        """build_workflow should raise if composite guard has no guards array."""
        from examples.swe_bench_pro.experiment_runner import (
            build_workflow,
            load_prompts,
        )

        from atomicguard.infrastructure.persistence.filesystem import (
            FilesystemArtifactDAG,
        )

        config = {
            "rmax": 3,
            "action_pairs": {
                "ap_test": {
                    "generator": "AnalysisGenerator",
                    "guard": "composite",
                    # Missing "guards" array
                }
            },
        }
        prompts = load_prompts()
        lang = get_language_config("python")
        dag = FilesystemArtifactDAG(str(tmp_path / "dag"))

        with pytest.raises(ValueError, match="requires 'guards' array"):
            build_workflow(
                config=config,
                prompts=prompts,
                lang_config=lang,
                model="test",
                base_url="http://localhost",
                artifact_dag=dag,
                api_key="test",
            )

    def test_composite_guard_unknown_subguard_raises(self, tmp_path):
        """build_workflow should raise if composite guard has unknown sub-guard."""
        from examples.swe_bench_pro.experiment_runner import (
            build_workflow,
            load_prompts,
        )

        from atomicguard.infrastructure.persistence.filesystem import (
            FilesystemArtifactDAG,
        )

        config = {
            "rmax": 3,
            "action_pairs": {
                "ap_test": {
                    "generator": "AnalysisGenerator",
                    "guard": "composite",
                    "guards": ["analysis", "nonexistent_guard"],
                }
            },
        }
        prompts = load_prompts()
        lang = get_language_config("python")
        dag = FilesystemArtifactDAG(str(tmp_path / "dag"))

        with pytest.raises(ValueError, match="Unknown guard"):
            build_workflow(
                config=config,
                prompts=prompts,
                lang_config=lang,
                model="test",
                base_url="http://localhost",
                artifact_dag=dag,
                api_key="test",
            )
