# SWE-Bench Pro Example: Implementation Plan

**Goal:** Create a working `examples/swe_bench_pro/` that runs AtomicGuard workflows
against SWE-Bench Pro instances and evaluates results using SWE-Bench Pro's own
evaluation infrastructure.

**Constraint:** Keep `examples/swe_bench_ablation/` intact. Reuse code from it where
possible via imports, not copies.

---

## SWE-Bench Pro vs Current Setup

| Aspect | SWE-Bench ablation (current) | SWE-Bench Pro (new) |
|--------|------------------------------|---------------------|
| Dataset | `AmazonScience/SWE-PolyBench` | `ScaleAI/SWE-bench_Pro` (731 public instances) |
| Languages | Python only | Python, Go, JavaScript, TypeScript |
| Dockerfile | Per-instance in dataset | Pre-built on DockerHub (`jefzda/sweap-images:...`) |
| Test command | Not in dataset | Managed via `run_scripts/` in eval repo |
| Eval harness | `swebench.harness.run_evaluation` (broken) | `swe_bench_pro_eval.py` from `scaleapi/SWE-bench_Pro-os` |
| Patch format | JSONL: `{instance_id, model_patch, model_name_or_path}` | JSON array: `[{instance_id, patch, prefix}]` |
| Extra fields | `task_category`, `dockerfile` | `requirements`, `interface`, `repo_language` |

---

## What Can Be Shared

**Import directly (no changes):**
- `AnalysisGenerator`, `LocalizationGenerator` — language-agnostic
- `AnalysisGuard`, `LocalizationGuard` — language-agnostic
- `PatchGuard` — already skips non-`.py` files in syntax check
- `analysis.py` — statistics and visualization operate on `ArmResult`
- `models.py` — Pydantic schemas (`Analysis`, `Patch`, `Localization`, etc.)
- `ArmResult` dataclass
- `EvalResult`, `write_eval_logs`, `format_instance_log` — per-instance logging
- `load_workflow_config`, `load_prompts`, `_topological_sort` — workflow construction
- Workflow JSON configs (`02_singleshot.json`, `03_s1_direct.json`, `04_s1_tdd.json`)

**Needs language-aware subclasses:**
- `PatchGenerator` — hardcodes `` ```python `` and "VALID PYTHON" in prompts
- `TestGenerator` — hardcodes pytest-specific instructions
- `TestSyntaxGuard` — uses `ast.parse()`, non-functional for Go/JS/TS

**Fully new:**
- Dataset loader (different schema)
- Evaluation module (different harness, different prediction format)
- Language configuration registry
- CLI entry point

---

## Directory Structure

```
examples/swe_bench_pro/
    __init__.py
    dataset.py              # SWE-Bench Pro dataset loader
    language.py             # Language config registry
    evaluation.py           # Wraps SWE-Bench Pro eval harness
    experiment_runner.py    # Orchestrates workflow runs
    demo.py                 # Click CLI
    prompts.json            # Language-adaptive prompt templates
    generators/
        __init__.py
        multilang_patch.py  # Language-aware PatchGenerator
        multilang_test.py   # Language-aware TestGenerator
    guards/
        __init__.py
        multilang_test_syntax.py  # Multi-language syntax check
```

---

## File-by-File Plan

### 1. `language.py` — Language Configuration Registry

Foundation module. No external dependencies.

```python
@dataclass(frozen=True)
class LanguageConfig:
    name: str                  # "python", "go", "javascript", "typescript"
    code_block_tag: str        # Markdown code fence language
    file_extensions: tuple[str, ...]
    test_framework: str        # "pytest", "go test", "jest"
    valid_code_label: str      # "VALID PYTHON", "VALID GO", etc.
    test_function_pattern: str # Regex for test function detection
    syntax_check_fn: Callable[[str], tuple[bool, str]] | None

LANGUAGE_CONFIGS: dict[str, LanguageConfig] = { ... }

def get_language_config(language: str) -> LanguageConfig: ...
```

Python gets `ast.parse` for syntax checking. Go/JS/TS get `None` (MVP — future
work can add `go vet`, `node --check`, `tsc --noEmit`).

### 2. `dataset.py` — Dataset Loader

```python
@dataclass(frozen=True)
class SWEBenchProInstance:
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    patch: str              # Gold patch
    test_patch: str
    fail_to_pass: list[str]
    pass_to_pass: list[str]
    repo_language: str      # "python", "go", "javascript", "typescript"
    requirements: str
    interface: str

def load_swe_bench_pro(
    split: str = "test",
    language: str | None = None,   # None = all languages
    max_instances: int | None = None,
) -> list[SWEBenchProInstance]: ...
```

Imports `_parse_test_list` from `swe_bench_ablation.dataset` for F2P/P2P parsing.

Key differences from ablation's `SWEInstance`: no `task_category`, no `dockerfile`;
has `repo_language`, `requirements`, `interface`.

### 3. `generators/multilang_patch.py` — Language-Aware Patch Generator

Subclasses `PatchGenerator` from ablation. Overrides `_build_prompt` to replace:
- `` ```python `` → `` ```{lang_config.code_block_tag} ``
- `"VALID PYTHON"` → `lang_config.valid_code_label`

```python
class MultiLangPatchGenerator(PatchGenerator):
    def __init__(self, language_config: LanguageConfig, **kwargs):
        super().__init__(**kwargs)
        self._lang = language_config
```

### 4. `generators/multilang_test.py` — Language-Aware Test Generator

Subclasses `TestGenerator`. Overrides `_build_prompt` to reference the correct
test framework (pytest vs `go test` vs Jest/Mocha).

### 5. `guards/multilang_test_syntax.py` — Multi-Language Syntax Check

Implements `GuardInterface`. For Python: `ast.parse()`. For others: regex check
that test patterns exist (`func Test\w+` for Go, `describe\(|it\(|test\(` for
JS/TS). Returns `passed=True` on heuristic match.

### 6. `prompts.json` — Language-Adaptive Prompts

Same structure as ablation's `prompts.json` but replaces Python-specific language
with generic references. Language-specific details are injected by the generator's
`_build_prompt`, not the templates.

### 7. `evaluation.py` — Wrap SWE-Bench Pro Harness

```python
def ensure_eval_repo(cache_dir: str = "~/.cache/swe_bench_pro") -> Path:
    """Clone or update scaleapi/SWE-bench_Pro-os. Pin to a known commit."""

def prepare_predictions(
    results: list[ArmResult], output_dir: str,
) -> dict[str, Path]:
    """Convert ArmResults to SWE-Bench Pro JSON format.
    Format: [{"instance_id": "...", "patch": "...", "prefix": "..."}]
    """

def run_evaluation(
    predictions_path: str | Path,
    eval_repo_path: str | Path,
    mode: str = "local",       # "local" (Docker) or "modal" (cloud)
    max_workers: int = 4,
    timeout: int = 7200,
    block_network: bool = True,
) -> dict[str, object]:
    """Invoke swe_bench_pro_eval.py via subprocess."""

def load_evaluation_results(
    results_dir: str, run_id: str = "swe_bench_pro",
    write_logs: bool = True,
) -> dict[str, bool]:
    """Parse eval_results.json, write per-instance logs via ablation's write_eval_logs."""
```

**Strategy: wrap their script, don't reimplement Docker logic.** Their pre-built
images + `run_scripts/` + `parser.py` represent significant engineering. Subprocess
invocation gives exact evaluation compatibility.

### 8. `experiment_runner.py` — Workflow Orchestration

```python
class SWEBenchProRunner:
    def __init__(self, model, base_url, api_key, output_dir, clone_dir): ...

    def _get_registries(self, lang_config: LanguageConfig) -> tuple[dict, dict]:
        """Build language-aware generator/guard registries."""

    def build_workflow(self, config, prompts, lang_config, ...):
        """Like ablation's build_workflow but with custom registries."""

    def run_instance(self, instance: SWEBenchProInstance, arm: str) -> ArmResult:
        """Inspect instance.repo_language, pick registries, run workflow."""

    def run_all(self, arms, split, language, max_instances, resume_from):
        """Same pattern as ablation: incremental JSONL persistence + resume."""
```

Key difference: `run_instance` inspects `instance.repo_language` and builds
registries dynamically. For Python instances, uses the standard ablation
generators/guards. For Go/JS/TS, uses the multi-language subclasses.

### 9. `demo.py` — CLI

```
python -m examples.swe_bench_pro.demo experiment \
    --model "..." --arms singleshot,s1_direct,s1_tdd \
    --language python --max-instances 10

python -m examples.swe_bench_pro.demo evaluate \
    --predictions-dir output/swe_bench_pro/predictions \
    --eval-mode local --max-workers 4

python -m examples.swe_bench_pro.demo visualize \
    --results output/swe_bench_pro/results.jsonl \
    --resolved output/swe_bench_pro/eval_results.json
```

Commands: `run`, `experiment`, `evaluate`, `visualize`, `list_instances`.

Loads workflow configs from ablation directory:
```python
ABLATION_DIR = Path(__file__).parent.parent / "swe_bench_ablation"
```

---

## Implementation Order

| Phase | Files | Dependencies |
|-------|-------|-------------|
| 1 | `language.py` | None |
| 2 | `dataset.py` | `_parse_test_list` from ablation |
| 3 | `generators/`, `guards/` | `language.py` + ablation generators/guards |
| 4 | `prompts.json` | None (standalone JSON) |
| 5 | `experiment_runner.py` | Phases 1-4 + ablation's workflow builder |
| 6 | `evaluation.py` | ablation's `EvalResult` + `write_eval_logs` |
| 7 | `demo.py` | Phases 5-6 |
| 8 | `__init__.py` | All modules |

Phases 5 and 6 can be developed in parallel.

---

## MVP Scope

**In scope:**
- Dataset loading with language filtering
- Language-aware generators/guards (Python full, Go/JS/TS heuristic)
- Workflow execution producing `ArmResult` JSONL
- Prediction export in SWE-Bench Pro JSON format
- Evaluation via their harness (local Docker mode)
- Per-instance evaluation logs
- Visualization reusing ablation's analysis module

**Out of scope for MVP:**
- Real syntax validators for Go/JS/TS (invoke compilers)
- Modal (cloud) evaluation
- Language-stratified analysis charts
- Docker container pooling / parallel evaluation tuning
- Eval repo version pinning UI

---

## Risks

1. **Cross-package imports** — `examples.swe_bench_ablation` must be importable when
   running `python -m examples.swe_bench_pro.demo`. Works because project root is on
   `sys.path` via `-m` invocation.

2. **Ablation `build_workflow` coupling** — uses hardcoded registries. The new runner
   needs its own `build_workflow` that accepts registries as parameters.

3. **Generator constructor mismatch** — `MultiLangPatchGenerator` takes `language_config`
   which standard workflow JSON doesn't reference. The runner must inject this when
   constructing generators.

4. **SWE-Bench Pro Docker images** — must exist on DockerHub. Missing images will
   produce evaluation errors. The eval module should handle partial results gracefully.

5. **Eval repo stability** — `scaleapi/SWE-bench_Pro-os` may change. `ensure_eval_repo`
   should pin to a specific commit hash.
