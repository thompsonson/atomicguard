# TDD Fine-Tuning: Implementation Plan

**Related:** `EXPERIMENT.md` (research design)
**Status:** Planning Phase

---

## Directory Structure

```
examples/advanced/tdd_finetuning/
├── EXPERIMENT.md                      # Research design (this experiment)
├── IMPLEMENTATION_PLAN.md             # This file (technical plan)
├── README.md                          # User-facing documentation
├── demo.py                            # CLI interface
├── workflow.json                      # Workflow definition
├── prompts.json                       # Generator prompts
├── tasks/                             # Task specifications
│   ├── stack.json
│   ├── queue.json
│   ├── calculator.json
│   ├── lru_cache.json
│   ├── template.json
│   └── password.json
├── generators/
│   ├── __init__.py
│   ├── test_generator.py             # g_test: spec → pytest tests
│   └── impl_generator.py             # g_impl: tests → implementation
├── guards/
│   ├── __init__.py
│   ├── syntax_guard.py               # Python AST validation
│   └── test_guard.py                 # Execute tests, verify pass
├── benchmarks/
│   ├── __init__.py
│   ├── runner.py                     # Run 50 trials per task
│   ├── metrics.py                    # Compute success rates
│   └── analysis.py                   # Statistical tests
├── training/
│   ├── __init__.py
│   ├── extractor.py                  # Extract training data from runs
│   ├── formatter.py                  # Format for Unsloth
│   └── finetune.py                   # LoRA fine-tuning script
└── output/                           # Generated (gitignored)
    ├── baseline_results/
    │   └── qwen2.5-coder_7b.json
    ├── training_data/
    │   └── test_generation.jsonl
    ├── models/
    │   └── test_agent_v1_ft/
    ├── finetuned_results/
    │   └── test_agent_v1_ft.json
    └── reports/
        └── comparison_report.md
```

---

## Implementation Phases

### Phase 0: Setup (Week 1, Day 1-2)

#### Task 0.1: Create Package Structure
```bash
cd examples/advanced/
mkdir -p tdd_finetuning/{tasks,generators,guards,benchmarks,training}
touch tdd_finetuning/{__init__.py,demo.py,workflow.json,prompts.json}
touch tdd_finetuning/generators/__init__.py
touch tdd_finetuning/guards/__init__.py
touch tdd_finetuning/benchmarks/__init__.py
touch tdd_finetuning/training/__init__.py
```

#### Task 0.2: Define Task Specifications
Create 6 JSON files in `tasks/` directory based on paper Section 4.3.

**Template** (`tasks/stack.json`):
```json
{
  "name": "Stack",
  "difficulty": "easy",
  "description": "Implement a stack data structure with standard operations",
  "signature": "class Stack",
  "requirements": [
    "push(item): Add item to top of stack",
    "pop(): Remove and return top item, raise exception if empty",
    "peek(): Return top item without removing, raise exception if empty",
    "is_empty(): Return True if stack is empty",
    "size(): Return number of items in stack"
  ],
  "edge_cases": [
    "pop() on empty stack must raise IndexError or custom exception",
    "peek() on empty stack must raise IndexError or custom exception",
    "Stack should handle None as a valid item"
  ],
  "expected_test_count": 8
}
```

**Action:** Create all 6 task files
- [ ] tasks/stack.json
- [ ] tasks/queue.json
- [ ] tasks/calculator.json
- [ ] tasks/lru_cache.json
- [ ] tasks/template.json
- [ ] tasks/password.json

---

### Phase 1: Core Workflow (Week 1, Day 3-5)

#### Task 1.1: Implement Generators

**File:** `generators/test_generator.py`
```python
"""
g_test: Generate pytest tests from specification.

Input: Task specification (JSON)
Output: Python test code (pytest format)
Guard: SyntaxGuard (validates Python AST)
"""

from pathlib import Path
from typing import Any
from openai import AsyncOpenAI

class TestGenerator:
    """Generate pytest tests from task specifications."""

    def __init__(self, model: str, base_url: str = "http://localhost:11434/v1"):
        self.model = model
        self.client = AsyncOpenAI(base_url=base_url, api_key="ollama")

    async def generate(
        self,
        task_spec: dict,
        workspace: Path,
        context: dict[str, Any]
    ) -> str:
        """
        Generate pytest tests for task.

        Returns: Python code as string
        """
        prompt = self._build_prompt(task_spec)

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )

        return response.choices[0].message.content or ""

    def _build_prompt(self, task_spec: dict) -> str:
        """Build prompt from task specification."""
        return f"""Generate comprehensive pytest unit tests for:

**Task:** {task_spec['name']}
**Description:** {task_spec['description']}

**Requirements:**
{chr(10).join(f"- {req}" for req in task_spec['requirements'])}

**Edge Cases:**
{chr(10).join(f"- {edge}" for edge in task_spec['edge_cases'])}

Generate ONLY the test code, no explanations.
Use pytest format with descriptive test names.
"""

SYSTEM_PROMPT = """You are an expert Python test engineer.
Generate comprehensive pytest test suites that:
1. Cover all requirements
2. Test edge cases and error conditions
3. Use descriptive test function names
4. Include docstrings explaining what each test validates
5. Use pytest.raises for exception testing

Output ONLY Python code, no markdown formatting."""
```

**File:** `generators/impl_generator.py`
```python
"""
g_impl: Generate implementation that passes tests.

Input: pytest test code
Output: Python implementation
Guard: TestGuard (runs tests, validates pass)
"""

class ImplGenerator:
    """Generate implementation from tests."""

    async def generate(
        self,
        tests: str,
        task_spec: dict,
        workspace: Path,
        context: dict[str, Any]
    ) -> str:
        """
        Generate implementation that passes tests.

        Args:
            tests: pytest test code (from g_test)
            task_spec: Original task specification
            workspace: Working directory

        Returns: Python implementation code
        """
        prompt = f"""Generate Python implementation for:

**Task:** {task_spec['name']}
{task_spec['description']}

**Tests to pass:**
```python
{tests}
```

Generate ONLY the implementation code that will make these tests pass.
"""

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": IMPL_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )

        return response.choices[0].message.content or ""

IMPL_SYSTEM_PROMPT = """You are an expert Python developer.
Generate clean, correct implementations that pass the provided tests.
Use type hints, docstrings, and follow Python best practices.
Output ONLY the implementation code, no markdown formatting."""
```

**Actions:**
- [ ] Implement TestGenerator
- [ ] Implement ImplGenerator
- [ ] Write unit tests for generators

#### Task 1.2: Implement Guards

**File:** `guards/syntax_guard.py`
```python
"""
SyntaxGuard: Validate Python code syntax.

Validates:
- Code parses as valid Python AST
- No syntax errors
- Code is executable (no undefined names at parse time)
"""

import ast
from dataclasses import dataclass

@dataclass
class GuardResult:
    passed: bool
    feedback: str

class SyntaxGuard:
    """Validate Python syntax using AST parsing."""

    def validate(self, code: str, context: dict = None) -> GuardResult:
        """
        Validate Python code syntax.

        Args:
            code: Python code string
            context: Additional context (unused)

        Returns:
            GuardResult with pass/fail and feedback
        """
        try:
            ast.parse(code)
            return GuardResult(passed=True, feedback="")
        except SyntaxError as e:
            return GuardResult(
                passed=False,
                feedback=f"Syntax error at line {e.lineno}: {e.msg}"
            )
        except Exception as e:
            return GuardResult(
                passed=False,
                feedback=f"Parse error: {str(e)}"
            )
```

**File:** `guards/test_guard.py`
```python
"""
TestGuard: Execute tests and validate they pass.

Validates:
- Tests execute without errors
- All tests pass
- No test failures or exceptions
"""

import subprocess
import tempfile
from pathlib import Path

class TestGuard:
    """Execute pytest tests and validate they pass."""

    def __init__(self, timeout: int = 60):
        self.timeout = timeout

    def validate(
        self,
        implementation: str,
        tests: str,
        context: dict = None
    ) -> GuardResult:
        """
        Run tests against implementation.

        Args:
            implementation: Python implementation code
            tests: pytest test code
            context: Additional context

        Returns:
            GuardResult with pass/fail and feedback
        """
        # Create temporary directory with both files
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Write implementation
            impl_file = tmpdir_path / "implementation.py"
            impl_file.write_text(implementation)

            # Write tests
            test_file = tmpdir_path / "test_implementation.py"
            # Inject import at top of tests
            test_code = f"from implementation import *\n\n{tests}"
            test_file.write_text(test_code)

            # Run pytest
            try:
                result = subprocess.run(
                    ["pytest", str(test_file), "-v", "--tb=short"],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    cwd=tmpdir
                )

                if result.returncode == 0:
                    return GuardResult(passed=True, feedback="")
                else:
                    # Extract failure info
                    output = result.stdout + result.stderr
                    return GuardResult(
                        passed=False,
                        feedback=f"Tests failed:\n{output[-500:]}"
                    )

            except subprocess.TimeoutExpired:
                return GuardResult(
                    passed=False,
                    feedback=f"Tests timed out after {self.timeout}s"
                )
            except Exception as e:
                return GuardResult(
                    passed=False,
                    feedback=f"Test execution error: {str(e)}"
                )
```

**Actions:**
- [ ] Implement SyntaxGuard
- [ ] Implement TestGuard
- [ ] Write unit tests for guards

---

### Phase 2: Benchmark Runner (Week 1-2)

#### Task 2.1: Implement Workflow Orchestrator

**File:** `benchmarks/runner.py`
```python
"""
Benchmark runner for TDD workflow.

Runs: g_test → g_impl workflow
Trials: 50 per task
Retry: Rmax=3 per generator
"""

from dataclasses import dataclass
from typing import Any
import asyncio
import json
from pathlib import Path

@dataclass
class TrialResult:
    """Result from a single trial."""
    trial_number: int
    success: bool

    # g_test results
    test_attempts: int
    test_passed: bool
    test_code: str
    test_feedback: str

    # g_impl results
    impl_attempts: int
    impl_passed: bool
    impl_code: str
    impl_feedback: str

    # Timing
    duration_seconds: float

class TDDWorkflowRunner:
    """Run TDD workflow benchmark."""

    def __init__(
        self,
        test_generator,
        impl_generator,
        syntax_guard,
        test_guard,
        rmax: int = 3
    ):
        self.test_gen = test_generator
        self.impl_gen = impl_generator
        self.syntax_guard = syntax_guard
        self.test_guard = test_guard
        self.rmax = rmax

    async def run_benchmark(
        self,
        task_specs: list[dict],
        trials: int = 50
    ) -> dict:
        """
        Run benchmark on all tasks.

        Args:
            task_specs: List of task specifications
            trials: Number of trials per task

        Returns:
            Dict of results by task
        """
        results = {}

        for task in task_specs:
            print(f"\n=== Task: {task['name']} ===")
            task_results = await self._run_task(task, trials)
            results[task['name']] = task_results

        return results

    async def _run_task(self, task: dict, trials: int) -> list[TrialResult]:
        """Run N trials for a single task."""
        results = []

        for trial_num in range(1, trials + 1):
            print(f"  Trial {trial_num}/{trials}...", end=" ", flush=True)
            result = await self._run_trial(task, trial_num)
            results.append(result)
            print("✓" if result.success else "✗")

        return results

    async def _run_trial(self, task: dict, trial_num: int) -> TrialResult:
        """Run single trial: g_test → g_impl."""
        import time
        start = time.time()

        # Phase 1: Generate tests (with retry)
        test_code, test_attempts, test_passed, test_feedback = await self._run_with_retry(
            generator=self.test_gen.generate,
            guard=self.syntax_guard.validate,
            args={"task_spec": task}
        )

        if not test_passed:
            # Failed to generate valid tests
            return TrialResult(
                trial_number=trial_num,
                success=False,
                test_attempts=test_attempts,
                test_passed=False,
                test_code=test_code,
                test_feedback=test_feedback,
                impl_attempts=0,
                impl_passed=False,
                impl_code="",
                impl_feedback="",
                duration_seconds=time.time() - start
            )

        # Phase 2: Generate implementation (with retry)
        impl_code, impl_attempts, impl_passed, impl_feedback = await self._run_with_retry(
            generator=self.impl_gen.generate,
            guard=lambda code: self.test_guard.validate(code, test_code),
            args={"tests": test_code, "task_spec": task}
        )

        return TrialResult(
            trial_number=trial_num,
            success=test_passed and impl_passed,
            test_attempts=test_attempts,
            test_passed=test_passed,
            test_code=test_code,
            test_feedback=test_feedback,
            impl_attempts=impl_attempts,
            impl_passed=impl_passed,
            impl_code=impl_code,
            impl_feedback=impl_feedback,
            duration_seconds=time.time() - start
        )

    async def _run_with_retry(
        self,
        generator,
        guard,
        args: dict
    ) -> tuple[str, int, bool, str]:
        """
        Run generator with guard validation and retry.

        Returns: (code, attempts, passed, feedback)
        """
        feedback = ""

        for attempt in range(1, self.rmax + 1):
            # Generate
            code = await generator(**args, workspace=Path("."), context={})

            # Validate
            result = guard(code)

            if result.passed:
                return (code, attempt, True, "")

            feedback = result.feedback
            # TODO: Add feedback to context for next attempt

        return (code, self.rmax, False, feedback)
```

**Actions:**
- [ ] Implement TDDWorkflowRunner
- [ ] Add metrics computation (success rate, avg attempts)
- [ ] Add results serialization (JSON)

---

### Phase 3: Training Pipeline (Week 2-3)

#### Task 3.1: Training Data Extractor

**File:** `training/extractor.py`
```python
"""
Extract training data from benchmark results.

Filters:
- First-attempt success (test_attempts == 1)
- Downstream validation (impl_passed == True)
"""

from typing import List
from dataclasses import dataclass

@dataclass
class TrainingSample:
    """Single training sample."""
    task_name: str
    task_spec: dict
    generated_tests: str
    metadata: dict

class TrainingDataExtractor:
    """Extract high-quality training samples from benchmark results."""

    def extract(self, benchmark_results: dict) -> List[TrainingSample]:
        """
        Extract training samples from benchmark results.

        Criteria:
        1. g_test passed on first attempt
        2. g_impl subsequently passed (validates test quality)

        Args:
            benchmark_results: Results from TDDWorkflowRunner

        Returns:
            List of training samples
        """
        samples = []

        for task_name, trials in benchmark_results.items():
            for trial in trials:
                if self._is_valid_sample(trial):
                    samples.append(TrainingSample(
                        task_name=task_name,
                        task_spec=trial.task_spec,
                        generated_tests=trial.test_code,
                        metadata={
                            "trial_number": trial.trial_number,
                            "test_attempts": trial.test_attempts,
                            "impl_attempts": trial.impl_attempts,
                            "duration": trial.duration_seconds,
                        }
                    ))

        return samples

    def _is_valid_sample(self, trial) -> bool:
        """Check if trial qualifies as training sample."""
        return (
            trial.test_attempts == 1 and  # First-attempt success
            trial.test_passed and         # Tests syntactically valid
            trial.impl_passed             # Implementation passed tests
        )
```

**Actions:**
- [ ] Implement TrainingDataExtractor
- [ ] Add sample deduplication
- [ ] Add quality metrics

#### Task 3.2: Fine-Tuning Script

**File:** `training/finetune.py`
```python
"""
Fine-tune model using Unsloth LoRA.

Hyperparameters:
- LoRA rank: 16
- Learning rate: 2e-4
- Epochs: 3
"""

from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

def finetune_test_generator(
    base_model: str,
    training_data: str,  # JSONL file
    output_dir: str
):
    """
    Fine-tune model for test generation.

    Args:
        base_model: Base model name (e.g., "qwen2.5-coder:7b")
        training_data: Path to training data JSONL
        output_dir: Output directory for fine-tuned model
    """
    # Load base model with Unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=4096,
        load_in_4bit=True,
    )

    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
    )

    # Load dataset
    from datasets import load_dataset
    dataset = load_dataset("json", data_files=training_data, split="train")

    # Train
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        max_seq_length=4096,
        args=TrainingArguments(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
            warmup_steps=10,
            num_train_epochs=3,
            learning_rate=2e-4,
            fp16=True,
            output_dir=output_dir,
            logging_steps=10,
        ),
    )

    trainer.train()

    # Save to GGUF
    model.save_pretrained_gguf(
        f"{output_dir}/gguf",
        tokenizer,
        quantization_method="q4_k_m"
    )
```

**Actions:**
- [ ] Implement fine-tuning script
- [ ] Add model export to GGUF
- [ ] Add training metrics logging

---

### Phase 4: Analysis & Reporting (Week 4)

#### Task 4.1: Statistical Analysis

**File:** `benchmarks/analysis.py`
```python
"""
Statistical analysis of results.

Tests:
- Fisher's exact test (significance)
- Cohen's h (effect size)
"""

from scipy.stats import fisher_exact
import numpy as np

def compare_models(baseline_results, finetuned_results):
    """
    Compare baseline vs fine-tuned model.

    Returns statistical comparison for each task.
    """
    comparisons = {}

    for task_name in baseline_results.keys():
        baseline = baseline_results[task_name]
        finetuned = finetuned_results[task_name]

        # Success counts
        baseline_success = sum(1 for t in baseline if t.success)
        finetuned_success = sum(1 for t in finetuned if t.success)

        # Fisher's exact test
        table = [
            [baseline_success, len(baseline) - baseline_success],
            [finetuned_success, len(finetuned) - finetuned_success]
        ]
        _, p_value = fisher_exact(table)

        # Cohen's h (effect size)
        p1 = baseline_success / len(baseline)
        p2 = finetuned_success / len(finetuned)
        cohens_h = 2 * (np.arcsin(np.sqrt(p2)) - np.arcsin(np.sqrt(p1)))

        comparisons[task_name] = {
            "baseline_success_rate": p1,
            "finetuned_success_rate": p2,
            "delta": p2 - p1,
            "p_value": p_value,
            "cohens_h": cohens_h,
            "significant": p_value < 0.05
        }

    return comparisons
```

**Actions:**
- [ ] Implement statistical tests
- [ ] Generate comparison tables (match paper format)
- [ ] Create visualizations (heatmaps, bar charts)

---

## Dependencies

### Python Packages
```
# Core
python = "^3.12"

# Existing AtomicGuard dependencies
(from pyproject.toml)

# New dependencies for this experiment
openai = "^1.0"          # LLM API client
unsloth = "^2024.1"      # LoRA fine-tuning
trl = "^0.7"             # SFTTrainer
scipy = "^1.11"          # Statistical tests
datasets = "^2.14"       # HuggingFace datasets
```

### External Tools
```
- Ollama >= 0.12.3 (for inference)
- pytest (for test execution)
```

---

## Validation Checkpoints

### After Phase 1 (Week 1)
- [ ] Can generate tests from task spec (manual inspection)
- [ ] Tests pass SyntaxGuard
- [ ] Can generate implementation from tests
- [ ] Implementation passes TestGuard
- [ ] Workflow completes end-to-end for at least 1 task

### After Phase 2 (Week 2)
- [ ] 300 trials completed (6 tasks × 50 trials)
- [ ] Extracted ≥50 training samples
- [ ] Training samples validated (manual review of 10)
- [ ] Baseline metrics computed
- [ ] Results match expected range from paper (57% ± 10%)

### After Phase 3 (Week 3)
- [ ] Fine-tuning completed without errors
- [ ] Model exported to GGUF
- [ ] Model loads in Ollama
- [ ] Smoke test: Model generates valid tests

### After Phase 4 (Week 4)
- [ ] 300 fine-tuned trials completed
- [ ] Statistical tests computed
- [ ] Comparison report generated
- [ ] Results documented in RESULTS.md

---

## Next Steps

1. Review this plan with stakeholders
2. Set up development environment
3. Begin Phase 0 (setup)
4. Track progress using git commits

**Status:** Ready for implementation
