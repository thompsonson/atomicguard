# TDD Workflow Fine-Tuning Experiment

**Date:** January 2026
**Status:** Planning
**Related Paper:** `docs/main.tex` (Section 4.3: TDD Workflow Benchmark)

---

## 1. Motivation

### 1.1 Context from the Paper

The AtomicGuard paper (Section 4.3, lines 897-956) presents a **TDD Workflow Benchmark** that evaluates multi-step workflows:

```
Workflow: g_test (generate tests) → g_impl (generate implementation)
Models tested: Qwen2.5-Coder (3B, 7B, 14B)
Tasks: 6 tasks (Stack, Queue, Calculator, LRU, Template, Password)
Results: 50 trials per model-task pair
```

**Key findings from paper:**
- 14B model: 70% overall success
- 7B model: 57% overall success
- 3B model: 36% overall success
- **Critical limitation (line 950-956)**: "LLM-generated tests frequently contain incorrect edge case expectations... The implementation then fails not because it is wrong, but because it must satisfy a flawed specification."

### 1.2 Our Research Question

**Can we extract training data from successful workflow executions and use it to improve model performance, demonstrating measurable learning in the AtomicGuard framework?**

This is **NOT** described in the paper. This is a **new experiment** to validate Extension 04 (Learning Loop) mentioned in the paper's Future Research section.

---

## 2. Experimental Design

### 2.1 Standards & Methodology

#### **Training Data Standards**

We define quality criteria for training samples (this is NEW, not from paper):

| Quality Tier | Definition | Selection Criteria |
|--------------|------------|-------------------|
| **First-Attempt Success** | g_test passed SyntaxGuard on first attempt | `attempt == 1 AND guard_passed == True` |
| **Validated Success** | Tests passed AND subsequent g_impl succeeded | `g_test.passed AND g_impl.passed` |
| **Workflow Success** | Full TDD workflow completed successfully | End-to-end success |

**Standard we will use:** First-Attempt Success + Validated Success

**Rationale:**
- First-attempt indicates model already knows the pattern (not learned through retries)
- Validated success confirms tests were correct (not flawed specs as paper identified)

#### **Evaluation Standards (From Paper)**

We maintain the paper's evaluation methodology (Section 4.3, Table 5):

| Metric | Definition | Source |
|--------|------------|--------|
| Success Rate | Percentage of trials where workflow completed | Paper, line 920-923 |
| Average Attempts | Mean number of generator attempts (g_test + g_impl) | Paper, line 937-939 |
| Average Duration | Mean time in seconds per trial | Paper, line 942-945 |

#### **Statistical Standards (From Paper)**

Following paper's methodology (Section 4.2, line 875):
- Fisher's exact test for significance
- p-value thresholds: *** p<0.001, ** p<0.01, * p<0.05
- Cohen's h for effect sizes

### 2.2 Three-Phase Protocol

#### **Phase 1: Baseline Evaluation & Data Collection**

**Objective:** Establish baseline performance and collect training data

```
Model: qwen2.5-coder:7b
Tasks: 6 (Stack, Queue, Calculator, LRU, Template, Password)
Trials: 50 per task (total: 300 runs)
Workflow: g_test → g_impl (same as paper Section 4.3)

Outputs:
1. Baseline metrics (for comparison)
2. Training dataset (extracted from successful runs)
```

**Expected training data yield:**
```
300 total runs
× 0.57 (expected 7B success rate from paper)
× 0.60 (estimated first-attempt success ratio)
≈ 100-120 training samples
```

**Data extraction criteria:**
```python
def is_valid_training_sample(run_result):
    """
    Extract sample if:
    1. g_test passed on first attempt (no retry loops)
    2. g_impl subsequently passed (validates test quality)
    """
    return (
        run_result.g_test.attempt == 1 and
        run_result.g_test.guard_passed and
        run_result.g_impl.passed
    )
```

#### **Phase 2: Fine-Tuning**

**Objective:** Specialize 7B model on successful test generation patterns

```
Base model: qwen2.5-coder:7b
Method: LoRA (Low-Rank Adaptation)
Hyperparameters:
  - LoRA rank (r): 16
  - LoRA alpha: 16
  - Learning rate: 2e-4
  - Epochs: 3
  - Batch size: 4

Training data: ~100-120 samples from Phase 1
Hardware: RTX A4000 (16GB VRAM)
Estimated time: 1 hour

Output: test_agent_v1_ft (GGUF format for Ollama)
```

**Training data format:**
```json
{
  "instruction": "Generate comprehensive pytest unit tests for the following specification:\n\n{task_spec}",
  "input": "",
  "output": "{generated_tests}",
  "metadata": {
    "task": "stack",
    "original_attempt": 1,
    "downstream_success": true
  }
}
```

#### **Phase 3: Comparative Evaluation**

**Objective:** Measure improvement against baseline

```
Models to compare:
1. qwen2.5-coder:7b (baseline - from Phase 1)
2. test_agent_v1_ft (fine-tuned - new run)
3. qwen2.5-coder:14b (reference - from paper)

Tasks: Same 6 tasks
Trials: 50 per task per model
Methodology: Identical to paper Section 4.3
```

**Comparison metrics:**
```python
@dataclass
class ComparisonResult:
    model: str
    task: str

    # From paper
    success_rate: float
    avg_attempts: float
    avg_duration: float

    # Improvement metrics (new)
    delta_success: float  # vs 7B baseline
    p_value: float        # Fisher's exact test
    cohens_h: float       # Effect size
```

---

## 3. Hypothesis

### 3.1 Primary Hypothesis

**H1:** Fine-tuning qwen2.5-coder:7b on first-attempt successful test generations will improve TDD workflow success rate by ≥5 percentage points (pp).

```
Null hypothesis (H0): success_rate(7B_ft) - success_rate(7B_baseline) ≤ 0
Alternative (H1): success_rate(7B_ft) - success_rate(7B_baseline) ≥ 5pp
Statistical test: Fisher's exact test, α=0.05
```

### 3.2 Secondary Hypothesis

**H2:** Fine-tuned 7B model will achieve performance comparable to baseline 14B model (within 5pp) on at least 3/6 tasks.

```
For each task:
  |success_rate(7B_ft) - success_rate(14B_baseline)| ≤ 5pp

Success criterion: True for ≥3 tasks
```

### 3.3 Expected Outcomes

Based on paper results (Table 5, lines 920-923):

| Model | Overall | Stack | Queue | Calc | LRU | Template | Password |
|-------|---------|-------|-------|------|-----|----------|----------|
| **7B Baseline** (paper) | 57% | 94% | 98% | 78% | 50% | 20% | 0% |
| **14B Baseline** (paper) | 70% | 88% | 98% | 84% | 66% | 68% | 14% |
| **7B Fine-tuned** (predicted) | **64±3%** | 96% | 100% | 82% | 58% | 32% | 6% |

**Predicted improvement:** +7pp overall (range: +4pp to +10pp)

**Tasks expected to improve most:**
1. Template (+12pp): Most sensitive to test quality
2. LRU (+8pp): Medium complexity, benefits from examples
3. Password (+6pp): Edge case coverage improvements

**Tasks expected to improve least:**
1. Queue (+2pp): Already at ceiling (98%)
2. Stack (+2pp): Already very high (94%)

---

## 4. Success Criteria

### 4.1 Minimum Viable Success

- [ ] Fine-tuned 7B shows **≥5pp improvement** over baseline 7B (overall)
- [ ] Improvement is **statistically significant** (p < 0.05) on ≥2 tasks
- [ ] Training data extraction yields **≥50 samples** (sufficient for LoRA)

**Decision:** If met, proceed to publication/blog post

### 4.2 Strong Success

- [ ] Fine-tuned 7B shows **≥8pp improvement** over baseline 7B
- [ ] Matches or exceeds 14B baseline on **≥3 tasks**
- [ ] Statistically significant (p < 0.01) on ≥3 tasks

**Decision:** If met, expand to additional models (3B, 14B) and document learning transfer

### 4.3 Exceptional Success

- [ ] Fine-tuned 7B **matches 14B baseline overall** (within 3pp)
- [ ] Demonstrates **parameter-efficient specialization** (70% performance with 50% parameters)
- [ ] Establishes methodology for **continuous improvement loop**

**Decision:** If met, this validates Extension 04 (Learning Loop) and warrants paper addendum

---

## 5. Risks & Mitigations

### 5.1 Insufficient Training Data

**Risk:** Phase 1 yields <50 samples (insufficient for LoRA)

**Mitigation:**
- Reduce quality threshold (accept attempt ≤2 instead of attempt==1)
- Run additional trials (extend from 50 to 100 per task)
- Supplement with human-written examples (gold standard)

### 5.2 No Measurable Improvement

**Risk:** Fine-tuned model shows <5pp improvement (hypothesis rejected)

**Mitigation:**
- Analyze failure modes (which tasks failed to improve?)
- Inspect training data quality (were samples truly representative?)
- Iterate on hyperparameters (learning rate, epochs, LoRA rank)

**Learning:** Even negative results validate that fine-tuning requires more data or different approach

### 5.3 Overfitting to Training Distribution

**Risk:** Model improves on trained tasks but doesn't generalize

**Test:** Evaluate on held-out task variations (e.g., Deque, PriorityQueue)

**Mitigation:** If overfitting detected, increase training data diversity

---

## 6. Timeline

### Week 1: Infrastructure
- [ ] Implement benchmark runner (reuse `examples/base`)
- [ ] Implement generators (g_test, g_impl)
- [ ] Implement guards (SyntaxGuard, TestGuard)
- [ ] Create task specifications (6 JSON files)

### Week 2: Phase 1 (Baseline + Data Collection)
- [ ] Run 6 tasks × 50 trials with qwen2.5-coder:7b
- [ ] Extract training data (~100-120 samples)
- [ ] Generate baseline metrics report
- [ ] Validate training data quality (manual review of 10 samples)

### Week 3: Phase 2 (Fine-Tuning)
- [ ] Format training data for Unsloth
- [ ] Fine-tune qwen2.5-coder:7b (LoRA)
- [ ] Export to GGUF for Ollama
- [ ] Smoke test fine-tuned model

### Week 4: Phase 3 (Evaluation + Analysis)
- [ ] Run 6 tasks × 50 trials with fine-tuned model
- [ ] Statistical comparison (Fisher's exact test, Cohen's h)
- [ ] Generate comparison tables (match paper format)
- [ ] Document findings

---

## 7. Deliverables

### 7.1 Code Artifacts

- [ ] `examples/advanced/tdd_finetuning/` (complete implementation)
- [ ] Benchmark runner with metrics collection
- [ ] Training data extraction pipeline
- [ ] Fine-tuning scripts (Unsloth)
- [ ] Statistical analysis tools

### 7.2 Data Artifacts

- [ ] Baseline results (qwen2.5-coder:7b, 6 tasks × 50 trials)
- [ ] Training dataset (~100-120 samples, JSONL format)
- [ ] Fine-tuned model (GGUF, ready for Ollama)
- [ ] Fine-tuned results (test_agent_v1_ft, 6 tasks × 50 trials)

### 7.3 Documentation

- [ ] `EXPERIMENT.md` (this document)
- [ ] `RESULTS.md` (findings, tables, statistical tests)
- [ ] `README.md` (setup, usage, reproduction instructions)
- [ ] Comparison tables matching paper format (Section 4.3)

---

## 8. Reproducibility

### 8.1 Seed Management

All random operations must be seeded:
```python
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
```

### 8.2 Versioning

Document all dependencies:
```
- Ollama version: 0.12.3+
- Model: qwen2.5-coder:7b (specific commit hash)
- Python: 3.12.11
- PyTorch: 2.x
- Unsloth: latest
```

### 8.3 Data Provenance

Each training sample must include:
```json
{
  "source": "baseline_run",
  "run_id": "uuid",
  "task": "stack",
  "trial_number": 23,
  "timestamp": "2026-01-15T10:23:45Z",
  "model": "qwen2.5-coder:7b",
  "attempt": 1,
  "guard_result": "PASS"
}
```

---

## 9. References

### 9.1 Paper References

- **Section 4.3** (lines 897-956): TDD Workflow Benchmark
- **Table 5** (lines 920-923): TDD workflow success rates
- **Table 6** (lines 937-945): TDD workflow efficiency metrics
- **Future Research** (Section 6): Continuous Learning via The Optimization Loop

### 9.2 Related Work

- Extension 04: Learning Loop (`docs/design/extensions/04_learning_loop.md`)
- Extension 05: Learning Implementation (`docs/design/extensions/05_learning_implementation.md`)

---

## 10. Open Questions

- [ ] Should we include human validation of training samples (quality check)?
- [ ] Should we fine-tune both g_test and g_impl, or just g_test?
- [ ] Should we test transfer learning (evaluate on new tasks not in training)?
- [ ] Should we document model card (ethical considerations, limitations)?

---

**Document Version:** 1.0
**Last Updated:** 2026-01-15
**Status:** Ready for review
