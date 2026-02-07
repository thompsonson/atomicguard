# SWE-Bench Pro Extended Experiment Results

**Date:** 2026-02-06
**Model:** DeepSeek Chat (`deepseek/deepseek-chat`) via OpenRouter
**Dataset:** SWE-Bench Pro (Python instances)
**Instances:** 30 Python instances
**Arms:** 3 (singleshot, s1_direct, s1_tdd)

## Executive Summary

This experiment evaluated three patch generation approaches on 30 Python instances from SWE-Bench Pro:

| Arm | Description | Patches Generated | Resolved | Resolution Rate |
|-----|-------------|-------------------|----------|-----------------|
| `02_singleshot` | Direct patch generation | 9/30 (30%) | 1/30 | **3.3%** |
| `03_s1_direct` | Analysis → Patch | 28/30 (93%) | 3/30 | **10.0%** |
| `04_s1_tdd` | Analysis → Test → Patch | 25/30 (83%) | 4/30 | **13.3%** |

Resolution rate is measured as `resolved / total instances` — the probability that a given instance is fixed end-to-end by the arm.

**Key Finding:** Progressive decomposition improves resolution rate at each step: singleshot (3.3%) → analysis + patch (10.0%) → analysis + test + patch (13.3%). The TDD arm's test-first constraint improves patch quality even though the generated tests are not executed, but the overall gains are modest at n=30 and should be validated at larger sample sizes.

> **Note on patch generation vs resolution:** The multi-step arms also dramatically improve *patch generation rate* (singleshot 30% → s1_direct 93% → s1_tdd 83%). However, generating a patch is necessary but not sufficient — resolution rate against the gold-standard evaluation harness is the metric that matters.

## Methodology

### Arms Description

1. **Singleshot (`02_singleshot`)**: Single-shot patch generation directly from the problem statement and a subset of repository files. No analysis step.

2. **S1 Direct (`03_s1_direct`)**: Two-step workflow:
   - **Analysis**: Generate bug analysis identifying root cause and affected files
   - **Patch**: Generate patch based on analysis

3. **S1 TDD (`04_s1_tdd`)**: Three-step workflow:
   - **Analysis**: Generate bug analysis
   - **Test**: Generate test cases that would fail before the fix
   - **Patch**: Generate patch that would make tests pass

### Configuration

- **Provider:** OpenRouter
- **Model:** `deepseek/deepseek-chat`
- **Max retries per step:** 4
- **Evaluation:** Docker-based SWE-Bench Pro evaluation harness

## Results

### Pass Rate Comparison

![Pass Rate Comparison](swe_bench_pro_images/pass_rate_comparison.png)

The pass rate comparison shows progressive improvement: singleshot (3.3%) → s1_direct (10.0%) → s1_tdd (13.3%). While s1_direct generates the most patches (93%), the TDD approach achieves the highest end-to-end resolution rate.

### Token Cost vs Pass Rate

![Token Cost vs Pass Rate](swe_bench_pro_images/token_cost_vs_pass_rate.png)

Multi-step approaches use more tokens but achieve significantly better patch generation rates. The TDD arm's additional test generation step provides the best return on token investment in terms of resolution rate.

### Per-Step Token Breakdown

![Per-Step Token Breakdown](swe_bench_pro_images/per_step_token_breakdown.png)

Token usage breakdown by generation step shows the relative cost of each phase in the multi-step pipelines.

### Wall Time Distribution

![Wall Time Distribution](swe_bench_pro_images/wall_time_distribution.png)

Wall time distribution across arms shows the time-quality tradeoff between approaches.

### Instance Outcome Heatmap

![Instance Outcome Heatmap](swe_bench_pro_images/instance_outcome_heatmap.png)

The heatmap shows per-instance outcomes across all arms, highlighting which instances were resolved by each approach.

### Pairwise Effect Sizes

![Pairwise Effect Sizes](swe_bench_pro_images/pairwise_effect_sizes.png)

Statistical comparison between arms showing effect sizes for resolution rate differences.

## Detailed Results

### Resolved Instances by Arm

**Singleshot (1 resolved):**
- `ansible__ansible-395e5e20...` - Play iterator state handling

**S1 Direct (3 resolved):**
- `ansible__ansible-395e5e20...` - Play iterator state handling
- `internetarchive__openlibrary-00bec1e7...` - Monitoring scheduler fix
- `internetarchive__openlibrary-4a5d2a7d...` - Wikidata entity handling

**S1 TDD (4 resolved):**
- `qutebrowser__qutebrowser-f91ace96...` - Qt warning hiding
- `qutebrowser__qutebrowser-96b99780...` - Duration parsing
- `internetarchive__openlibrary-4a5d2a7d...` - Wikidata entity handling
- `internetarchive__openlibrary-5069b09e...` - Booknotes work ID update

### Failure Analysis

**Singleshot Failures (21/30 no patch):**
- Primary cause: Search string mismatches (model generates code that doesn't match actual file content)
- Secondary cause: Invalid JSON output format
- The singleshot approach doesn't read the actual file content before generating patches

**S1 Direct Failures (2/30 no patch):**
- Files not found in repository (model hallucinates file paths in analysis)

**S1 TDD Failures (5/30 no patch):**
- Analysis guard rejects files not found in repository
- Test generation complexity increases failure rate

## Key Observations

1. **Multi-step approaches dramatically improve patch generation:**
   - Singleshot: 30% success in generating valid patches
   - S1 Direct: 93% success
   - S1 TDD: 83% success

2. **TDD improves end-to-end resolution rate:**
   - Despite generating fewer patches than s1_direct, TDD resolves more instances
   - Resolution rate (resolved/total): TDD (13.3%) > Direct (10.0%) > Singleshot (3.3%)
   - The difference between direct and TDD is a single instance (3 vs 4 at n=30) — directional but not yet statistically significant

3. **Analysis step is critical:**
   - The analysis step helps the model understand the codebase before attempting patches
   - Guards catch and allow retry of invalid analyses/patches

4. **Search string matching remains challenging:**
   - Models struggle to match exact code content without reading files
   - This is the primary failure mode for singleshot

## Recommendations

1. **Prefer multi-step pipelines** for production use — the 3x improvement in resolution rate from singleshot (3.3%) to s1_direct (10.0%) justifies the additional token cost
2. **Validate TDD gains at larger sample sizes** — the TDD advantage (13.3% vs 10.0%) rests on a single additional resolved instance; running on the full 731-instance SWE-Bench Pro set will clarify whether this holds
3. **Investigate execution-verified arms (05, 06)** — the current TDD arm generates tests but does not execute them; adding Docker-based G_ver guards (TestRedGuard, TestGreenGuard) should amplify the TDD benefit by rejecting tests that don't actually fail on the buggy code
4. **Explore ensemble approaches** — the resolved sets across arms have minimal overlap (only 1 instance resolved by both s1_direct and s1_tdd), suggesting complementary strengths that an ensemble or voting arm could exploit

## Files

- Results: `output/swe_bench_pro_code_aware/results.jsonl`
- Predictions: `output/swe_bench_pro_code_aware/predictions/`
- Evaluation logs: `output/swe_bench_pro_code_aware/predictions/eval_logs/`
- Visualizations: `output/swe_bench_pro_code_aware/*.png`
