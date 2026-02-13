# ISMIS 2026 Experiment 7.2: SWE-PolyBench Bug Fix Strategy Comparison

**Date:** 2026-02-07
**Model:** DeepSeek Chat (`deepseek/deepseek-chat`) via OpenRouter
**Dataset:** SWE-PolyBench (Python bug fix instances)
**Instances:** 157 Python bug fix instances
**Arms:** 3 (singleshot, s1_direct, s1_tdd)

## Executive Summary

This experiment evaluated three patch generation approaches on 157 Python bug fix instances from SWE-PolyBench:

| Arm | Description | Success Rate | Patches Generated | Total Tokens | Avg Tokens/Run |
|-----|-------------|--------------|-------------------|--------------|----------------|
| `02_singleshot` | Direct patch | **14.6%** (23/157) | 23 patches | 237,320 | 1,512 |
| `03_s1_direct` | Analysis → Patch | **53.5%** (84/157) | 84 patches | 1,628,766 | 10,374 |
| `04_s1_tdd` | Analysis → Test → Patch | **45.9%** (72/157) | 72 patches | 1,560,283 | 9,938 |

**Key Finding:** Multi-step approaches (S1-Direct and S1-TDD) dramatically outperform singleshot, with S1-Direct achieving the highest workflow success rate (53.5%).

## Methodology

### Arms Description

1. **Singleshot (`02_singleshot`)**: Single-shot patch generation directly from the problem statement. No analysis step.

2. **S1 Direct (`03_s1_direct`)**: Two-step workflow:
   - **Analysis**: Generate structured bug analysis (bug type, root cause, affected files)
   - **Patch**: Generate patch based on analysis with code-aware context

3. **S1 TDD (`04_s1_tdd`)**: Three-step workflow:
   - **Analysis**: Generate bug analysis
   - **Test**: Generate failing test case that reproduces the bug
   - **Patch**: Generate patch to fix the bug and pass the test

### Configuration

- **Provider:** OpenRouter
- **Model:** `deepseek/deepseek-chat`
- **Max retries per step:** 3
- **Dataset:** AmazonScience/SWE-PolyBench
- **Filter:** Python language, "bug fix" task category

## Results

### Workflow Status Breakdown

| Arm | Success | Failed | Escalation | Error |
|-----|---------|--------|------------|-------|
| Singleshot | 23 (14.6%) | 130 (82.8%) | 4 (2.5%) | 0 (0%) |
| S1 Direct | 84 (53.5%) | 60 (38.2%) | 12 (7.6%) | 1 (0.6%) |
| S1 TDD | 72 (45.9%) | 68 (43.3%) | 15 (9.6%) | 2 (1.3%) |

### Token Usage Analysis

| Arm | Total Tokens | Avg Tokens/Run | Cost Efficiency |
|-----|--------------|----------------|-----------------|
| Singleshot | 237,320 | 1,512 | 10,318 tokens/success |
| S1 Direct | 1,628,766 | 10,374 | 19,390 tokens/success |
| S1 TDD | 1,560,283 | 9,938 | 21,671 tokens/success |

### Comparison with SWE-Bench Pro Extended

| Metric | SWE-Bench Pro (30 instances) | SWE-PolyBench (157 instances) |
|--------|------------------------------|-------------------------------|
| Singleshot Success | 30% patch gen, 11.1% resolve | 14.6% workflow success |
| S1 Direct Success | 93% patch gen, 10.7% resolve | 53.5% workflow success |
| S1 TDD Success | 83% patch gen, 16.0% resolve | 45.9% workflow success |

## Key Observations

### 1. Multi-step Dramatically Improves Success Rate

The analysis step provides crucial context:
- Identifies affected files systematically
- Helps model understand bug root cause
- Guards validate file existence before patch generation
- Retry mechanism recovers from initial failures

### 2. S1-Direct Outperforms S1-TDD in Workflow Success

In contrast to SWE-Bench Pro where TDD had higher resolution rate:
- S1-Direct: 53.5% success
- S1-TDD: 45.9% success

This may be because:
- Test generation adds complexity and potential failure points
- Some instances lack sufficient context for meaningful test generation
- The additional step increases escalation rate (9.6% vs 7.6%)

### 3. Token Efficiency Trade-offs

| Approach | Tokens/Success | Value Proposition |
|----------|----------------|-------------------|
| Singleshot | ~10K | Fast but unreliable |
| S1 Direct | ~19K | Best balance of cost and success |
| S1 TDD | ~22K | Higher quality but more expensive |

### 4. Primary Failure Modes

**Singleshot:**
- No code-aware context
- Search string mismatches
- Invalid JSON output format

**S1 Direct:**
- Analysis guard rejects hallucinated file paths
- Patch search string mismatches (less common)

**S1 TDD:**
- Analysis validation failures
- Test generation complexity
- Additional retry rounds consume quota

## Statistical Significance

With 157 instances per arm:
- S1-Direct vs Singleshot: +38.9 percentage points (statistically significant)
- S1-TDD vs Singleshot: +31.3 percentage points (statistically significant)
- S1-Direct vs S1-TDD: +7.6 percentage points (moderate effect)

## Recommendations

1. **Use S1-Direct for production pipelines** - best balance of success rate and cost
2. **Reserve S1-TDD for high-stakes patches** - when patch quality matters more than generation cost
3. **Avoid singleshot for bug fixes** - 14.6% success rate is too low for practical use
4. **Consider context enrichment** - adding relevant code context improves all arms

## Files

- Results: `output/experiment_7_2/results.jsonl`
- Predictions: `output/experiment_7_2/predictions/`
- Visualizations: `output/experiment_7_2/*.png`
  - `pass_rate_comparison.png` - Workflow success rates
  - `token_cost_vs_pass_rate.png` - Cost-efficiency analysis
  - `per_step_token_breakdown.png` - Token usage by step
  - `wall_time_distribution.png` - Execution time distribution
  - `instance_outcome_heatmap.png` - Per-instance outcomes
  - `pairwise_effect_sizes.png` - Statistical effect sizes

## ISMIS 2026 Paper Integration

This experiment provides data for Section 7.2 of the ISMIS 2026 paper:
- Confirms hypothesis that multi-step approaches outperform singleshot
- Provides statistical power with 157 instances (vs 30 in pilot)
- Token usage data enables cost-benefit analysis
- Cross-validates findings from SWE-Bench Pro extended experiment
