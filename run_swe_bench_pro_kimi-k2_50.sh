#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="output/swe_bench_pro_kimi-k2_50_20260204"

# Ensure LLM_API_KEY is set
if [[ -z "${LLM_API_KEY:-}" ]]; then
    echo "Error: LLM_API_KEY is not set. Export your OpenRouter API key first." >&2
    exit 1
fi

# 1. Run experiment (50 instances, all 3 arms)
uv run python -m examples.swe_bench_pro.demo --debug --log-file "${OUTPUT_DIR}/experiment.log" experiment \
    --provider openrouter \
    --base-url https://openrouter.ai/api/v1 \
    --arms singleshot,s1_direct,s1_tdd \
    --max-instances 50 \
    --max-workers 4 \
    --output-dir "${OUTPUT_DIR}"

# 2. Evaluate patches in Docker
uv run python -m examples.swe_bench_pro.demo evaluate \
    --predictions-dir "${OUTPUT_DIR}/predictions" \
    --max-workers 4

# 3. Visualize results
uv run python -m examples.swe_bench_pro.demo visualize \
    --results "${OUTPUT_DIR}/results.jsonl" \
    --resolved "${OUTPUT_DIR}/predictions/eval_output/eval_results.json"
