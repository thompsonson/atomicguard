#!/usr/bin/env bash
set -euo pipefail

# Run TDD Decomposed Backtracking Workflow (Arm 07) with Qwen3-Coder-Next
#
# This script runs the s1_decomposed workflow on a single SWE-Bench Pro instance
# using qwen/qwen3-coder-next from OpenRouter, then exports an HTML visualization
# of the workflow execution showing all 11 steps, retries, and backtracking events.
#
# Usage: ./run_decomposed_backtrack.sh

# Source .env for API keys
if [[ -f .env ]]; then
    # shellcheck disable=SC1091
    source .env
else
    echo "Error: .env file not found. Please create one with LLM_API_KEY set."
    exit 1
fi

# Configuration
MODEL="qwen/qwen3-coder-next"
PROVIDER="openrouter"
OUTPUT_DIR="output/decomposed_qwen3_coder_$(date +%Y%m%d_%H%M%S)"
ARM="s1_decomposed"

echo "=============================================="
echo "Running TDD Decomposed Backtracking Workflow"
echo "=============================================="
echo "Model:  $MODEL"
echo "Arm:    $ARM"
echo "Output: $OUTPUT_DIR"
echo ""

# Run experiment (1 instance for verification)
uv run python -m examples.swe_bench_pro.demo --debug experiment \
    --provider "$PROVIDER" \
    --model "$MODEL" \
    --base-url https://openrouter.ai/api/v1 \
    --arms "$ARM" \
    --language python \
    --max-instances 1 \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "=============================================="
echo "Generating Workflow DAG Visualization"
echo "=============================================="

# Export workflow DAG HTML visualization for each instance/arm
uv run python -c "
from atomicguard import export_workflow_html
from atomicguard.infrastructure.persistence.filesystem import FilesystemArtifactDAG
from pathlib import Path

output_dir = Path('$OUTPUT_DIR')
dags_dir = output_dir / 'artifact_dags'

if not dags_dir.exists():
    print(f'DAG directory not found: {dags_dir}')
    raise SystemExit(1)

found = False
for dag_dir in sorted(dags_dir.rglob('index.json')):
    dag_dir = dag_dir.parent
    dag = FilesystemArtifactDAG(str(dag_dir))
    artifacts = dag.get_all()
    if not artifacts:
        continue
    found = True
    workflow_id = artifacts[0].workflow_id
    rel = dag_dir.relative_to(dags_dir)
    out_path = output_dir / f'workflow_dag_{\"_\".join(rel.parts)}.html'
    out = export_workflow_html(dag, workflow_id=workflow_id, output_path=str(out_path))
    print(f'Workflow DAG visualization: {out}')

if not found:
    print('No artifacts found in any DAG')
"

echo ""
echo "=============================================="
echo "Done!"
echo "=============================================="
echo "Open the workflow visualization(s) in your browser:"
echo "  ls $OUTPUT_DIR/workflow_dag_*.html"
