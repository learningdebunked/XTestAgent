#!/bin/bash
# Quick script to validate all TestAgentX paper claims
# Usage: bash scripts/validate_paper_claims.sh [project_path]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PROJECT_PATH="${1:-sample_project}"
OUTPUT_DIR="evaluation_results/$(date +%Y%m%d_%H%M%S)"

echo "=========================================="
echo "TestAgentX Paper Claims Validation"
echo "=========================================="
echo "Project: $PROJECT_PATH"
echo "Output:  $OUTPUT_DIR"
echo "=========================================="
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run full evaluation
echo "Running full evaluation pipeline..."
python3 "$PROJECT_ROOT/evaluation/run_full_evaluation.py" \
    --project "$PROJECT_PATH" \
    --output "$OUTPUT_DIR" \
    2>&1 | tee "$OUTPUT_DIR/evaluation.log"

# Check if summary report was generated
if [ -f "$OUTPUT_DIR/summary_report.json" ]; then
    echo ""
    echo "=========================================="
    echo "Validation Complete!"
    echo "=========================================="
    echo ""
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "Summary:"
    python3 -c "
import json
with open('$OUTPUT_DIR/summary_report.json', 'r') as f:
    data = json.load(f)
    
print('Metric                    | Target  | Measured | Status')
print('-' * 60)

comparison = data.get('comparison', {})
for metric, values in comparison.items():
    name = metric.replace('_', ' ').title()
    target = values['paper_claim']
    measured = values['measured']
    status = '✅ PASS' if values['meets_claim'] else '❌ FAIL'
    print(f'{name:<25} | {target:>6.1f}% | {measured:>7.1f}% | {status}')
"
    echo ""
    echo "Full report: $OUTPUT_DIR/summary_report.json"
else
    echo "ERROR: Evaluation failed. Check $OUTPUT_DIR/evaluation.log"
    exit 1
fi
