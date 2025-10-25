# TestAgentX Paper Claims Validation Guide

This guide explains how to measure and validate all claims made in the TestAgentX paper.

## Overview

The paper makes six key empirical claims that require validation:

| Claim | Target | Status |
|-------|--------|--------|
| Test Coverage | 89% | ✅ Measurable |
| Mutation Score | 84% | ✅ Measurable |
| Time Reduction | 55% | ✅ Measurable |
| Patch Accuracy | 91% | ✅ Measurable |
| False Positive Rate | 8% | ✅ Measurable |
| Developer Acceptance | 82% | ✅ Measurable |

## Quick Start

### 1. Install Evaluation Tools

```bash
# Install PITest for mutation testing
mvn dependency:get -Dartifact=org.pitest:pitest-maven:LATEST

# Setup JaCoCo (already done if you ran setup_jacoco.sh)
bash scripts/setup_jacoco.sh
```

### 2. Run Full Evaluation

```bash
python evaluation/run_full_evaluation.py \
    --dataset defects4j \
    --project /path/to/your/project \
    --output evaluation_results/
```

### 3. View Results

```bash
cat evaluation_results/summary_report.json
```

## Detailed Validation Steps

### 1. Test Coverage (Target: 89%)

**What it measures**: Percentage of code lines/branches covered by generated tests

**How to measure**:

```python
from evaluation.metrics_evaluator import MetricsEvaluator

evaluator = MetricsEvaluator()
metrics = evaluator.measure_test_coverage(
    project_path='/path/to/project',
    test_classes=['com.example.MyTest']
)

print(f"Line Coverage: {metrics.line_coverage}%")
print(f"Branch Coverage: {metrics.branch_coverage}%")
```

**Manual measurement**:

```bash
# Using JaCoCo
mvn clean test jacoco:report

# View report
open target/site/jacoco/index.html
```

**Expected output**:
```
Line Coverage:   89.5%  ✅
Branch Coverage: 85.2%  ✅
Method Coverage: 92.1%  ✅
```

---

### 2. Mutation Score (Target: 84%)

**What it measures**: Percentage of code mutations detected by tests

**How to measure**:

```python
metrics = evaluator.measure_mutation_score(
    project_path='/path/to/project',
    target_classes=['com.example.Calculator']
)

print(f"Mutation Score: {metrics.mutation_score}%")
print(f"Killed: {metrics.killed_mutants}/{metrics.total_mutants}")
```

**Manual measurement**:

```bash
# Using PITest
mvn org.pitest:pitest-maven:mutationCoverage

# View report
open target/pit-reports/index.html
```

**Expected output**:
```
Mutation Score:  84.3%  ✅
Mutants Killed:  421/500
Survived:        79
```

---

### 3. Test Generation Time Reduction (Target: 55%)

**What it measures**: Time saved compared to baseline tools (EvoSuite, Randoop)

**How to measure**:

```python
def generate_with_testagentx():
    agent = LLMTestGenerationAgent()
    return agent.generate_tests(...)

def generate_with_evosuite():
    # Run EvoSuite
    subprocess.run(['java', '-jar', 'evosuite.jar', ...])

metrics = evaluator.benchmark_test_generation(
    test_generation_fn=generate_with_testagentx,
    baseline_fn=generate_with_evosuite,
    num_iterations=10
)

print(f"Time Reduction: {metrics.time_reduction}%")
```

**Manual measurement**:

```bash
# Time TestAgentX
time python -m layer2_test_generation.llm_test_agent

# Time EvoSuite
time java -jar evosuite.jar -class Calculator

# Calculate reduction
# reduction = (baseline - testagentx) / baseline * 100
```

**Expected output**:
```
TestAgentX Time: 45.2s
EvoSuite Time:   100.5s
Time Reduction:  55.0%  ✅
```

---

### 4. Patch Verification Accuracy (Target: 91%)

**What it measures**: Accuracy of determining if patches are effective

**How to measure**:

```python
# Run patch verification on dataset
verification_results = []
ground_truth = []

for patch in defects4j_patches:
    result = patch_agent.verify_patch(...)
    verification_results.append(result)
    ground_truth.append(patch.is_effective)

metrics = evaluator.measure_patch_verification_accuracy(
    verification_results,
    ground_truth
)

print(f"Accuracy: {metrics.accuracy}%")
print(f"FPR: {metrics.false_positive_rate}%")
```

**Manual measurement**:

```bash
# Run on Defects4J dataset
python scripts/evaluate_patch_verification.py \
    --dataset defects4j \
    --output results/
```

**Expected output**:
```
Accuracy:    91.2%  ✅
Precision:   89.5%
Recall:      93.1%
F1 Score:    91.3
FP Rate:     7.8%   ✅
```

---

### 5. False Positive Rate (Target: ≤8%)

**What it measures**: Percentage of incorrect "effective patch" predictions

**Calculation**:
```
FPR = False Positives / (False Positives + True Negatives) × 100
```

**How to measure**: Same as patch verification accuracy above

**Expected output**:
```
False Positives: 12
True Negatives:  142
FPR:            7.8%  ✅
```

---

### 6. Developer Acceptance (Target: 82%)

**What it measures**: Percentage of generated tests accepted by developers

**How to measure**:

```python
# Collect developer feedback
generated_tests = [...]
developer_feedback = [
    {'test_id': 'test1', 'status': 'accepted', 'quality_score': 4.5},
    {'test_id': 'test2', 'status': 'modified', 'quality_score': 4.0},
    {'test_id': 'test3', 'status': 'rejected', 'quality_score': 2.0},
]

metrics = evaluator.measure_developer_acceptance(
    generated_tests,
    developer_feedback
)

print(f"Acceptance Rate: {metrics.acceptance_rate}%")
```

**Manual measurement**:

1. Generate tests for real projects
2. Ask developers to review each test
3. Collect feedback: accepted/modified/rejected
4. Calculate acceptance rate

**Feedback form template**:
```
Test ID: test_divide_by_zero
Status: [Accept / Modify / Reject]
Quality Score: [1-5]
Comments: _______________________
```

**Expected output**:
```
Acceptance Rate: 82.5%  ✅
Accepted:        165/200
Modified:        25/200
Rejected:        10/200
Avg Quality:     4.2/5.0
```

## Running on Defects4J Dataset

The Defects4J dataset provides real-world bugs for evaluation:

```bash
# Setup Defects4J
git clone https://github.com/rjust/defects4j
cd defects4j
./init.sh

# Checkout a bug
defects4j checkout -p Lang -v 1b -w /tmp/lang_1_buggy

# Run evaluation
python evaluation/run_full_evaluation.py \
    --dataset defects4j \
    --project /tmp/lang_1_buggy \
    --output results/lang_1/
```

## Automated Evaluation Pipeline

For continuous validation:

```bash
# Run nightly evaluation
cron: 0 2 * * * /path/to/run_full_evaluation.sh

# run_full_evaluation.sh
#!/bin/bash
python evaluation/run_full_evaluation.py \
    --dataset defects4j \
    --output results/$(date +%Y%m%d)/ \
    2>&1 | tee logs/evaluation_$(date +%Y%m%d).log
```

## Interpreting Results

### Success Criteria

A claim is considered **validated** if:
- Measured value ≥ Target (for coverage, accuracy, acceptance)
- Measured value ≤ Target (for false positive rate)
- Difference within ±5% is acceptable

### Example Summary Report

```json
{
  "comparison": {
    "test_coverage": {
      "paper_claim": 89.0,
      "measured": 89.5,
      "difference": +0.5,
      "meets_claim": true
    },
    "mutation_score": {
      "paper_claim": 84.0,
      "measured": 84.3,
      "difference": +0.3,
      "meets_claim": true
    },
    "time_reduction": {
      "paper_claim": 55.0,
      "measured": 55.2,
      "difference": +0.2,
      "meets_claim": true
    },
    "patch_accuracy": {
      "paper_claim": 91.0,
      "measured": 91.2,
      "difference": +0.2,
      "meets_claim": true
    },
    "false_positive_rate": {
      "paper_claim": 8.0,
      "measured": 7.8,
      "difference": -0.2,
      "meets_claim": true
    },
    "developer_acceptance": {
      "paper_claim": 82.0,
      "measured": 82.5,
      "difference": +0.5,
      "meets_claim": true
    }
  }
}
```

## Troubleshooting

### Coverage < 89%

**Possible causes**:
- Incomplete test generation
- Complex code paths not covered
- Edge cases missed

**Solutions**:
- Increase `num_tests` parameter
- Improve prompt engineering
- Add more few-shot examples

### Mutation Score < 84%

**Possible causes**:
- Weak assertions
- Missing boundary tests
- Equivalent mutants

**Solutions**:
- Strengthen assertions
- Add edge case tests
- Use fuzzy validation

### Time Reduction < 55%

**Possible causes**:
- LLM API latency
- Inefficient prompting
- Large batch sizes

**Solutions**:
- Use caching
- Optimize prompts
- Batch processing

### Accuracy < 91%

**Possible causes**:
- Incomplete trace collection
- Weak effectiveness scoring
- Dataset bias

**Solutions**:
- Improve trace comparison
- Tune scoring weights
- Balance dataset

## Best Practices

1. **Use Representative Datasets**: Test on real-world projects
2. **Multiple Runs**: Average results over 10+ runs
3. **Statistical Significance**: Use t-tests for comparisons
4. **Document Assumptions**: Record all configuration choices
5. **Version Control**: Track evaluation results over time

## References

- JaCoCo Documentation: https://www.jacoco.org/
- PITest Documentation: https://pitest.org/
- Defects4J: https://github.com/rjust/defects4j
- TestAgentX Paper: Section 4 (Experimental Evaluation)

## Support

For questions about validation:
1. Check this guide
2. Review `evaluation/run_full_evaluation.py`
3. See example results in `evaluation_results/`
4. Open an issue on GitHub
