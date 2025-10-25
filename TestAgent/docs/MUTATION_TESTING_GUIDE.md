# Mutation Testing Guide for TestAgentX

## Overview

This module implements mutation testing to evaluate test suite quality as described in **Section 4.2** of the TestAgentX paper: *"Mutation testing is used to assess the quality of generated test suites."*

Mutation testing introduces small changes (mutations) to source code and checks if tests detect these changes. A high mutation score indicates a strong test suite.

## Key Concepts

### What is Mutation Testing?

Mutation testing evaluates test quality by:
1. **Creating mutants**: Making small changes to source code
2. **Running tests**: Executing test suite against each mutant
3. **Checking results**: Determining if tests catch the mutation
4. **Calculating score**: Percentage of mutants killed by tests

### Mutation Score

```
Mutation Score = (Killed Mutants / Total Valid Mutants) × 100%
```

**Target**: 80%+ mutation score indicates high-quality tests

## Components

### 1. Mutation Operators (`mutation_operators.py`)

**9 Types of Operators**:

| Operator | Code | Description | Example |
|----------|------|-------------|---------|
| AOR | Arithmetic | Replace +, -, *, / | `a + b` → `a - b` |
| ROR | Relational | Replace <, >, ==, != | `x < 10` → `x <= 10` |
| COR | Conditional | Replace &&, \|\| | `a && b` → `a \|\| b` |
| UOI | Unary Insert | Insert negation | `if (x)` → `if (!x)` |
| SDL | Statement Delete | Remove statements | `x++;` → `// x++;` |
| CRP | Constant Replace | Change constants | `return 0` → `return 1` |
| RVR | Return Replace | Change returns | `return true` → `return false` |
| NMR | Null Replace | Return null | `return obj` → `return null` |
| BVA | Boundary Adjust | Adjust boundaries | `i < n` → `i <= n` |

### 2. Mutation Engine (`mutation_engine.py`)

Generates and executes mutants:
- Automatic mutant generation
- Test execution
- Result collection
- PITest integration

### 3. Mutation Analyzer (`mutation_analyzer.py`)

Analyzes results:
- Identifies weak spots
- Operator effectiveness
- Per-file scores
- Recommendations

### 4. Test Quality Assessor (`test_quality_assessor.py`)

Assesses test quality:
- Overall quality score
- Letter grade (A-F)
- Improvement recommendations
- Suite comparison

## Quick Start

### Basic Mutation Testing

```python
from mutation_testing import MutationEngine

# Initialize engine
engine = MutationEngine()

# Run mutation testing
result = engine.run_mutation_testing(
    source_file='Calculator.java',
    test_command='mvn test',
    language='java',
    project_path='/path/to/project'
)

print(f"Mutation Score: {result.mutation_score:.2f}%")
print(f"Killed: {result.killed_mutants}/{result.total_mutants}")
```

### Using PITest (Java)

```python
# Run PITest for Java projects
result = engine.run_pitest(
    project_path='/path/to/maven/project',
    target_classes=['com.example.Calculator'],
    test_classes=['com.example.CalculatorTest']
)

print(f"Mutation Score: {result.mutation_score:.2f}%")
```

### Analyze Results

```python
from mutation_testing import MutationAnalyzer

analyzer = MutationAnalyzer()

# Analyze mutation results
report = analyzer.analyze(result)

print(f"Weak Spots: {len(report.weak_spots)}")
for spot in report.weak_spots[:3]:
    print(f"  {spot['file']}:{spot['line']} - {spot['survived_mutants']} survived")

# Generate HTML report
analyzer.generate_html_report(report, 'mutation_report.html')
```

### Assess Test Quality

```python
from mutation_testing import TestQualityAssessor

assessor = TestQualityAssessor()

# Assess quality
metrics = assessor.assess_quality(result, code_coverage=85.0)

print(f"Grade: {metrics.grade}")
print(f"Overall Score: {metrics.overall_quality_score:.1f}%")
print(f"Mutation Score: {metrics.mutation_score:.1f}%")
print(f"Test Effectiveness: {metrics.test_effectiveness:.1f}%")

# Get recommendations
recommendations = assessor.recommend_improvements(metrics, report)
for rec in recommendations:
    print(f"  - {rec}")
```

## Integration with TestAgentX

### Assess Generated Tests

```python
from layer2_test_generation.llm_test_agent import LLMTestGenerationAgent
from mutation_testing import TestQualityAssessor

# Generate tests
agent = LLMTestGenerationAgent()
tests = agent.generate_tests(
    method_signature="public int add(int a, int b)",
    method_source="...",
    num_tests=5
)

# Assess quality using mutation testing
assessor = TestQualityAssessor()
metrics = assessor.assess_generated_tests(
    generated_tests=tests,
    source_file='Calculator.java',
    test_command='mvn test',
    language='java'
)

print(f"Generated tests quality: {metrics.grade}")
```

### Compare Test Suites

```python
# Compare TestAgentX vs baseline
testagentx_result = engine.run_mutation_testing(...)
baseline_result = engine.run_mutation_testing(...)

comparison = assessor.compare_test_suites(
    suite1_result=baseline_result,
    suite2_result=testagentx_result
)

print(f"Winner: {comparison['winner']}")
print(f"Score difference: {comparison['score_difference']:.1f}%")
print(comparison['summary'])
```

## Advanced Usage

### Custom Mutation Operators

```python
from mutation_testing.mutation_operators import MutationOperator, MutationOperatorType

def custom_replacement(line: str) -> str:
    # Custom mutation logic
    return line.replace('public', 'private')

custom_operator = MutationOperator(
    operator_type=MutationOperatorType.SDL,
    name="Visibility Mutation",
    description="Change method visibility",
    language="java",
    pattern=r'public',
    replacement_fn=custom_replacement
)
```

### Selective Mutation

```python
# Generate mutants for specific operators only
from mutation_testing.mutation_operators import MutationOperatorType, get_operator_by_type

operators = [
    get_operator_by_type(MutationOperatorType.AOR, 'java'),
    get_operator_by_type(MutationOperatorType.ROR, 'java')
]

# Use only these operators
for operator in operators:
    mutated_code = operator.apply(source_code, line_number)
```

### Parallel Execution

```python
import concurrent.futures

def test_mutant(mutant):
    # Test single mutant
    return engine.run_mutation_testing(...)

# Test mutants in parallel
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(test_mutant, mutants))
```

## Best Practices

### 1. Start with High-Value Code

Focus mutation testing on critical code:
```python
# Test core business logic first
result = engine.run_mutation_testing(
    source_file='BusinessLogic.java',
    test_command='mvn test',
    language='java'
)
```

### 2. Set Realistic Targets

| Code Type | Target Score |
|-----------|--------------|
| Critical Business Logic | 90%+ |
| Core Functionality | 80%+ |
| Utility Code | 70%+ |
| UI Code | 60%+ |

### 3. Iterate on Weak Spots

```python
# Identify weak spots
report = analyzer.analyze(result)

# Add tests for weak spots
for spot in report.weak_spots:
    print(f"Add tests for {spot['file']}:{spot['line']}")
    # Generate targeted tests
    
# Re-run mutation testing
new_result = engine.run_mutation_testing(...)
```

### 4. Monitor Trends

```python
# Track mutation score over time
scores = []
for commit in commits:
    result = engine.run_mutation_testing(...)
    scores.append({
        'commit': commit,
        'score': result.mutation_score,
        'date': datetime.now()
    })

# Plot trend
import matplotlib.pyplot as plt
plt.plot([s['date'] for s in scores], [s['score'] for s in scores])
plt.title('Mutation Score Trend')
plt.show()
```

### 5. Integrate into CI/CD

```yaml
# .github/workflows/mutation-testing.yml
name: Mutation Testing

on: [push, pull_request]

jobs:
  mutation-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run Mutation Testing
        run: |
          python -m mutation_testing.run_mutation_tests
      - name: Check Score
        run: |
          if [ $(cat mutation_score.txt) -lt 80 ]; then
            echo "Mutation score below 80%"
            exit 1
          fi
```

## Interpreting Results

### Mutation Score

- **90-100%**: Excellent test suite
- **80-89%**: Good test suite
- **70-79%**: Adequate test suite
- **60-69%**: Needs improvement
- **<60%**: Poor test suite

### Mutant Status

| Status | Meaning | Action |
|--------|---------|--------|
| KILLED | Test detected mutation | ✅ Good |
| SURVIVED | Test missed mutation | ❌ Add test |
| TIMEOUT | Test took too long | ⚠️ Optimize |
| ERROR | Test execution failed | ⚠️ Fix test |

### Weak Spots

Weak spots indicate areas needing more tests:

```python
# Example weak spot
{
    'file': 'Calculator.java',
    'line': 42,
    'survived_mutants': 3,
    'operators': ['AOR', 'ROR', 'BVA'],
    'severity': 'HIGH'
}
```

**Action**: Add tests specifically for line 42 covering arithmetic, relational, and boundary conditions.

## Common Issues

### Low Mutation Score

**Problem**: Mutation score < 60%

**Solutions**:
1. Add more test cases
2. Strengthen assertions
3. Test edge cases
4. Add negative tests

### High Timeout Rate

**Problem**: Many mutants timeout

**Solutions**:
1. Increase timeout limit
2. Optimize test execution
3. Use test parallelization
4. Skip slow tests for mutation

### Equivalent Mutants

**Problem**: Mutants that don't change behavior

**Solutions**:
1. Manually identify equivalent mutants
2. Exclude from score calculation
3. Use advanced mutation operators

## Performance Optimization

### 1. Selective Testing

```python
# Test only changed files
changed_files = get_changed_files()
for file in changed_files:
    result = engine.run_mutation_testing(file, ...)
```

### 2. Incremental Mutation

```python
# Only test new mutants
previous_mutants = load_previous_mutants()
new_mutants = [m for m in mutants if m not in previous_mutants]
```

### 3. Test Prioritization

```python
# Run fast tests first
fast_tests = ['UnitTest1', 'UnitTest2']
result = engine.run_mutation_testing(
    test_command=f'mvn test -Dtest={",".join(fast_tests)}'
)
```

## Example: Complete Workflow

```python
#!/usr/bin/env python3
"""
Complete mutation testing workflow
"""

from mutation_testing import (
    MutationEngine,
    MutationAnalyzer,
    TestQualityAssessor
)

def main():
    # 1. Run mutation testing
    engine = MutationEngine()
    result = engine.run_mutation_testing(
        source_file='Calculator.java',
        test_command='mvn test',
        language='java',
        project_path='/path/to/project'
    )
    
    # 2. Analyze results
    analyzer = MutationAnalyzer()
    report = analyzer.analyze(result)
    
    # 3. Assess quality
    assessor = TestQualityAssessor()
    metrics = assessor.assess_quality(result, code_coverage=85.0)
    
    # 4. Generate reports
    analyzer.generate_html_report(report, 'mutation_report.html')
    assessor.generate_quality_report(metrics, 'quality_report.txt')
    
    # 5. Print summary
    print("\n" + "="*60)
    print("MUTATION TESTING SUMMARY")
    print("="*60)
    print(f"Mutation Score:  {result.mutation_score:.1f}%")
    print(f"Quality Grade:   {metrics.grade}")
    print(f"Killed Mutants:  {result.killed_mutants}/{result.total_mutants}")
    print(f"Weak Spots:      {len(report.weak_spots)}")
    print("\nTop Recommendations:")
    for rec in report.recommendations[:5]:
        print(f"  - {rec}")
    
    # 6. Exit with appropriate code
    if metrics.grade in ['A', 'B']:
        return 0
    elif metrics.grade == 'C':
        print("\n⚠️  Warning: Test quality needs improvement")
        return 0
    else:
        print("\n❌ Error: Test quality below acceptable threshold")
        return 1

if __name__ == "__main__":
    exit(main())
```

## References

- TestAgentX Paper: Section 4.2 (Mutation Testing)
- PITest: https://pitest.org/
- Mutation Testing: https://en.wikipedia.org/wiki/Mutation_testing
- MutPy (Python): https://github.com/mutpy/mutpy

## Support

For questions about mutation testing:
1. Review this guide
2. Check example scripts in `examples/mutation_testing/`
3. See test results in `mutation_results/`
4. Open an issue on GitHub
