# Quick Start: Patch Verification

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Setup JaCoCo

```bash
bash scripts/setup_jacoco.sh
```

This downloads and installs:
- `lib/jacocoagent.jar`
- `lib/jacococli.jar`

## Basic Usage

### Simple Example

```python
from src.layer4_patch_regression.patch_verification_agent import PatchVerificationAgent

# Initialize agent
agent = PatchVerificationAgent(
    epsilon=0.1,
    config={
        'jacoco_agent': 'lib/jacocoagent.jar',
        'jacoco_cli': 'lib/jacococli.jar',
        'timeout_seconds': 300
    }
)

# Define test cases
test_cases = [
    {
        'id': 'test_1',
        'class_name': 'com.example.CalculatorTest',
        'method_name': 'testDivideByZero'
    }
]

# Verify patch
result = agent.verify_patch(
    project_path='/path/to/project',
    test_cases=test_cases,
    patch_file='fix.patch'
)

# Check results
if result.is_effective:
    print(f"✅ Patch is effective (score: {result.effectiveness_score:.2f})")
else:
    print(f"❌ Patch is not effective (score: {result.effectiveness_score:.2f})")

# View trace differences
print("\nTrace Differences:")
for test_id, diff_data in result.trace_differences.items():
    print(f"\nTest: {test_id}")
    diff = diff_data['differences']
    print(f"  Lines added: {len(diff['line_coverage']['added'])}")
    print(f"  Lines removed: {len(diff['line_coverage']['removed'])}")
    print(f"  Branches added: {len(diff['branch_coverage']['added'])}")
    print(f"  Methods added: {len(diff['method_calls']['added'])}")
```

## Advanced Usage

### With Custom Configuration

```python
config = {
    'jacoco_agent': 'lib/jacocoagent.jar',
    'jacoco_cli': 'lib/jacococli.jar',
    'java_home': '/usr/lib/jvm/java-11-openjdk',
    'timeout_seconds': 600,
    'memory_limit_mb': 8192
}

agent = PatchVerificationAgent(epsilon=0.15, config=config)
```

### Using Patch Content Directly

```python
patch_content = """
diff --git a/Calculator.java b/Calculator.java
index 1234567..89abcde 100644
--- a/Calculator.java
+++ b/Calculator.java
@@ -10,6 +10,9 @@ public class Calculator {
     }
     
     public int divide(int a, int b) {
+        if (b == 0) {
+            throw new IllegalArgumentException("Division by zero");
+        }
         return a / b;
     }
"""

result = agent.verify_patch(
    project_path='/path/to/project',
    test_cases=test_cases,
    patch_content=patch_content
)
```

### Comparing Specific Versions

```python
result = agent.verify_patch(
    project_path='/path/to/patched',
    test_cases=test_cases,
    buggy_version_path='/path/to/buggy'
)
```

## Understanding Results

### PatchVerificationResult Fields

```python
result.is_effective          # bool: Is patch effective?
result.effectiveness_score   # float: Score 0.0-1.0
result.trace_differences     # dict: Detailed differences
result.execution_time        # float: Total execution time (seconds)
result.memory_usage          # float: Peak memory usage (MB)
```

### Trace Differences Structure

```python
{
    'test_id': {
        'differences': {
            'line_coverage': {
                'added': [6, 7, 8],      # New lines covered
                'removed': [4, 5],       # Lines no longer covered
                'common': [1, 2, 3]      # Lines covered in both
            },
            'branch_coverage': {
                'added': ['2-4'],        # New branches covered
                'removed': ['2-3'],      # Branches no longer covered
                'changed': ['1-2: 0.5 -> 0.8']  # Changed coverage
            },
            'method_calls': {
                'added': ['methodC'],    # New methods called
                'removed': ['methodB']   # Methods no longer called
            },
            'execution_time_diff': 0.1,  # Time difference (seconds)
            'memory_usage_diff': 5.0     # Memory difference (MB)
        },
        'effectiveness_score': 0.85
    }
}
```

## Effectiveness Scoring

The effectiveness score is calculated based on:

| Factor | Weight | Impact |
|--------|--------|--------|
| New lines covered | +0.3 | Positive |
| New branches covered | +0.3 | Positive |
| Execution time increase > 1s | -0.1 | Negative |
| Memory increase > 10MB | -0.1 | Negative |

**Interpretation**:
- **0.8 - 1.0**: Highly effective patch
- **0.6 - 0.8**: Moderately effective patch
- **0.4 - 0.6**: Marginally effective patch
- **0.0 - 0.4**: Ineffective patch

## Troubleshooting

### JaCoCo Not Found

```bash
# Re-run setup
bash scripts/setup_jacoco.sh

# Verify installation
ls -l lib/jacoco*.jar
```

### Empty Coverage Reports

Check:
1. JaCoCo agent path is correct
2. Java project is compiled (`mvn compile`)
3. Test classes are in `target/test-classes`
4. Source files are in `src/main/java`

### Tests Timeout

Increase timeout in config:
```python
config = {'timeout_seconds': 900}  # 15 minutes
```

### Memory Issues

Increase memory limit:
```python
config = {'memory_limit_mb': 8192}  # 8GB
```

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: Patch Verification

on: [pull_request]

jobs:
  verify:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          bash scripts/setup_jacoco.sh
      
      - name: Verify patch
        run: |
          python scripts/verify_pr_patch.py
```

## Best Practices

1. **Run on Representative Tests**: Use tests that cover the patched code
2. **Set Appropriate Timeouts**: Balance thoroughness with speed
3. **Monitor Resource Usage**: Track memory and execution time
4. **Cache JaCoCo Reports**: Reuse reports when possible
5. **Analyze Failures**: Investigate low effectiveness scores
6. **Automate in CI/CD**: Integrate into your development workflow

## Examples

See `examples/patch_verification_example.py` for complete working examples.

## Documentation

- Full documentation: `docs/patch_verification_implementation.md`
- Implementation details: `PATCH_VERIFICATION_FIXES.md`
- API reference: `docs/api_reference.md`

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review the full documentation
3. Run tests: `pytest tests/test_patch_verification.py -v`
4. Check logs for detailed error messages
