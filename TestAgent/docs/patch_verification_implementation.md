# Patch Verification Implementation

## Overview

This document describes the complete implementation of Layer 4: Patch Verification in TestAgentX, which implements **Equation (8)** from the paper:

```
Δ_trace = Trace(P_f, t_j) - Trace(P_b, t_j)
```

Where:
- `P_f` = Fixed/patched version of code
- `P_b` = Buggy version of code
- `t_j` = Test case j
- `Δ_trace` = Difference in execution traces

## Architecture

### Components

1. **ExecutionTrace**: Data structure capturing test execution details
   - Method calls made during execution
   - Line coverage information
   - Branch decisions (taken/not taken)
   - Exceptions raised

2. **PatchVerificationAgent**: Main agent for patch verification
   - Collects traces from both versions
   - Compares traces to assess patch effectiveness
   - Measures performance impact

3. **JaCoCo Integration**: Code coverage analysis
   - Binary `.exec` file generation
   - XML report conversion
   - Line and branch coverage extraction

## Implementation Details

### 1. Execution Trace Collection

The agent runs tests on both buggy and patched versions, collecting:

```python
trace = ExecutionTrace(
    method_calls=[...],      # Methods called during execution
    line_coverage=[...],     # Lines executed
    branch_decisions=[...],  # Branch paths taken
    exceptions=[...]         # Exceptions raised
)
```

### 2. JaCoCo Integration

#### Setup

Run the setup script to download JaCoCo:

```bash
bash scripts/setup_jacoco.sh
```

This downloads:
- `jacocoagent.jar` - Java agent for instrumentation
- `jacococli.jar` - Command-line tool for report generation

#### Coverage Collection

The agent:
1. Runs tests with JaCoCo agent attached
2. Generates binary `.exec` files
3. Converts to XML using JaCoCo CLI
4. Parses XML to extract coverage data

```python
# Run test with JaCoCo
cmd = [
    "java",
    f"-javaagent:{jacoco_agent}=destfile={output_file}",
    "-cp", classpath,
    "org.junit.runner.JUnitCore",
    f"{test_class}#{test_method}"
]
```

#### XML Report Parsing

The implementation parses JaCoCo XML reports to extract:

```xml
<line nr="42" mi="0" ci="5" mb="0" cb="2"/>
```

Where:
- `nr` = Line number
- `ci` = Covered instructions
- `mi` = Missed instructions
- `cb` = Covered branches
- `mb` = Missed branches

### 3. Method Call Extraction

Extracts method calls from stack traces using regex patterns:

```python
pattern = r'at\s+([\w\.]+)\(([\w\.]+):(\d+)\)'
```

Filters out framework methods (JUnit, reflection) to focus on application code.

### 4. Branch Decision Tracking

Converts branch coverage data into decision tuples:

```python
decisions = [
    (line_num, True),   # Branch taken
    (line_num, False),  # Branch not taken
]
```

### 5. Exception Extraction

Identifies exceptions in test output:

```python
pattern = r'([\w\.]+Exception|[\w\.]+Error):\s*(.+?)(?=\n|$)'
```

### 6. Trace Comparison

Compares traces between versions:

```python
diff = {
    'line_coverage': {
        'added': [...],    # New lines covered
        'removed': [...],  # Lines no longer covered
        'common': [...]    # Lines covered in both
    },
    'branch_coverage': {
        'added': [...],    # New branches covered
        'removed': [...],  # Branches no longer covered
        'changed': [...]   # Branches with different coverage
    },
    'method_calls': {
        'added': [...],    # New methods called
        'removed': [...]   # Methods no longer called
    }
}
```

### 7. Effectiveness Scoring

Calculates patch effectiveness (0.0 to 1.0):

```python
score = 0.0

# Reward for increased coverage
if new_lines_covered:
    score += 0.3
if new_branches_covered:
    score += 0.3

# Penalize performance regression
if execution_time_increase > threshold:
    score -= 0.1
if memory_increase > threshold:
    score -= 0.1

effectiveness = max(0.0, min(1.0, score))
```

## Usage Example

```python
from layer4_patch_regression.patch_verification_agent import PatchVerificationAgent

# Initialize agent
agent = PatchVerificationAgent(
    epsilon=0.1,
    config={
        'jacoco_agent': 'lib/jacocoagent.jar',
        'jacoco_cli': 'lib/jacococli.jar',
        'timeout_seconds': 300,
        'memory_limit_mb': 4096
    }
)

# Verify a patch
result = agent.verify_patch(
    project_path='/path/to/project',
    test_cases=[
        {
            'id': 'test_1',
            'class_name': 'com.example.CalculatorTest',
            'method_name': 'testDivideByZero'
        }
    ],
    patch_file='fix.patch',
    buggy_version_path='/path/to/buggy'
)

# Check results
print(f"Patch is effective: {result.is_effective}")
print(f"Effectiveness score: {result.effectiveness_score:.2f}")
print(f"Trace differences: {result.trace_differences}")
```

## Configuration

### Required Files

1. **JaCoCo Agent** (`lib/jacocoagent.jar`)
   - Java agent for bytecode instrumentation
   - Collects coverage during test execution

2. **JaCoCo CLI** (`lib/jacococli.jar`)
   - Converts binary `.exec` to XML reports
   - Provides detailed coverage analysis

### Environment Variables

```bash
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk
export JACOCO_AGENT=lib/jacocoagent.jar
export JACOCO_CLI=lib/jacococli.jar
```

### Agent Configuration

```python
config = {
    'jacoco_agent': 'lib/jacocoagent.jar',
    'jacoco_cli': 'lib/jacococli.jar',
    'java_home': '/usr/lib/jvm/default-java',
    'timeout_seconds': 300,
    'memory_limit_mb': 4096
}
```

## Performance Considerations

### Memory Usage

- Measured using `psutil` library
- Tracks RSS (Resident Set Size) before/after test execution
- Reports delta in MB

### Execution Time

- Measured using `time.time()`
- Includes test execution and coverage collection
- Reported in seconds

### Optimization Tips

1. **Batch Processing**: Process multiple tests in parallel
2. **Caching**: Cache parsed coverage reports
3. **Incremental Analysis**: Only analyze changed files
4. **Timeout Management**: Set appropriate timeouts for long-running tests

## Troubleshooting

### JaCoCo Not Found

```bash
# Run setup script
bash scripts/setup_jacoco.sh

# Verify installation
ls -l lib/jacoco*.jar
```

### Coverage Report Empty

Check:
1. JaCoCo agent is properly attached
2. Class files are in expected location
3. Source files are accessible
4. XML report generation succeeded

### Method Calls Not Extracted

Ensure:
1. Test output includes stack traces
2. Logging is enabled for method calls
3. Regex patterns match your output format

## Testing

Run the test suite:

```bash
pytest tests/test_patch_verification.py -v
```

## Future Enhancements

1. **Dynamic Instrumentation**: Use Java agents for real-time trace collection
2. **Distributed Tracing**: Support for microservices architectures
3. **Machine Learning**: Learn optimal effectiveness thresholds
4. **Visualization**: Generate trace diff visualizations
5. **Integration**: Connect with CI/CD pipelines

## References

- [JaCoCo Documentation](https://www.jacoco.org/jacoco/trunk/doc/)
- TestAgentX Paper: Section 3.5 (Patch Verification)
- Equation (8): Trace-based patch effectiveness measurement
